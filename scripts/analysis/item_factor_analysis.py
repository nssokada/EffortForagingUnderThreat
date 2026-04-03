"""
Item-level factor analysis on mental health questionnaire data.

Loads item-level responses from Stage 4, cleans them, and runs
exploratory factor analysis (EFA). Use this for clinical regressions
that need factor scores rather than summed scale totals.

Usage:
  python scripts/analysis/item_factor_analysis.py --stage4_dir <path>
  python scripts/analysis/item_factor_analysis.py  # auto-detect latest
"""

import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from factor_analyzer import FactorAnalyzer
from factor_analyzer.factor_analyzer import calculate_bartlett_sphericity, calculate_kmo


def find_stage4_dir():
    """Auto-detect latest stage4 output."""
    base = Path("data/exploratory_350/processed")
    candidates = sorted(base.glob("stage4_*"))
    if candidates:
        return candidates[-1]
    # Fall back: item data might be in stage5 dir
    candidates = sorted(base.glob("stage5_*"))
    return candidates[-1] if candidates else None


def load_item_data(stage4_dir):
    """Load item-level mental health data from Stage 4."""
    stage4_dir = Path(stage4_dir)
    item_path = stage4_dir / "mental_health_items_wide.csv"
    if not item_path.exists():
        item_path = stage4_dir / "mental_health_items_wide.pkl"
        if item_path.exists():
            return pd.read_pickle(item_path)
        raise FileNotFoundError(f"No item-level data found in {stage4_dir}")
    return pd.read_csv(item_path)


def clean_items(df):
    """Clean item-level data: identify questionnaire items, check ranges, drop STICSA."""
    id_col = "participantID" if "participantID" in df.columns else "subj"

    # Identify item columns (not ID, not total scores)
    item_cols = [c for c in df.columns if c != id_col and "_" in c
                 and not c.endswith("_total") and not c.endswith("_mean")]

    # Exclude STICSA items
    item_cols = [c for c in item_cols if not c.upper().startswith("STICSA")]

    # Scale maxima for range validation
    SCALE_MAX = {
        "AMI": 4, "DASS21": 3, "MFIS": 4,
        "OASIS": 4, "PHQ9": 3, "STAI": 4,
    }

    item_df = df[[id_col] + item_cols].copy()

    # Clip to valid ranges
    for col in item_cols:
        prefix = col.split("_")[0].upper()
        max_val = SCALE_MAX.get(prefix)
        if max_val:
            item_df[col] = item_df[col].clip(0, max_val)

    # Drop participants with >20% missing items
    n_missing = item_df[item_cols].isnull().sum(axis=1)
    threshold = 0.2 * len(item_cols)
    item_df = item_df[n_missing <= threshold].copy()

    # Impute remaining NaNs with item median
    for col in item_cols:
        item_df[col] = item_df[col].fillna(item_df[col].median())

    print(f"Items: {len(item_cols)} (excl. STICSA)")
    print(f"Participants: {len(item_df)} (after missingness filter)")
    return item_df, item_cols, id_col


def run_efa(item_df, item_cols, n_factors=None, max_factors=8):
    """Run exploratory factor analysis with parallel analysis for n_factors."""
    X = item_df[item_cols].values

    # KMO and Bartlett's test
    chi2, p = calculate_bartlett_sphericity(X)
    kmo_all, kmo_model = calculate_kmo(X)
    print(f"\nBartlett's test: χ²={chi2:.0f}, p={p:.2e}")
    print(f"KMO: {kmo_model:.3f}")

    if kmo_model < 0.5:
        print("WARNING: KMO < 0.5, factor analysis may not be appropriate")

    # Parallel analysis to determine n_factors
    if n_factors is None:
        fa_test = FactorAnalyzer(rotation=None, n_factors=min(max_factors, len(item_cols)))
        fa_test.fit(X)
        eigenvalues = fa_test.get_eigenvalues()[0]

        # Random eigenvalues (95th percentile from shuffled data)
        n_iter = 100
        random_eigs = np.zeros((n_iter, len(eigenvalues)))
        for i in range(n_iter):
            X_rand = np.column_stack([np.random.permutation(X[:, j]) for j in range(X.shape[1])])
            fa_rand = FactorAnalyzer(rotation=None, n_factors=min(max_factors, len(item_cols)))
            fa_rand.fit(X_rand)
            random_eigs[i] = fa_rand.get_eigenvalues()[0]

        random_95 = np.percentile(random_eigs, 95, axis=0)
        n_factors = int(np.sum(eigenvalues > random_95))
        n_factors = max(1, min(n_factors, max_factors))

        print(f"\nParallel analysis suggests {n_factors} factors")
        print(f"  Eigenvalues: {eigenvalues[:max_factors].round(2)}")
        print(f"  95th rand:   {random_95[:max_factors].round(2)}")

    # Fit final model
    fa = FactorAnalyzer(rotation="promax", n_factors=n_factors)
    fa.fit(X)

    # Loadings
    loadings = pd.DataFrame(
        fa.loadings_, index=item_cols,
        columns=[f"Factor{i+1}" for i in range(n_factors)]
    )

    # Factor scores
    scores = fa.transform(X)
    scores_df = item_df[["participantID" if "participantID" in item_df.columns else "subj"]].copy()
    for i in range(n_factors):
        scores_df[f"Factor{i+1}"] = scores[:, i]

    # Variance explained
    var_explained = fa.get_factor_variance()
    print(f"\nVariance explained: {var_explained[1].round(3)} (proportion)")
    print(f"Cumulative: {var_explained[2].round(3)}")

    return loadings, scores_df, fa


def main():
    parser = argparse.ArgumentParser(description="Item-level factor analysis")
    parser.add_argument("--stage4_dir", type=str, default=None)
    parser.add_argument("--n_factors", type=int, default=None,
                        help="Number of factors (default: parallel analysis)")
    parser.add_argument("--output_dir", type=str, default="results/stats/factor_analysis")
    args = parser.parse_args()

    stage4_dir = args.stage4_dir or find_stage4_dir()
    if stage4_dir is None:
        print("ERROR: No stage4 output found. Run preprocessing first.")
        return

    print(f"Loading items from: {stage4_dir}")
    df = load_item_data(stage4_dir)
    item_df, item_cols, id_col = clean_items(df)

    loadings, scores_df, fa = run_efa(item_df, item_cols, n_factors=args.n_factors)

    # Save
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    loadings.to_csv(out / "factor_loadings.csv")
    scores_df.to_csv(out / "factor_scores.csv", index=False)
    print(f"\nSaved to {out}/")


if __name__ == "__main__":
    main()
