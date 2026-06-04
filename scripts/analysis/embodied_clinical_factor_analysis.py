"""
HiTOP-style factor analysis of the clinical symptom panel, then regress
factor scores on (ω, κ).

The single-subscale analyses in result_604 found at most 1–2 significant
hits out of 51 tests. Anxiety and depression scales are highly correlated
(typically r > 0.5 in non-clinical samples), so the marginal regressions
dilute any specific signal across redundant outcomes. This script tests
whether the latent dimensional structure of the symptom panel reveals
(ω, κ) effects the subscale analyses miss.

Steps:
  1. Build subject × subscale matrix (pooled N = 571, 14 subscales)
  2. Parallel analysis to determine the number of factors
  3. EFA with oblimin rotation (correlated factors)
  4. Optional bifactor structure (general distress + specific factors)
  5. Regress factor scores on omega_z + kappa_z + omega_z:kappa_z
  6. Compare to subscale-level results

Outputs:
  results/stats/clinical/factor_loadings.csv
  results/stats/clinical/factor_scores.csv
  results/stats/clinical/factor_param_regressions.csv
  results/stats/clinical/parallel_analysis.csv
"""

from __future__ import annotations

import os
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
os.chdir(REPO_ROOT)
NB_DIR = REPO_ROOT / "notebooks" / "analysis"
sys.path.insert(0, str(NB_DIR))

import numpy as np
import pandas as pd
import bambi as bmb
import arviz as az
from scipy.stats import zscore
from sklearn.decomposition import FactorAnalysis
from sklearn.preprocessing import StandardScaler

from config import BKW  # type: ignore


# Use only true subscales (not totals — to avoid colinearity with components)
SUBSCALES = [
    "DASS21_Anxiety", "DASS21_Depression", "DASS21_Stress",
    "PHQ9_Total",  # PHQ9 only has Total available
    "OASIS_Total",  # OASIS only has Total available
    "STAI_Trait", "STAI_State",
    "STICSA_Total",  # STICSA only has Total
    "AMI_Behavioural", "AMI_Social", "AMI_Emotional",
    "MFIS_Physical", "MFIS_Cognitive", "MFIS_Psychosocial",
]


def load_pooled():
    rows = []
    for sample, paths in {
        "exploratory": {
            "psych": "data/exploratory_350/processed/stage5_filtered_data_20260403_133425/psych.csv",
            "m4_params": "results/stats/joint_optimal/exploratory/mcmc_m4_params.csv",
        },
        "confirmatory": {
            "psych": "data/confirmatory_350/processed/stage5_filtered_data_20260403_142413/psych.csv",
            "m4_params": "results/stats/joint_optimal/confirmatory/mcmc_m4_params.csv",
        },
    }.items():
        psych = pd.read_csv(paths["psych"])
        m4 = pd.read_csv(paths["m4_params"])
        df = psych.merge(m4, on="subj", how="inner")
        df["sample"] = sample
        rows.append(df)
    pooled = pd.concat(rows, ignore_index=True)
    pooled["log_omega"] = np.log(pooled["omega"])
    pooled["log_kappa"] = np.log(pooled["kappa"])
    pooled["omega_z"] = zscore(pooled["log_omega"].values, nan_policy="omit")
    pooled["kappa_z"] = zscore(pooled["log_kappa"].values, nan_policy="omit")
    return pooled


def parallel_analysis(X, n_iter=500, percentile=95, seed=42):
    """Horn's parallel analysis — compare observed eigenvalues to random
    eigenvalues. Number of factors = count of observed eigvals above the
    percentile-th random eigenvalue."""
    rng = np.random.default_rng(seed)
    n, p = X.shape
    # Observed eigenvalues of correlation matrix
    obs_eigs = np.sort(np.linalg.eigvalsh(np.corrcoef(X.T)))[::-1]
    # Random eigenvalues from shuffled data
    rand_eigs = np.zeros((n_iter, p))
    for i in range(n_iter):
        X_rand = np.empty_like(X)
        for j in range(p):
            X_rand[:, j] = rng.permutation(X[:, j])
        rand_eigs[i] = np.sort(np.linalg.eigvalsh(np.corrcoef(X_rand.T)))[::-1]
    rand_pct = np.percentile(rand_eigs, percentile, axis=0)
    n_factors = int(np.sum(obs_eigs > rand_pct))
    return obs_eigs, rand_pct, n_factors


def main():
    print("=" * 70)
    print("HiTOP-style factor analysis of clinical symptom panel")
    print("=" * 70)
    pooled = load_pooled()
    print(f"Pooled N before NA-drop: {len(pooled)}")

    # Build the symptom matrix
    avail = [s for s in SUBSCALES if s in pooled.columns]
    print(f"Available subscales: {len(avail)} of {len(SUBSCALES)} requested")
    print(f"  {avail}")

    X_df = pooled[avail + ["subj", "sample", "omega_z", "kappa_z"]].dropna().copy()
    X = X_df[avail].values
    print(f"After NA-drop: N = {len(X_df)}, {X.shape[1]} subscales")

    # Step 1: parallel analysis
    print("\n--- STEP 1: Parallel analysis ---")
    obs, rand_pct, n_fact = parallel_analysis(X, n_iter=500, percentile=95)
    print(f"  Observed eigenvalues (top 8): {np.round(obs[:8], 3)}")
    print(f"  Random 95th percentile  (top 8): {np.round(rand_pct[:8], 3)}")
    print(f"  Number of factors recommended: {n_fact}")

    # Save parallel-analysis results
    pa_df = pd.DataFrame({
        "factor": np.arange(1, len(obs) + 1),
        "observed_eig": obs,
        "random_95pct_eig": rand_pct,
        "retain": obs > rand_pct,
    })
    out_dir = REPO_ROOT / "results" / "stats" / "clinical"
    out_dir.mkdir(parents=True, exist_ok=True)
    pa_df.to_csv(out_dir / "parallel_analysis.csv", index=False)

    # Step 2: EFA with varimax rotation (sklearn — orthogonal)
    print(f"\n--- STEP 2: EFA with varimax rotation, n_factors = {n_fact} ---")
    # Standardise X first (FactorAnalysis assumes unit variance)
    scaler = StandardScaler()
    X_std = scaler.fit_transform(X)
    fa = FactorAnalysis(n_components=n_fact, rotation="varimax", random_state=42)
    fa.fit(X_std)
    loadings = pd.DataFrame(
        fa.components_.T,  # sklearn returns (n_components, n_features); we want (n_features, n_components)
        index=avail,
        columns=[f"F{i+1}" for i in range(n_fact)],
    )
    print("\nLoadings (after varimax rotation):")
    print(loadings.round(3).to_string())

    # Step 3: factor scores per subject
    print("\n--- STEP 3: Computing factor scores per subject ---")
    scores = fa.transform(X_std)
    scores_df = pd.DataFrame(scores, index=X_df.index, columns=loadings.columns)
    scores_df["subj"] = X_df["subj"].values
    scores_df["sample"] = X_df["sample"].values
    scores_df["omega_z"] = X_df["omega_z"].values
    scores_df["kappa_z"] = X_df["kappa_z"].values

    # Save loadings and scores
    loadings.to_csv(out_dir / "factor_loadings.csv")
    scores_df.to_csv(out_dir / "factor_scores.csv", index=False)
    print(f"  Saved loadings: {out_dir / 'factor_loadings.csv'}")
    print(f"  Saved scores: {out_dir / 'factor_scores.csv'}")

    # Step 4: regress factor scores on (omega_z, kappa_z, omega_z:kappa_z)
    print("\n--- STEP 4: Factor scores ~ omega_z + kappa_z + omega_z:kappa_z ---")
    records = []
    for fac in loadings.columns:
        df = scores_df[[fac, "omega_z", "kappa_z"]].dropna()
        # z-score the factor score (already centered but may not be unit-SD)
        df[f"{fac}_z"] = zscore(df[fac].values, nan_policy="omit")
        print(f"\n  Factor {fac} (n = {len(df)}):")
        m = bmb.Model(f"{fac}_z ~ omega_z + kappa_z + omega_z:kappa_z", data=df)
        s = az.summary(m.fit(**BKW), hdi_prob=0.95)
        for term in ["omega_z", "kappa_z", "omega_z:kappa_z"]:
            if term not in s.index:
                continue
            beta = s.loc[term, "mean"]
            lo = s.loc[term, "hdi_2.5%"]
            hi = s.loc[term, "hdi_97.5%"]
            sig = (lo > 0) or (hi < 0)
            star = "★" if sig else " "
            print(f"    {term:20s} β = {beta:+.4f} [{lo:+.4f}, {hi:+.4f}] {star}")
            records.append({
                "factor": fac, "term": term, "n": len(df),
                "beta": beta, "hdi_lo": lo, "hdi_hi": hi, "sig": sig,
            })
    pd.DataFrame(records).to_csv(out_dir / "factor_param_regressions.csv", index=False)

    # Step 5: Print loadings interpretation
    print("\n" + "=" * 70)
    print("INTERPRETATION — which subscales load on which factors?")
    print("=" * 70)
    # For each factor, sort loadings descending and show the strongest items
    for fac in loadings.columns:
        ranked = loadings[fac].abs().sort_values(ascending=False)
        print(f"\n{fac} — top-loading subscales (|loading| sorted):")
        for sub in ranked.head(6).index:
            ld = loadings.loc[sub, fac]
            print(f"  {sub:25s} {ld:+.3f}")

    # Step 6: headline summary
    print("\n" + "=" * 70)
    print("HEADLINE — significant (ω, κ) effects on factor scores")
    print("=" * 70)
    rec_df = pd.DataFrame(records)
    sig = rec_df[rec_df["sig"]]
    if len(sig) == 0:
        print("  NO significant (ω, κ) effects on any factor score.")
    else:
        print(sig.to_string(index=False))


if __name__ == "__main__":
    main()
