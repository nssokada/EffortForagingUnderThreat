"""
Per-cookie-type affect slopes: fit response ~ T + D separately for heavy and
light probe trials, then test which features predict ω and κ.

For each subject × question (anxiety, confidence):
  - HEAVY probe trials only: fit response ~ T + D → intercept_H, slope_T_H, slope_D_H
  - LIGHT probe trials only: fit response ~ T + D → intercept_L, slope_T_L, slope_D_L

Yields 12 features per subject (3 per cookie × 2 cookies × 2 questions).

Second-level regression within each sample:
  ω_z ~ 12 features
  κ_z ~ 12 features

Output: results/stats/affect_analysis/affect_TD_by_cookie_predict_params.csv
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
import statsmodels.api as sm
from scipy.stats import zscore

from load_data import load_both  # type: ignore


SAMPLES = {
    "exploratory": "data/exploratory_350/processed/stage5_filtered_data_20260403_133425",
    "confirmatory": "data/confirmatory_350/processed/stage5_filtered_data_20260403_142413",
}

MIN_OBS = 4  # need at least 4 trials per (subj, question, cookie) to fit a 3-param regression


def per_subject_cookie_stratified_slopes():
    """Per-subject regression slopes of affect on (T, D), separately for heavy
    and light probe trials."""
    rows = []
    for sample, path in SAMPLES.items():
        f = pd.read_csv(Path(path) / "feelings.csv", low_memory=False)
        # Cookie weight: heavy = reward 5, light = reward 1
        f["cookie"] = np.where(f["trialCookie_rewardValue"] == 5.0, "heavy", "light")
        for q in ["anxiety", "confidence"]:
            for cookie in ["heavy", "light"]:
                sub_qc = f[(f["questionLabel"] == q) & (f["cookie"] == cookie)].dropna(subset=["response"])
                for subj, g in sub_qc.groupby("subj"):
                    if len(g) < MIN_OBS:
                        continue
                    preds = []
                    for c in ["threat", "distance"]:
                        if c in g.columns and g[c].nunique() > 1:
                            preds.append(c)
                    if len(preds) < 1:
                        continue
                    X = sm.add_constant(g[preds].values)
                    try:
                        res = sm.OLS(g["response"].values, X).fit()
                        row = {
                            "subj": int(subj), "sample": sample,
                            "question": q, "cookie": cookie, "n_obs": int(len(g)),
                            "intercept": float(res.params[0]),
                        }
                        for i, p in enumerate(preds):
                            row[f"slope_{p}"] = float(res.params[i + 1])
                        rows.append(row)
                    except Exception:
                        pass
    return pd.DataFrame(rows)


def widen_to_subject(long):
    """Pivot the long table to one row per (subj, sample) with all 12 features."""
    out = long.pivot_table(
        index=["subj", "sample"],
        columns=["question", "cookie"],
        values=["intercept", "slope_threat", "slope_distance"],
    )
    out.columns = [f"{q}_{c}_{kind}".replace("slope_threat", "slopeT").replace("slope_distance", "slopeD")
                   for kind, q, c in out.columns]
    out = out.reset_index()
    return out


def build_master():
    exp, conf = load_both()
    rows = []
    for sample, d in [("exploratory", exp), ("confirmatory", conf)]:
        m = d["master"].reset_index().rename(columns={"index": "subj"}).copy()
        m["sample"] = sample
        rows.append(m[["subj", "sample", "omega", "kappa"]])
    master = pd.concat(rows, ignore_index=True)
    master["omega_z"] = np.nan
    master["kappa_z"] = np.nan
    for s in master["sample"].unique():
        mask = master["sample"] == s
        master.loc[mask, "omega_z"] = zscore(np.log(master.loc[mask, "omega"]).values)
        master.loc[mask, "kappa_z"] = zscore(np.log(master.loc[mask, "kappa"]).values)
    return master


def fit(df, outcome, predictors, sample_name):
    sub = df[df["sample"] == sample_name].copy()
    sub = sub[[outcome] + predictors].dropna()
    if len(sub) < 30:
        return None
    for p in predictors:
        sub[p] = zscore(sub[p].values, nan_policy="omit")
    X = sm.add_constant(sub[predictors].values)
    return sm.OLS(sub[outcome].values, X).fit(), len(sub)


def main():
    print("=" * 78)
    print("AFFECT (T, D) by COOKIE TYPE → ω, κ")
    print("=" * 78)

    print("\nComputing per-subject, per-cookie-type affect slopes...")
    long = per_subject_cookie_stratified_slopes()
    print(f"  long rows: {len(long)}")
    wide = widen_to_subject(long)
    print(f"  wide rows: {len(wide)}")
    print(f"  columns: {[c for c in wide.columns if c not in ('subj', 'sample')]}")

    master = build_master()
    df = master.merge(wide, on=["subj", "sample"], how="inner")
    print(f"\nMerged N: {len(df)}")
    print(f"  exp: {(df['sample']=='exploratory').sum()}, conf: {(df['sample']=='confirmatory').sum()}")

    predictors = [c for c in wide.columns if c not in ("subj", "sample")]
    print(f"\nPredictors ({len(predictors)}):")
    for p in predictors:
        print(f"  {p}")

    rows = []
    for outcome in ["omega_z", "kappa_z"]:
        print("\n" + "=" * 78)
        print(f"OUTCOME: {outcome}")
        print("=" * 78)
        for sample in ["exploratory", "confirmatory"]:
            fit_res = fit(df, outcome, predictors, sample)
            if fit_res is None: continue
            res, N = fit_res
            print(f"\n  --- {sample} (N = {N}) ---")
            print(f"      R² = {res.rsquared:.4f}    adj R² = {res.rsquared_adj:.4f}    "
                  f"F = {res.fvalue:.2f}, model p = {res.f_pvalue:.4g}")
            for i, name in enumerate(predictors):
                beta = res.params[i + 1]
                p = res.pvalues[i + 1]
                sig = "★" if p < 0.05 else " "
                sig = "★★" if p < 0.01 else sig
                sig = "★★★" if p < 0.001 else sig
                print(f"      {name:35s} β={beta:+.3f}  p={p:.4g} {sig}")
                rows.append({
                    "outcome": outcome, "sample": sample, "N": N, "R2": res.rsquared,
                    "predictor": name, "beta": float(beta), "se": float(res.bse[i+1]),
                    "t": float(res.tvalues[i+1]), "p": float(p),
                })

    out_path = REPO_ROOT / "results" / "stats" / "affect_analysis" / "affect_TD_by_cookie_predict_params.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out_path, index=False)
    print(f"\nSaved: {out_path}")

    # Replication summary
    print("\n" + "=" * 78)
    print("REPLICATION SUMMARY")
    print("=" * 78)
    rdf = pd.DataFrame(rows)
    for outcome in ["omega_z", "kappa_z"]:
        print(f"\n  Outcome: {outcome}")
        for predictor in predictors:
            er = rdf[(rdf["outcome"] == outcome) & (rdf["sample"] == "exploratory") &
                    (rdf["predictor"] == predictor)]
            cr = rdf[(rdf["outcome"] == outcome) & (rdf["sample"] == "confirmatory") &
                    (rdf["predictor"] == predictor)]
            if len(er) == 0 or len(cr) == 0:
                continue
            eb, ep = er["beta"].iloc[0], er["p"].iloc[0]
            cb, cp = cr["beta"].iloc[0], cr["p"].iloc[0]
            replicates = (ep < 0.05) and (cp < 0.05) and (eb * cb > 0)
            tag = "★ REPLICATES" if replicates else ""
            print(f"    {predictor:35s}  exp β={eb:+.3f} p={ep:.4g}    "
                  f"conf β={cb:+.3f} p={cp:.4g}    {tag}")


if __name__ == "__main__":
    main()
