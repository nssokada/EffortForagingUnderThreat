"""
Simpler version of affect_features_predict_params: drop cookie reward.

For each subject, per-trial regression:
    response ~ threat + distance     (one per question: anxiety, confidence)
gives 3 features per question = 6 affect features total:
  anxiety_intercept, anxiety_slope_T, anxiety_slope_D,
  confidence_intercept, confidence_slope_T, confidence_slope_D

Second-level regression within each sample:
  ω_z ~ 6 affect features + sample
  κ_z ~ 6 affect features + sample

Replication = significant at p < 0.05 in both samples with same sign.

Slopes already computed in result_510 pipeline and stored in
results/stats/clinical/phenotype_metacog_slopes_subjects.csv.

Output: results/stats/affect_analysis/affect_TD_predict_params.csv
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


SLOPES_CSV = REPO_ROOT / "results" / "stats" / "clinical" / "phenotype_metacog_slopes_subjects.csv"
PREDICTORS = [
    "anxiety_slope_T", "anxiety_slope_D", "anxiety_intercept",
    "confidence_slope_T", "confidence_slope_D", "confidence_intercept",
]


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


def fit_within_sample(df, outcome, sample_name):
    sub = df[df["sample"] == sample_name].copy()
    sub = sub[[outcome] + PREDICTORS].dropna()
    if len(sub) < 30:
        return None, sub
    for p in PREDICTORS:
        sub[p] = zscore(sub[p].values, nan_policy="omit")
    X = sm.add_constant(sub[PREDICTORS].values)
    res = sm.OLS(sub[outcome].values, X).fit()
    return res, sub


def main():
    print("=" * 78)
    print("AFFECT (T, D only) → ω, κ — metacognitive substrate, no cookie reward")
    print("=" * 78)

    master = build_master()
    slopes = pd.read_csv(SLOPES_CSV)
    keep = ["subj", "sample"] + PREDICTORS
    slopes = slopes[keep]

    df = master.merge(slopes, on=["subj", "sample"], how="inner")
    print(f"\nMerged N: {len(df)}")
    print(f"  exp: {(df['sample']=='exploratory').sum()}, conf: {(df['sample']=='confirmatory').sum()}")

    # Sanity check
    print("\n[Sanity] Pooled mean of predictors:")
    for p in PREDICTORS:
        v = df[p].dropna()
        print(f"  {p:30s} mean = {v.mean():+.3f}    median = {v.median():+.3f}")

    rows = []
    for outcome in ["omega_z", "kappa_z"]:
        print("\n" + "=" * 78)
        print(f"OUTCOME: {outcome}")
        print("=" * 78)
        for sample in ["exploratory", "confirmatory"]:
            res, sub = fit_within_sample(df, outcome, sample)
            if res is None:
                continue
            print(f"\n  --- {sample} (N = {len(sub)}) ---")
            print(f"      R² = {res.rsquared:.4f}    adj R² = {res.rsquared_adj:.4f}    "
                  f"F = {res.fvalue:.2f}, model p = {res.f_pvalue:.4g}")
            for i, name in enumerate(PREDICTORS):
                beta = res.params[i + 1]
                se = res.bse[i + 1]
                t = res.tvalues[i + 1]
                p = res.pvalues[i + 1]
                sig = "★" if p < 0.05 else " "
                sig = "★★" if p < 0.01 else sig
                sig = "★★★" if p < 0.001 else sig
                print(f"      {name:30s} β={beta:+.3f}  SE={se:.3f}  t={t:+.2f}  p={p:.4g} {sig}")
                rows.append({
                    "outcome": outcome, "sample": sample, "N": len(sub), "R2": res.rsquared,
                    "predictor": name, "beta": float(beta), "se": float(se),
                    "t": float(t), "p": float(p),
                })

    out_path = REPO_ROOT / "results" / "stats" / "affect_analysis" / "affect_TD_predict_params.csv"
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
        sub = rdf[rdf["outcome"] == outcome]
        for predictor in PREDICTORS:
            exp_row = sub[(sub["sample"] == "exploratory") & (sub["predictor"] == predictor)]
            conf_row = sub[(sub["sample"] == "confirmatory") & (sub["predictor"] == predictor)]
            if len(exp_row) == 0 or len(conf_row) == 0:
                continue
            exp_beta = exp_row["beta"].iloc[0]
            conf_beta = conf_row["beta"].iloc[0]
            exp_p = exp_row["p"].iloc[0]
            conf_p = conf_row["p"].iloc[0]
            both_sig = (exp_p < 0.05) and (conf_p < 0.05)
            same_sign = (exp_beta * conf_beta) > 0
            replicates = both_sig and same_sign
            tag = "★ REPLICATES" if replicates else ""
            print(f"    {predictor:30s}  exp β={exp_beta:+.3f} p={exp_p:.4g}    "
                  f"conf β={conf_beta:+.3f} p={conf_p:.4g}    {tag}")


if __name__ == "__main__":
    main()
