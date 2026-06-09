"""
Same as affect_TD_predict_params.py but run anxiety and confidence in
SEPARATE regressions instead of jointly. Reveals whether anxiety effects
are being masked by confidence in the joint model (and vice versa).

Three models per (outcome × sample):
  Model A: ω_z or κ_z ~ anxiety_intercept + anxiety_slope_T + anxiety_slope_D
  Model B: ω_z or κ_z ~ confidence_intercept + confidence_slope_T + confidence_slope_D
  Model C: joint (all 6, same as affect_TD_predict_params.py for comparison)

Output: results/stats/affect_analysis/affect_TD_predict_params_separate.csv
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
from scipy.stats import zscore, pearsonr

from load_data import load_both  # type: ignore


SLOPES_CSV = REPO_ROOT / "results" / "stats" / "clinical" / "phenotype_metacog_slopes_subjects.csv"
ANX_PREDS = ["anxiety_intercept", "anxiety_slope_T", "anxiety_slope_D"]
CONF_PREDS = ["confidence_intercept", "confidence_slope_T", "confidence_slope_D"]
ALL_PREDS = ANX_PREDS + CONF_PREDS


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
    print("AFFECT → ω, κ — SEPARATE anxiety vs confidence regressions")
    print("=" * 78)

    master = build_master()
    slopes = pd.read_csv(SLOPES_CSV)
    df = master.merge(slopes[["subj", "sample"] + ALL_PREDS], on=["subj", "sample"])
    print(f"\nMerged N: {len(df)} (exp {(df['sample']=='exploratory').sum()}, "
          f"conf {(df['sample']=='confirmatory').sum()})")

    # Inter-affect correlations
    print("\n[Sanity] anxiety × confidence correlations (pooled):")
    for ap in ANX_PREDS:
        for cp in CONF_PREDS:
            r, p = pearsonr(df[ap].dropna(), df[cp].dropna())
            print(f"  {ap:25s} × {cp:25s}  r = {r:+.3f}  p = {p:.4g}")

    rows = []
    for outcome in ["omega_z", "kappa_z"]:
        print("\n" + "=" * 78)
        print(f"OUTCOME: {outcome}")
        print("=" * 78)
        for sample in ["exploratory", "confirmatory"]:
            print(f"\n  ── {sample} ──")
            for model_label, preds in [("anxiety_only", ANX_PREDS),
                                        ("confidence_only", CONF_PREDS),
                                        ("joint", ALL_PREDS)]:
                fit_res = fit(df, outcome, preds, sample)
                if fit_res is None: continue
                res, N = fit_res
                print(f"\n    [{model_label}] N={N}  R² = {res.rsquared:.4f}  "
                      f"adj R² = {res.rsquared_adj:.4f}  F p = {res.f_pvalue:.4g}")
                for i, name in enumerate(preds):
                    beta = res.params[i + 1]
                    p = res.pvalues[i + 1]
                    sig = "★" if p < 0.05 else " "
                    sig = "★★" if p < 0.01 else sig
                    sig = "★★★" if p < 0.001 else sig
                    print(f"      {name:30s} β={beta:+.3f}  p={p:.4g} {sig}")
                    rows.append({
                        "outcome": outcome, "sample": sample, "model": model_label,
                        "predictor": name, "N": N, "R2": res.rsquared,
                        "beta": float(beta), "se": float(res.bse[i+1]),
                        "t": float(res.tvalues[i+1]), "p": float(p),
                    })

    out_path = REPO_ROOT / "results" / "stats" / "affect_analysis" / "affect_TD_predict_params_separate.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out_path, index=False)
    print(f"\nSaved: {out_path}")

    # Replication summary across the three model variants
    print("\n" + "=" * 78)
    print("REPLICATION SUMMARY (separate vs joint)")
    print("=" * 78)
    rdf = pd.DataFrame(rows)
    for outcome in ["omega_z", "kappa_z"]:
        print(f"\n  Outcome: {outcome}")
        for model_label in ["anxiety_only", "confidence_only", "joint"]:
            print(f"\n    [{model_label}]")
            for predictor in (ANX_PREDS if model_label == "anxiety_only" else
                               CONF_PREDS if model_label == "confidence_only" else ALL_PREDS):
                exp_row = rdf[(rdf["outcome"] == outcome) & (rdf["sample"] == "exploratory") &
                              (rdf["model"] == model_label) & (rdf["predictor"] == predictor)]
                conf_row = rdf[(rdf["outcome"] == outcome) & (rdf["sample"] == "confirmatory") &
                               (rdf["model"] == model_label) & (rdf["predictor"] == predictor)]
                if len(exp_row) == 0 or len(conf_row) == 0:
                    continue
                eb, ep = exp_row["beta"].iloc[0], exp_row["p"].iloc[0]
                cb, cp = conf_row["beta"].iloc[0], conf_row["p"].iloc[0]
                replicates = (ep < 0.05) and (cp < 0.05) and (eb * cb > 0)
                tag = "★ REPLICATES" if replicates else ""
                print(f"      {predictor:30s}  exp β={eb:+.3f} p={ep:.4g}    "
                      f"conf β={cb:+.3f} p={cp:.4g}    {tag}")


if __name__ == "__main__":
    main()
