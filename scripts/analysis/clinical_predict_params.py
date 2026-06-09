"""
Do clinical scales JOINTLY predict ω or κ?

User corrected: Test C should be ω ~ clinical scales (joint), not clinical ~ ω.
This asks: is variation in ω explained by mental-health profile?

Two regressions per sample:
  ω_z ~ {all clinical scales}
  κ_z ~ {all clinical scales}

Both within-sample replication (exploratory N=290 / confirmatory N=281).

Also report two variants:
  (a) clinical scales ONLY — raw association
  (b) clinical scales + affect_contrasts — does clinical add beyond affect?

Output: results/stats/affect_analysis/clinical_predict_params.csv
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
MIN_OBS = 4

CLINICAL_SCALES = [
    "DASS21_Anxiety", "DASS21_Depression", "DASS21_Stress",
    "OASIS_Total", "STICSA_Total",
    "AMI_Behavioural", "AMI_Social", "AMI_Emotional",
    "MFIS_Total", "PHQ9_Total",
]
CONTRASTS = [
    "anxiety_intercept_HvL", "anxiety_slopeT_HvL", "anxiety_slopeD_HvL",
    "confidence_intercept_HvL", "confidence_slopeT_HvL", "confidence_slopeD_HvL",
]


def per_subject_cookie_stratified():
    rows = []
    for sample, path in SAMPLES.items():
        f = pd.read_csv(Path(path) / "feelings.csv", low_memory=False)
        f["cookie"] = np.where(f["trialCookie_rewardValue"] == 5.0, "heavy", "light")
        for q in ["anxiety", "confidence"]:
            for cookie in ["heavy", "light"]:
                sub_qc = f[(f["questionLabel"] == q) & (f["cookie"] == cookie)].dropna(subset=["response"])
                for subj, g in sub_qc.groupby("subj"):
                    if len(g) < MIN_OBS: continue
                    preds = [c for c in ["threat", "distance"] if c in g.columns and g[c].nunique() > 1]
                    if len(preds) < 1: continue
                    X = sm.add_constant(g[preds].values)
                    try:
                        res = sm.OLS(g["response"].values, X).fit()
                        row = {"subj": int(subj), "sample": sample, "question": q,
                               "cookie": cookie, "intercept": float(res.params[0])}
                        for i, p in enumerate(preds):
                            row[f"slope_{p}"] = float(res.params[i + 1])
                        rows.append(row)
                    except Exception: pass
    return pd.DataFrame(rows)


def build_contrasts():
    long = per_subject_cookie_stratified()
    wide = long.pivot_table(index=["subj", "sample"], columns=["question", "cookie"],
                            values=["intercept", "slope_threat", "slope_distance"])
    wide.columns = [f"{q}_{c}_{kind}".replace("slope_threat", "slopeT").replace("slope_distance", "slopeD")
                    for kind, q, c in wide.columns]
    wide = wide.reset_index()
    for q in ["anxiety", "confidence"]:
        for feat in ["intercept", "slopeT", "slopeD"]:
            h, l = f"{q}_heavy_{feat}", f"{q}_light_{feat}"
            if h in wide.columns and l in wide.columns:
                wide[f"{q}_{feat}_HvL"] = wide[h] - wide[l]
    return wide


def build_master():
    exp, conf = load_both()
    rows = []
    for sample, d in [("exploratory", exp), ("confirmatory", conf)]:
        m = d["master"].reset_index().rename(columns={"index": "subj"}).copy()
        m["sample"] = sample
        rows.append(m)
    master = pd.concat(rows, ignore_index=True)
    master["omega_z"] = np.nan
    master["kappa_z"] = np.nan
    for s in master["sample"].unique():
        mask = master["sample"] == s
        master.loc[mask, "omega_z"] = zscore(np.log(master.loc[mask, "omega"]).values)
        master.loc[mask, "kappa_z"] = zscore(np.log(master.loc[mask, "kappa"]).values)
    return master


def fit(df, outcome_z, predictors, sample_name):
    sub = df[df["sample"] == sample_name].copy()
    sub = sub[[outcome_z] + predictors].dropna()
    if len(sub) < 30:
        return None
    for p in predictors:
        sub[p] = zscore(sub[p].values, nan_policy="omit")
    X = sm.add_constant(sub[predictors].values)
    return sm.OLS(sub[outcome_z].values, X).fit(), len(sub)


def main():
    print("=" * 78)
    print("Do CLINICAL SCALES jointly predict ω and κ?")
    print("=" * 78)

    master = build_master()
    contrasts = build_contrasts()
    df = master.merge(contrasts, on=["subj", "sample"], how="inner")
    print(f"\nMerged N: {len(df)} (exp {(df['sample']=='exploratory').sum()}, "
          f"conf {(df['sample']=='confirmatory').sum()})")

    rows = []
    for outcome in ["omega_z", "kappa_z"]:
        print("\n" + "=" * 78)
        print(f"OUTCOME: {outcome}")
        print("=" * 78)

        # Model 1: clinical scales only
        print("\n--- Model 1: clinical scales ONLY ---")
        for sample in ["exploratory", "confirmatory"]:
            res_pack = fit(df, outcome, CLINICAL_SCALES, sample)
            if res_pack is None: continue
            res, N = res_pack
            print(f"\n  [{sample}] N={N}  R²={res.rsquared:.4f}  "
                  f"adj R²={res.rsquared_adj:.4f}  F p={res.f_pvalue:.4g}")
            for i, name in enumerate(CLINICAL_SCALES):
                beta = res.params[i + 1]
                p = res.pvalues[i + 1]
                sig = "★" if p < 0.05 else " "
                sig = "★★" if p < 0.01 else sig
                sig = "★★★" if p < 0.001 else sig
                print(f"    {name:24s} β={beta:+.3f}  p={p:.4g} {sig}")
                rows.append({"outcome": outcome, "sample": sample, "model": "clinical_only",
                             "N": N, "R2": res.rsquared, "F_p": res.f_pvalue,
                             "predictor": name, "beta": float(beta), "p": float(p)})

        # Model 2: clinical scales + affect contrasts
        print("\n--- Model 2: clinical scales + affect contrasts ---")
        for sample in ["exploratory", "confirmatory"]:
            preds = CLINICAL_SCALES + CONTRASTS
            res_pack = fit(df, outcome, preds, sample)
            if res_pack is None: continue
            res, N = res_pack
            print(f"\n  [{sample}] N={N}  R²={res.rsquared:.4f}  "
                  f"adj R²={res.rsquared_adj:.4f}  F p={res.f_pvalue:.4g}")
            for i, name in enumerate(preds):
                beta = res.params[i + 1]
                p = res.pvalues[i + 1]
                if p < 0.1:
                    sig = "★" if p < 0.05 else " "
                    sig = "★★" if p < 0.01 else sig
                    sig = "★★★" if p < 0.001 else sig
                    print(f"    {name:32s} β={beta:+.3f}  p={p:.4g} {sig}")
                rows.append({"outcome": outcome, "sample": sample,
                             "model": "clinical_plus_affect",
                             "N": N, "R2": res.rsquared, "F_p": res.f_pvalue,
                             "predictor": name, "beta": float(beta), "p": float(p)})

    out_path = REPO_ROOT / "results" / "stats" / "affect_analysis" / "clinical_predict_params.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out_path, index=False)
    print(f"\nSaved: {out_path}")

    # Replication summary
    print("\n" + "=" * 78)
    print("REPLICATION SUMMARY (both samples p<0.05, same sign)")
    print("=" * 78)
    rdf = pd.DataFrame(rows)
    for outcome in ["omega_z", "kappa_z"]:
        for model_label in ["clinical_only", "clinical_plus_affect"]:
            sub = rdf[(rdf["outcome"] == outcome) & (rdf["model"] == model_label)]
            # Joint model F-test summary
            f_exp = sub[sub["sample"] == "exploratory"]["F_p"].iloc[0] if len(sub[sub["sample"] == "exploratory"]) else None
            f_conf = sub[sub["sample"] == "confirmatory"]["F_p"].iloc[0] if len(sub[sub["sample"] == "confirmatory"]) else None
            print(f"\n  [{outcome}] [{model_label}]  joint F-test p: exp={f_exp:.4g}  conf={f_conf:.4g}")
            for predictor in sub["predictor"].unique():
                er = sub[(sub["sample"] == "exploratory") & (sub["predictor"] == predictor)]
                cr = sub[(sub["sample"] == "confirmatory") & (sub["predictor"] == predictor)]
                if len(er) == 0 or len(cr) == 0: continue
                eb, ep = er["beta"].iloc[0], er["p"].iloc[0]
                cb, cp = cr["beta"].iloc[0], cr["p"].iloc[0]
                replicates = (ep < 0.05) and (cp < 0.05) and (eb * cb > 0)
                if replicates:
                    print(f"    {predictor:32s}  exp β={eb:+.3f} p={ep:.4g}    "
                          f"conf β={cb:+.3f} p={cp:.4g}    ★ REPLICATES")


if __name__ == "__main__":
    main()
