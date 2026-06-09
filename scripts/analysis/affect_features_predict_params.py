"""
Affect features as metacognitive substrate of (ω, κ).

For each subject we have:
  - ω, κ from M4 model fits (computational parameters)
  - Per-trial anxiety and confidence ratings on probe trials
  - Per-subject regression slopes of anxiety/confidence on (T, D, cookie reward)

Question: do affect's structural responses to specific task features predict
the fitted computational parameters? That is, are ω and κ metacognitively
substantiated by HOW subjects' anxiety and confidence respond to threat,
distance, and reward in the task?

Regressions (within each sample):
  ω_z ~ anx_slope_T + anx_slope_D + anx_slope_reward
        + conf_slope_T + conf_slope_D + conf_slope_reward
        + anx_intercept + conf_intercept

  κ_z ~ same predictors

A finding "replicates" if hits at p < 0.05 in both samples with same sign.

Output:
  results/stats/affect_analysis/affect_features_predict_params.csv
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


def per_subject_affect_slopes_on_features():
    """Per-subject regression slope of affect on (T, D, cookie reward) within question.

    For each subject and each question (anxiety, confidence), fits:
        response ~ threat + distance + trialCookie_rewardValue
    Returns DataFrame with one row per (subj, sample, question) and columns
    for intercept and slopes on each predictor.
    """
    rows = []
    for sample, path in SAMPLES.items():
        f = pd.read_csv(Path(path) / "feelings.csv", low_memory=False)
        for q in ["anxiety", "confidence"]:
            sub_q = f[f["questionLabel"] == q].dropna(subset=["response"]).copy()
            for subj, g in sub_q.groupby("subj"):
                if len(g) < 5:
                    continue
                preds = []
                for c in ["threat", "distance", "trialCookie_rewardValue"]:
                    if c in g.columns and g[c].nunique() > 1:
                        preds.append(c)
                if not preds:
                    continue
                X = sm.add_constant(g[preds].values)
                try:
                    res = sm.OLS(g["response"].values, X).fit()
                    row = {
                        "subj": int(subj),
                        "sample": sample,
                        "question": q,
                        "n_obs": int(len(g)),
                        "intercept": float(res.params[0]),
                    }
                    for i, p in enumerate(preds):
                        row[f"slope_{p}"] = float(res.params[i + 1])
                    rows.append(row)
                except Exception:
                    pass
    return pd.DataFrame(rows)


def build_master():
    exp, conf = load_both()
    rows = []
    for sample, d in [("exploratory", exp), ("confirmatory", conf)]:
        m = d["master"].reset_index().rename(columns={"index": "subj"}).copy()
        m["sample"] = sample
        rows.append(m[["subj", "sample", "omega", "kappa"]])
    master = pd.concat(rows, ignore_index=True)
    # Z-score log parameters within each sample so within-sample regressions are clean
    master["omega_z"] = np.nan
    master["kappa_z"] = np.nan
    for s in master["sample"].unique():
        mask = master["sample"] == s
        master.loc[mask, "omega_z"] = zscore(np.log(master.loc[mask, "omega"]).values)
        master.loc[mask, "kappa_z"] = zscore(np.log(master.loc[mask, "kappa"]).values)
    return master


def widen_affect(affect_long):
    """Convert long per-(subj, sample, question) affect data to wide per-(subj, sample).

    Columns: anxiety_intercept, anxiety_slope_threat, anxiety_slope_distance,
             anxiety_slope_reward, confidence_intercept, confidence_slope_threat,
             confidence_slope_distance, confidence_slope_reward.
    """
    out = affect_long.pivot_table(
        index=["subj", "sample"],
        columns="question",
        values=["intercept", "slope_threat", "slope_distance", "slope_trialCookie_rewardValue"],
    )
    # Flatten MultiIndex columns: (value, question) -> "{question}_{value}"
    out.columns = [f"{q}_{v}" for v, q in out.columns]
    out = out.reset_index()
    # Normalize column names
    rename_map = {}
    for c in out.columns:
        if c in ("subj", "sample"):
            continue
        rename_map[c] = (
            c.replace("intercept", "intercept")
            .replace("slope_threat", "slope_T")
            .replace("slope_distance", "slope_D")
            .replace("slope_trialCookie_rewardValue", "slope_reward")
        )
    out = out.rename(columns=rename_map)
    return out


PREDICTORS = [
    "anxiety_slope_T", "anxiety_slope_D", "anxiety_slope_reward",
    "confidence_slope_T", "confidence_slope_D", "confidence_slope_reward",
    "anxiety_intercept", "confidence_intercept",
]


def fit_within_sample(df, outcome, sample_name):
    sub = df[df["sample"] == sample_name].copy()
    cols = [outcome] + PREDICTORS
    sub = sub[cols].dropna().copy()
    if len(sub) < 30:
        return None, sub
    # z-score predictors within this sample
    for p in PREDICTORS:
        sub[p] = zscore(sub[p].values, nan_policy="omit")
    X = sm.add_constant(sub[PREDICTORS].values)
    res = sm.OLS(sub[outcome].values, X).fit()
    return res, sub


def main():
    print("=" * 78)
    print("AFFECT FEATURES PREDICT ω AND κ (metacognitive substrate analysis)")
    print("=" * 78)

    print("\nLoading per-subject (ω, κ)...")
    master = build_master()
    print(f"  Master: {len(master)} rows (exp {(master['sample']=='exploratory').sum()}, "
          f"conf {(master['sample']=='confirmatory').sum()})")
    print(f"  ω median (raw): {master['omega'].median():.3f}")
    print(f"  κ median (raw): {master['kappa'].median():.3f}")

    print("\nComputing per-subject affect slopes on (T, D, cookie reward)...")
    affect_long = per_subject_affect_slopes_on_features()
    affect_wide = widen_affect(affect_long)
    print(f"  Affect-feature rows: {len(affect_wide)}")

    # Sanity check: directional means of slopes
    print("\n[Sanity] mean slopes across pooled subjects:")
    for c in PREDICTORS:
        if c in affect_wide.columns:
            v = affect_wide[c].dropna()
            print(f"    {c:30s} mean = {v.mean():+.3f}   median = {v.median():+.3f}   n = {len(v)}")

    print("\nMerging master + affect on (subj, sample)...")
    df = master.merge(affect_wide, on=["subj", "sample"], how="inner")
    print(f"  Merged N: {len(df)}")
    print(f"  Per sample: exp {(df['sample']=='exploratory').sum()}, "
          f"conf {(df['sample']=='confirmatory').sum()}")

    # ── Run regressions ───────────────────────────────────────────────────
    rows = []
    for outcome in ["omega_z", "kappa_z"]:
        print("\n" + "=" * 78)
        print(f"OUTCOME: {outcome}")
        print("=" * 78)
        for sample in ["exploratory", "confirmatory"]:
            res, sub = fit_within_sample(df, outcome, sample)
            if res is None:
                print(f"\n  [{sample}] not enough data (N={len(sub)})")
                continue
            print(f"\n  --- {sample} (N = {len(sub)}) ---")
            print(f"      R² = {res.rsquared:.4f}    adj R² = {res.rsquared_adj:.4f}    "
                  f"F = {res.fvalue:.2f}, p = {res.f_pvalue:.4g}")
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
                    "outcome": outcome,
                    "sample": sample,
                    "N": len(sub),
                    "R2": res.rsquared,
                    "predictor": name,
                    "beta": float(beta),
                    "se": float(se),
                    "t": float(t),
                    "p": float(p),
                })

    # ── Save table ────────────────────────────────────────────────────────
    out_path = REPO_ROOT / "results" / "stats" / "affect_analysis" / "affect_features_predict_params.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out_path, index=False)
    print(f"\nSaved: {out_path}")

    # ── Replication summary ───────────────────────────────────────────────
    print("\n" + "=" * 78)
    print("REPLICATION SUMMARY — predictors hitting p < 0.05 in BOTH samples with same sign")
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
