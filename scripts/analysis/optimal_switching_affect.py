"""
Are humans approximately optimal in switching choice + effort across conditions,
and do affect-reactivity profiles explain the deviations?

The proposed paper framing:
  - Optimal: switch to light cookie when dangerous (high T), heavy when safe (low T).
  - Optimal: press harder when dangerous, softer when safe.
  - Humans approximately optimal at group level — but deviate individually.
  - Deviation is driven by how anxiety/confidence respond to T, D, and reward (cookie weight).

This script tests:
  1. Group-level adaptive switching (P(heavy) by T; vigor by T)
  2. Per-subject optimality (pct_opt)
  3. Per-subject affect-reactivity slopes on T, D, and cookie reward
     - Do these predict pct_opt BEYOND ω, κ?
     - This is the user's core claim: affect reactivity explains residual deviation

Outputs: results/stats/joint_optimal/optimal_switching_affect.csv
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


SAMPLES = {
    "exploratory": "data/exploratory_350/processed/stage5_filtered_data_20260403_133425",
    "confirmatory": "data/confirmatory_350/processed/stage5_filtered_data_20260403_142413",
}


def per_subject_affect_slopes_on_reward():
    """Per-subject regression slope of affect on cookie reward, in addition to T, D."""
    rows = []
    for sample, path in SAMPLES.items():
        f = pd.read_csv(Path(path) / "feelings.csv", low_memory=False)
        for col in ["threat", "distance", "trialCookie_rewardValue"]:
            if col not in f.columns:
                continue
        # Per subject × question
        for q in ["anxiety", "confidence"]:
            sub_q = f[f["questionLabel"] == q].dropna(subset=["response"]).copy()
            for subj, g in sub_q.groupby("subj"):
                if len(g) < 5:
                    continue
                # Z-score predictors within question (not within subject — gives slope on natural scale)
                preds = []
                for c in ["threat", "distance", "trialCookie_rewardValue"]:
                    if c in g.columns and g[c].nunique() > 1:
                        preds.append(c)
                if not preds:
                    continue
                X = sm.add_constant(g[preds].values)
                try:
                    r = sm.OLS(g["response"].values, X).fit()
                    row = {"subj": int(subj), "sample": sample, "question": q,
                           "n_obs": int(len(g)), "intercept": float(r.params[0])}
                    for i, p in enumerate(preds):
                        row[f"slope_{p}"] = float(r.params[i+1])
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
        rows.append(m)
    m = pd.concat(rows, ignore_index=True)
    m["omega_z"] = zscore(np.log(m["omega"]).values)
    m["kappa_z"] = zscore(np.log(m["kappa"]).values)
    m["sample_dummy"] = (m["sample"] == "confirmatory").astype(int)
    return m


def per_subject_p_heavy_by_T(exp, conf):
    rows = []
    for sample, d in [("exploratory", exp), ("confirmatory", conf)]:
        v = d["vigor"][["subj", "T_round", "is_heavy"]].copy()
        v["sample"] = sample
        rows.append(v)
    beh = pd.concat(rows, ignore_index=True)
    g = beh.groupby(["subj", "sample", "T_round"])["is_heavy"].mean().reset_index()
    wide = g.pivot_table(index=["subj", "sample"], columns="T_round", values="is_heavy").reset_index()
    wide.columns = ["subj", "sample"] + [f"p_heavy_T{c}" for c in wide.columns[2:]]
    wide["p_heavy_slope_T"] = wide["p_heavy_T0.9"] - wide["p_heavy_T0.1"]
    return wide


def per_subject_vigor_by_T(exp, conf):
    rows = []
    for sample, d in [("exploratory", exp), ("confirmatory", conf)]:
        v = d["vigor"][["subj", "T_round", "norm_rate"]].copy()
        v["sample"] = sample
        rows.append(v)
    beh = pd.concat(rows, ignore_index=True)
    g = beh.groupby(["subj", "sample", "T_round"])["norm_rate"].mean().reset_index()
    wide = g.pivot_table(index=["subj", "sample"], columns="T_round", values="norm_rate").reset_index()
    wide.columns = ["subj", "sample"] + [f"vigor_T{c}" for c in wide.columns[2:]]
    wide["vigor_slope_T"] = wide["vigor_T0.9"] - wide["vigor_T0.1"]
    return wide


def fit_z(df, lhs, rhs):
    sub = df[[lhs] + rhs + ["sample_dummy"]].dropna().copy()
    if len(sub) < 30:
        return None
    sub[lhs + "_z"] = zscore(sub[lhs].values, nan_policy="omit")
    for r in rhs:
        sub[r] = zscore(sub[r].values, nan_policy="omit")
    X = sm.add_constant(sub[rhs + ["sample_dummy"]].values)
    return sm.OLS(sub[lhs + "_z"].values, X).fit()


def main():
    exp, conf = load_both()
    master = build_master()
    print(f"N master: {len(master)}")

    ph_wide = per_subject_p_heavy_by_T(exp, conf)
    v_wide = per_subject_vigor_by_T(exp, conf)
    master = master.merge(ph_wide, on=["subj", "sample"]).merge(v_wide, on=["subj", "sample"])

    # ── 1. ARE HUMANS APPROXIMATELY OPTIMAL? ─────────────────────────────
    print("\n" + "=" * 78)
    print("1. GROUP-LEVEL ADAPTIVE SWITCHING")
    print("=" * 78)
    print("\n  P(heavy) by threat level (group mean):")
    for col in ["p_heavy_T0.1", "p_heavy_T0.5", "p_heavy_T0.9"]:
        print(f"    {col:18s}  mean = {master[col].mean():.3f}  SD = {master[col].std():.3f}")
    print(f"\n  Vigor by threat level (group mean):")
    for col in ["vigor_T0.1", "vigor_T0.5", "vigor_T0.9"]:
        print(f"    {col:18s}  mean = {master[col].mean():.3f}  SD = {master[col].std():.3f}")
    print(f"\n  pct_opt distribution: mean = {master['pct_opt'].mean():.3f}  "
          f"median = {master['pct_opt'].median():.3f}  "
          f"SD = {master['pct_opt'].std():.3f}")
    print(f"  fraction above 0.5: {(master['pct_opt'] > 0.5).mean():.3f}")
    print(f"  fraction above 0.7: {(master['pct_opt'] > 0.7).mean():.3f}")

    # ── 2. PER-SUBJECT AFFECT SLOPES on (T, D, reward) ─────────────────
    print("\n" + "=" * 78)
    print("2. PER-SUBJECT AFFECT SLOPES on (T, D, cookie reward)")
    print("=" * 78)
    aff_slopes = per_subject_affect_slopes_on_reward()
    print(f"\n  affect-slope rows: {len(aff_slopes)}")
    print(f"  unique (subj, sample, question): {len(aff_slopes.drop_duplicates(['subj','sample','question']))}")
    # Pivot wide: one row per (subj, sample) with all slopes from both questions
    pivot = aff_slopes.pivot_table(
        index=["subj", "sample"],
        columns="question",
        values=[c for c in aff_slopes.columns if c.startswith("slope_") or c == "intercept"],
    )
    pivot.columns = [f"{q}_{c}" for c, q in pivot.columns]
    pivot = pivot.reset_index()
    master = master.merge(pivot, on=["subj", "sample"], how="left")

    # ── 3. DO AFFECT SLOPES PREDICT OPTIMALITY BEYOND ω, κ? ─────────────
    print("\n" + "=" * 78)
    print("3. AFFECT REACTIVITY → OPTIMALITY (beyond ω, κ)")
    print("=" * 78)

    # Base: pct_opt ~ ω + κ
    print("\n  --- Base: pct_opt ~ ω_z + κ_z + sample ---")
    res_base = fit_z(master, "pct_opt", ["omega_z", "kappa_z"])
    if res_base is not None:
        print(f"    R² = {res_base.rsquared:.4f}")
        for i, n in enumerate(["omega_z", "kappa_z"]):
            print(f"    {n:14s} β={res_base.params[i+1]:+.3f}  p={res_base.pvalues[i+1]:.4g}")

    # Add affect slopes
    affect_predictors = [c for c in master.columns
                          if c.startswith(("anxiety_slope_", "confidence_slope_", "anxiety_intercept", "confidence_intercept"))
                          and "T_round" not in c]
    # only keep ones that exist with enough data
    valid_aff = [c for c in affect_predictors if c in master.columns
                  and master[c].notna().sum() > 100]
    print(f"\n  Affect predictors available: {valid_aff}")

    # Full model: pct_opt ~ ω + κ + each affect predictor (one at a time first)
    print("\n  --- Each affect predictor ALONE controlling for ω, κ ---")
    rows = []
    for p in valid_aff:
        res = fit_z(master, "pct_opt", ["omega_z", "kappa_z", p])
        if res is None: continue
        sig = "★" if res.pvalues[3] < 0.05 else " "
        sig2 = "★★" if res.pvalues[3] < 0.01 else sig
        sig3 = "★★★" if res.pvalues[3] < 0.001 else sig2
        print(f"    pct_opt ~ ω + κ + {p:30s}  ω β={res.params[1]:+.3f} p={res.pvalues[1]:.4g}    "
              f"{p[:14]} β={res.params[3]:+.3f} p={res.pvalues[3]:.4g} {sig3}")
        rows.append({"predictor": p, "beta_omega": res.params[1], "p_omega": res.pvalues[1],
                     "beta_affect": res.params[3], "p_affect": res.pvalues[3], "R2": res.rsquared})

    # Full model: pct_opt ~ ω + κ + all anxiety slopes + all confidence slopes
    print("\n  --- ALL anxiety slopes together (controlling ω, κ) ---")
    anx_slopes = [p for p in valid_aff if p.startswith("anxiety_slope_")]
    if anx_slopes:
        res = fit_z(master, "pct_opt", ["omega_z", "kappa_z"] + anx_slopes)
        if res is not None:
            print(f"    R² = {res.rsquared:.4f}  (base R² = {res_base.rsquared:.4f})")
            print(f"    ΔR² from adding anxiety slopes: {res.rsquared - res_base.rsquared:+.4f}")
            for i, n in enumerate(["omega_z", "kappa_z"] + anx_slopes):
                print(f"      {n:28s} β={res.params[i+1]:+.3f}  p={res.pvalues[i+1]:.4g}")

    print("\n  --- ALL confidence slopes together (controlling ω, κ) ---")
    conf_slopes = [p for p in valid_aff if p.startswith("confidence_slope_")]
    if conf_slopes:
        res = fit_z(master, "pct_opt", ["omega_z", "kappa_z"] + conf_slopes)
        if res is not None:
            print(f"    R² = {res.rsquared:.4f}  (base R² = {res_base.rsquared:.4f})")
            print(f"    ΔR² from adding confidence slopes: {res.rsquared - res_base.rsquared:+.4f}")
            for i, n in enumerate(["omega_z", "kappa_z"] + conf_slopes):
                print(f"      {n:28s} β={res.params[i+1]:+.3f}  p={res.pvalues[i+1]:.4g}")

    # All affect predictors together
    print("\n  --- ALL affect slopes + intercepts together ---")
    res = fit_z(master, "pct_opt", ["omega_z", "kappa_z"] + valid_aff)
    if res is not None:
        print(f"    R² = {res.rsquared:.4f}  ΔR² vs base: {res.rsquared - res_base.rsquared:+.4f}")
        for i, n in enumerate(["omega_z", "kappa_z"] + valid_aff):
            sig = "★" if res.pvalues[i+1] < 0.05 else " "
            print(f"      {n:28s} β={res.params[i+1]:+.3f}  p={res.pvalues[i+1]:.4g} {sig}")

    # Save
    out_path = REPO_ROOT / "results" / "stats" / "joint_optimal" / "optimal_switching_affect.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out_path, index=False)
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
