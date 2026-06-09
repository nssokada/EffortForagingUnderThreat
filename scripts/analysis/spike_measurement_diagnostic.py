"""
Is the anxiety_slope_T → smaller reactive spike finding a real defensive
budget / front-loading effect, or a measurement artifact from baseline ceiling?

The original measure: spike_mag_peak = peak_strike_effort − mean_preEncounter_effort.
If anxiety-reactive subjects already have HIGH anticipatory baseline, they have less
headroom to surge — the smaller "spike" could be mechanical.

We disambiguate with 5 alternative spike measures:

  1. peak_strike_effort (ABSOLUTE peak, no baseline subtraction)
     If anxiety_slope_T still negatively predicts → real, not artifact
     If anxiety_slope_T positive or null → was an artifact

  2. peak_strike_effort / mean_preEncounter_effort (ratio, accounts for proportional rise)
     Tests multiplicative spike

  3. peak_strike_effort / calibrationMax (proportion of subject's max capacity)
     Normalizes by individual capacity ceiling

  4. peak_strike_effort − peak_preEncounter_effort (peak-to-peak, not peak-to-mean)
     Cleaner difference if anticipatory variability matters

  5. spike_mag_peak controlling for baseline as covariate
     If anxiety_slope_T effect disappears with baseline controlled → mediated by baseline (ceiling-like)
     If anxiety_slope_T effect persists → independent of baseline

Output: results/stats/joint_optimal/spike_measurement_diagnostic.csv
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


def per_subject_spike_measures():
    exp, conf = load_both()
    rows = []
    for sample, d in [("exploratory", exp), ("confirmatory", conf)]:
        beh = d["beh"][[
            "subj", "T_round", "isAttackTrial",
            "mean_preEncounter_effort", "peak_preEncounter_effort",
            "mean_strike_effort", "peak_strike_effort",
            "mean_postEncounter_effort", "peak_postEncounter_effort",
            "calibrationMax",
        ]].copy()
        beh["sample"] = sample
        beh = beh.dropna(subset=["mean_preEncounter_effort"])
        for subj, g in beh.groupby("subj"):
            att = g[g["isAttackTrial"] == 1]
            if len(att) < 5:
                continue
            row = {"subj": int(subj), "sample": sample, "n_attack": len(att)}

            # Anticipatory baseline (the potential confound)
            row["pre_mean"] = float(att["mean_preEncounter_effort"].mean())
            row["pre_peak"] = float(att["peak_preEncounter_effort"].mean())

            # 1. Absolute peak strike effort
            row["abs_peak_strike"] = float(att["peak_strike_effort"].mean())

            # Original spike measure (peak − pre mean)
            row["spike_mag_peak"] = float(
                (att["peak_strike_effort"] - att["mean_preEncounter_effort"]).mean()
            )

            # 2. Ratio peak / pre mean
            row["spike_ratio_mean"] = float(
                (att["peak_strike_effort"] / att["mean_preEncounter_effort"].clip(lower=0.01)).mean()
            )

            # 3. Proportion of calibration max
            calmax = att["calibrationMax"].iloc[0]
            if calmax > 0:
                row["abs_peak_norm"] = float(att["peak_strike_effort"].mean() / calmax)
                row["calibrationMax"] = float(calmax)

            # 4. Peak-to-peak (peak strike − peak pre)
            row["spike_peak_to_peak"] = float(
                (att["peak_strike_effort"] - att["peak_preEncounter_effort"]).mean()
            )

            rows.append(row)
    return pd.DataFrame(rows)


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


def fit_z(df, outcome, predictors, sample_name):
    sub = df[df["sample"] == sample_name].copy()
    sub = sub[[outcome] + predictors].dropna()
    if len(sub) < 30:
        return None
    sub[outcome + "_z"] = zscore(sub[outcome].values, nan_policy="omit")
    for p in predictors:
        sub[p] = zscore(sub[p].values, nan_policy="omit")
    X = sm.add_constant(sub[predictors].values)
    return sm.OLS(sub[outcome + "_z"].values, X).fit(), len(sub)


def main():
    print("=" * 78)
    print("SPIKE MEASUREMENT DIAGNOSTIC — artifact or real?")
    print("=" * 78)

    spike = per_subject_spike_measures()
    master = build_master()
    slopes = pd.read_csv(SLOPES_CSV)
    affect_cols = ["anxiety_intercept", "anxiety_slope_T", "anxiety_slope_D",
                   "confidence_intercept", "confidence_slope_T", "confidence_slope_D"]
    affect_cols = [c for c in affect_cols if c in slopes.columns]
    df = master.merge(spike, on=["subj", "sample"]).merge(
        slopes[["subj", "sample"] + affect_cols], on=["subj", "sample"], how="left")
    print(f"\nN: {len(df)} (exp {(df['sample']=='exploratory').sum()}, conf {(df['sample']=='confirmatory').sum()})")

    print("\n[Sanity] correlations between baseline (pre_mean) and spike measures (pooled):")
    for outcome in ["abs_peak_strike", "spike_mag_peak", "spike_ratio_mean", "abs_peak_norm", "spike_peak_to_peak"]:
        if outcome in df.columns:
            sub = df[[outcome, "pre_mean"]].dropna()
            r = np.corrcoef(sub[outcome], sub["pre_mean"])[0, 1]
            print(f"  {outcome:22s} × pre_mean: r = {r:+.3f}")

    rows = []

    # ── For each spike measure, test anxiety_slope_T effect ──────────────
    outcomes = ["abs_peak_strike", "spike_mag_peak", "spike_ratio_mean",
                "abs_peak_norm", "spike_peak_to_peak"]
    for outcome in outcomes:
        if outcome not in df.columns: continue
        print("\n" + "=" * 78)
        print(f"OUTCOME: {outcome}")
        print("=" * 78)

        for sample in ["exploratory", "confirmatory"]:
            res = fit_z(df, outcome, ["omega_z", "kappa_z", "anxiety_slope_T",
                                       "confidence_slope_D"], sample)
            if res is None: continue
            r, n = res
            print(f"\n  [{sample}] N={n}  R²={r.rsquared:.4f}")
            for i, p in enumerate(["omega_z", "kappa_z", "anxiety_slope_T", "confidence_slope_D"]):
                beta = r.params[i+1]
                pv = r.pvalues[i+1]
                sig = "★" if pv < 0.05 else " "
                sig = "★★" if pv < 0.01 else sig
                sig = "★★★" if pv < 0.001 else sig
                if abs(beta) > 0.05 or pv < 0.1:
                    print(f"    {p:24s} β={beta:+.3f}  p={pv:.4g} {sig}")
                rows.append({
                    "outcome": outcome, "sample": sample, "predictor": p,
                    "beta": float(beta), "p": float(pv), "R2": float(r.rsquared),
                })

    # ── Key test: spike_mag_peak controlling for pre_mean ────────────────
    print("\n" + "=" * 78)
    print("KEY TEST: spike_mag_peak ~ ω + κ + anxiety_slope_T + pre_mean (baseline as covariate)")
    print("If anxiety_slope_T effect persists after controlling for baseline → real")
    print("If anxiety_slope_T effect disappears → mediated by baseline (ceiling-like)")
    print("=" * 78)
    for sample in ["exploratory", "confirmatory"]:
        res = fit_z(df, "spike_mag_peak",
                    ["omega_z", "kappa_z", "anxiety_slope_T", "confidence_slope_D", "pre_mean"],
                    sample)
        if res is None: continue
        r, n = res
        print(f"\n  [{sample}] N={n}  R²={r.rsquared:.4f}")
        for i, p in enumerate(["omega_z", "kappa_z", "anxiety_slope_T", "confidence_slope_D", "pre_mean"]):
            beta = r.params[i+1]
            pv = r.pvalues[i+1]
            sig = "★" if pv < 0.05 else " "
            sig = "★★" if pv < 0.01 else sig
            sig = "★★★" if pv < 0.001 else sig
            print(f"    {p:24s} β={beta:+.3f}  p={pv:.4g} {sig}")
            rows.append({
                "outcome": "spike_mag_peak_with_baseline_covariate", "sample": sample,
                "predictor": p, "beta": float(beta), "p": float(pv), "R2": float(r.rsquared),
            })

    # Save
    out = REPO_ROOT / "results" / "stats" / "joint_optimal" / "spike_measurement_diagnostic.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out, index=False)
    print(f"\nSaved: {out}")


if __name__ == "__main__":
    main()
