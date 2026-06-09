"""
Do affect features (anxiety/confidence reactivity, intercepts) modulate the
within-trial vigor dynamics that (ω, κ) only partially explain?

Specifically:
  1. Does anxiety_slope_T predict the reactive spike (above and beyond ω, κ)?
     Hypothesis (user): subjects with steeper anxiety reactivity to threat
     might have a stronger reactive motor response when the predator appears.

  2. Does confidence_slope_T modulate anticipatory steepness or baseline?
     Hypothesis: confidence may dampen anticipatory vigor or modulate baseline.

  3. Do the affect-modulation findings replicate across both samples?

For each per-subject dynamics feature, we fit:
  feature_z ~ ω_z + κ_z + affect features

And test which affect features survive after parameters are controlled.

Output: results/stats/affect_analysis/affect_modulates_dynamics.csv
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

# Dynamics features
DYNAMICS = ["pre_slope_T", "pre_at_lowT", "pre_at_midT", "pre_at_highT",
            "spike_mag_peak", "spike_mag_mean", "post_minus_pre"]

# Affect features
AFFECT = [
    "anxiety_intercept", "anxiety_slope_T", "anxiety_slope_D",
    "confidence_intercept", "confidence_slope_T", "confidence_slope_D",
]


def per_subject_vigor_dynamics():
    exp, conf = load_both()
    rows = []
    for sample, d in [("exploratory", exp), ("confirmatory", conf)]:
        beh = d["beh"][[
            "subj", "T_round", "isAttackTrial",
            "mean_preEncounter_effort", "peak_preEncounter_effort",
            "mean_postEncounter_effort", "peak_postEncounter_effort",
            "mean_strike_effort", "peak_strike_effort",
        ]].copy()
        beh["sample"] = sample
        beh = beh.dropna(subset=["mean_preEncounter_effort"])
        for subj, g in beh.groupby("subj"):
            row = {"subj": int(subj), "sample": sample}
            try:
                X = sm.add_constant(g["T_round"].astype(float).values)
                res = sm.OLS(g["mean_preEncounter_effort"].values, X).fit()
                row["pre_slope_T"] = float(res.params[1])
            except Exception:
                pass
            for T_level, label in [(0.1, "lowT"), (0.5, "midT"), (0.9, "highT")]:
                sub = g[g["T_round"] == T_level]
                if len(sub) > 0:
                    row[f"pre_at_{label}"] = float(sub["mean_preEncounter_effort"].mean())
            att = g[g["isAttackTrial"] == 1]
            if len(att) > 0:
                row["spike_mag_peak"] = float(
                    (att["peak_strike_effort"] - att["mean_preEncounter_effort"]).mean())
                row["spike_mag_mean"] = float(
                    (att["mean_strike_effort"] - att["mean_preEncounter_effort"]).mean())
                row["post_minus_pre"] = float(
                    (att["mean_postEncounter_effort"] - att["mean_preEncounter_effort"]).mean())
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


def print_result(res_pack, label, predictors):
    if res_pack is None:
        print(f"  [{label}] insufficient data")
        return
    res, n = res_pack
    print(f"  [{label}] N={n}  R²={res.rsquared:.4f}  F p={res.f_pvalue:.4g}")
    for i, p in enumerate(predictors):
        beta = res.params[i + 1]
        pv = res.pvalues[i + 1]
        sig = "★" if pv < 0.05 else " "
        sig = "★★" if pv < 0.01 else sig
        sig = "★★★" if pv < 0.001 else sig
        if abs(beta) > 0.05 or pv < 0.1:
            print(f"    {p:24s} β={beta:+.3f}  p={pv:.4g} {sig}")


def main():
    print("=" * 78)
    print("AFFECT MODULATES DYNAMICS — does affect explain residual variance?")
    print("=" * 78)

    dyn = per_subject_vigor_dynamics()
    master = build_master()
    df = master.merge(dyn, on=["subj", "sample"])
    slopes = pd.read_csv(SLOPES_CSV)
    keep = [c for c in AFFECT if c in slopes.columns]
    df = df.merge(slopes[["subj", "sample"] + keep], on=["subj", "sample"], how="left")
    print(f"\nMerged N: {len(df)} (exp {(df['sample']=='exploratory').sum()}, conf {(df['sample']=='confirmatory').sum()})")

    rows_summary = []

    for outcome in DYNAMICS:
        if outcome not in df.columns: continue
        print("\n" + "=" * 78)
        print(f"OUTCOME: {outcome}")
        print("=" * 78)

        # Model 1: just params
        print(f"\n  Model A: {outcome} ~ ω + κ (baseline)")
        for sample in ["exploratory", "confirmatory"]:
            res = fit_z(df, outcome, ["omega_z", "kappa_z"], sample)
            if res is None: continue
            r, n = res
            r2_base = r.rsquared
            print(f"    [{sample}] R² = {r2_base:.4f}")
            rows_summary.append({"outcome": outcome, "model": "params_only",
                                 "sample": sample, "R2": float(r2_base)})

        # Model 2: params + affect
        print(f"\n  Model B: {outcome} ~ ω + κ + affect features")
        for sample in ["exploratory", "confirmatory"]:
            res = fit_z(df, outcome, ["omega_z", "kappa_z"] + keep, sample)
            if res is None: continue
            r, n = res
            print_result(res, sample, ["omega_z", "kappa_z"] + keep)
            rows_summary.append({"outcome": outcome, "model": "params_plus_affect",
                                 "sample": sample, "R2": float(r.rsquared)})
            for i, p in enumerate(["omega_z", "kappa_z"] + keep):
                rows_summary.append({
                    "outcome": outcome, "model": "params_plus_affect", "sample": sample,
                    "predictor": p, "beta": float(r.params[i+1]), "p": float(r.pvalues[i+1]),
                })

    # Save
    out_path = REPO_ROOT / "results" / "stats" / "affect_analysis" / "affect_modulates_dynamics.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows_summary).to_csv(out_path, index=False)
    print(f"\nSaved: {out_path}")

    # Replication summary — focus on affect predictors only
    print("\n" + "=" * 78)
    print("REPLICATION SUMMARY — affect predictors of dynamics (controlling for ω, κ)")
    print("=" * 78)
    rdf = pd.DataFrame(rows_summary)
    rdf = rdf[rdf["model"] == "params_plus_affect"]
    if "predictor" in rdf.columns:
        rdf = rdf[rdf["predictor"].isin(keep)]
        for outcome in DYNAMICS:
            for predictor in keep:
                er = rdf[(rdf["outcome"] == outcome) & (rdf["sample"] == "exploratory") & (rdf["predictor"] == predictor)]
                cr = rdf[(rdf["outcome"] == outcome) & (rdf["sample"] == "confirmatory") & (rdf["predictor"] == predictor)]
                if len(er) == 0 or len(cr) == 0: continue
                eb, ep = er["beta"].iloc[0], er["p"].iloc[0]
                cb, cp = cr["beta"].iloc[0], cr["p"].iloc[0]
                replicates_sig = (ep < 0.05) and (cp < 0.05) and (eb * cb > 0)
                if replicates_sig:
                    print(f"  ★ {outcome:20s} ← {predictor:24s} "
                          f"exp β={eb:+.3f} p={ep:.4g}    conf β={cb:+.3f} p={cp:.4g}")


if __name__ == "__main__":
    main()
