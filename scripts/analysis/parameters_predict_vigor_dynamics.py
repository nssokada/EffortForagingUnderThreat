"""
Do (ω, κ) predict the temporal dynamics of vigor — not just averages?

Three preregistration-extension analyses:

  H1 (predicted POSITIVE): ω predicts anticipatory steepness — i.e., the slope of
      pre-encounter pressing rate on T per subject. High-ω subjects should
      accelerate their anticipatory vigor more steeply as threat rises.

  H2 (predicted NEGATIVE): κ predicts baseline anticipatory vigor at low T —
      high-κ subjects should stay closer to the effort floor when threat is
      negligible.

  H3 (predicted NULL for both ω and κ): The reactive spike (peak strike effort
      minus pre-encounter rate, on attack trials) should NOT scale with the
      computational parameters — if it's a genuinely Pavlovian/reflexive
      response to predator detection.

The dissociation pattern (H1+H2 significant, H3 null) would show that the
parameters control the STRATEGIC anticipatory component of embodied
defensive computation but not the REACTIVE component — mapping cleanly onto
the predatory imminence continuum.

Output: results/stats/joint_optimal/parameters_predict_vigor_dynamics.csv
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


def per_subject_vigor_dynamics():
    """Compute per-subject features from phase-segmented effort columns."""
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
            row = {"subj": int(subj), "sample": sample, "n_trials": len(g)}

            # H1: per-subject slope of pre-encounter vigor on T
            try:
                X = sm.add_constant(g["T_round"].astype(float).values)
                res = sm.OLS(g["mean_preEncounter_effort"].values, X).fit()
                row["pre_slope_T"] = float(res.params[1])
                row["pre_intercept"] = float(res.params[0])
            except Exception:
                pass

            # H2: baseline anticipatory vigor at low T
            low_T = g[g["T_round"] == 0.1]
            if len(low_T) > 0:
                row["pre_at_lowT"] = float(low_T["mean_preEncounter_effort"].mean())

            # Also mid- and high-T anticipatory for sanity
            mid_T = g[g["T_round"] == 0.5]
            high_T = g[g["T_round"] == 0.9]
            if len(mid_T) > 0:
                row["pre_at_midT"] = float(mid_T["mean_preEncounter_effort"].mean())
            if len(high_T) > 0:
                row["pre_at_highT"] = float(high_T["mean_preEncounter_effort"].mean())

            # H3: reactive spike magnitude on attack trials
            att = g[g["isAttackTrial"] == 1]
            if len(att) > 0:
                row["spike_mag_peak"] = float(
                    (att["peak_strike_effort"] - att["mean_preEncounter_effort"]).mean()
                )
                row["spike_mag_mean"] = float(
                    (att["mean_strike_effort"] - att["mean_preEncounter_effort"]).mean()
                )
                row["post_minus_pre"] = float(
                    (att["mean_postEncounter_effort"] - att["mean_preEncounter_effort"]).mean()
                )
                row["n_attack_trials"] = int(len(att))

            rows.append(row)
    return pd.DataFrame(rows)


def build_master():
    exp, conf = load_both()
    rows = []
    for sample, d in [("exploratory", exp), ("confirmatory", conf)]:
        m = d["master"].reset_index().rename(columns={"index": "subj"}).copy()
        m["sample"] = sample
        rows.append(m[["subj", "sample", "omega", "kappa", "escape_rate", "earnings"]])
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
        se = res.bse[i + 1]
        pv = res.pvalues[i + 1]
        sig = "★" if pv < 0.05 else " "
        sig = "★★" if pv < 0.01 else sig
        sig = "★★★" if pv < 0.001 else sig
        print(f"    {p:14s} β={beta:+.3f}  SE={se:.3f}  p={pv:.4g} {sig}")


def main():
    print("=" * 78)
    print("PARAMETERS PREDICT VIGOR DYNAMICS — embodied imminence test")
    print("=" * 78)

    dyn = per_subject_vigor_dynamics()
    print(f"\nDynamics features computed for {len(dyn)} subjects")
    print(f"  exp: {(dyn['sample']=='exploratory').sum()}, conf: {(dyn['sample']=='confirmatory').sum()}")

    print("\n[Sanity] feature means (pooled):")
    for c in ["pre_slope_T", "pre_at_lowT", "pre_at_midT", "pre_at_highT",
              "spike_mag_peak", "spike_mag_mean", "post_minus_pre"]:
        if c in dyn.columns:
            v = dyn[c].dropna()
            print(f"  {c:20s} mean = {v.mean():+.4f}    median = {v.median():+.4f}    n = {len(v)}")

    master = build_master()
    df = master.merge(dyn, on=["subj", "sample"])
    print(f"\nMerged N: {len(df)}")

    rows_summary = []

    # H1: anticipatory steepness ~ ω + κ
    print("\n" + "=" * 78)
    print("H1: anticipatory steepness (pre_slope_T) ~ ω + κ  [predict ω POSITIVE]")
    print("=" * 78)
    for sample in ["exploratory", "confirmatory"]:
        res = fit_z(df, "pre_slope_T", ["omega_z", "kappa_z"], sample)
        print(f"\n  --- {sample} ---")
        print_result(res, sample, ["omega_z", "kappa_z"])
        if res is not None:
            r, n = res
            rows_summary.append({"test": "H1_anticipatory_slope", "sample": sample,
                                 "predictor": "omega_z", "beta": float(r.params[1]),
                                 "p": float(r.pvalues[1])})
            rows_summary.append({"test": "H1_anticipatory_slope", "sample": sample,
                                 "predictor": "kappa_z", "beta": float(r.params[2]),
                                 "p": float(r.pvalues[2])})

    # H2: baseline anticipatory at low T ~ ω + κ
    print("\n" + "=" * 78)
    print("H2: baseline anticipatory (pre_at_lowT) ~ ω + κ  [predict κ NEGATIVE]")
    print("=" * 78)
    for sample in ["exploratory", "confirmatory"]:
        res = fit_z(df, "pre_at_lowT", ["omega_z", "kappa_z"], sample)
        print(f"\n  --- {sample} ---")
        print_result(res, sample, ["omega_z", "kappa_z"])
        if res is not None:
            r, n = res
            rows_summary.append({"test": "H2_baseline_lowT", "sample": sample,
                                 "predictor": "omega_z", "beta": float(r.params[1]),
                                 "p": float(r.pvalues[1])})
            rows_summary.append({"test": "H2_baseline_lowT", "sample": sample,
                                 "predictor": "kappa_z", "beta": float(r.params[2]),
                                 "p": float(r.pvalues[2])})

    # H3: reactive spike ~ ω + κ  [predict NULL for both]
    print("\n" + "=" * 78)
    print("H3: reactive spike magnitude ~ ω + κ  [predict NULL — strategic/reactive dissociation]")
    print("=" * 78)
    for spike_metric in ["spike_mag_peak", "spike_mag_mean", "post_minus_pre"]:
        print(f"\n  Spike metric: {spike_metric}")
        for sample in ["exploratory", "confirmatory"]:
            res = fit_z(df, spike_metric, ["omega_z", "kappa_z"], sample)
            print(f"    --- {sample} ---")
            print_result(res, sample, ["omega_z", "kappa_z"])
            if res is not None:
                r, n = res
                rows_summary.append({"test": f"H3_{spike_metric}", "sample": sample,
                                     "predictor": "omega_z", "beta": float(r.params[1]),
                                     "p": float(r.pvalues[1])})
                rows_summary.append({"test": f"H3_{spike_metric}", "sample": sample,
                                     "predictor": "kappa_z", "beta": float(r.params[2]),
                                     "p": float(r.pvalues[2])})

    # Save
    out_path = REPO_ROOT / "results" / "stats" / "joint_optimal" / "parameters_predict_vigor_dynamics.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows_summary).to_csv(out_path, index=False)
    print(f"\nSaved: {out_path}")

    # Replication summary
    print("\n" + "=" * 78)
    print("REPLICATION SUMMARY (p < 0.05 both samples, same sign)")
    print("=" * 78)
    rdf = pd.DataFrame(rows_summary)
    for test in rdf["test"].unique():
        for pred in ["omega_z", "kappa_z"]:
            er = rdf[(rdf["test"] == test) & (rdf["sample"] == "exploratory") & (rdf["predictor"] == pred)]
            cr = rdf[(rdf["test"] == test) & (rdf["sample"] == "confirmatory") & (rdf["predictor"] == pred)]
            if len(er) == 0 or len(cr) == 0: continue
            eb, ep = er["beta"].iloc[0], er["p"].iloc[0]
            cb, cp = cr["beta"].iloc[0], cr["p"].iloc[0]
            replicates_sig = (ep < 0.05) and (cp < 0.05) and (eb * cb > 0)
            replicates_null = (ep > 0.05) and (cp > 0.05)
            if replicates_sig:
                tag = "★ REPLICATES SIGNIFICANT"
            elif replicates_null:
                tag = "○ REPLICATES NULL"
            else:
                tag = ""
            print(f"  {test:24s} {pred:10s} exp β={eb:+.3f} p={ep:.4g}   conf β={cb:+.3f} p={cp:.4g}   {tag}")


if __name__ == "__main__":
    main()
