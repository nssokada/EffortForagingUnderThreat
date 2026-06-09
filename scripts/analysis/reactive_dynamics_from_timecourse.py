"""
Recover the reactive signal using temporal/derivative measures that are
INDEPENDENT of baseline ceiling.

Subtractive spike measures suffer from baseline-ceiling artifact. Acceleration
and latency measures are independent of baseline magnitude.

For each subject × attack trial:
  - Align vigor timecourse to encounterTime
  - Compute reactive measures:
    1. peak_post: peak vigor in [encounterTime, encounterTime+1.5s] window
    2. time_to_peak_from_encounter: when does the post-encounter peak occur?
    3. accel_post: slope of vigor over [encounterTime, encounterTime+0.5s] window
    4. latency_to_onset: time from encounter to first rise (>1.1 × pre-encounter mean)
    5. pre_mean_500ms: vigor mean in 500ms BEFORE encounter (local baseline)

Aggregate per subject (mean across attack trials), then test (ω, κ, anxiety_slope_T)
predictors. Compare baseline-correlations across measures to identify which
ones are baseline-independent.

This is exploratory-sample only (confirmatory vigor timeseries not processed).
Replication confirmation deferred until conf is processed through the pipeline.

Output: results/stats/joint_optimal/reactive_dynamics_timecourse.csv
"""

from __future__ import annotations

import os
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
os.chdir(REPO_ROOT)

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.stats import zscore


VIGOR_TS_PATH = REPO_ROOT / "data/exploratory_350/processed/vigor_processed/smoothed_vigor_ts.parquet"
M4_PARAMS = REPO_ROOT / "results/stats/joint_optimal/exploratory/mcmc_m4_params.csv"
SLOPES_CSV = REPO_ROOT / "results/stats/clinical/phenotype_metacog_slopes_subjects.csv"


def per_subject_reactive_dynamics():
    """Compute timecourse-based reactive measures per subject."""
    print("Loading smoothed vigor timeseries...")
    ts = pd.read_parquet(VIGOR_TS_PATH)
    print(f"  shape: {ts.shape}")

    # Only attack trials
    ts_att = ts[ts["isAttackTrial"] == 1].copy()
    print(f"  attack trial rows: {len(ts_att)}")

    rows = []
    n_subjects = ts_att["subj"].nunique()
    print(f"  processing {n_subjects} subjects...")

    for si, (subj, sub) in enumerate(ts_att.groupby("subj")):
        per_trial = []
        for (trial,), trial_df in sub.groupby(["trial"]):
            enc_t = trial_df["encounterTime"].iloc[0]
            if not (0 < enc_t < 30):
                continue
            t = trial_df["t"].values
            v = trial_df["vigor_norm"].values
            if len(t) < 20:
                continue

            # Pre-encounter window: [enc_t - 0.5, enc_t]
            pre_mask = (t >= enc_t - 0.5) & (t < enc_t)
            if pre_mask.sum() < 3:
                continue
            pre_mean = float(v[pre_mask].mean())

            # Post-encounter window: [enc_t, enc_t + 1.5]
            post_mask = (t >= enc_t) & (t < enc_t + 1.5)
            if post_mask.sum() < 5:
                continue
            t_post = t[post_mask]
            v_post = v[post_mask]

            # Peak in post window
            peak_idx = int(np.argmax(v_post))
            peak_post = float(v_post[peak_idx])
            time_to_peak = float(t_post[peak_idx] - enc_t)

            # Acceleration: slope of v over first 500ms post-encounter
            accel_mask = (t >= enc_t) & (t < enc_t + 0.5)
            if accel_mask.sum() < 3:
                accel_post = np.nan
            else:
                t_acc = t[accel_mask] - enc_t
                v_acc = v[accel_mask]
                try:
                    slope = np.polyfit(t_acc, v_acc, 1)[0]
                    accel_post = float(slope)
                except Exception:
                    accel_post = np.nan

            # Latency to onset: first time v exceeds 1.1 × pre_mean
            threshold = pre_mean * 1.1 + 0.05  # small additive offset
            try:
                onset_idx = np.where(v_post > threshold)[0]
                latency = float(t_post[onset_idx[0]] - enc_t) if len(onset_idx) > 0 else np.nan
            except Exception:
                latency = np.nan

            per_trial.append({
                "trial": int(trial), "pre_mean": pre_mean, "peak_post": peak_post,
                "time_to_peak": time_to_peak, "accel_post": accel_post, "latency": latency,
            })

        if not per_trial:
            continue
        td = pd.DataFrame(per_trial)
        rows.append({
            "subj": int(subj),
            "n_att_trials": len(td),
            "pre_mean": float(td["pre_mean"].mean()),
            "peak_post": float(td["peak_post"].mean()),
            "time_to_peak": float(td["time_to_peak"].mean()),
            "accel_post": float(td["accel_post"].mean()),
            "latency": float(td["latency"].mean()),
            "subtractive_spike": float((td["peak_post"] - td["pre_mean"]).mean()),
        })
        if (si + 1) % 50 == 0:
            print(f"  {si+1}/{n_subjects} done")
    return pd.DataFrame(rows)


def main():
    print("=" * 78)
    print("REACTIVE DYNAMICS FROM TIMECOURSE — baseline-independent measures")
    print("=" * 78)

    feat = per_subject_reactive_dynamics()
    print(f"\nSubjects with features: {len(feat)}")
    print(f"  sample features (mean, sd, n):")
    for c in ["pre_mean", "peak_post", "time_to_peak", "accel_post", "latency", "subtractive_spike"]:
        v = feat[c].dropna()
        print(f"    {c:20s} mean = {v.mean():+.3f}  sd = {v.std():.3f}  n = {len(v)}")

    # Sanity: correlation with pre_mean (the suspect baseline)
    print("\n[Sanity] correlation of each reactive measure with pre_mean (baseline):")
    for c in ["peak_post", "time_to_peak", "accel_post", "latency", "subtractive_spike"]:
        sub = feat[[c, "pre_mean"]].dropna()
        r = np.corrcoef(sub[c], sub["pre_mean"])[0, 1]
        print(f"  {c:20s} × pre_mean: r = {r:+.3f}")

    # Merge with M4 params + affect slopes
    print("\nMerging with parameters and affect features...")
    m4 = pd.read_csv(M4_PARAMS)
    m4["omega_z"] = zscore(np.log(m4["omega"]).values)
    m4["kappa_z"] = zscore(np.log(m4["kappa"]).values)

    slopes = pd.read_csv(SLOPES_CSV)
    slopes_exp = slopes[slopes["sample"] == "exploratory"].copy()
    affect_cols = ["anxiety_intercept", "anxiety_slope_T", "anxiety_slope_D",
                   "confidence_intercept", "confidence_slope_T", "confidence_slope_D"]
    df = feat.merge(m4[["subj", "omega_z", "kappa_z"]], on="subj")
    df = df.merge(slopes_exp[["subj"] + affect_cols], on="subj", how="left")
    print(f"  Merged N: {len(df)}")

    # Test ω, κ, affect predictors for each measure
    rows = []
    for outcome in ["peak_post", "time_to_peak", "accel_post", "latency", "subtractive_spike"]:
        print("\n" + "=" * 78)
        print(f"OUTCOME: {outcome}")
        print("=" * 78)
        for predictors_label, preds in [
            ("params_only", ["omega_z", "kappa_z"]),
            ("params + anxiety_slope_T", ["omega_z", "kappa_z", "anxiety_slope_T"]),
            ("params + full affect", ["omega_z", "kappa_z"] + affect_cols),
        ]:
            sub = df[[outcome] + preds].dropna()
            if len(sub) < 50:
                continue
            sub[outcome + "_z"] = zscore(sub[outcome].values, nan_policy="omit")
            for p in preds:
                sub[p] = zscore(sub[p].values, nan_policy="omit")
            X = sm.add_constant(sub[preds].values)
            res = sm.OLS(sub[outcome + "_z"].values, X).fit()
            print(f"\n  Model: {predictors_label} | N = {len(sub)} | R² = {res.rsquared:.4f}")
            for i, p in enumerate(preds):
                beta = res.params[i + 1]
                pv = res.pvalues[i + 1]
                sig = "★" if pv < 0.05 else " "
                sig = "★★" if pv < 0.01 else sig
                sig = "★★★" if pv < 0.001 else sig
                if abs(beta) > 0.05 or pv < 0.1:
                    print(f"    {p:28s} β={beta:+.3f}  p={pv:.4g} {sig}")
                rows.append({
                    "outcome": outcome, "model": predictors_label, "predictor": p,
                    "beta": float(beta), "p": float(pv), "R2": float(res.rsquared),
                })

    out = REPO_ROOT / "results/stats/joint_optimal/reactive_dynamics_timecourse.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out, index=False)
    print(f"\nSaved: {out}")


if __name__ == "__main__":
    main()
