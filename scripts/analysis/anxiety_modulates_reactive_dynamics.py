"""
Does anxiety modulate the reactive dynamics measures?

Two tests:

  1. DIRECT: Does anxiety predict reactive dynamics (acceleration, peak, time-to-peak,
     latency) above and beyond (ω, κ)? Tested with multiple anxiety operationalizations:
     intercept, slope_T, slope_D, mean, variance.

  2. INTERACTION: Does anxiety MODULATE the parameter-dynamics coupling? I.e., does
     the ω → acceleration effect depend on the subject's anxiety profile?
     Test ω × anxiety_intercept and ω × anxiety_slope_T interactions.

Output: results/stats/joint_optimal/anxiety_modulates_reactive_dynamics.csv
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


REACTIVE_CSV = REPO_ROOT / "results/stats/joint_optimal/reactive_dynamics_timecourse.csv"
DYNAMICS_OUTPUTS = ["accel_post", "peak_post", "time_to_peak", "latency"]


def per_subject_anxiety_spread():
    """Compute per-subject anxiety variance and mean across probe trials (exp only)."""
    f = pd.read_csv("data/exploratory_350/processed/stage5_filtered_data_20260403_133425/feelings.csv",
                    low_memory=False)
    f = f[f["questionLabel"] == "anxiety"].dropna(subset=["response"])
    out = f.groupby("subj").agg(
        anxiety_mean=("response", "mean"),
        anxiety_sd=("response", "std"),
        anxiety_range=("response", lambda x: x.max() - x.min()),
        anxiety_n=("response", "count"),
    ).reset_index()
    return out


def compute_reactive_features():
    """Recompute reactive dynamics features (mirrors previous script)."""
    import os as _os
    _os.chdir('/Users/nokada/Documents/CALTECH/EffortForagingUnderThreat')
    print("Loading smoothed vigor timeseries (exploratory)...")
    ts = pd.read_parquet("data/exploratory_350/processed/vigor_processed/smoothed_vigor_ts.parquet")
    ts_att = ts[ts["isAttackTrial"] == 1].copy()
    rows = []
    for subj, sub in ts_att.groupby("subj"):
        per_trial = []
        for (trial,), trial_df in sub.groupby(["trial"]):
            enc_t = trial_df["encounterTime"].iloc[0]
            if not (0 < enc_t < 30):
                continue
            t = trial_df["t"].values
            v = trial_df["vigor_norm"].values
            if len(t) < 20:
                continue
            pre_mask = (t >= enc_t - 0.5) & (t < enc_t)
            if pre_mask.sum() < 3:
                continue
            pre_mean = float(v[pre_mask].mean())
            post_mask = (t >= enc_t) & (t < enc_t + 1.5)
            if post_mask.sum() < 5:
                continue
            t_post = t[post_mask]
            v_post = v[post_mask]
            peak_idx = int(np.argmax(v_post))
            peak_post = float(v_post[peak_idx])
            time_to_peak = float(t_post[peak_idx] - enc_t)
            accel_mask = (t >= enc_t) & (t < enc_t + 0.5)
            if accel_mask.sum() < 3:
                accel_post = np.nan
            else:
                t_acc = t[accel_mask] - enc_t
                v_acc = v[accel_mask]
                try:
                    accel_post = float(np.polyfit(t_acc, v_acc, 1)[0])
                except Exception:
                    accel_post = np.nan
            threshold = pre_mean * 1.1 + 0.05
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
            "pre_mean": float(td["pre_mean"].mean()),
            "peak_post": float(td["peak_post"].mean()),
            "time_to_peak": float(td["time_to_peak"].mean()),
            "accel_post": float(td["accel_post"].mean()),
            "latency": float(td["latency"].mean()),
        })
    return pd.DataFrame(rows)


def fit_z(df, outcome, predictors):
    sub = df[[outcome] + predictors].dropna()
    if len(sub) < 30:
        return None
    sub[outcome + "_z"] = zscore(sub[outcome].values, nan_policy="omit")
    for p in predictors:
        sub[p] = zscore(sub[p].values, nan_policy="omit")
    X = sm.add_constant(sub[predictors].values)
    return sm.OLS(sub[outcome + "_z"].values, X).fit(), len(sub)


def main():
    print("=" * 78)
    print("DOES ANXIETY MODULATE REACTIVE DYNAMICS?")
    print("=" * 78)

    feat = compute_reactive_features()
    print(f"  Reactive features computed for {len(feat)} subjects")

    # Anxiety spread + intercept/slope features
    m4 = pd.read_csv("results/stats/joint_optimal/exploratory/mcmc_m4_params.csv")
    m4["omega_z"] = zscore(np.log(m4["omega"]).values)
    m4["kappa_z"] = zscore(np.log(m4["kappa"]).values)

    slopes = pd.read_csv("results/stats/clinical/phenotype_metacog_slopes_subjects.csv")
    slopes_exp = slopes[slopes["sample"] == "exploratory"]
    anx_cols = ["anxiety_intercept", "anxiety_slope_T", "anxiety_slope_D"]
    df = feat.merge(m4[["subj", "omega", "kappa", "omega_z", "kappa_z"]], on="subj")
    df = df.merge(slopes_exp[["subj"] + anx_cols], on="subj", how="left")

    spread = per_subject_anxiety_spread()
    df = df.merge(spread, on="subj", how="left")
    print(f"  Merged N: {len(df)}")
    print(f"  Sample of new anxiety features:")
    for c in ["anxiety_mean", "anxiety_sd", "anxiety_range"]:
        if c in df.columns:
            v = df[c].dropna()
            print(f"    {c:20s} mean = {v.mean():.3f}  sd = {v.std():.3f}  n = {len(v)}")

    rows = []

    # ── DIRECT: each anxiety feature vs each reactive measure, controlling ω, κ ──
    print("\n" + "=" * 78)
    print("TEST 1 — DIRECT: anxiety → reactive measure | ω + κ")
    print("=" * 78)
    anx_predictors = ["anxiety_intercept", "anxiety_slope_T", "anxiety_slope_D",
                      "anxiety_mean", "anxiety_sd", "anxiety_range"]
    for outcome in DYNAMICS_OUTPUTS:
        print(f"\n  Outcome: {outcome}")
        for anx in anx_predictors:
            res = fit_z(df, outcome, ["omega_z", "kappa_z", anx])
            if res is None:
                continue
            r, n = res
            anx_beta = r.params[3]
            anx_p = r.pvalues[3]
            sig = "★" if anx_p < 0.05 else " "
            sig = "★★" if anx_p < 0.01 else sig
            tag = "★ SIGNIFICANT" if anx_p < 0.05 else ""
            print(f"    {anx:20s} β={anx_beta:+.3f}  p={anx_p:.4g}    R²={r.rsquared:.4f}  {tag}")
            rows.append({"test": "direct", "outcome": outcome, "predictor": anx,
                         "beta": float(anx_beta), "p": float(anx_p), "R2": float(r.rsquared)})

    # ── INTERACTION: ω × anxiety, κ × anxiety on accel_post ──
    print("\n" + "=" * 78)
    print("TEST 2 — INTERACTION: does anxiety modulate parameter-reactive coupling?")
    print("(Outcome: accel_post; focus on ω × anxiety interactions)")
    print("=" * 78)
    for anx in anx_predictors:
        sub = df[["accel_post", "omega_z", "kappa_z", anx]].dropna().copy()
        if len(sub) < 30: continue
        sub["accel_post_z"] = zscore(sub["accel_post"].values, nan_policy="omit")
        sub[anx + "_z"] = zscore(sub[anx].values, nan_policy="omit")
        sub["omega_x_anx"] = sub["omega_z"] * sub[anx + "_z"]
        sub["kappa_x_anx"] = sub["kappa_z"] * sub[anx + "_z"]
        X = sm.add_constant(sub[["omega_z", "kappa_z", anx + "_z", "omega_x_anx", "kappa_x_anx"]].values)
        try:
            res = sm.OLS(sub["accel_post_z"].values, X).fit()
            print(f"\n  {anx} interactions (N={len(sub)}, R²={res.rsquared:.4f}):")
            for i, n in enumerate(["omega_z", "kappa_z", anx, "ω×anx", "κ×anx"]):
                b = res.params[i+1]
                p = res.pvalues[i+1]
                sig = "★" if p < 0.05 else " "
                sig = "★★" if p < 0.01 else sig
                if abs(b) > 0.05 or p < 0.1:
                    print(f"    {n:14s} β={b:+.3f}  p={p:.4g} {sig}")
                rows.append({"test": "interaction", "outcome": "accel_post",
                             "predictor": n + "(" + anx + ")",
                             "beta": float(b), "p": float(p), "R2": float(res.rsquared)})
        except Exception as e:
            print(f"  Error fitting {anx}: {e}")

    # Save
    out = REPO_ROOT / "results/stats/joint_optimal/anxiety_modulates_reactive_dynamics.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out, index=False)
    print(f"\nSaved: {out}")

    # Summary
    print("\n" + "=" * 78)
    print("SUMMARY — does anxiety modulate reactive dynamics?")
    print("=" * 78)
    rdf = pd.DataFrame(rows)
    sig_direct = rdf[(rdf["test"] == "direct") & (rdf["p"] < 0.05)]
    sig_int = rdf[(rdf["test"] == "interaction") & (rdf["p"] < 0.05)]
    print(f"\n  Significant direct effects of anxiety on reactive measures (controlling ω, κ):")
    if len(sig_direct) > 0:
        for _, r in sig_direct.iterrows():
            print(f"    {r['outcome']:14s} ← {r['predictor']:24s} β={r['beta']:+.3f} p={r['p']:.4g}")
    else:
        print("    (none)")
    print(f"\n  Significant interactions on accel_post:")
    if len(sig_int) > 0:
        for _, r in sig_int.iterrows():
            print(f"    {r['predictor']:30s} β={r['beta']:+.3f} p={r['p']:.4g}")
    else:
        print("    (none)")


if __name__ == "__main__":
    main()
