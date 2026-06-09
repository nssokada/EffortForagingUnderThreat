"""
The anxiety_intercept → peak_post effect (β = -0.143, p = 0.005) deserves
careful disentangling because:

1. peak_post correlates with pre_mean baseline (r = +0.64)
2. anxiety_intercept correlates with anticipatory baseline (β ≈ +0.16)
3. So baseline alone would predict anxiety_intercept → POSITIVE on peak_post
4. But we observe NEGATIVE (β = -0.143, p = 0.005)
5. The effect is going AGAINST the confound direction

This means the TOTAL anxiety effect could be even larger negative if we
partial out the baseline-mediated positive contribution. Test:

  peak_post ~ ω + κ + anxiety_intercept + pre_mean (baseline as covariate)

If anxiety_intercept β becomes MORE negative → real direct effect being
masked by baseline. If it disappears → baseline-mediated entirely.

Repeat for: anxiety_slope_T, anxiety_intercept, anxiety_mean, anxiety_sd,
anxiety_range. Also for confidence features.

Output: results/stats/joint_optimal/anxiety_peak_disentangle.csv
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


def per_subject_features():
    """Mirror the previous reactive-dynamics computation."""
    print("Loading smoothed vigor timeseries (exploratory)...")
    ts = pd.read_parquet("data/exploratory_350/processed/vigor_processed/smoothed_vigor_ts.parquet")
    ts_att = ts[ts["isAttackTrial"] == 1].copy()
    rows = []
    for subj, sub in ts_att.groupby("subj"):
        per_trial = []
        for (trial,), trial_df in sub.groupby(["trial"]):
            enc_t = trial_df["encounterTime"].iloc[0]
            if not (0 < enc_t < 30): continue
            t = trial_df["t"].values
            v = trial_df["vigor_norm"].values
            if len(t) < 20: continue
            pre_mask = (t >= enc_t - 0.5) & (t < enc_t)
            if pre_mask.sum() < 3: continue
            pre_mean = float(v[pre_mask].mean())
            post_mask = (t >= enc_t) & (t < enc_t + 1.5)
            if post_mask.sum() < 5: continue
            t_post = t[post_mask]
            v_post = v[post_mask]
            peak_idx = int(np.argmax(v_post))
            peak_post = float(v_post[peak_idx])
            time_to_peak = float(t_post[peak_idx] - enc_t)
            accel_mask = (t >= enc_t) & (t < enc_t + 0.5)
            accel_post = np.nan
            if accel_mask.sum() >= 3:
                try:
                    accel_post = float(np.polyfit(t[accel_mask] - enc_t, v[accel_mask], 1)[0])
                except Exception:
                    pass
            per_trial.append({
                "trial": int(trial), "pre_mean": pre_mean, "peak_post": peak_post,
                "time_to_peak": time_to_peak, "accel_post": accel_post,
            })
        if not per_trial: continue
        td = pd.DataFrame(per_trial)
        rows.append({
            "subj": int(subj),
            "pre_mean": float(td["pre_mean"].mean()),
            "peak_post": float(td["peak_post"].mean()),
            "time_to_peak": float(td["time_to_peak"].mean()),
            "accel_post": float(td["accel_post"].mean()),
        })
    return pd.DataFrame(rows)


def per_subject_anxiety_spread():
    f = pd.read_csv("data/exploratory_350/processed/stage5_filtered_data_20260403_133425/feelings.csv",
                    low_memory=False)
    out = {}
    for q in ["anxiety", "confidence"]:
        sub = f[f["questionLabel"] == q].dropna(subset=["response"])
        agg = sub.groupby("subj").agg(
            **{f"{q}_mean": ("response", "mean"),
               f"{q}_sd": ("response", "std"),
               f"{q}_range": ("response", lambda x: x.max() - x.min())}
        ).reset_index()
        out[q] = agg
    return out["anxiety"].merge(out["confidence"], on="subj")


def fit_z(df, outcome, predictors):
    sub = df[[outcome] + predictors].dropna()
    if len(sub) < 30: return None
    sub[outcome + "_z"] = zscore(sub[outcome].values, nan_policy="omit")
    for p in predictors:
        sub[p] = zscore(sub[p].values, nan_policy="omit")
    X = sm.add_constant(sub[predictors].values)
    return sm.OLS(sub[outcome + "_z"].values, X).fit(), len(sub)


def main():
    print("=" * 78)
    print("ANXIETY EFFECT ON PEAK_POST — partial out baseline")
    print("=" * 78)

    feat = per_subject_features()
    m4 = pd.read_csv("results/stats/joint_optimal/exploratory/mcmc_m4_params.csv")
    m4["omega_z"] = zscore(np.log(m4["omega"]).values)
    m4["kappa_z"] = zscore(np.log(m4["kappa"]).values)
    slopes = pd.read_csv("results/stats/clinical/phenotype_metacog_slopes_subjects.csv")
    slopes_exp = slopes[slopes["sample"] == "exploratory"]
    affect_cols_slopes = ["anxiety_intercept", "anxiety_slope_T", "anxiety_slope_D",
                          "confidence_intercept", "confidence_slope_T", "confidence_slope_D"]
    df = feat.merge(m4[["subj", "omega_z", "kappa_z"]], on="subj")
    df = df.merge(slopes_exp[["subj"] + affect_cols_slopes], on="subj", how="left")
    spread = per_subject_anxiety_spread()
    df = df.merge(spread, on="subj", how="left")
    print(f"\nN = {len(df)}")

    affect_preds = ["anxiety_intercept", "anxiety_slope_T", "anxiety_slope_D",
                    "anxiety_mean", "anxiety_sd", "anxiety_range",
                    "confidence_intercept", "confidence_slope_T", "confidence_slope_D",
                    "confidence_mean", "confidence_sd", "confidence_range"]

    rows = []

    for outcome in ["peak_post", "accel_post", "time_to_peak"]:
        print("\n" + "=" * 78)
        print(f"OUTCOME: {outcome}")
        print("=" * 78)
        for affect in affect_preds:
            if affect not in df.columns: continue

            # Model A: no baseline control
            res_a = fit_z(df, outcome, ["omega_z", "kappa_z", affect])
            # Model B: baseline (pre_mean) as covariate
            res_b = fit_z(df, outcome, ["omega_z", "kappa_z", affect, "pre_mean"])
            if res_a is None or res_b is None: continue

            r_a, n_a = res_a
            r_b, n_b = res_b
            beta_a = r_a.params[3]
            p_a = r_a.pvalues[3]
            beta_b = r_b.params[3]
            p_b = r_b.pvalues[3]
            # Change in beta when baseline added
            delta_beta = beta_b - beta_a

            # Tag rows by behavior
            tag = ""
            if p_a < 0.05 and p_b < 0.05 and (beta_a * beta_b > 0):
                if abs(beta_b) > abs(beta_a):
                    tag = "★ ROBUST + STRENGTHENED (real, mask removed)"
                elif abs(beta_b) > abs(beta_a) * 0.7:
                    tag = "★ ROBUST"
                else:
                    tag = " ↘ ROBUST but attenuated"
            elif p_a < 0.05 and p_b >= 0.05:
                tag = " ✗ MEDIATED BY BASELINE"
            elif p_a >= 0.05 and p_b < 0.05:
                tag = " ★ EMERGES with baseline control"

            print(f"\n  {affect}:")
            print(f"    no baseline   β={beta_a:+.3f}  p={p_a:.4g}")
            print(f"    + baseline    β={beta_b:+.3f}  p={p_b:.4g}  Δβ={delta_beta:+.3f}  {tag}")

            rows.append({
                "outcome": outcome, "affect": affect,
                "beta_no_baseline": float(beta_a), "p_no_baseline": float(p_a),
                "beta_with_baseline": float(beta_b), "p_with_baseline": float(p_b),
                "delta_beta": float(delta_beta), "tag": tag,
            })

    out = REPO_ROOT / "results/stats/joint_optimal/anxiety_peak_disentangle.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out, index=False)
    print(f"\nSaved: {out}")

    # Summary of robust effects
    print("\n" + "=" * 78)
    print("SUMMARY: which affect→peak/accel/time-to-peak effects are ROBUST to baseline?")
    print("=" * 78)
    rdf = pd.DataFrame(rows)
    robust = rdf[(rdf["p_no_baseline"] < 0.05) & (rdf["p_with_baseline"] < 0.05) &
                 (rdf["beta_no_baseline"] * rdf["beta_with_baseline"] > 0)]
    print(f"\n  Robust (significant both with and without baseline control, same sign):")
    if len(robust) > 0:
        for _, r in robust.iterrows():
            print(f"    {r['outcome']:14s} ← {r['affect']:24s} no-base β={r['beta_no_baseline']:+.3f}  "
                  f"with-base β={r['beta_with_baseline']:+.3f}  Δβ={r['delta_beta']:+.3f}  {r['tag']}")
    else:
        print("    (none)")

    # Mediated by baseline
    mediated = rdf[(rdf["p_no_baseline"] < 0.05) & (rdf["p_with_baseline"] >= 0.05)]
    print(f"\n  Mediated by baseline (significant without, null with):")
    if len(mediated) > 0:
        for _, r in mediated.iterrows():
            print(f"    {r['outcome']:14s} ← {r['affect']:24s} no-base β={r['beta_no_baseline']:+.3f}  "
                  f"with-base β={r['beta_with_baseline']:+.3f}")
    else:
        print("    (none)")


if __name__ == "__main__":
    main()
