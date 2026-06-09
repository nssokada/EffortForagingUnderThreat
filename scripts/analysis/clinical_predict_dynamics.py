"""
Do psychiatric questionnaire scales predict vigor dynamics (anticipatory baseline,
anticipatory steepness, absolute peak strike effort, reactive peak/acceleration)?

This fills the gap in the affect/clinical/dynamics matrix. We've tested:
  - clinical → (ω, κ) parameters [§4.6, mostly null]
  - clinical → behavior via CCA [null cross-sample]
  - affect probes (anxiety/confidence) → dynamics [partial signals, mostly artifact]

But never: clinical scales → dynamics measures directly.

Scales tested (z-scored within sample):
  DASS21_Anxiety, DASS21_Depression, DASS21_Stress
  PHQ9_Total, OASIS_Total, STAI_Trait, STAI_State, STICSA_Total
  AMI_Total, AMI_Behavioural, AMI_Social, AMI_Emotional
  MFIS_Total
  F1 (general distress factor, from §4.6 EFA)
  F2 (apathy/fatigue factor, from §4.6 EFA)

Outcomes:
  Anticipatory (both samples, from beh):
    pre_at_lowT, pre_at_midT, pre_at_highT, pre_slope_T, abs_peak_strike
  Reactive (exploratory only, from smoothed_vigor_ts.parquet):
    peak_post_baseline_controlled, accel_post

Regressions (within each sample):
  anticipatory:    outcome_z ~ scale_z + omega_z + kappa_z
  peak_post:       peak_post_z ~ scale_z + pre_mean_z + omega_z + kappa_z   (baseline as covariate)
  accel_post:      accel_post_z ~ scale_z + omega_z + kappa_z

A finding "replicates" if p<0.05 in BOTH samples with same sign (anticipatory only;
reactive measures are exploratory-only because confirmatory vigor_ts not yet processed).

Output: results/stats/clinical/clinical_predict_dynamics.csv
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
VIGOR_TS_PATH = REPO_ROOT / "data/exploratory_350/processed/vigor_processed/smoothed_vigor_ts.parquet"
FACTOR_SCORES = REPO_ROOT / "results/stats/clinical/factor_scores.csv"

CLINICAL_SCALES = [
    "DASS21_Anxiety", "DASS21_Depression", "DASS21_Stress",
    "PHQ9_Total", "OASIS_Total", "STAI_Trait", "STAI_State", "STICSA_Total",
    "AMI_Total", "AMI_Behavioural", "AMI_Social", "AMI_Emotional",
    "MFIS_Total",
]
FACTOR_COLS = ["F1", "F2"]

ANTICIPATORY_OUTCOMES = ["pre_at_lowT", "pre_at_midT", "pre_at_highT",
                         "pre_slope_T", "abs_peak_strike"]
REACTIVE_OUTCOMES = ["peak_post", "accel_post"]


def per_subject_anticipatory():
    """From behavior table: anticipatory baseline by T, slope, and absolute peak strike."""
    exp, conf = load_both()
    rows = []
    for sample, d in [("exploratory", exp), ("confirmatory", conf)]:
        beh = d["beh"][[
            "subj", "T_round", "isAttackTrial",
            "mean_preEncounter_effort", "peak_strike_effort",
        ]].copy()
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
                row["abs_peak_strike"] = float(att["peak_strike_effort"].mean())
            rows.append(row)
    return pd.DataFrame(rows)


def per_subject_reactive_exploratory():
    """From smoothed_vigor_ts (exp only): pre_mean, peak_post, accel_post."""
    print("  loading smoothed_vigor_ts.parquet ...")
    ts = pd.read_parquet(VIGOR_TS_PATH)
    ts_att = ts[ts["isAttackTrial"] == 1].copy()

    rows = []
    for subj, sub in ts_att.groupby("subj"):
        per_trial = []
        for (trial,), tdf in sub.groupby(["trial"]):
            enc_t = tdf["encounterTime"].iloc[0]
            if not (0 < enc_t < 30):
                continue
            t = tdf["t"].values
            v = tdf["vigor_norm"].values
            if len(t) < 20:
                continue
            pre_mask = (t >= enc_t - 0.5) & (t < enc_t)
            if pre_mask.sum() < 3:
                continue
            pre_mean = float(v[pre_mask].mean())
            post_mask = (t >= enc_t) & (t < enc_t + 1.5)
            if post_mask.sum() < 5:
                continue
            v_post = v[post_mask]
            peak_post = float(v_post.max())
            accel_mask = (t >= enc_t) & (t < enc_t + 0.5)
            if accel_mask.sum() < 3:
                accel_post = np.nan
            else:
                try:
                    accel_post = float(np.polyfit(t[accel_mask] - enc_t, v[accel_mask], 1)[0])
                except Exception:
                    accel_post = np.nan
            per_trial.append({"pre_mean": pre_mean, "peak_post": peak_post,
                              "accel_post": accel_post})
        if not per_trial:
            continue
        td = pd.DataFrame(per_trial)
        rows.append({
            "subj": int(subj),
            "sample": "exploratory",
            "pre_mean": float(td["pre_mean"].mean()),
            "peak_post": float(td["peak_post"].mean()),
            "accel_post": float(td["accel_post"].mean()),
        })
    return pd.DataFrame(rows)


def per_subject_params():
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


def load_clinical_long():
    rows = []
    for sample, path in SAMPLES.items():
        psych = pd.read_csv(Path(path) / "psych.csv")
        cols = [c for c in CLINICAL_SCALES if c in psych.columns]
        sub = psych[["subj"] + cols].copy()
        sub["sample"] = sample
        rows.append(sub)
    return pd.concat(rows, ignore_index=True)


def load_factor_scores():
    if not FACTOR_SCORES.exists():
        return pd.DataFrame(columns=["subj", "sample", "F1", "F2"])
    f = pd.read_csv(FACTOR_SCORES)[["subj", "sample", "F1", "F2"]]
    return f


def fit_one(df, outcome, scale, sample, baseline_covariate=False):
    sub = df[df["sample"] == sample].copy()
    cols = [outcome, scale, "omega_z", "kappa_z"]
    if baseline_covariate:
        cols.append("pre_mean")
    sub = sub[cols].dropna()
    if len(sub) < 30:
        return None
    sub[outcome + "_z"] = zscore(sub[outcome].values, nan_policy="omit")
    sub[scale + "_z"] = zscore(sub[scale].values, nan_policy="omit")
    preds = [scale + "_z", "omega_z", "kappa_z"]
    if baseline_covariate:
        sub["pre_mean_z"] = zscore(sub["pre_mean"].values, nan_policy="omit")
        preds.append("pre_mean_z")
    X = sm.add_constant(sub[preds].values)
    res = sm.OLS(sub[outcome + "_z"].values, X).fit()
    return {
        "scale": scale,
        "outcome": outcome,
        "sample": sample,
        "N": len(sub),
        "R2": float(res.rsquared),
        "beta_scale": float(res.params[1]),
        "se_scale": float(res.bse[1]),
        "p_scale": float(res.pvalues[1]),
        "beta_omega": float(res.params[2]),
        "p_omega": float(res.pvalues[2]),
        "beta_kappa": float(res.params[3]),
        "p_kappa": float(res.pvalues[3]),
        "baseline_controlled": baseline_covariate,
    }


def main():
    print("=" * 78)
    print("CLINICAL SCALES → VIGOR DYNAMICS")
    print("=" * 78)

    print("\n[1] Per-subject anticipatory dynamics (both samples)...")
    ant = per_subject_anticipatory()
    print(f"    rows: {len(ant)}    exp: {(ant['sample']=='exploratory').sum()}    "
          f"conf: {(ant['sample']=='confirmatory').sum()}")

    print("\n[2] Per-subject reactive dynamics (exploratory only, from smoothed_vigor_ts)...")
    react = per_subject_reactive_exploratory()
    print(f"    rows: {len(react)}")

    print("\n[3] Per-subject (ω, κ) ...")
    params = per_subject_params()
    print(f"    rows: {len(params)}")

    print("\n[4] Clinical scales (psych.csv per sample) + EFA factors (F1, F2)...")
    clin = load_clinical_long()
    fac = load_factor_scores()
    print(f"    clinical rows: {len(clin)}    factor rows: {len(fac)}")

    print("\n[5] Merging on (subj, sample)...")
    df = params.merge(ant, on=["subj", "sample"], how="inner")
    df = df.merge(react, on=["subj", "sample"], how="left")
    df = df.merge(clin, on=["subj", "sample"], how="inner")
    if len(fac) > 0:
        df = df.merge(fac, on=["subj", "sample"], how="left")
    print(f"    final N: {len(df)}    exp: {(df['sample']=='exploratory').sum()}    "
          f"conf: {(df['sample']=='confirmatory').sum()}")

    all_scales = [c for c in CLINICAL_SCALES if c in df.columns] + \
                 [c for c in FACTOR_COLS if c in df.columns]

    # ── Fit regressions ───────────────────────────────────────────────────
    rows = []
    for scale in all_scales:
        for outcome in ANTICIPATORY_OUTCOMES:
            for sample in ["exploratory", "confirmatory"]:
                r = fit_one(df, outcome, scale, sample, baseline_covariate=False)
                if r is not None:
                    rows.append(r)
        # Reactive (exp only)
        for outcome in REACTIVE_OUTCOMES:
            baseline = (outcome == "peak_post")
            r = fit_one(df, outcome, scale, "exploratory", baseline_covariate=baseline)
            if r is not None:
                rows.append(r)

    rdf = pd.DataFrame(rows)
    out_path = REPO_ROOT / "results" / "stats" / "clinical" / "clinical_predict_dynamics.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    rdf.to_csv(out_path, index=False)
    print(f"\nSaved: {out_path}")
    print(f"Total tests: {len(rdf)}")

    # ── Replication summary (anticipatory) ────────────────────────────────
    print("\n" + "=" * 78)
    print("REPLICATION SUMMARY — ANTICIPATORY (p<0.05 in BOTH samples, same sign)")
    print("=" * 78)
    replicates_rows = []
    for scale in all_scales:
        for outcome in ANTICIPATORY_OUTCOMES:
            exp_r = rdf[(rdf["scale"] == scale) & (rdf["outcome"] == outcome) &
                        (rdf["sample"] == "exploratory")]
            conf_r = rdf[(rdf["scale"] == scale) & (rdf["outcome"] == outcome) &
                         (rdf["sample"] == "confirmatory")]
            if len(exp_r) == 0 or len(conf_r) == 0:
                continue
            eb = exp_r["beta_scale"].iloc[0]
            cb = conf_r["beta_scale"].iloc[0]
            ep = exp_r["p_scale"].iloc[0]
            cp = conf_r["p_scale"].iloc[0]
            if (ep < 0.05) and (cp < 0.05) and (eb * cb > 0):
                replicates_rows.append({
                    "scale": scale, "outcome": outcome,
                    "exp_beta": eb, "exp_p": ep,
                    "conf_beta": cb, "conf_p": cp,
                })
    if replicates_rows:
        for r in replicates_rows:
            print(f"  {r['scale']:20s} → {r['outcome']:18s}  "
                  f"exp β={r['exp_beta']:+.3f} p={r['exp_p']:.4g}    "
                  f"conf β={r['conf_beta']:+.3f} p={r['conf_p']:.4g}")
    else:
        print("  (none)")

    # ── Single-sample hits (anticipatory) ─────────────────────────────────
    print("\n" + "-" * 78)
    print("ANTICIPATORY single-sample p<0.01 hits (not replicating but worth noting)")
    print("-" * 78)
    notes = 0
    for scale in all_scales:
        for outcome in ANTICIPATORY_OUTCOMES:
            for sample in ["exploratory", "confirmatory"]:
                r = rdf[(rdf["scale"] == scale) & (rdf["outcome"] == outcome) &
                        (rdf["sample"] == sample)]
                if len(r) == 0:
                    continue
                p = r["p_scale"].iloc[0]
                b = r["beta_scale"].iloc[0]
                if p < 0.01:
                    other = "confirmatory" if sample == "exploratory" else "exploratory"
                    o_r = rdf[(rdf["scale"] == scale) & (rdf["outcome"] == outcome) &
                              (rdf["sample"] == other)]
                    o_p = o_r["p_scale"].iloc[0] if len(o_r) else np.nan
                    o_b = o_r["beta_scale"].iloc[0] if len(o_r) else np.nan
                    if not ((o_p < 0.05) and (b * o_b > 0)):
                        print(f"  [{sample[:3]}] {scale:20s} → {outcome:18s}  "
                              f"β={b:+.3f} p={p:.4g}  (other: β={o_b:+.3f} p={o_p:.4g})")
                        notes += 1
    if notes == 0:
        print("  (none)")

    # ── Reactive (exp only) ───────────────────────────────────────────────
    print("\n" + "=" * 78)
    print("REACTIVE (exploratory only) — p<0.05 hits")
    print("=" * 78)
    hit = 0
    for scale in all_scales:
        for outcome in REACTIVE_OUTCOMES:
            r = rdf[(rdf["scale"] == scale) & (rdf["outcome"] == outcome) &
                    (rdf["sample"] == "exploratory")]
            if len(r) == 0:
                continue
            p = r["p_scale"].iloc[0]
            b = r["beta_scale"].iloc[0]
            note = " (baseline-controlled)" if outcome == "peak_post" else ""
            if p < 0.05:
                sig = "★★★" if p < 0.001 else ("★★" if p < 0.01 else "★")
                print(f"  {scale:20s} → {outcome:18s}{note}  β={b:+.3f}  p={p:.4g}  {sig}")
                hit += 1
    if hit == 0:
        print("  (none)")

    print("\nDone.")


if __name__ == "__main__":
    main()
