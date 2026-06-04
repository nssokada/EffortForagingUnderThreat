"""
Strategic-angle analysis: does the (ω, κ) angle — the strategic style of
avoidance — predict clinical and metacognitive correlates that raw
parameter levels do not?

Polar decomposition of (ω, κ):
  angle    = atan2(κ_z, ω_z)   — low = threat-driven, high = effort-driven
  magnitude = sqrt(ω_z² + κ_z²) — total avoidance intensity

This decouples *strategic style* (angle) from *avoidance intensity* (magnitude).
Two subjects with identical P(heavy) can have arrived there via different angles.

Three analysis families:

  ANALYSIS 1 — Polar predictors of clinical scales:
    scale_z ~ angle_z + magnitude_z + angle_z:magnitude_z
    Tests whether anxiety/depression/apathy scales preferentially load
    on the strategic-style axis (angle) over and above intensity.

  ANALYSIS 2 — Angle → metacognitive variables:
    mean_confidence ~ angle_z + magnitude_z
    mean_anxiety    ~ angle_z + magnitude_z
    anx_calibration ~ angle_z + magnitude_z
    Tests whether subjective monitoring correlates with strategic style.

  ANALYSIS 3 — Angle × anxiety calibration on optimality:
    pct_opt ~ angle_z * anx_calibration_z + magnitude_z
    Tests whether well-calibrated anxiety MODERATES the optimality cost
    of strategic style — the embodied metacognitive interaction.

Outputs:
  results/stats/clinical/strategic_angle_clinical.csv
  results/stats/clinical/strategic_angle_metacog.csv
  results/stats/clinical/strategic_angle_optimality.csv
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
import bambi as bmb
import arviz as az
from scipy.stats import zscore

from config import BKW  # type: ignore
from load_data import load_both  # type: ignore


CLINICAL_SCALES = [
    "DASS21_Anxiety", "DASS21_Depression", "DASS21_Stress",
    "PHQ9_Total", "OASIS_Total",
    "STAI_Trait", "STAI_State", "STICSA_Total",
    "AMI_Behavioural", "AMI_Social", "AMI_Emotional", "AMI_Total",
    "MFIS_Physical", "MFIS_Cognitive", "MFIS_Psychosocial", "MFIS_Total",
]

SCALE_TYPE = {
    "DASS21_Anxiety": "anxiety", "OASIS_Total": "anxiety", "STAI_Trait": "anxiety",
    "STAI_State": "anxiety", "STICSA_Total": "anxiety",
    "DASS21_Depression": "depression", "PHQ9_Total": "depression",
    "DASS21_Stress": "stress",
    "AMI_Behavioural": "apathy", "AMI_Social": "apathy", "AMI_Emotional": "apathy",
    "AMI_Total": "apathy",
    "MFIS_Physical": "fatigue", "MFIS_Cognitive": "fatigue",
    "MFIS_Psychosocial": "fatigue", "MFIS_Total": "fatigue",
}


def build_pooled() -> pd.DataFrame:
    print("Loading samples via load_both()...")
    exp_data, conf_data = load_both()
    exp_master = exp_data["master"].reset_index().rename(columns={"index": "subj"}).copy()
    conf_master = conf_data["master"].reset_index().rename(columns={"index": "subj"}).copy()
    exp_master["sample"] = "exploratory"
    conf_master["sample"] = "confirmatory"
    pooled = pd.concat([exp_master, conf_master], ignore_index=True)

    # Polar decomposition on pooled (ω, κ)
    if "omega_z" not in pooled.columns and "omega" in pooled.columns:
        pooled["omega_z"] = zscore(np.log(pooled["omega"]).values, nan_policy="omit")
        pooled["kappa_z"] = zscore(np.log(pooled["kappa"]).values, nan_policy="omit")
    pooled["angle"] = np.arctan2(pooled["kappa_z"], pooled["omega_z"])
    pooled["angle_z"] = zscore(pooled["angle"].values, nan_policy="omit")
    pooled["magnitude"] = np.sqrt(pooled["omega_z"] ** 2 + pooled["kappa_z"] ** 2)
    pooled["magnitude_z"] = zscore(pooled["magnitude"].values, nan_policy="omit")
    return pooled


def fit_regression(formula: str, data: pd.DataFrame, terms: list[str]) -> dict:
    model = bmb.Model(formula, data=data)
    result = model.fit(**BKW)
    s = az.summary(result, hdi_prob=0.95)
    out = {}
    for term in terms:
        if term in s.index:
            beta = s.loc[term, "mean"]
            lo = s.loc[term, "hdi_2.5%"]
            hi = s.loc[term, "hdi_97.5%"]
            out[term] = {"beta": beta, "hdi_lo": lo, "hdi_hi": hi,
                          "sig": (lo > 0) or (hi < 0)}
    return out


def analysis_1_polar_clinical(pooled: pd.DataFrame) -> pd.DataFrame:
    print("\n" + "=" * 70)
    print("ANALYSIS 1 — Polar predictors of clinical scales (pooled N ≈ 571)")
    print("=" * 70)
    records = []
    for scale in CLINICAL_SCALES:
        if scale not in pooled.columns:
            continue
        df = pooled[[scale, "angle_z", "magnitude_z", "sample"]].dropna().copy()
        if len(df) < 50:
            continue
        df[f"{scale}_z"] = zscore(df[scale].values, nan_policy="omit")
        formula = f"{scale}_z ~ angle_z + magnitude_z + angle_z:magnitude_z"
        terms = ["angle_z", "magnitude_z", "angle_z:magnitude_z"]
        print(f"\n--- {scale} (n={len(df)}) ---")
        res = fit_regression(formula, df, terms)
        for t in terms:
            if t in res:
                r = res[t]
                star = "★" if r["sig"] else " "
                print(f"  {t:24s} β = {r['beta']:+.4f} [{r['hdi_lo']:+.4f}, {r['hdi_hi']:+.4f}] {star}")
                records.append({
                    "scale": scale, "scale_type": SCALE_TYPE.get(scale, "unknown"),
                    "term": t, "n": len(df),
                    "beta": r["beta"], "hdi_lo": r["hdi_lo"], "hdi_hi": r["hdi_hi"],
                    "sig": r["sig"],
                })
    return pd.DataFrame(records)


def analysis_2_metacog(pooled: pd.DataFrame) -> pd.DataFrame:
    print("\n" + "=" * 70)
    print("ANALYSIS 2 — Polar predictors of metacognitive variables")
    print("=" * 70)
    records = []
    targets = [c for c in ["mean_confidence", "mean_anxiety", "anx_calibration"] if c in pooled.columns]
    if not targets:
        print("  No metacognitive variables found in pooled master")
        return pd.DataFrame()
    for target in targets:
        df = pooled[[target, "angle_z", "magnitude_z"]].dropna().copy()
        if len(df) < 50:
            continue
        df[f"{target}_z"] = zscore(df[target].values, nan_policy="omit")
        formula = f"{target}_z ~ angle_z + magnitude_z + angle_z:magnitude_z"
        terms = ["angle_z", "magnitude_z", "angle_z:magnitude_z"]
        print(f"\n--- {target} (n={len(df)}) ---")
        res = fit_regression(formula, df, terms)
        for t in terms:
            if t in res:
                r = res[t]
                star = "★" if r["sig"] else " "
                print(f"  {t:24s} β = {r['beta']:+.4f} [{r['hdi_lo']:+.4f}, {r['hdi_hi']:+.4f}] {star}")
                records.append({
                    "target": target, "term": t, "n": len(df),
                    "beta": r["beta"], "hdi_lo": r["hdi_lo"], "hdi_hi": r["hdi_hi"],
                    "sig": r["sig"],
                })
    return pd.DataFrame(records)


def analysis_3_angle_calibration_optimality(pooled: pd.DataFrame) -> pd.DataFrame:
    print("\n" + "=" * 70)
    print("ANALYSIS 3 — pct_opt ~ angle × anxiety_calibration + magnitude (THE KEY TEST)")
    print("=" * 70)
    records = []
    if "anx_calibration" not in pooled.columns or "pct_opt" not in pooled.columns:
        print("  Missing anx_calibration or pct_opt; skipping")
        return pd.DataFrame()
    df = pooled[["pct_opt", "angle_z", "magnitude_z", "anx_calibration"]].dropna().copy()
    df["pct_opt_z"] = zscore(df["pct_opt"].values, nan_policy="omit")
    df["anx_cal_z"] = zscore(df["anx_calibration"].values, nan_policy="omit")
    print(f"  N = {len(df)}")
    formula = "pct_opt_z ~ angle_z * anx_cal_z + magnitude_z"
    terms = ["angle_z", "anx_cal_z", "angle_z:anx_cal_z", "magnitude_z"]
    res = fit_regression(formula, df, terms)
    print(f"\n  Formula: {formula}")
    for t in terms:
        if t in res:
            r = res[t]
            star = "★" if r["sig"] else " "
            print(f"  {t:24s} β = {r['beta']:+.4f} [{r['hdi_lo']:+.4f}, {r['hdi_hi']:+.4f}] {star}")
            records.append({
                "outcome": "pct_opt", "term": t, "n": len(df),
                "beta": r["beta"], "hdi_lo": r["hdi_lo"], "hdi_hi": r["hdi_hi"],
                "sig": r["sig"],
            })
    # Also test on earnings if available
    if "earnings" in pooled.columns:
        df2 = pooled[["earnings", "angle_z", "magnitude_z", "anx_calibration"]].dropna().copy()
        df2["earnings_z"] = zscore(df2["earnings"].values, nan_policy="omit")
        df2["anx_cal_z"] = zscore(df2["anx_calibration"].values, nan_policy="omit")
        print(f"\n--- earnings ~ angle × anx_cal + magnitude (n={len(df2)}) ---")
        res2 = fit_regression("earnings_z ~ angle_z * anx_cal_z + magnitude_z", df2, terms)
        for t in terms:
            if t in res2:
                r = res2[t]
                star = "★" if r["sig"] else " "
                print(f"  {t:24s} β = {r['beta']:+.4f} [{r['hdi_lo']:+.4f}, {r['hdi_hi']:+.4f}] {star}")
                records.append({
                    "outcome": "earnings", "term": t, "n": len(df2),
                    "beta": r["beta"], "hdi_lo": r["hdi_lo"], "hdi_hi": r["hdi_hi"],
                    "sig": r["sig"],
                })
    # And on escape_rate
    if "escape_rate" in pooled.columns:
        df3 = pooled[["escape_rate", "angle_z", "magnitude_z", "anx_calibration"]].dropna().copy()
        df3["escape_z"] = zscore(df3["escape_rate"].values, nan_policy="omit")
        df3["anx_cal_z"] = zscore(df3["anx_calibration"].values, nan_policy="omit")
        print(f"\n--- escape_rate ~ angle × anx_cal + magnitude (n={len(df3)}) ---")
        res3 = fit_regression("escape_z ~ angle_z * anx_cal_z + magnitude_z", df3, terms)
        for t in terms:
            if t in res3:
                r = res3[t]
                star = "★" if r["sig"] else " "
                print(f"  {t:24s} β = {r['beta']:+.4f} [{r['hdi_lo']:+.4f}, {r['hdi_hi']:+.4f}] {star}")
                records.append({
                    "outcome": "escape_rate", "term": t, "n": len(df3),
                    "beta": r["beta"], "hdi_lo": r["hdi_lo"], "hdi_hi": r["hdi_hi"],
                    "sig": r["sig"],
                })
    return pd.DataFrame(records)


def main():
    pooled = build_pooled()
    print(f"\nPooled N: {len(pooled)}")
    print(f"Columns relevant: {[c for c in pooled.columns if any(s in c for s in ['omega','kappa','angle','magnitude','mean_','anx','pct','earnings','escape'])][:20]}")

    out_dir = REPO_ROOT / "results" / "stats" / "clinical"
    out_dir.mkdir(parents=True, exist_ok=True)

    df1 = analysis_1_polar_clinical(pooled)
    df2 = analysis_2_metacog(pooled)
    df3 = analysis_3_angle_calibration_optimality(pooled)

    df1.to_csv(out_dir / "strategic_angle_clinical.csv", index=False)
    df2.to_csv(out_dir / "strategic_angle_metacog.csv", index=False)
    df3.to_csv(out_dir / "strategic_angle_optimality.csv", index=False)
    print(f"\nSaved: {out_dir / 'strategic_angle_clinical.csv'}")
    print(f"Saved: {out_dir / 'strategic_angle_metacog.csv'}")
    print(f"Saved: {out_dir / 'strategic_angle_optimality.csv'}")

    # Summary of significant terms
    print("\n" + "=" * 70)
    print("HEADLINE — significant terms across all three analyses")
    print("=" * 70)

    if len(df1) > 0:
        sig1 = df1[df1["sig"]]
        print(f"\nClinical scales (Analysis 1): {len(sig1)} of {len(df1)} terms significant")
        if len(sig1) > 0:
            print(sig1[["scale", "scale_type", "term", "beta", "hdi_lo", "hdi_hi"]].to_string(index=False))

    if len(df2) > 0:
        sig2 = df2[df2["sig"]]
        print(f"\nMetacognitive (Analysis 2): {len(sig2)} of {len(df2)} terms significant")
        if len(sig2) > 0:
            print(sig2[["target", "term", "beta", "hdi_lo", "hdi_hi"]].to_string(index=False))

    if len(df3) > 0:
        sig3 = df3[df3["sig"]]
        print(f"\nAngle × calibration (Analysis 3): {len(sig3)} of {len(df3)} terms significant")
        if len(sig3) > 0:
            print(sig3[["outcome", "term", "beta", "hdi_lo", "hdi_hi"]].to_string(index=False))


if __name__ == "__main__":
    main()
