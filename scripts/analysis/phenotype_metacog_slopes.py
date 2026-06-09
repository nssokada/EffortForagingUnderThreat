"""
Slope-corrected phenotype × metacognition profile.

Earlier phenotype_metacognition_profile.py used per-subject MEAN confidence
and MEAN anxiety as metacognitive measures. This is methodologically incomplete:
mean ratings conflate baseline level with reactivity to task conditions. A
subject who is "overconfident" could be (a) genuinely uncalibrated — high
confidence regardless of threat — or (b) just optimistic on the scale but
appropriately responsive. The mean rating can't distinguish these.

This script computes regression-derived measures per subject:

  For each affect channel (confidence, anxiety), within-subject:
    response ~ T_z + D_z + Intercept

  Extract:
    intercept        — baseline level after partialing task conditions
    slope_T          — reactivity to threat (within-subject)
    slope_D          — reactivity to distance (within-subject)
    calibration_T    — pearsonr(threat, response) within-subject
    calibration_D    — pearsonr(distance, response) within-subject

Then re-run the phenotype profile using these regression-derived measures.

Outputs:
  results/stats/clinical/phenotype_metacog_slopes_profile.csv
  results/stats/clinical/phenotype_metacog_slopes_anova.csv
  results/stats/clinical/phenotype_metacog_slopes_subjects.csv (per-subject)
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
from scipy.stats import zscore, pearsonr, f_oneway, linregress

from config import BKW  # type: ignore
from load_data import load_both  # type: ignore


SAMPLES = {
    "exploratory": {
        "stage5": "data/exploratory_350/processed/stage5_filtered_data_20260403_133425",
    },
    "confirmatory": {
        "stage5": "data/confirmatory_350/processed/stage5_filtered_data_20260403_142413",
    },
}


def per_subject_affect_regression(feelings: pd.DataFrame, channel: str) -> pd.DataFrame:
    """For each subject, fit response ~ T + D and extract intercept, slopes, calibrations."""
    sub_data = feelings[feelings["questionLabel"] == channel].copy()
    sub_data = sub_data.dropna(subset=["response", "threat", "distance"])
    sub_data["T"] = sub_data["threat"].astype(float)
    sub_data["D"] = sub_data["distance"].astype(float)

    records = []
    for subj, group in sub_data.groupby("subj"):
        if len(group) < 5:
            continue
        # Within-subject regression
        T_arr = group["T"].values
        D_arr = group["D"].values
        y = group["response"].values

        # Check we have variance in both predictors
        if T_arr.std() < 1e-6 or D_arr.std() < 1e-6:
            continue

        # Z-score predictors WITHIN subject so slopes are comparable
        T_z = (T_arr - T_arr.mean()) / T_arr.std()
        D_z = (D_arr - D_arr.mean()) / D_arr.std()

        # Simple OLS (no penalization needed — small N per subject)
        X = np.column_stack([np.ones_like(T_z), T_z, D_z])
        try:
            coef, *_ = np.linalg.lstsq(X, y, rcond=None)
            intercept, slope_T, slope_D = coef
        except np.linalg.LinAlgError:
            continue

        # Calibrations
        try:
            cal_T, _ = pearsonr(T_arr, y)
            cal_D, _ = pearsonr(D_arr, y)
        except (ValueError, RuntimeError):
            cal_T = np.nan
            cal_D = np.nan

        # Raw mean (for comparison with old analysis)
        mean_resp = float(y.mean())
        sd_resp = float(y.std())

        records.append({
            "subj": subj,
            f"{channel}_mean": mean_resp,
            f"{channel}_sd": sd_resp,
            f"{channel}_intercept": float(intercept),
            f"{channel}_slope_T": float(slope_T),
            f"{channel}_slope_D": float(slope_D),
            f"{channel}_cal_T": float(cal_T) if not np.isnan(cal_T) else np.nan,
            f"{channel}_cal_D": float(cal_D) if not np.isnan(cal_D) else np.nan,
            f"{channel}_n_probes": len(group),
        })
    return pd.DataFrame(records)


def build_pooled_with_slopes():
    print("=" * 70)
    print("Computing per-subject affect regressions...")
    print("=" * 70)
    exp_data, conf_data = load_both()
    masters = {
        "exploratory": exp_data["master"].reset_index().rename(columns={"index": "subj"}).copy(),
        "confirmatory": conf_data["master"].reset_index().rename(columns={"index": "subj"}).copy(),
    }
    pieces = []
    for sample, paths in SAMPLES.items():
        feelings = pd.read_csv(Path(paths["stage5"]) / "feelings.csv", low_memory=False)
        print(f"\n--- {sample}: {len(feelings)} probe trials, {feelings['subj'].nunique()} subjects ---")
        anx_df = per_subject_affect_regression(feelings, "anxiety")
        conf_df = per_subject_affect_regression(feelings, "confidence")
        merged = anx_df.merge(conf_df, on="subj", how="outer")
        merged["sample"] = sample
        pieces.append(merged)
    all_slopes = pd.concat(pieces, ignore_index=True)

    # Merge with master
    masters["exploratory"]["sample"] = "exploratory"
    masters["confirmatory"]["sample"] = "confirmatory"
    pooled_master = pd.concat([masters["exploratory"], masters["confirmatory"]], ignore_index=True)
    pooled = pooled_master.merge(all_slopes, on=["subj", "sample"], how="inner")

    # Compute pooled-z parameters
    pooled["omega_z_pool"] = zscore(np.log(pooled["omega"]).values, nan_policy="omit")
    pooled["kappa_z_pool"] = zscore(np.log(pooled["kappa"]).values, nan_policy="omit")

    print(f"\nPooled N with slope measures: {len(pooled)}")
    return pooled


def define_phenotypes(pooled: pd.DataFrame) -> pd.DataFrame:
    med_choice = pooled["p_heavy"].median()
    med_vigor = pooled["mean_vigor"].median()
    pooled["high_choice"] = (pooled["p_heavy"] > med_choice).astype(int)
    pooled["high_vigor"] = (pooled["mean_vigor"] > med_vigor).astype(int)
    def lbl(r):
        if r["high_choice"] == 1 and r["high_vigor"] == 1: return "HH"
        if r["high_choice"] == 1 and r["high_vigor"] == 0: return "HL"
        if r["high_choice"] == 0 and r["high_vigor"] == 1: return "LH"
        return "LL"
    pooled["phenotype"] = pooled.apply(lbl, axis=1)
    print(f"\nPhenotype counts:")
    print(pooled["phenotype"].value_counts())
    return pooled


def phenotype_profile(pooled: pd.DataFrame) -> pd.DataFrame:
    metacog_vars = [
        "confidence_mean", "confidence_intercept", "confidence_slope_T", "confidence_slope_D",
        "confidence_cal_T", "confidence_cal_D",
        "anxiety_mean", "anxiety_intercept", "anxiety_slope_T", "anxiety_slope_D",
        "anxiety_cal_T", "anxiety_cal_D",
    ]
    other_vars = [
        "p_heavy", "mean_vigor", "omega", "kappa",
        "earnings", "escape_rate", "pct_opt",
    ]
    rows = []
    for phen in ["HH", "HL", "LH", "LL"]:
        sub = pooled[pooled["phenotype"] == phen]
        rec = {"phenotype": phen, "N": len(sub)}
        for v in other_vars + metacog_vars:
            if v in sub.columns:
                rec[f"{v}_mean"] = sub[v].mean()
                rec[f"{v}_sd"] = sub[v].std()
        rows.append(rec)
    return pd.DataFrame(rows)


def anova_table(pooled: pd.DataFrame, vars_to_test: list[str]) -> pd.DataFrame:
    records = []
    for v in vars_to_test:
        if v not in pooled.columns:
            continue
        groups = [pooled[pooled["phenotype"] == p][v].dropna().values for p in ["HH", "HL", "LH", "LL"]]
        if any(len(g) < 5 for g in groups):
            continue
        F, p = f_oneway(*groups)
        records.append({
            "var": v, "F": F, "p": p,
            "HH": groups[0].mean(), "HL": groups[1].mean(),
            "LH": groups[2].mean(), "LL": groups[3].mean(),
        })
    return pd.DataFrame(records)


def regression_phenotype_metacog(pooled: pd.DataFrame, outcome: str, formula: str, label: str):
    print(f"\n--- {label} ---")
    print(f"  formula: {formula}")
    df = pooled.dropna(subset=[outcome] + [v.strip() for v in formula.split("~")[1].replace("+", " ").replace("*", " ").replace(":", " ").split()]).copy()
    # Z-score outcome
    df[f"{outcome}_z"] = zscore(df[outcome].values, nan_policy="omit")
    full_formula = f"{outcome}_z ~ " + formula.split("~", 1)[1]
    try:
        m = bmb.Model(full_formula, data=df)
        s = az.summary(m.fit(**BKW), hdi_prob=0.95)
        print(s[["mean", "hdi_2.5%", "hdi_97.5%"]].round(3).to_string())
        return s
    except Exception as e:
        print(f"  ERROR: {e}")
        return None


def main():
    pooled = build_pooled_with_slopes()
    pooled = define_phenotypes(pooled)

    print("\n" + "=" * 70)
    print("STEP 1 — Phenotype profile with regression-derived metacognitive measures")
    print("=" * 70)
    profile = phenotype_profile(pooled)
    print(profile.round(3).to_string(index=False))
    profile.to_csv(REPO_ROOT / "results" / "stats" / "clinical" / "phenotype_metacog_slopes_profile.csv", index=False)

    print("\n" + "=" * 70)
    print("STEP 2 — One-way ANOVA across phenotypes")
    print("=" * 70)
    test_vars = [
        "confidence_mean", "confidence_intercept", "confidence_slope_T", "confidence_slope_D",
        "confidence_cal_T", "confidence_cal_D",
        "anxiety_mean", "anxiety_intercept", "anxiety_slope_T", "anxiety_slope_D",
        "anxiety_cal_T", "anxiety_cal_D",
    ]
    anova = anova_table(pooled, test_vars)
    print(anova.round(3).to_string(index=False))
    anova.to_csv(REPO_ROOT / "results" / "stats" / "clinical" / "phenotype_metacog_slopes_anova.csv", index=False)

    # Save the subject-level table for downstream use
    keep_cols = ["subj", "sample", "phenotype", "p_heavy", "mean_vigor", "omega", "kappa",
                  "omega_z_pool", "kappa_z_pool",
                  "earnings", "escape_rate", "pct_opt",
                  "confidence_mean", "confidence_intercept", "confidence_slope_T", "confidence_slope_D",
                  "confidence_cal_T", "confidence_cal_D",
                  "anxiety_mean", "anxiety_intercept", "anxiety_slope_T", "anxiety_slope_D",
                  "anxiety_cal_T", "anxiety_cal_D"]
    keep_cols = [c for c in keep_cols if c in pooled.columns]
    pooled[keep_cols].to_csv(REPO_ROOT / "results" / "stats" / "clinical" / "phenotype_metacog_slopes_subjects.csv", index=False)

    print("\n" + "=" * 70)
    print("STEP 3 — Regression-based test: phenotype + slope measures → outcomes")
    print("=" * 70)

    print("\n*** outcome = earnings ***")
    # earnings ~ phenotype + confidence_slope_T + anxiety_slope_T + confidence_intercept + anxiety_intercept
    regression_phenotype_metacog(
        pooled, "earnings",
        "earnings ~ phenotype + confidence_slope_T + anxiety_slope_T + confidence_intercept + anxiety_intercept",
        "earnings ~ phenotype + slope_T + intercept (both channels)"
    )

    print("\n*** outcome = pct_opt ***")
    regression_phenotype_metacog(
        pooled, "pct_opt",
        "pct_opt ~ phenotype + confidence_slope_T + anxiety_slope_T + confidence_intercept + anxiety_intercept",
        "pct_opt ~ phenotype + slope_T + intercept (both channels)"
    )

    print("\n*** outcome = escape_rate ***")
    regression_phenotype_metacog(
        pooled, "escape_rate",
        "escape_rate ~ phenotype + confidence_slope_T + anxiety_slope_T + confidence_intercept + anxiety_intercept",
        "escape_rate ~ phenotype + slope_T + intercept (both channels)"
    )

    print("\nDone. CSVs saved to results/stats/clinical/")


if __name__ == "__main__":
    main()
