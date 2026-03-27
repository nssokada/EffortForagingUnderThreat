#!/usr/bin/env python3
"""
07_affect.py — Full affect and metacognition analysis for EVC+gamma model.

Analyses:
  4.1  EVC-derived survival predicts affect (anxiety, confidence) via LMMs
  4.2  Metacognitive miscalibration (confidence vs choice/vigor quality)
  4.3  Calibration vs discrepancy double dissociation

Outputs:
  results/stats/evc_affect_lmm.csv
  results/stats/evc_metacognition.csv
  results/stats/evc_double_dissociation.csv
  results/figs/paper/fig_metacognition.png
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import pearsonr, spearmanr
import statsmodels.formula.api as smf

def safe_pearsonr(x, y):
    """Pearson r dropping NaN pairs."""
    mask = np.isfinite(x) & np.isfinite(y)
    if mask.sum() < 10:
        return np.nan, np.nan
    return pearsonr(x[mask], y[mask])
import statsmodels.api as sm
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# ── Paths ────────────────────────────────────────────────────────────────────
DATA_DIR  = "/workspace/data/exploratory_350/processed/stage5_filtered_data_20260320_191950"
PARAM_F   = "/workspace/results/stats/oc_evc_gamma_params.csv"
OUT_STATS = "/workspace/results/stats"
OUT_FIGS  = "/workspace/results/figs/paper"

# ── Model constants ──────────────────────────────────────────────────────────
GAMMA = 0.283
P_ESC = 0.6

# ── Load data ────────────────────────────────────────────────────────────────
print("=" * 70)
print("LOADING DATA")
print("=" * 70)

feelings = pd.read_csv(f"{DATA_DIR}/feelings.csv")
behavior = pd.read_csv(f"{DATA_DIR}/behavior.csv")
psych    = pd.read_csv(f"{DATA_DIR}/psych.csv")
params   = pd.read_csv(PARAM_F)

print(f"Feelings: {feelings.shape[0]} rows, {feelings['subj'].nunique()} subjects")
print(f"Behavior: {behavior.shape[0]} rows, {behavior['subj'].nunique()} subjects")
print(f"Psych:    {psych.shape[0]} rows")
print(f"Params:   {params.shape[0]} rows")

# ── Compute EVC survival for each probe trial ───────────────────────────────
# S = (1 - T^gamma) + epsilon_i * T^gamma * p_esc
# where T = threat, gamma = 0.283, p_esc = 0.6, epsilon_i = subject parameter

feelings = feelings.merge(params[["subj", "epsilon"]], on="subj", how="left")

# Drop subjects missing from params (if any)
n_before = feelings["subj"].nunique()
feelings = feelings.dropna(subset=["epsilon"])
n_after = feelings["subj"].nunique()
print(f"Subjects with params: {n_after}/{n_before}")

T = feelings["threat"].values
eps = feelings["epsilon"].values
T_gamma = T ** GAMMA
feelings["S"] = (1 - T_gamma) + eps * T_gamma * P_ESC
feelings["S_z"] = (feelings["S"] - feelings["S"].mean()) / feelings["S"].std()

# Split by question type
anxiety_df    = feelings[feelings["questionLabel"] == "anxiety"].copy()
confidence_df = feelings[feelings["questionLabel"] == "confidence"].copy()
print(f"Anxiety probes:    {anxiety_df.shape[0]}")
print(f"Confidence probes: {confidence_df.shape[0]}")


# ══════════════════════════════════════════════════════════════════════════════
# 4.1  EVC-DERIVED SURVIVAL PREDICTS AFFECT
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("4.1  EVC-DERIVED SURVIVAL PREDICTS AFFECT")
print("=" * 70)

lmm_results = []

for label, df in [("anxiety", anxiety_df), ("confidence", confidence_df)]:
    print(f"\n--- {label.upper()} ~ S_z ---")
    df = df.copy()
    df["subj_str"] = df["subj"].astype(str)

    # Fit LMM: response ~ S_z + (1 + S_z | subject)
    try:
        md = smf.mixedlm(
            "response ~ S_z",
            data=df,
            groups=df["subj_str"],
            re_formula="~S_z"
        )
        mdf = md.fit(reml=True, method="lbfgs", maxiter=500)

        beta = mdf.fe_params["S_z"]
        se   = mdf.bse_fe["S_z"]
        t    = mdf.tvalues["S_z"]
        p    = mdf.pvalues["S_z"]

        # Random effect variances
        re_var_intercept = mdf.cov_re.iloc[0, 0]
        re_var_slope     = mdf.cov_re.iloc[1, 1] if mdf.cov_re.shape[0] > 1 else np.nan

        print(f"  Fixed effect S_z: beta={beta:.4f}, SE={se:.4f}, t={t:.2f}, p={p:.2e}")
        print(f"  Random effect variance: intercept={re_var_intercept:.4f}, slope={re_var_slope:.4f}")

        lmm_results.append({
            "outcome": label, "predictor": "S_z",
            "beta": beta, "SE": se, "t": t, "p": p,
            "re_var_intercept": re_var_intercept,
            "re_var_slope": re_var_slope,
            "n_obs": len(df), "n_subj": df["subj"].nunique(),
            "converged": mdf.converged
        })
    except Exception as e:
        print(f"  LMM failed: {e}")
        # Fallback: OLS per subject, then aggregate
        print("  Falling back to subject-level OLS + t-test")
        slopes = []
        for s, sdf in df.groupby("subj"):
            if sdf["S_z"].std() > 0:
                b = np.polyfit(sdf["S_z"], sdf["response"], 1)
                slopes.append(b[0])
        slopes = np.array(slopes)
        t_stat, p_val = stats.ttest_1samp(slopes, 0)
        print(f"  Mean slope: {slopes.mean():.4f}, t={t_stat:.2f}, p={p_val:.2e}")
        lmm_results.append({
            "outcome": label, "predictor": "S_z",
            "beta": slopes.mean(), "SE": slopes.std() / np.sqrt(len(slopes)),
            "t": t_stat, "p": p_val,
            "re_var_intercept": np.nan, "re_var_slope": np.var(slopes),
            "n_obs": len(df), "n_subj": df["subj"].nunique(),
            "converged": "OLS_fallback"
        })

# ── Epsilon moderation: median split ──
print("\n--- EPSILON MODERATION ---")
eps_median = params["epsilon"].median()
print(f"Epsilon median: {eps_median:.4f}")

high_eps = params[params["epsilon"] >= eps_median]["subj"].values
low_eps  = params[params["epsilon"] <  eps_median]["subj"].values

for label, df in [("anxiety", anxiety_df)]:
    slopes_high, slopes_low = [], []
    for s, sdf in df.groupby("subj"):
        if sdf["S_z"].std() > 0:
            b = np.polyfit(sdf["S_z"], sdf["response"], 1)
            if s in high_eps:
                slopes_high.append(b[0])
            else:
                slopes_low.append(b[0])

    slopes_high = np.array(slopes_high)
    slopes_low  = np.array(slopes_low)
    t_mod, p_mod = stats.ttest_ind(slopes_high, slopes_low)
    print(f"  {label}: high-eps slope={slopes_high.mean():.4f} (n={len(slopes_high)}), "
          f"low-eps slope={slopes_low.mean():.4f} (n={len(slopes_low)})")
    print(f"  Moderation: t={t_mod:.2f}, p={p_mod:.4f}")

    lmm_results.append({
        "outcome": f"{label}_eps_high", "predictor": "S_z",
        "beta": slopes_high.mean(), "SE": slopes_high.std()/np.sqrt(len(slopes_high)),
        "t": np.nan, "p": np.nan,
        "re_var_intercept": np.nan, "re_var_slope": np.var(slopes_high),
        "n_obs": np.nan, "n_subj": len(slopes_high), "converged": "OLS_split"
    })
    lmm_results.append({
        "outcome": f"{label}_eps_low", "predictor": "S_z",
        "beta": slopes_low.mean(), "SE": slopes_low.std()/np.sqrt(len(slopes_low)),
        "t": np.nan, "p": np.nan,
        "re_var_intercept": np.nan, "re_var_slope": np.var(slopes_low),
        "n_obs": np.nan, "n_subj": len(slopes_low), "converged": "OLS_split"
    })
    lmm_results.append({
        "outcome": f"{label}_eps_moderation", "predictor": "eps_group",
        "beta": slopes_high.mean() - slopes_low.mean(),
        "SE": np.nan, "t": t_mod, "p": p_mod,
        "re_var_intercept": np.nan, "re_var_slope": np.nan,
        "n_obs": np.nan, "n_subj": len(slopes_high) + len(slopes_low),
        "converged": "ttest"
    })

lmm_df = pd.DataFrame(lmm_results)
lmm_df.to_csv(f"{OUT_STATS}/evc_affect_lmm.csv", index=False)
print(f"\nSaved: {OUT_STATS}/evc_affect_lmm.csv")


# ══════════════════════════════════════════════════════════════════════════════
# 4.2  METACOGNITIVE MISCALIBRATION
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("4.2  METACOGNITIVE MISCALIBRATION")
print("=" * 70)

# ── Compute EV for each option in each behavioral trial ──
# EV = R * S - C * (1 - S)  where S = survival probability
# Using the EVC model: S = (1 - T^gamma) + epsilon * T^gamma * p_esc
# For simplicity, use condition-level survival (not subject-specific for "objective" EV)
# But for choice quality, we need to know which option has higher EV

# Distance mapping: behavior uses 1,2,3; map to game units 5,7,9
dist_map = {1: 5, 2: 7, 3: 9}

# Reward: R_H=5, R_L=1; Capture cost C=5
R_H, R_L, C = 5.0, 1.0, 5.0

# For each trial, compute EV of both options using population-average survival
# Actually, let's use subject-specific epsilon for a fair comparison
beh = behavior.merge(params[["subj", "epsilon"]], on="subj", how="inner")

# Survival function per option
# L option: always distance=1 (game units = 5), effort=0.4
# H option: varies
# But survival depends on threat only in this model (distance affects effort, not survival directly)
# S = (1 - T^gamma) + epsilon * T^gamma * p_esc
# This is the same for both options at the same threat level!

# Wait -- both options face the same threat T on a given trial. The difference is in reward and effort.
# EV_H = R_H * S - C * (1-S) - c_effort * effort_H * distance_H
# EV_L = R_L * S - C * (1-S) - c_effort * effort_L * distance_L
# The survival term cancels in the comparison! So EV-maximizing depends on reward vs effort cost.

# Actually let me reconsider. The model is about choosing which cookie to pursue.
# Both cookies have the same threat on a given trial, but different reward and effort.
# The "EV-maximizing" choice is the one with higher net value.
# Using the EVC model: V = R * S - C*(1-S) - c_effort * E * D
# where E = effort level, D = distance

# Merge c_effort too
beh = beh.copy()
beh["c_effort"] = beh["subj"].map(params.set_index("subj")["c_effort"])
beh["c_death"]  = beh["subj"].map(params.set_index("subj")["c_death"])

T_b = beh["threat"].values
eps_b = beh["epsilon"].values
T_gamma_b = T_b ** GAMMA
S_b = (1 - T_gamma_b) + eps_b * T_gamma_b * P_ESC

# Value of each option (using subject parameters)
# V = R * S - c_death * (1-S) - c_effort * effort * dist_game_units
beh["S"] = S_b
beh["V_H"] = R_H * S_b - beh["c_death"] * (1 - S_b) - beh["c_effort"] * beh["effort_H"] * beh["distance_H"].map(dist_map)
beh["V_L"] = R_L * S_b - beh["c_death"] * (1 - S_b) - beh["c_effort"] * beh["effort_L"] * beh["distance_L"].map(dist_map)

# EV-maximizing: choose H if V_H > V_L, i.e., choice=1
beh["ev_max_choice"] = (beh["V_H"] > beh["V_L"]).astype(int)
# choice=1 means chose heavy (H), choice=0 means chose light (L)
beh["correct_choice"] = (beh["choice"] == beh["ev_max_choice"]).astype(int)

# ── Subject-level metrics ──
subj_metrics = beh.groupby("subj").agg(
    choice_quality=("correct_choice", "mean"),
    survival_rate=("outcome", lambda x: 1 - x.mean()),  # outcome=1 means caught
    n_trials=("choice", "count"),
    mean_choice=("choice", "mean"),  # proportion choosing heavy
).reset_index()

# Mean confidence per subject
conf_subj = confidence_df.groupby("subj")["response"].mean().reset_index()
conf_subj.columns = ["subj", "mean_confidence"]

# Mean anxiety per subject
anx_subj = anxiety_df.groupby("subj")["response"].mean().reset_index()
anx_subj.columns = ["subj", "mean_anxiety"]

subj_metrics = subj_metrics.merge(conf_subj, on="subj", how="inner")
subj_metrics = subj_metrics.merge(anx_subj, on="subj", how="inner")
subj_metrics = subj_metrics.merge(params[["subj", "epsilon", "c_effort", "c_death"]], on="subj", how="inner")

print(f"Subject-level metrics: n={len(subj_metrics)}")
print(f"  Choice quality: mean={subj_metrics['choice_quality'].mean():.3f}, "
      f"SD={subj_metrics['choice_quality'].std():.3f}")
print(f"  Survival rate:  mean={subj_metrics['survival_rate'].mean():.3f}, "
      f"SD={subj_metrics['survival_rate'].std():.3f}")
print(f"  Mean confidence: mean={subj_metrics['mean_confidence'].mean():.3f}, "
      f"SD={subj_metrics['mean_confidence'].std():.3f}")

# ── Correlations ──
r_conf_choice, p_conf_choice = pearsonr(subj_metrics["mean_confidence"], subj_metrics["choice_quality"])
r_conf_surv,   p_conf_surv   = pearsonr(subj_metrics["mean_confidence"], subj_metrics["survival_rate"])

print(f"\n  r(confidence, choice_quality) = {r_conf_choice:.3f}, p = {p_conf_choice:.4f}")
print(f"  r(confidence, survival_rate)  = {r_conf_surv:.3f}, p = {p_conf_surv:.4f}")

# ── Steiger's test for dependent correlations ──
def steiger_test(r12, r13, r23, n):
    """Test H0: r12 = r13 where variable 1 is the common variable."""
    rm = (r12 + r13) / 2
    f = (1 - r23) / (2 * (1 - rm**2))
    h = (1 - f * rm**2) / (1 - rm**2)
    z1 = np.arctanh(r12)
    z2 = np.arctanh(r13)
    z = (z1 - z2) * np.sqrt((n - 3) / (2 * (1 - r23) * h))
    p = 2 * (1 - stats.norm.cdf(abs(z)))
    return z, p

r_choice_surv, _ = pearsonr(subj_metrics["choice_quality"], subj_metrics["survival_rate"])
z_steiger, p_steiger = steiger_test(r_conf_choice, r_conf_surv, r_choice_surv, len(subj_metrics))
print(f"\n  Steiger's test (r_conf_choice vs r_conf_surv): z={z_steiger:.3f}, p={p_steiger:.4f}")

# ── 2x2: median split choice quality x survival rate ──
cq_med = subj_metrics["choice_quality"].median()
sr_med = subj_metrics["survival_rate"].median()

subj_metrics["cq_group"] = np.where(subj_metrics["choice_quality"] >= cq_med, "high_CQ", "low_CQ")
subj_metrics["sr_group"] = np.where(subj_metrics["survival_rate"] >= sr_med, "high_SR", "low_SR")

print("\n  2x2 Confidence by Choice Quality x Survival Rate:")
crosstab = subj_metrics.groupby(["cq_group", "sr_group"])["mean_confidence"].agg(["mean", "std", "count"])
print(crosstab.to_string())

# ANOVA
from itertools import product
cells = {}
for cq, sr in product(["low_CQ", "high_CQ"], ["low_SR", "high_SR"]):
    mask = (subj_metrics["cq_group"] == cq) & (subj_metrics["sr_group"] == sr)
    cells[f"{cq}_{sr}"] = subj_metrics.loc[mask, "mean_confidence"].values

# Two-way ANOVA via OLS
subj_metrics["cq_high"] = (subj_metrics["cq_group"] == "high_CQ").astype(int)
subj_metrics["sr_high"] = (subj_metrics["sr_group"] == "high_SR").astype(int)
anova_mod = smf.ols("mean_confidence ~ cq_high * sr_high", data=subj_metrics).fit()
print("\n  2x2 ANOVA:")
print(sm.stats.anova_lm(anova_mod, typ=2).to_string())

# Save metacognition results
meta_results = {
    "r_conf_choice_quality": r_conf_choice, "p_conf_choice_quality": p_conf_choice,
    "r_conf_survival_rate": r_conf_surv, "p_conf_survival_rate": p_conf_surv,
    "r_choice_surv": r_choice_surv,
    "steiger_z": z_steiger, "steiger_p": p_steiger,
    "choice_quality_mean": subj_metrics["choice_quality"].mean(),
    "choice_quality_sd": subj_metrics["choice_quality"].std(),
    "survival_rate_mean": subj_metrics["survival_rate"].mean(),
    "survival_rate_sd": subj_metrics["survival_rate"].std(),
    "mean_confidence_mean": subj_metrics["mean_confidence"].mean(),
    "mean_confidence_sd": subj_metrics["mean_confidence"].std(),
    "n_subjects": len(subj_metrics),
}
meta_df = pd.DataFrame([meta_results])
meta_df.to_csv(f"{OUT_STATS}/evc_metacognition.csv", index=False)
print(f"\nSaved: {OUT_STATS}/evc_metacognition.csv")


# ══════════════════════════════════════════════════════════════════════════════
# 4.3  CALIBRATION vs DISCREPANCY DOUBLE DISSOCIATION
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("4.3  CALIBRATION vs DISCREPANCY DOUBLE DISSOCIATION")
print("=" * 70)

# For each subject, compute:
# Calibration: within-subject r(anxiety_rating, 1-S)
# Discrepancy: mean anxiety residual after regressing out S

calib_list = []
for s, sdf in anxiety_df.groupby("subj"):
    T_s = sdf["threat"].values
    eps_s = sdf["epsilon"].values[0]
    T_gamma_s = T_s ** GAMMA
    S_s = (1 - T_gamma_s) + eps_s * T_gamma_s * P_ESC
    danger = 1 - S_s  # higher = more dangerous
    anxiety = sdf["response"].values

    # Calibration: correlation between anxiety and objective danger
    if np.std(danger) > 0 and np.std(anxiety) > 0:
        r_cal, _ = pearsonr(anxiety, danger)
    else:
        r_cal = np.nan

    # Discrepancy: mean residual of anxiety after regressing on S
    # anxiety_pred = a + b*S; discrepancy = mean(anxiety - anxiety_pred)
    # Equivalently, discrepancy = mean(anxiety) - (a + b*mean(S))
    # But simpler: fit OLS, get residuals, take mean
    if np.std(S_s) > 0:
        slope, intercept = np.polyfit(S_s, anxiety, 1)
        predicted = slope * S_s + intercept
        residuals = anxiety - predicted
        discrepancy = residuals.mean()  # should be ~0 per subject, need another approach
        # Better: discrepancy = mean anxiety - mean predicted from POPULATION model
        # Use discrepancy = mean(anxiety) adjusted for mean danger level
    else:
        discrepancy = np.nan

    calib_list.append({"subj": s, "calibration": r_cal, "discrepancy_residual": discrepancy})

# Better discrepancy measure: fit population model, then compute per-subject residual
# Population model: anxiety ~ S
pop_slope, pop_intercept = np.polyfit(anxiety_df["S"].values, anxiety_df["response"].values, 1)
print(f"Population model: anxiety = {pop_intercept:.3f} + {pop_slope:.3f} * S")

# Per-subject mean residual from population model
for entry in calib_list:
    s = entry["subj"]
    sdf = anxiety_df[anxiety_df["subj"] == s]
    predicted = pop_slope * sdf["S"].values + pop_intercept
    entry["discrepancy"] = (sdf["response"].values - predicted).mean()

calib_df = pd.DataFrame(calib_list)
calib_df = calib_df.dropna(subset=["calibration", "discrepancy"])
print(f"Subjects with valid calibration/discrepancy: {len(calib_df)}")

# Merge with subject metrics and psych
calib_df = calib_df.merge(subj_metrics, on="subj", how="inner")
calib_df = calib_df.merge(psych[["subj", "OASIS_Total", "STAI_State", "STICSA_Total",
                                  "PHQ9_Total", "DASS21_Anxiety", "DASS21_Stress",
                                  "DASS21_Depression", "STAI_Trait"]], on="subj", how="inner")

# ── Test calibration and discrepancy are orthogonal ──
r_cd, p_cd = safe_pearsonr(calib_df["calibration"].values, calib_df["discrepancy"].values)
print(f"\nr(calibration, discrepancy) = {r_cd:.3f}, p = {p_cd:.4f}")

# ── Performance metrics ──
# Earnings proxy: use survival_rate (higher = more earnings)
# Could also compute total earnings but survival_rate is a good proxy

# Correlate calibration with performance
r_cal_cq, p_cal_cq = safe_pearsonr(calib_df["calibration"].values, calib_df["choice_quality"].values)
r_cal_sr, p_cal_sr = safe_pearsonr(calib_df["calibration"].values, calib_df["survival_rate"].values)
print(f"\nCalibration -> Performance:")
print(f"  r(calibration, choice_quality) = {r_cal_cq:.3f}, p = {p_cal_cq:.4f}")
print(f"  r(calibration, survival_rate)  = {r_cal_sr:.3f}, p = {p_cal_sr:.4f}")

# Correlate discrepancy with performance (should be NULL)
r_dis_cq, p_dis_cq = safe_pearsonr(calib_df["discrepancy"].values, calib_df["choice_quality"].values)
r_dis_sr, p_dis_sr = safe_pearsonr(calib_df["discrepancy"].values, calib_df["survival_rate"].values)
print(f"\nDiscrepancy -> Performance (expect null):")
print(f"  r(discrepancy, choice_quality) = {r_dis_cq:.3f}, p = {p_dis_cq:.4f}")
print(f"  r(discrepancy, survival_rate)  = {r_dis_sr:.3f}, p = {p_dis_sr:.4f}")

# ── Clinical correlates ──
clinical_vars = ["OASIS_Total", "STAI_State", "STICSA_Total", "PHQ9_Total",
                 "DASS21_Anxiety", "DASS21_Stress", "DASS21_Depression", "STAI_Trait"]

print(f"\nCalibration -> Clinical (expect null):")
cal_clin_results = []
for var in clinical_vars:
    r, p = safe_pearsonr(calib_df["calibration"].values, calib_df[var].values)
    print(f"  r(calibration, {var}) = {r:.3f}, p = {p:.4f}")
    cal_clin_results.append({"measure": var, "r": r, "p": p, "predictor": "calibration"})

print(f"\nDiscrepancy -> Clinical:")
dis_clin_results = []
for var in clinical_vars:
    r, p = safe_pearsonr(calib_df["discrepancy"].values, calib_df[var].values)
    print(f"  r(discrepancy, {var}) = {r:.3f}, p = {p:.4f}")
    dis_clin_results.append({"measure": var, "r": r, "p": p, "predictor": "discrepancy"})

# ── Formal double dissociation test ──
print("\n--- DOUBLE DISSOCIATION TEST ---")
# Calibration: predicts performance (p<.05) but NOT clinical (p>.1)
# Discrepancy: predicts clinical (p<.05) but NOT performance (p>.1)

# Best performance metric
perf_metric = "survival_rate"  # or choice_quality
r_cal_perf, p_cal_perf = safe_pearsonr(calib_df["calibration"].values, calib_df[perf_metric].values)
r_dis_perf, p_dis_perf = safe_pearsonr(calib_df["discrepancy"].values, calib_df[perf_metric].values)

# Best clinical metric: OASIS (anxiety-specific)
r_cal_oasis, p_cal_oasis = safe_pearsonr(calib_df["calibration"].values, calib_df["OASIS_Total"].values)
r_dis_oasis, p_dis_oasis = safe_pearsonr(calib_df["discrepancy"].values, calib_df["OASIS_Total"].values)

# Also STAI_Trait
r_cal_stai, p_cal_stai = safe_pearsonr(calib_df["calibration"].values, calib_df["STAI_Trait"].values)
r_dis_stai, p_dis_stai = safe_pearsonr(calib_df["discrepancy"].values, calib_df["STAI_Trait"].values)

print(f"  Calibration -> {perf_metric}: r={r_cal_perf:.3f}, p={p_cal_perf:.4f}")
print(f"  Calibration -> OASIS:         r={r_cal_oasis:.3f}, p={p_cal_oasis:.4f}")
print(f"  Calibration -> STAI_Trait:    r={r_cal_stai:.3f}, p={p_cal_stai:.4f}")
print(f"  Discrepancy -> {perf_metric}: r={r_dis_perf:.3f}, p={p_dis_perf:.4f}")
print(f"  Discrepancy -> OASIS:         r={r_dis_oasis:.3f}, p={p_dis_oasis:.4f}")
print(f"  Discrepancy -> STAI_Trait:    r={r_dis_stai:.3f}, p={p_dis_stai:.4f}")

dd_pass_cal_perf  = p_cal_perf < 0.05 or p_cal_cq < 0.05
dd_pass_cal_clin  = all(r["p"] > 0.1 for r in cal_clin_results if np.isfinite(r["p"]))
dd_pass_dis_clin  = any(r["p"] < 0.05 for r in dis_clin_results if np.isfinite(r["p"]))
dd_pass_dis_perf  = p_dis_perf > 0.1 and p_dis_cq > 0.1

print(f"\n  Double dissociation criteria:")
print(f"    Calibration -> performance (p<.05): {'PASS' if dd_pass_cal_perf else 'FAIL'}")
print(f"    Calibration -> clinical (all p>.1): {'PASS' if dd_pass_cal_clin else 'FAIL'}")
print(f"    Discrepancy -> clinical (any p<.05): {'PASS' if dd_pass_dis_clin else 'FAIL'}")
print(f"    Discrepancy -> performance (all p>.1): {'PASS' if dd_pass_dis_perf else 'FAIL'}")
print(f"    OVERALL: {'DOUBLE DISSOCIATION SUPPORTED' if (dd_pass_cal_perf and dd_pass_cal_clin and dd_pass_dis_clin and dd_pass_dis_perf) else 'DOUBLE DISSOCIATION NOT FULLY SUPPORTED'}")

# Save double dissociation results
dd_rows = []
# Performance
dd_rows.append({"predictor": "calibration", "outcome": "choice_quality", "r": r_cal_cq, "p": p_cal_cq, "domain": "performance"})
dd_rows.append({"predictor": "calibration", "outcome": "survival_rate", "r": r_cal_sr, "p": p_cal_sr, "domain": "performance"})
dd_rows.append({"predictor": "discrepancy", "outcome": "choice_quality", "r": r_dis_cq, "p": p_dis_cq, "domain": "performance"})
dd_rows.append({"predictor": "discrepancy", "outcome": "survival_rate", "r": r_dis_sr, "p": p_dis_sr, "domain": "performance"})
# Clinical
for res in cal_clin_results + dis_clin_results:
    dd_rows.append({"predictor": res["predictor"], "outcome": res["measure"], "r": res["r"], "p": res["p"], "domain": "clinical"})
# Orthogonality
dd_rows.append({"predictor": "calibration", "outcome": "discrepancy", "r": r_cd, "p": p_cd, "domain": "orthogonality"})

dd_df = pd.DataFrame(dd_rows)
dd_df.to_csv(f"{OUT_STATS}/evc_double_dissociation.csv", index=False)
print(f"\nSaved: {OUT_STATS}/evc_double_dissociation.csv")


# ══════════════════════════════════════════════════════════════════════════════
# FIGURE: Multi-panel metacognition figure
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("GENERATING FIGURE")
print("=" * 70)

fig = plt.figure(figsize=(16, 5))
gs = gridspec.GridSpec(1, 3, wspace=0.35)

# ── Panel A: S -> anxiety and confidence ──
ax_a = fig.add_subplot(gs[0])

# Plot anxiety by S bins
for label, df, color, marker in [("Anxiety", anxiety_df, "#E63946", "o"),
                                   ("Confidence", confidence_df, "#457B9D", "s")]:
    df = df.copy()
    df["S_bin"] = pd.qcut(df["S"], q=5, duplicates="drop")
    bin_means = df.groupby("S_bin").agg(
        S_mean=("S", "mean"),
        resp_mean=("response", "mean"),
        resp_se=("response", lambda x: x.std() / np.sqrt(len(x)))
    ).reset_index()

    ax_a.errorbar(bin_means["S_mean"], bin_means["resp_mean"],
                  yerr=bin_means["resp_se"], fmt=marker + "-", color=color,
                  label=label, markersize=8, capsize=3, linewidth=2)

ax_a.set_xlabel("EVC Survival Probability (S)", fontsize=12)
ax_a.set_ylabel("Rating (0-7)", fontsize=12)
ax_a.set_title("A. Survival Predicts Affect", fontsize=13, fontweight="bold")
ax_a.legend(fontsize=10)
ax_a.spines["top"].set_visible(False)
ax_a.spines["right"].set_visible(False)

# ── Panel B: Confidence vs choice quality and survival rate ──
ax_b = fig.add_subplot(gs[1])

ax_b.scatter(subj_metrics["choice_quality"], subj_metrics["mean_confidence"],
             alpha=0.4, s=20, color="#2A9D8F", label=f"Choice quality\nr={r_conf_choice:.2f}")
# Fit line
x_cq = subj_metrics["choice_quality"]
slope_cq, intercept_cq = np.polyfit(x_cq, subj_metrics["mean_confidence"], 1)
x_range = np.linspace(x_cq.min(), x_cq.max(), 50)
ax_b.plot(x_range, slope_cq * x_range + intercept_cq, color="#2A9D8F", linewidth=2)

ax_b.scatter(subj_metrics["survival_rate"], subj_metrics["mean_confidence"],
             alpha=0.4, s=20, color="#E9C46A", marker="^",
             label=f"Survival rate\nr={r_conf_surv:.2f}")
x_sr = subj_metrics["survival_rate"]
slope_sr, intercept_sr = np.polyfit(x_sr, subj_metrics["mean_confidence"], 1)
x_range2 = np.linspace(x_sr.min(), x_sr.max(), 50)
ax_b.plot(x_range2, slope_sr * x_range2 + intercept_sr, color="#E9C46A", linewidth=2)

ax_b.set_xlabel("Performance Metric", fontsize=12)
ax_b.set_ylabel("Mean Confidence", fontsize=12)
ax_b.set_title("B. Metacognitive Calibration", fontsize=13, fontweight="bold")
ax_b.legend(fontsize=9, loc="upper left")
ax_b.spines["top"].set_visible(False)
ax_b.spines["right"].set_visible(False)

# ── Panel C: Double dissociation ──
ax_c = fig.add_subplot(gs[2])

# Left side: Calibration -> performance; Right side: Discrepancy -> clinical
# Show as bar chart of correlation strengths

bar_data = [
    ("Cal->Perf", r_cal_sr, p_cal_sr, "#264653"),
    ("Cal->Clinical", np.mean([r["r"] for r in cal_clin_results]),
     np.mean([r["p"] for r in cal_clin_results]), "#264653"),
    ("Disc->Perf", r_dis_sr, p_dis_sr, "#E76F51"),
    ("Disc->Clinical", np.mean([r["r"] for r in dis_clin_results]),
     np.mean([r["p"] for r in dis_clin_results]), "#E76F51"),
]

x_pos = np.arange(len(bar_data))
colors = [d[3] for d in bar_data]
hatches = ["", "//", "", "//"]
bars = ax_c.bar(x_pos, [abs(d[1]) for d in bar_data], color=colors, alpha=0.7,
                edgecolor="black", linewidth=0.5)
for bar, h in zip(bars, hatches):
    bar.set_hatch(h)

# Add significance markers
for i, (name, r_val, p_val, _) in enumerate(bar_data):
    sig = "***" if p_val < 0.001 else "**" if p_val < 0.01 else "*" if p_val < 0.05 else "n.s."
    ax_c.text(i, abs(r_val) + 0.01, sig, ha="center", va="bottom", fontsize=11)
    ax_c.text(i, abs(r_val) / 2, f"r={r_val:.2f}", ha="center", va="center",
              fontsize=8, color="white", fontweight="bold")

ax_c.set_xticks(x_pos)
ax_c.set_xticklabels([d[0] for d in bar_data], rotation=30, ha="right", fontsize=9)
ax_c.set_ylabel("|r|", fontsize=12)
ax_c.set_title("C. Double Dissociation", fontsize=13, fontweight="bold")
ax_c.spines["top"].set_visible(False)
ax_c.spines["right"].set_visible(False)
ax_c.set_ylim(0, max(abs(d[1]) for d in bar_data) + 0.08)

plt.savefig(f"{OUT_FIGS}/fig_metacognition.png", dpi=300, bbox_inches="tight")
plt.close()
print(f"Saved: {OUT_FIGS}/fig_metacognition.png")

print("\n" + "=" * 70)
print("ALL ANALYSES COMPLETE")
print("=" * 70)
