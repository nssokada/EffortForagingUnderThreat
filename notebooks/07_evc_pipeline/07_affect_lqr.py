#!/usr/bin/env python3
"""
Affect & Metacognition analysis for the EVC-LQR model.

Analyses:
  4.1  EVC-derived survival predicts affect (anxiety, confidence) via LMMs
  4.2  Metacognitive miscalibration (confidence tracks choice quality, not survival)
  4.3  Calibration vs discrepancy double dissociation

Output:
  results/stats/evc_lqr_affect.csv
  results/stats/evc_lqr_metacognition.csv
  results/stats/evc_lqr_dissociation.csv
  results/figs/paper/fig_lqr_metacognition.png
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import pearsonr, spearmanr
from scipy.special import expit
import statsmodels.formula.api as smf
import statsmodels.api as sm
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

plt.rcParams.update({
    'font.sans-serif': ['DejaVu Sans', 'Arial', 'sans-serif'],
    'font.family': 'sans-serif',
    'figure.dpi': 150,
    'axes.spines.right': False,
    'axes.spines.top': False,
})

# ── Paths ────────────────────────────────────────────────────────────────────
DATA_DIR  = "/workspace/data/exploratory_350/processed/stage5_filtered_data_20260320_191950"
PARAM_F   = "/workspace/results/stats/oc_evc_lqr_final_params.csv"
POP_F     = "/workspace/results/stats/oc_evc_lqr_final_population.csv"
OUT_STATS = "/workspace/results/stats"
OUT_FIGS  = "/workspace/results/figs/paper"

# ── Model constants ──────────────────────────────────────────────────────────
pop = pd.read_csv(POP_F)
CE = float(pop['c_effort'].iloc[0])
GAMMA = float(pop['gamma'].iloc[0])
P_ESC = 0.6

def safe_pearsonr(x, y):
    mask = np.isfinite(x) & np.isfinite(y)
    if mask.sum() < 10:
        return np.nan, np.nan
    return pearsonr(x[mask], y[mask])

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
print(f"Params:   {params.shape[0]} rows")

# ── Compute S for each probe trial ──────────────────────────────────────────
# S = (1 - T^gamma) + epsilon * T^gamma * p_esc
feelings = feelings.merge(params[["subj", "c_death", "epsilon"]], on="subj", how="left")
n_before = feelings["subj"].nunique()
feelings = feelings.dropna(subset=["epsilon"])
n_after = feelings["subj"].nunique()
print(f"Subjects with params: {n_after}/{n_before}")

T = feelings["threat"].values
eps = feelings["epsilon"].values
T_gamma = T ** GAMMA
feelings["S"] = (1 - T_gamma) + eps * T_gamma * P_ESC
feelings["S_z"] = (feelings["S"] - feelings["S"].mean()) / feelings["S"].std()

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

        re_var_intercept = mdf.cov_re.iloc[0, 0]
        re_var_slope = mdf.cov_re.iloc[1, 1] if mdf.cov_re.shape[0] > 1 else np.nan

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
        # Fallback: subject-level OLS
        slopes = []
        for s, sdf in df.groupby("subj"):
            if sdf["S_z"].std() > 0:
                b = np.polyfit(sdf["S_z"], sdf["response"], 1)
                slopes.append(b[0])
        slopes = np.array(slopes)
        t_stat, p_val = stats.ttest_1samp(slopes, 0)
        print(f"  Fallback: Mean slope: {slopes.mean():.4f}, t={t_stat:.2f}, p={p_val:.2e}")
        lmm_results.append({
            "outcome": label, "predictor": "S_z",
            "beta": slopes.mean(), "SE": slopes.std() / np.sqrt(len(slopes)),
            "t": t_stat, "p": p_val,
            "re_var_intercept": np.nan, "re_var_slope": np.var(slopes),
            "n_obs": len(df), "n_subj": df["subj"].nunique(),
            "converged": "OLS_fallback"
        })

lmm_df = pd.DataFrame(lmm_results)
lmm_df.to_csv(f"{OUT_STATS}/evc_lqr_affect.csv", index=False)
print(f"\nSaved: {OUT_STATS}/evc_lqr_affect.csv")


# ══════════════════════════════════════════════════════════════════════════════
# 4.2  METACOGNITIVE MISCALIBRATION
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("4.2  METACOGNITIVE MISCALIBRATION")
print("=" * 70)

# Compute EV for each behavioral trial using EVC-LQR model
beh = behavior.merge(params[["subj", "epsilon", "c_death"]], on="subj", how="inner")

T_b = beh["threat"].values
eps_b = beh["epsilon"].values
cd_b = beh["c_death"].values
T_gamma_b = T_b ** GAMMA
S_b = (1 - T_gamma_b) + eps_b * T_gamma_b * P_ESC

# Value of each option using LQR choice formula
# V_H = S * 5 - (1-S) * cd * 10 - ce * 0.81 * dist_H
# V_L = S * 1 - (1-S) * cd * 6  - ce * 0.16
beh["S"] = S_b
beh["V_H"] = S_b * 5 - (1 - S_b) * cd_b * 10 - CE * 0.81 * beh["distance_H"]
beh["V_L"] = S_b * 1 - (1 - S_b) * cd_b * 6  - CE * 0.16

beh["ev_max_choice"] = (beh["V_H"] > beh["V_L"]).astype(int)
beh["correct_choice"] = (beh["choice"] == beh["ev_max_choice"]).astype(int)

# Subject-level metrics
subj_metrics = beh.groupby("subj").agg(
    choice_quality=("correct_choice", "mean"),
    survival_rate=("outcome", lambda x: 1 - x.mean()),
    n_trials=("choice", "count"),
    mean_choice=("choice", "mean"),
).reset_index()

# Mean confidence and anxiety per subject
conf_subj = confidence_df.groupby("subj")["response"].mean().reset_index()
conf_subj.columns = ["subj", "mean_confidence"]
anx_subj = anxiety_df.groupby("subj")["response"].mean().reset_index()
anx_subj.columns = ["subj", "mean_anxiety"]

subj_metrics = subj_metrics.merge(conf_subj, on="subj", how="inner")
subj_metrics = subj_metrics.merge(anx_subj, on="subj", how="inner")
subj_metrics = subj_metrics.merge(params[["subj", "epsilon", "c_death"]], on="subj", how="inner")

print(f"Subject-level metrics: n={len(subj_metrics)}")
print(f"  Choice quality: mean={subj_metrics['choice_quality'].mean():.3f}, "
      f"SD={subj_metrics['choice_quality'].std():.3f}")
print(f"  Survival rate:  mean={subj_metrics['survival_rate'].mean():.3f}, "
      f"SD={subj_metrics['survival_rate'].std():.3f}")
print(f"  Mean confidence: mean={subj_metrics['mean_confidence'].mean():.3f}, "
      f"SD={subj_metrics['mean_confidence'].std():.3f}")

# Correlations
r_conf_choice, p_conf_choice = pearsonr(subj_metrics["mean_confidence"], subj_metrics["choice_quality"])
r_conf_surv,   p_conf_surv   = pearsonr(subj_metrics["mean_confidence"], subj_metrics["survival_rate"])

print(f"\n  r(confidence, choice_quality) = {r_conf_choice:.3f}, p = {p_conf_choice:.4f}")
print(f"  r(confidence, survival_rate)  = {r_conf_surv:.3f}, p = {p_conf_surv:.4f}")

# Steiger's test for dependent correlations
def steiger_test(r12, r13, r23, n):
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

# Save metacognition results
meta_results = {
    "r_conf_choice_quality": r_conf_choice, "p_conf_choice_quality": p_conf_choice,
    "r_conf_survival_rate": r_conf_surv, "p_conf_survival_rate": p_conf_surv,
    "r_choice_surv": r_choice_surv,
    "steiger_z": z_steiger, "steiger_p": p_steiger,
    "choice_quality_mean": subj_metrics["choice_quality"].mean(),
    "survival_rate_mean": subj_metrics["survival_rate"].mean(),
    "mean_confidence_mean": subj_metrics["mean_confidence"].mean(),
    "n_subjects": len(subj_metrics),
}
pd.DataFrame([meta_results]).to_csv(f"{OUT_STATS}/evc_lqr_metacognition.csv", index=False)
print(f"\nSaved: {OUT_STATS}/evc_lqr_metacognition.csv")


# ══════════════════════════════════════════════════════════════════════════════
# 4.3  CALIBRATION vs DISCREPANCY DOUBLE DISSOCIATION
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("4.3  CALIBRATION vs DISCREPANCY DOUBLE DISSOCIATION")
print("=" * 70)

# Per-subject calibration: within-subject r(anxiety, 1-S)
# Per-subject discrepancy: mean anxiety residual from population model

calib_list = []
for s, sdf in anxiety_df.groupby("subj"):
    T_s = sdf["threat"].values
    eps_s = sdf["epsilon"].values[0]
    T_gamma_s = T_s ** GAMMA
    S_s = (1 - T_gamma_s) + eps_s * T_gamma_s * P_ESC
    danger = 1 - S_s
    anxiety = sdf["response"].values

    if np.std(danger) > 0 and np.std(anxiety) > 0:
        r_cal, _ = pearsonr(anxiety, danger)
    else:
        r_cal = np.nan

    calib_list.append({"subj": s, "calibration": r_cal})

# Population model for discrepancy
pop_slope, pop_intercept = np.polyfit(anxiety_df["S"].values, anxiety_df["response"].values, 1)
print(f"Population model: anxiety = {pop_intercept:.3f} + {pop_slope:.3f} * S")

for entry in calib_list:
    s = entry["subj"]
    sdf = anxiety_df[anxiety_df["subj"] == s]
    predicted = pop_slope * sdf["S"].values + pop_intercept
    entry["discrepancy"] = (sdf["response"].values - predicted).mean()

calib_df = pd.DataFrame(calib_list)
calib_df = calib_df.dropna(subset=["calibration", "discrepancy"])
print(f"Subjects with valid calibration/discrepancy: {len(calib_df)}")

# Merge with metrics and psych
calib_df = calib_df.merge(subj_metrics, on="subj", how="inner")
calib_df = calib_df.merge(psych[["subj", "OASIS_Total", "STAI_State", "STICSA_Total",
                                  "PHQ9_Total", "DASS21_Anxiety", "DASS21_Stress",
                                  "DASS21_Depression", "STAI_Trait"]], on="subj", how="inner")

# Test calibration-discrepancy orthogonality
r_cd_disc, p_cd_disc = safe_pearsonr(calib_df["calibration"].values, calib_df["discrepancy"].values)
print(f"\nr(calibration, discrepancy) = {r_cd_disc:.3f}, p = {p_cd_disc:.4f}")

# Performance correlations
r_cal_cq, p_cal_cq = safe_pearsonr(calib_df["calibration"].values, calib_df["choice_quality"].values)
r_cal_sr, p_cal_sr = safe_pearsonr(calib_df["calibration"].values, calib_df["survival_rate"].values)
print(f"\nCalibration -> Performance:")
print(f"  r(calibration, choice_quality) = {r_cal_cq:.3f}, p = {p_cal_cq:.4f}")
print(f"  r(calibration, survival_rate)  = {r_cal_sr:.3f}, p = {p_cal_sr:.4f}")

r_dis_cq, p_dis_cq = safe_pearsonr(calib_df["discrepancy"].values, calib_df["choice_quality"].values)
r_dis_sr, p_dis_sr = safe_pearsonr(calib_df["discrepancy"].values, calib_df["survival_rate"].values)
print(f"\nDiscrepancy -> Performance (expect null):")
print(f"  r(discrepancy, choice_quality) = {r_dis_cq:.3f}, p = {p_dis_cq:.4f}")
print(f"  r(discrepancy, survival_rate)  = {r_dis_sr:.3f}, p = {p_dis_sr:.4f}")

# Clinical correlates
clinical_vars = ["OASIS_Total", "STAI_State", "STICSA_Total", "PHQ9_Total",
                 "DASS21_Anxiety", "DASS21_Stress", "DASS21_Depression", "STAI_Trait"]

print(f"\nCalibration -> Clinical (expect null):")
cal_clin_results = []
for var in clinical_vars:
    r, p = safe_pearsonr(calib_df["calibration"].values, calib_df[var].values)
    print(f"  r(calibration, {var:25s}) = {r:.3f}, p = {p:.4f}")
    cal_clin_results.append({"measure": var, "r": r, "p": p, "predictor": "calibration"})

print(f"\nDiscrepancy -> Clinical:")
dis_clin_results = []
for var in clinical_vars:
    r, p = safe_pearsonr(calib_df["discrepancy"].values, calib_df[var].values)
    print(f"  r(discrepancy, {var:25s}) = {r:.3f}, p = {p:.4f}")
    dis_clin_results.append({"measure": var, "r": r, "p": p, "predictor": "discrepancy"})

# Double dissociation evaluation
print("\n--- DOUBLE DISSOCIATION TEST ---")
dd_pass_cal_perf  = p_cal_sr < 0.05 or p_cal_cq < 0.05
dd_pass_cal_clin  = all(r["p"] > 0.1 for r in cal_clin_results if np.isfinite(r["p"]))
dd_pass_dis_clin  = any(r["p"] < 0.05 for r in dis_clin_results if np.isfinite(r["p"]))
dd_pass_dis_perf  = p_dis_sr > 0.1 and p_dis_cq > 0.1

print(f"  Calibration -> performance (p<.05): {'PASS' if dd_pass_cal_perf else 'FAIL'}")
print(f"  Calibration -> clinical (all p>.1): {'PASS' if dd_pass_cal_clin else 'FAIL'}")
print(f"  Discrepancy -> clinical (any p<.05): {'PASS' if dd_pass_dis_clin else 'FAIL'}")
print(f"  Discrepancy -> performance (all p>.1): {'PASS' if dd_pass_dis_perf else 'FAIL'}")
dd_overall = dd_pass_cal_perf and dd_pass_cal_clin and dd_pass_dis_clin and dd_pass_dis_perf
print(f"  OVERALL: {'DOUBLE DISSOCIATION SUPPORTED' if dd_overall else 'DOUBLE DISSOCIATION NOT FULLY SUPPORTED'}")

# Save dissociation results
dd_rows = []
dd_rows.append({"predictor": "calibration", "outcome": "choice_quality", "r": r_cal_cq, "p": p_cal_cq, "domain": "performance"})
dd_rows.append({"predictor": "calibration", "outcome": "survival_rate", "r": r_cal_sr, "p": p_cal_sr, "domain": "performance"})
dd_rows.append({"predictor": "discrepancy", "outcome": "choice_quality", "r": r_dis_cq, "p": p_dis_cq, "domain": "performance"})
dd_rows.append({"predictor": "discrepancy", "outcome": "survival_rate", "r": r_dis_sr, "p": p_dis_sr, "domain": "performance"})
for res in cal_clin_results + dis_clin_results:
    dd_rows.append({"predictor": res["predictor"], "outcome": res["measure"], "r": res["r"], "p": res["p"], "domain": "clinical"})
dd_rows.append({"predictor": "calibration", "outcome": "discrepancy", "r": r_cd_disc, "p": p_cd_disc, "domain": "orthogonality"})

dd_df = pd.DataFrame(dd_rows)
dd_df.to_csv(f"{OUT_STATS}/evc_lqr_dissociation.csv", index=False)
print(f"\nSaved: {OUT_STATS}/evc_lqr_dissociation.csv")


# ══════════════════════════════════════════════════════════════════════════════
# FIGURE: 4-panel metacognition figure
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("GENERATING FIGURE")
print("=" * 70)

fig = plt.figure(figsize=(16, 10))
gs = gridspec.GridSpec(2, 2, wspace=0.35, hspace=0.4)

# ── Panel A: S -> anxiety and confidence ──
ax_a = fig.add_subplot(gs[0, 0])

for label, df_plot, color, marker in [("Anxiety", anxiety_df, "#E63946", "o"),
                                       ("Confidence", confidence_df, "#457B9D", "s")]:
    df_plot = df_plot.copy()
    df_plot["S_bin"] = pd.qcut(df_plot["S"], q=5, duplicates="drop")
    bin_means = df_plot.groupby("S_bin").agg(
        S_mean=("S", "mean"),
        resp_mean=("response", "mean"),
        resp_se=("response", lambda x: x.std() / np.sqrt(len(x)))
    ).reset_index()

    ax_a.errorbar(bin_means["S_mean"], bin_means["resp_mean"],
                  yerr=bin_means["resp_se"], fmt=marker + "-", color=color,
                  label=label, markersize=8, capsize=3, linewidth=2)

ax_a.set_xlabel("EVC Survival Probability (S)", fontsize=12)
ax_a.set_ylabel("Rating (0-7)", fontsize=12)
ax_a.set_title("A  Survival Predicts Affect", fontsize=13, fontweight="bold", loc="left")
ax_a.legend(fontsize=10, frameon=False)
ax_a.spines["top"].set_visible(False)
ax_a.spines["right"].set_visible(False)

# ── Panel B: Confidence vs choice quality and survival rate ──
ax_b = fig.add_subplot(gs[0, 1])

ax_b.scatter(subj_metrics["choice_quality"], subj_metrics["mean_confidence"],
             alpha=0.4, s=20, color="#2A9D8F", label=f"Choice quality (r={r_conf_choice:.2f})")
x_cq = subj_metrics["choice_quality"]
slope_cq, intercept_cq = np.polyfit(x_cq, subj_metrics["mean_confidence"], 1)
x_range = np.linspace(x_cq.min(), x_cq.max(), 50)
ax_b.plot(x_range, slope_cq * x_range + intercept_cq, color="#2A9D8F", linewidth=2)

ax_b.scatter(subj_metrics["survival_rate"], subj_metrics["mean_confidence"],
             alpha=0.4, s=20, color="#E9C46A", marker="^",
             label=f"Survival rate (r={r_conf_surv:.2f})")
x_sr = subj_metrics["survival_rate"]
slope_sr, intercept_sr = np.polyfit(x_sr, subj_metrics["mean_confidence"], 1)
x_range2 = np.linspace(x_sr.min(), x_sr.max(), 50)
ax_b.plot(x_range2, slope_sr * x_range2 + intercept_sr, color="#E9C46A", linewidth=2)

ax_b.set_xlabel("Performance Metric", fontsize=12)
ax_b.set_ylabel("Mean Confidence", fontsize=12)
ax_b.set_title("B  Metacognitive Calibration", fontsize=13, fontweight="bold", loc="left")
ax_b.legend(fontsize=9, loc="upper left", frameon=False)
ax_b.spines["top"].set_visible(False)
ax_b.spines["right"].set_visible(False)

# ── Panel C: Double dissociation bar chart ──
ax_c = fig.add_subplot(gs[1, 0])

# Performance associations
perf_data = [
    ("Cal -> CQ", r_cal_cq, p_cal_cq),
    ("Cal -> SR", r_cal_sr, p_cal_sr),
    ("Disc -> CQ", r_dis_cq, p_dis_cq),
    ("Disc -> SR", r_dis_sr, p_dis_sr),
]

x_pos = np.arange(len(perf_data))
colors_perf = ['#264653' if 'Cal' in d[0] else '#E76F51' for d in perf_data]
bars = ax_c.bar(x_pos, [d[1] for d in perf_data], color=colors_perf, alpha=0.7, edgecolor='none')
ax_c.set_xticks(x_pos)
ax_c.set_xticklabels([d[0] for d in perf_data], fontsize=9, rotation=30, ha='right')
ax_c.axhline(0, color='black', linewidth=0.5)
ax_c.set_ylabel("Pearson r", fontsize=11)
ax_c.set_title("C  Performance Associations", fontsize=13, fontweight="bold", loc="left")

# Add significance markers
for i, d in enumerate(perf_data):
    if d[2] < 0.001:
        ax_c.text(i, d[1] + 0.01, '***', ha='center', fontsize=10, fontweight='bold')
    elif d[2] < 0.01:
        ax_c.text(i, d[1] + 0.01, '**', ha='center', fontsize=10, fontweight='bold')
    elif d[2] < 0.05:
        ax_c.text(i, d[1] + 0.01, '*', ha='center', fontsize=10)

ax_c.spines["top"].set_visible(False)
ax_c.spines["right"].set_visible(False)

# ── Panel D: Clinical associations ──
ax_d = fig.add_subplot(gs[1, 1])

# Show best clinical variables
clinical_short = {
    "OASIS_Total": "OASIS",
    "STAI_Trait": "STAI-T",
    "DASS21_Anxiety": "DASS-A",
    "PHQ9_Total": "PHQ-9",
    "STAI_State": "STAI-S",
}

cal_vals = []
dis_vals = []
labels = []
for var in ["OASIS_Total", "STAI_Trait", "DASS21_Anxiety", "PHQ9_Total", "STAI_State"]:
    labels.append(clinical_short[var])
    cal_r = next(r["r"] for r in cal_clin_results if r["measure"] == var)
    dis_r = next(r["r"] for r in dis_clin_results if r["measure"] == var)
    cal_vals.append(cal_r)
    dis_vals.append(dis_r)

x_clin = np.arange(len(labels))
width = 0.35
ax_d.bar(x_clin - width/2, cal_vals, width, color='#264653', alpha=0.7, label='Calibration', edgecolor='none')
ax_d.bar(x_clin + width/2, dis_vals, width, color='#E76F51', alpha=0.7, label='Discrepancy', edgecolor='none')
ax_d.set_xticks(x_clin)
ax_d.set_xticklabels(labels, fontsize=10)
ax_d.axhline(0, color='black', linewidth=0.5)
ax_d.set_ylabel("Pearson r", fontsize=11)
ax_d.set_title("D  Clinical Associations", fontsize=13, fontweight="bold", loc="left")
ax_d.legend(fontsize=9, frameon=False)
ax_d.spines["top"].set_visible(False)
ax_d.spines["right"].set_visible(False)

plt.savefig(f"{OUT_FIGS}/fig_lqr_metacognition.png", dpi=150, bbox_inches="tight", facecolor="white")
print(f"\nSaved: {OUT_FIGS}/fig_lqr_metacognition.png")
print("\nDone!")
