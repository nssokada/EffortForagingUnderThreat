#!/usr/bin/env python3
"""
16_residual_suboptimality.py — Residual suboptimality and affect
================================================================

KEY analysis: does affective discrepancy predict choice suboptimality
that ce and cd CANNOT explain?

Tests:
  1. Calibration & discrepancy computation
  2. Residual overcaution (after removing ce + cd effects)
  3. Discrepancy → residual overcaution
  4. Calibration → policy alignment
  5. Discrepancy → excess vigor at low threat
  6. Four-route independence matrix
  7. Clinical convergence

Output:
  results/stats/residual_suboptimality.csv
  results/stats/four_routes.csv
  results/figs/paper/fig_residual_affect.png
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import pearsonr
import statsmodels.api as sm
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from pathlib import Path

plt.rcParams.update({
    'font.sans-serif': ['DejaVu Sans', 'Arial', 'sans-serif'],
    'font.family': 'sans-serif',
    'figure.dpi': 150,
    'axes.spines.right': False,
    'axes.spines.top': False,
})

# ── Paths ──
DATA_DIR = Path("/workspace/data/exploratory_350/processed/stage5_filtered_data_20260320_191950")
PARAMS_FILE = Path("/workspace/results/stats/oc_evc_final_params.csv")
POP_FILE = Path("/workspace/results/stats/oc_evc_final_81_population.csv")
DEVIATIONS_FILE = Path("/workspace/results/stats/per_subject_deviations.csv")
POLICY_FILE = Path("/workspace/results/stats/optimal_policy.csv")
VIGOR_TS = Path("/workspace/data/exploratory_350/processed/vigor_processed/smoothed_vigor_ts.parquet")
DYNAMICS_FILE = Path("/workspace/results/stats/evc_vigor_dynamics.csv")
OUT_STATS = Path("/workspace/results/stats/residual_suboptimality.csv")
OUT_ROUTES = Path("/workspace/results/stats/four_routes.csv")
OUT_FIG = Path("/workspace/results/figs/paper/fig_residual_affect.png")
OUT_FIG.parent.mkdir(parents=True, exist_ok=True)

def safe_pearsonr(x, y):
    mask = np.isfinite(x) & np.isfinite(y)
    if mask.sum() < 10:
        return np.nan, np.nan
    return pearsonr(x[mask], y[mask])

def partial_corr(x, y, Z):
    """Partial correlation of x and y, controlling for columns in Z (2D array)."""
    mask = np.isfinite(x) & np.isfinite(y) & np.all(np.isfinite(Z), axis=1)
    x, y, Z = x[mask], y[mask], Z[mask]
    rx = sm.OLS(x, sm.add_constant(Z)).fit().resid
    ry = sm.OLS(y, sm.add_constant(Z)).fit().resid
    return pearsonr(rx, ry)

# ── Load data ──
print("=" * 70)
print("16. RESIDUAL SUBOPTIMALITY AND AFFECT")
print("=" * 70)

feelings = pd.read_csv(DATA_DIR / "feelings.csv")
behavior = pd.read_csv(DATA_DIR / "behavior.csv")
psych = pd.read_csv(DATA_DIR / "psych.csv")
params = pd.read_csv(PARAMS_FILE)
pop = pd.read_csv(POP_FILE)
dev = pd.read_csv(DEVIATIONS_FILE)
policy = pd.read_csv(POLICY_FILE)

GAMMA = float(pop["gamma"].iloc[0])
EPSILON = float(pop["epsilon"].iloc[0])
P_ESC = float(pop["p_esc"].iloc[0])

print(f"Population: gamma={GAMMA:.4f}, epsilon={EPSILON:.4f}, p_esc={P_ESC:.4f}")
print(f"Feelings: {feelings.shape[0]} rows, {feelings['subj'].nunique()} subjects")
print(f"Behavior: {behavior.shape[0]} trials, {behavior['subj'].nunique()} subjects")

results = []

# ══════════════════════════════════════════════════════════════════════════
# 1. CALIBRATION AND DISCREPANCY
# ══════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("1. CALIBRATION AND DISCREPANCY")
print("=" * 70)

# Compute S for each probe trial
anxiety_df = feelings[feelings["questionLabel"] == "anxiety"].copy()
T = anxiety_df["threat"].values
T_gamma = T ** GAMMA
anxiety_df["S"] = (1 - T_gamma) + EPSILON * T_gamma * P_ESC

# Population-level regression: anxiety ~ S
pop_slope, pop_intercept = np.polyfit(anxiety_df["S"].values, anxiety_df["response"].values, 1)
print(f"Population model: anxiety = {pop_intercept:.3f} + {pop_slope:.3f} * S")

# Per-subject calibration and discrepancy
calib_list = []
for s, sdf in anxiety_df.groupby("subj"):
    T_s = sdf["threat"].values
    T_gamma_s = T_s ** GAMMA
    S_s = (1 - T_gamma_s) + EPSILON * T_gamma_s * P_ESC
    danger = 1 - S_s
    anxiety = sdf["response"].values

    # Calibration: within-subject r(anxiety, 1-S)
    if np.std(danger) > 0 and np.std(anxiety) > 0:
        r_cal, _ = pearsonr(anxiety, danger)
    else:
        r_cal = np.nan

    # Discrepancy: mean residual from population regression
    predicted = pop_slope * S_s + pop_intercept
    disc = (anxiety - predicted).mean()

    calib_list.append({"subj": s, "calibration": r_cal, "discrepancy": disc})

calib_df = pd.DataFrame(calib_list)
calib_df = calib_df.dropna(subset=["calibration", "discrepancy"])
print(f"Subjects with valid calibration/discrepancy: {len(calib_df)}")

r_cal_disc, p_cal_disc = safe_pearsonr(calib_df["calibration"].values, calib_df["discrepancy"].values)
print(f"r(calibration, discrepancy) = {r_cal_disc:.3f}, p = {p_cal_disc:.4f}")
results.append({"test": "calibration_discrepancy_orthogonality", "r": r_cal_disc, "p": p_cal_disc,
                "n": len(calib_df), "note": "orthogonal if |r|<0.15"})

# ══════════════════════════════════════════════════════════════════════════
# 2. RESIDUAL CHOICE SUBOPTIMALITY
# ══════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("2. RESIDUAL CHOICE SUBOPTIMALITY")
print("=" * 70)

# Merge all per-subject data
df = dev.merge(params, on="subj").merge(calib_df, on="subj", how="inner")
df["log_ce"] = np.log(df["c_effort"])
df["log_cd"] = np.log(df["c_death"])

print(f"Subjects with all data: {len(df)}")

# Regress overcautious_rate on log(ce) + log(cd), take residuals
X_base = sm.add_constant(df[["log_ce", "log_cd"]].values)
model_base = sm.OLS(df["overcautious_rate"].values, X_base).fit()
df["residual_overcaution"] = model_base.resid

print(f"\nBase model: overcautious_rate ~ log(ce) + log(cd)")
print(f"  R² = {model_base.rsquared:.4f}")
print(f"  log(ce): beta={model_base.params[1]:.4f}, p={model_base.pvalues[1]:.2e}")
print(f"  log(cd): beta={model_base.params[2]:.4f}, p={model_base.pvalues[2]:.2e}")
results.append({"test": "base_model_overcaution_R2", "r": model_base.rsquared,
                "p": model_base.f_pvalue, "n": len(df)})

# ══════════════════════════════════════════════════════════════════════════
# 3. DISCREPANCY → RESIDUAL OVERCAUTION
# ══════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("3. DISCREPANCY → RESIDUAL OVERCAUTION")
print("=" * 70)

r_disc_resid, p_disc_resid = safe_pearsonr(df["discrepancy"].values, df["residual_overcaution"].values)
print(f"Pearson r(discrepancy, residual_overcaution) = {r_disc_resid:.3f}, p = {p_disc_resid:.4f}")
results.append({"test": "discrepancy_residual_overcaution", "r": r_disc_resid,
                "p": p_disc_resid, "n": len(df),
                "note": "KEY TEST: does affect explain what params cannot?"})

# Hierarchical regression: Step 1 = log(ce) + log(cd), Step 2 = + discrepancy
X_full = sm.add_constant(df[["log_ce", "log_cd", "discrepancy"]].values)
model_full = sm.OLS(df["overcautious_rate"].values, X_full).fit()

delta_R2 = model_full.rsquared - model_base.rsquared
# F-change test for adding discrepancy
n = len(df)
k_base = 3  # intercept + 2 predictors
k_full = 4  # intercept + 3 predictors
f_change = (delta_R2 / (k_full - k_base)) / ((1 - model_full.rsquared) / (n - k_full))
from scipy.stats import f as f_dist
p_f_change = 1 - f_dist.cdf(f_change, k_full - k_base, n - k_full)

print(f"\nHierarchical regression:")
print(f"  Step 1 R² (ce+cd):          {model_base.rsquared:.4f}")
print(f"  Step 2 R² (ce+cd+disc):     {model_full.rsquared:.4f}")
print(f"  ΔR²:                         {delta_R2:.4f}")
print(f"  F-change:                     {f_change:.3f}")
print(f"  p(F-change):                  {p_f_change:.4f}")
print(f"  Discrepancy beta:             {model_full.params[3]:.4f}")
print(f"  Discrepancy t:                {model_full.tvalues[3]:.3f}")
print(f"  Discrepancy p:                {model_full.pvalues[3]:.4f}")
results.append({"test": "hierarchical_discrepancy_delta_R2", "r": delta_R2,
                "p": p_f_change, "n": n,
                "note": f"F_change={f_change:.3f}"})

if p_disc_resid < 0.05:
    print("\n  *** DISCREPANCY PREDICTS RESIDUAL OVERCAUTION ***")
else:
    print("\n  *** DISCREPANCY DOES NOT PREDICT RESIDUAL OVERCAUTION (p>.05) ***")

# ══════════════════════════════════════════════════════════════════════════
# 4. CALIBRATION → POLICY ALIGNMENT
# ══════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("4. CALIBRATION → POLICY ALIGNMENT")
print("=" * 70)

# Policy alignment: how closely actual choices match model predictions
# For each subject, compute |P(heavy)_observed - P(heavy)_predicted| per condition
# Model predicted: P(heavy) = sigmoid(tau * (EV_heavy - EV_light))
# where EV_x = S * R_x - (1-S) * C, S from subject's condition
# Actually simpler: policy alignment = 1 - mean |choice - optimal_choice| = optimality_rate
# But we want something MORE specific: per-condition alignment

# Create per-subject per-condition choice rates
subj_cond_rates = behavior.groupby(["subj", "threat", "distance_H"])["choice"].mean().reset_index()
subj_cond_rates.columns = ["subj", "threat", "distance_H", "obs_heavy_rate"]

# Model predicted rate per condition (using population model)
# P(heavy) = sigmoid(tau * (S*5 - (1-S)*5 - S*1 + (1-S)*5))
# = sigmoid(tau * (S*4)) = sigmoid(4*tau*S)
# Wait, that's the choice between heavy and light given same S
# Actually in the model, the choice utility for cookie j is:
# U_j = S * R_j - (1-S) * C - c_effort * effort_j - c_death * (1-S)
# The subject-level ce and cd vary. Let me use a simpler measure.
#
# Policy alignment = 1 - mean(|observed_choice - optimal_choice|) across trials
# This IS the optimality rate, which we already have.
# Let me instead use: per-condition |obs_rate - optimal_indicator| averaged
# This captures how CLOSELY the subject's choice distribution matches optimal

policy_opt = policy[["threat", "distance_H", "optimal_choice"]].copy()
policy_opt["optimal_heavy_indicator"] = (policy_opt["optimal_choice"] == "heavy").astype(float)

subj_alignment = subj_cond_rates.merge(policy_opt[["threat", "distance_H", "optimal_heavy_indicator"]],
                                        on=["threat", "distance_H"])
subj_alignment["abs_dev"] = np.abs(subj_alignment["obs_heavy_rate"] - subj_alignment["optimal_heavy_indicator"])
policy_alignment_per_subj = subj_alignment.groupby("subj")["abs_dev"].mean().reset_index()
policy_alignment_per_subj.columns = ["subj", "policy_misalignment"]
policy_alignment_per_subj["policy_alignment"] = 1 - policy_alignment_per_subj["policy_misalignment"]

df = df.merge(policy_alignment_per_subj, on="subj", how="left")
print(f"Policy alignment: mean={df['policy_alignment'].mean():.3f}, sd={df['policy_alignment'].std():.3f}")

# Calibration → policy alignment
r_cal_pa, p_cal_pa = safe_pearsonr(df["calibration"].values, df["policy_alignment"].values)
print(f"Pearson r(calibration, policy_alignment) = {r_cal_pa:.3f}, p = {p_cal_pa:.4f}")
results.append({"test": "calibration_policy_alignment_bivariate", "r": r_cal_pa,
                "p": p_cal_pa, "n": len(df)})

# Controlling for ce + cd
r_cal_pa_part, p_cal_pa_part = partial_corr(
    df["calibration"].values, df["policy_alignment"].values,
    df[["log_ce", "log_cd"]].values
)
print(f"Partial r(calibration, policy_alignment | ce+cd) = {r_cal_pa_part:.3f}, p = {p_cal_pa_part:.4f}")
results.append({"test": "calibration_policy_alignment_partial", "r": r_cal_pa_part,
                "p": p_cal_pa_part, "n": len(df)})

# Hierarchical regression for calibration
model_base_pa = sm.OLS(df["policy_alignment"].values, X_base).fit()
X_full_cal = sm.add_constant(df[["log_ce", "log_cd", "calibration"]].values)
model_full_cal = sm.OLS(df["policy_alignment"].values, X_full_cal).fit()

delta_R2_cal = model_full_cal.rsquared - model_base_pa.rsquared
f_change_cal = (delta_R2_cal / 1) / ((1 - model_full_cal.rsquared) / (n - 4))
p_f_change_cal = 1 - f_dist.cdf(f_change_cal, 1, n - 4)

print(f"\nHierarchical regression (policy alignment):")
print(f"  Step 1 R² (ce+cd):          {model_base_pa.rsquared:.4f}")
print(f"  Step 2 R² (ce+cd+cal):      {model_full_cal.rsquared:.4f}")
print(f"  ΔR²:                         {delta_R2_cal:.4f}")
print(f"  F-change:                     {f_change_cal:.3f}")
print(f"  p(F-change):                  {p_f_change_cal:.4f}")
results.append({"test": "hierarchical_calibration_delta_R2", "r": delta_R2_cal,
                "p": p_f_change_cal, "n": n})

# ══════════════════════════════════════════════════════════════════════════
# 5. DISCREPANCY → EXCESS VIGOR AT LOW THREAT
# ══════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("5. DISCREPANCY → EXCESS VIGOR AT LOW THREAT")
print("=" * 70)

# Load vigor and compute mean excess vigor at T=0.1 per subject
print("Loading vigor timeseries...")
vigor = pd.read_parquet(VIGOR_TS)
low_threat_vigor = vigor[vigor["threat"].round(1) == 0.1].groupby("subj")["vigor_resid"].mean().reset_index()
low_threat_vigor.columns = ["subj", "excess_vigor_low_T"]
df = df.merge(low_threat_vigor, on="subj", how="left")

print(f"Excess vigor at T=0.1: mean={df['excess_vigor_low_T'].mean():.4f}, sd={df['excess_vigor_low_T'].std():.4f}")

# Residualize on log(cd)
mask_v = df["excess_vigor_low_T"].notna()
if mask_v.sum() > 10:
    resid_vigor = sm.OLS(
        df.loc[mask_v, "excess_vigor_low_T"].values,
        sm.add_constant(df.loc[mask_v, "log_cd"].values)
    ).fit().resid

    r_disc_vigor, p_disc_vigor = pearsonr(
        df.loc[mask_v, "discrepancy"].values, resid_vigor
    )
    print(f"r(discrepancy, residual_excess_vigor_lowT) = {r_disc_vigor:.3f}, p = {p_disc_vigor:.4f}")
    results.append({"test": "discrepancy_excess_vigor_lowT", "r": r_disc_vigor,
                    "p": p_disc_vigor, "n": mask_v.sum()})
else:
    print("Insufficient data for vigor analysis")
    r_disc_vigor, p_disc_vigor = np.nan, np.nan

# ══════════════════════════════════════════════════════════════════════════
# 6. FOUR-ROUTE INDEPENDENCE MATRIX
# ══════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("6. FOUR-ROUTE INDEPENDENCE MATRIX")
print("=" * 70)

# Load encounter reactivity from dynamics results
dynamics = pd.read_csv(DYNAMICS_FILE)

# Get per-subject reactivity from vigor timeseries
# reactivity = post-encounter minus pre-encounter vigor
# The dynamics file has population-level stats, not per-subject
# Let me compute per-subject reactivity

vigor_atk = vigor[vigor["isAttackTrial"] == 1].copy()
vigor_atk = vigor_atk.dropna(subset=["encounterTime"])

# Encounter-aligned time
vigor_atk["t_enc"] = vigor_atk["t"] - vigor_atk["encounterTime"]

# Pre: t_enc in [-2, 0], Post: t_enc in [0, 2]
pre_vigor = vigor_atk[(vigor_atk["t_enc"] >= -2) & (vigor_atk["t_enc"] < 0)].groupby("subj")["vigor_resid"].mean()
post_vigor = vigor_atk[(vigor_atk["t_enc"] >= 0) & (vigor_atk["t_enc"] <= 2)].groupby("subj")["vigor_resid"].mean()
reactivity = (post_vigor - pre_vigor).reset_index()
reactivity.columns = ["subj", "encounter_reactivity"]

df = df.merge(reactivity, on="subj", how="left")
print(f"Encounter reactivity: mean={df['encounter_reactivity'].mean():.4f}, sd={df['encounter_reactivity'].std():.4f}")

# Compute vigor gap (mean |vigor_resid|)
vigor_gap_subj = vigor.groupby("subj")["vigor_resid"].apply(lambda x: np.abs(x).mean()).reset_index()
vigor_gap_subj.columns = ["subj", "vigor_gap"]
df = df.merge(vigor_gap_subj, on="subj", how="left")

# Now build the 5×5 correlation matrix
predictors = ["log_ce", "log_cd", "calibration", "discrepancy", "encounter_reactivity"]
outcomes = ["overcautious_rate", "survival_rate", "policy_alignment", "residual_overcaution", "total_earnings"]

# Ensure all columns are present
for col in predictors + outcomes:
    if col not in df.columns:
        print(f"WARNING: {col} not in df!")

print(f"\nFull correlation matrix ({len(df)} subjects with complete data):")
print("-" * 100)

corr_matrix = []
for pred in predictors:
    row = {}
    for out in outcomes:
        mask = df[pred].notna() & df[out].notna()
        if mask.sum() >= 10:
            r, p = pearsonr(df.loc[mask, pred], df.loc[mask, out])
            row[out] = {"r": r, "p": p}
        else:
            row[out] = {"r": np.nan, "p": np.nan}
    corr_matrix.append(row)

# Print matrix
header = f"{'Predictor':>25s}" + "".join(f"{o:>20s}" for o in outcomes)
print(header)
for i, pred in enumerate(predictors):
    line = f"{pred:>25s}"
    for out in outcomes:
        r = corr_matrix[i][out]["r"]
        p = corr_matrix[i][out]["p"]
        star = "***" if p < 0.001 else ("**" if p < 0.01 else ("*" if p < 0.05 else ""))
        line += f"  {r:>7.3f}{star:<4s}  "
    print(line)

# Save as flat dataframe
routes_list = []
for i, pred in enumerate(predictors):
    for out in outcomes:
        routes_list.append({
            "predictor": pred,
            "outcome": out,
            "r": corr_matrix[i][out]["r"],
            "p": corr_matrix[i][out]["p"],
        })
routes_df = pd.DataFrame(routes_list)

# Identify primary outcomes
print("\nPrimary pathway identification:")
for i, pred in enumerate(predictors):
    best_out = max(outcomes, key=lambda o: abs(corr_matrix[i][o]["r"]) if not np.isnan(corr_matrix[i][o]["r"]) else 0)
    best_r = corr_matrix[i][best_out]["r"]
    best_p = corr_matrix[i][best_out]["p"]
    print(f"  {pred:>25s} → {best_out:<25s} r={best_r:.3f}, p={best_p:.2e}")

# ══════════════════════════════════════════════════════════════════════════
# 7. CLINICAL CONVERGENCE
# ══════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("7. CLINICAL CONVERGENCE")
print("=" * 70)

df = df.merge(psych[["subj", "STAI_Trait", "STAI_State", "OASIS_Total",
                       "DASS21_Anxiety", "DASS21_Depression", "DASS21_Stress",
                       "PHQ9_Total", "AMI_Total", "MFIS_Total", "STICSA_Total"]],
               on="subj", how="left")

clinical_tests = [
    ("discrepancy", "STAI_Trait"), ("discrepancy", "OASIS_Total"),
    ("discrepancy", "DASS21_Anxiety"), ("discrepancy", "STICSA_Total"),
    ("encounter_reactivity", "AMI_Total"),
    ("calibration", "STAI_Trait"), ("calibration", "OASIS_Total"),
]

print(f"\nClinical associations (N≈{len(df)}):")
for pred, clin in clinical_tests:
    mask = df[pred].notna() & df[clin].notna()
    if mask.sum() >= 10:
        r, p = pearsonr(df.loc[mask, pred], df.loc[mask, clin])
        star = "***" if p < 0.001 else ("**" if p < 0.01 else ("*" if p < 0.05 else ""))
        print(f"  r({pred}, {clin}) = {r:.3f}, p = {p:.4f} {star}")
        results.append({"test": f"clinical_{pred}_{clin}", "r": r, "p": p, "n": mask.sum()})
    else:
        print(f"  {pred} vs {clin}: insufficient data")

# ══════════════════════════════════════════════════════════════════════════
# 8. FIGURE
# ══════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("8. GENERATING FIGURE")
print("=" * 70)

fig = plt.figure(figsize=(16, 12))
gs = GridSpec(2, 2, hspace=0.35, wspace=0.35)

# Panel A: Discrepancy → residual overcaution scatter
ax_a = fig.add_subplot(gs[0, 0])
mask_a = df["discrepancy"].notna() & df["residual_overcaution"].notna()
ax_a.scatter(df.loc[mask_a, "discrepancy"], df.loc[mask_a, "residual_overcaution"],
             alpha=0.3, s=15, c="crimson")
m, b_val = np.polyfit(df.loc[mask_a, "discrepancy"], df.loc[mask_a, "residual_overcaution"], 1)
x_range = np.linspace(df["discrepancy"].min(), df["discrepancy"].max(), 100)
ax_a.plot(x_range, m * x_range + b_val, "k-", lw=2)
ax_a.set_xlabel("Affective discrepancy")
ax_a.set_ylabel("Residual overcaution\n(after removing ce + cd)")
ax_a.set_title(f"A. Discrepancy → residual overcaution\nr={r_disc_resid:.3f}, p={p_disc_resid:.4f}",
               fontweight="bold")
ax_a.axhline(0, color="gray", ls="--", lw=0.5)
ax_a.axvline(0, color="gray", ls="--", lw=0.5)

# Panel B: Calibration → policy alignment scatter
ax_b = fig.add_subplot(gs[0, 1])
mask_b = df["calibration"].notna() & df["policy_alignment"].notna()
ax_b.scatter(df.loc[mask_b, "calibration"], df.loc[mask_b, "policy_alignment"],
             alpha=0.3, s=15, c="steelblue")
m2, b2 = np.polyfit(df.loc[mask_b, "calibration"], df.loc[mask_b, "policy_alignment"], 1)
x_range2 = np.linspace(df["calibration"].min(), df["calibration"].max(), 100)
ax_b.plot(x_range2, m2 * x_range2 + b2, "k-", lw=2)
ax_b.set_xlabel("Affective calibration\n(within-subject r(anxiety, danger))")
ax_b.set_ylabel("Policy alignment")
ax_b.set_title(f"B. Calibration → policy alignment\nr={r_cal_pa:.3f}, p={p_cal_pa:.4f}",
               fontweight="bold")

# Panel C: Four-route heatmap
ax_c = fig.add_subplot(gs[1, 0])
r_matrix = np.zeros((len(predictors), len(outcomes)))
p_matrix = np.zeros((len(predictors), len(outcomes)))
for i, pred in enumerate(predictors):
    for j, out in enumerate(outcomes):
        r_matrix[i, j] = corr_matrix[i][out]["r"]
        p_matrix[i, j] = corr_matrix[i][out]["p"]

im = ax_c.imshow(r_matrix, cmap="RdBu_r", vmin=-0.5, vmax=0.5, aspect="auto")
for i in range(len(predictors)):
    for j in range(len(outcomes)):
        r_val = r_matrix[i, j]
        p_val = p_matrix[i, j]
        star = "***" if p_val < 0.001 else ("**" if p_val < 0.01 else ("*" if p_val < 0.05 else ""))
        color = "white" if abs(r_val) > 0.25 else "black"
        ax_c.text(j, i, f"{r_val:.2f}{star}", ha="center", va="center",
                  fontsize=7, color=color)

pred_labels = ["log(ce)", "log(cd)", "calibration", "discrepancy", "reactivity"]
out_labels = ["overcaution", "survival", "alignment", "resid_OC", "earnings"]
ax_c.set_xticks(range(len(outcomes)))
ax_c.set_xticklabels(out_labels, fontsize=8, rotation=30, ha="right")
ax_c.set_yticks(range(len(predictors)))
ax_c.set_yticklabels(pred_labels, fontsize=8)
ax_c.set_title("C. Five-route correlation matrix", fontweight="bold")
plt.colorbar(im, ax=ax_c, label="Pearson r", shrink=0.8)

# Panel D: Hierarchical R² bar chart
ax_d = fig.add_subplot(gs[1, 1])

# Three models for overcaution
r2_base = model_base.rsquared
r2_disc = model_full.rsquared  # + discrepancy

# Rerun with calibration for comparison
X_full_cal_oc = sm.add_constant(df[["log_ce", "log_cd", "calibration"]].values)
model_cal_oc = sm.OLS(df["overcautious_rate"].values, X_full_cal_oc).fit()
r2_cal_oc = model_cal_oc.rsquared

# Also both
X_both = sm.add_constant(df[["log_ce", "log_cd", "discrepancy", "calibration"]].values)
model_both = sm.OLS(df["overcautious_rate"].values, X_both).fit()
r2_both = model_both.rsquared

# For policy alignment
r2_pa_base = model_base_pa.rsquared
r2_pa_cal = model_full_cal.rsquared

labels = ["ce+cd\n(overcaution)", "+discrepancy", "+calibration", "+both",
          "ce+cd\n(alignment)", "+calibration"]
r2_values = [r2_base, r2_disc, r2_cal_oc, r2_both, r2_pa_base, r2_pa_cal]
colors = ["lightgray", "crimson", "steelblue", "purple", "lightgray", "steelblue"]

bars = ax_d.bar(range(len(labels)), r2_values, color=colors, edgecolor="black", linewidth=0.5)
ax_d.set_xticks(range(len(labels)))
ax_d.set_xticklabels(labels, fontsize=7, rotation=30, ha="right")
ax_d.set_ylabel("R²")
ax_d.set_title("D. Hierarchical R² comparison", fontweight="bold")

# Add value labels
for i, (bar, val) in enumerate(zip(bars, r2_values)):
    ax_d.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.002,
              f"{val:.3f}", ha="center", va="bottom", fontsize=7)

# Add divider
ax_d.axvline(3.5, color="gray", ls="--", lw=0.5)

fig.savefig(OUT_FIG, dpi=200, bbox_inches="tight")
print(f"Saved figure: {OUT_FIG}")

# ── Save ──
results_df = pd.DataFrame(results)
results_df.to_csv(OUT_STATS, index=False)
routes_df.to_csv(OUT_ROUTES, index=False)
print(f"Saved: {OUT_STATS}")
print(f"Saved: {OUT_ROUTES}")

# ══════════════════════════════════════════════════════════════════════════
# SUMMARY
# ══════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("SUMMARY OF KEY RESULTS")
print("=" * 70)
print(f"\n1. Calibration-Discrepancy orthogonality: r={r_cal_disc:.3f}, p={p_cal_disc:.4f}")
print(f"   {'PASS: orthogonal' if abs(r_cal_disc) < 0.15 else 'FAIL: not orthogonal'}")
print(f"\n2. Discrepancy → residual overcaution: r={r_disc_resid:.3f}, p={p_disc_resid:.4f}")
print(f"   {'PASS' if p_disc_resid < 0.05 else 'FAIL'}")
print(f"   Hierarchical ΔR² = {delta_R2:.4f}, F={f_change:.3f}, p={p_f_change:.4f}")
print(f"\n3. Calibration → policy alignment: r={r_cal_pa:.3f}, p={p_cal_pa:.4f}")
print(f"   Partial (controlling ce+cd): r={r_cal_pa_part:.3f}, p={p_cal_pa_part:.4f}")
print(f"   {'PASS' if p_cal_pa_part < 0.05 else 'FAIL'}")
print(f"\n4. Discrepancy → excess vigor at low T: r={r_disc_vigor:.3f}, p={p_disc_vigor:.4f}")

print("\nDone.")
