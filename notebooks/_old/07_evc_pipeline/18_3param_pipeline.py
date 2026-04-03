#!/usr/bin/env python3
"""
18_3param_pipeline.py — Full analysis pipeline for the 3-param v2 model (k + beta + cd)
========================================================================================

Runs all downstream analyses with the new model parameters:
  1. Optimal policy & deviations
  2. k→distance-overcaution, β→threat-overcaution, cd→vigor gap (TRIPLE dissociation)
  3. Affect: calibration & discrepancy (no gamma/epsilon in S for affect)
  4. Clinical associations
  5. Encounter dynamics with new params
  6. New explorations: k×β profiles, β→anxiety, triple dissociation matrix

Uses params from: results/stats/oc_evc_3param_v2_params.csv
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import pearsonr, f as f_dist
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
PARAMS_FILE = Path("/workspace/results/stats/oc_evc_3param_v2_params.csv")
POP_FILE = Path("/workspace/results/stats/oc_evc_3param_v2_population.csv")
VIGOR_TS = Path("/workspace/data/exploratory_350/processed/vigor_processed/smoothed_vigor_ts.parquet")
OUT_DIR = Path("/workspace/results/stats")
FIG_DIR = Path("/workspace/results/figs/paper")
FIG_DIR.mkdir(parents=True, exist_ok=True)

def safe_pearsonr(x, y):
    mask = np.isfinite(x) & np.isfinite(y)
    if mask.sum() < 10:
        return np.nan, np.nan
    return pearsonr(x[mask], y[mask])

def partial_corr(x, y, Z):
    """Partial correlation of x and y, controlling for columns in Z."""
    if Z.ndim == 1:
        Z = Z.reshape(-1, 1)
    mask = np.isfinite(x) & np.isfinite(y) & np.all(np.isfinite(Z), axis=1)
    x, y, Z = x[mask], y[mask], Z[mask]
    rx = sm.OLS(x, sm.add_constant(Z)).fit().resid
    ry = sm.OLS(y, sm.add_constant(Z)).fit().resid
    return pearsonr(rx, ry)

# ── Load data ──
print("=" * 70)
print("3-PARAM v2 PIPELINE (k + β + cd, no γ/ε)")
print("=" * 70)

behavior = pd.read_csv(DATA_DIR / "behavior.csv")
behavior_rich = pd.read_csv(DATA_DIR / "behavior_rich.csv", low_memory=False)
feelings = pd.read_csv(DATA_DIR / "feelings.csv")
psych = pd.read_csv(DATA_DIR / "psych.csv")
params = pd.read_csv(PARAMS_FILE)
pop = pd.read_csv(POP_FILE)

params.rename(columns={"subj": "subj"}, inplace=True)
params["log_k"] = np.log(params["k"])
params["log_beta"] = np.log(params["beta"])
params["log_cd"] = np.log(params["c_death"])

TAU = float(pop["tau"].iloc[0])
P_ESC = float(pop["p_esc"].iloc[0])

print(f"Behavior: {behavior.shape[0]} trials, {behavior['subj'].nunique()} subjects")
print(f"Params: {len(params)} subjects (k, beta, cd)")
print(f"Population: tau={TAU:.3f}, p_esc={P_ESC:.4f}")

all_results = []

# ══════════════════════════════════════════════════════════════════════════
# SECTION 1: OPTIMAL POLICY AND DEVIATIONS
# ══════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("SECTION 1: OPTIMAL POLICY")
print("=" * 70)

behavior_rich["survived"] = (behavior_rich["trialEndState"] == "escaped").astype(int)

conditions = []
for t in [0.1, 0.5, 0.9]:
    for d in [1, 2, 3]:
        mask_h = (behavior_rich["threat"].round(1) == t) & (behavior_rich["choice"] == 1) & (behavior_rich["distance_H"] == d)
        mask_l = (behavior_rich["threat"].round(1) == t) & (behavior_rich["choice"] == 0) & (behavior_rich["distance_H"] == d)
        surv_h = behavior_rich.loc[mask_h, "survived"].mean() if mask_h.sum() > 0 else np.nan
        surv_l = behavior_rich.loc[mask_l, "survived"].mean() if mask_l.sum() > 0 else np.nan
        ev_h = surv_h * 5 - (1 - surv_h) * 5 if not np.isnan(surv_h) else np.nan
        ev_l = surv_l * 1 - (1 - surv_l) * 5 if not np.isnan(surv_l) else np.nan
        optimal = "heavy" if (not np.isnan(ev_h) and not np.isnan(ev_l) and ev_h > ev_l) else "light"
        conditions.append({
            "threat": t, "distance_H": d,
            "surv_heavy": surv_h, "surv_light": surv_l,
            "ev_heavy": ev_h, "ev_light": ev_l,
            "optimal_choice": optimal,
            "ev_advantage": ev_h - ev_l if not (np.isnan(ev_h) or np.isnan(ev_l)) else np.nan,
        })

policy_df = pd.DataFrame(conditions)

print("\nEV-optimal policy:")
for _, row in policy_df.iterrows():
    print(f"  T={row['threat']:.1f} D={row['distance_H']:.0f}: EV_H={row['ev_heavy']:+.2f} EV_L={row['ev_light']:+.2f} → {row['optimal_choice']}")

n_obj_heavy = (policy_df["optimal_choice"] == "heavy").sum()
print(f"\n{n_obj_heavy}/9 conditions favor heavy objectively")

# Classify trials
optimal_lookup = {}
ev_lookup = {}
for _, row in policy_df.iterrows():
    key = (row["threat"], int(row["distance_H"]))
    optimal_lookup[key] = 1 if row["optimal_choice"] == "heavy" else 0
    ev_lookup[key] = {"ev_heavy": row["ev_heavy"], "ev_light": row["ev_light"]}

behavior["optimal_choice"] = behavior.apply(
    lambda r: optimal_lookup.get((r["threat"], r["distance_H"]), np.nan), axis=1)
behavior["is_overcautious"] = ((behavior["choice"] == 0) & (behavior["optimal_choice"] == 1)).astype(int)
behavior["is_overrisky"] = ((behavior["choice"] == 1) & (behavior["optimal_choice"] == 0)).astype(int)
behavior["is_optimal"] = (behavior["choice"] == behavior["optimal_choice"]).astype(int)

# EV loss
def compute_ev_loss(row):
    evs = ev_lookup.get((row["threat"], row["distance_H"]), {})
    ev_chosen = evs.get("ev_heavy", 0) if row["choice"] == 1 else evs.get("ev_light", 0)
    ev_optimal = max(evs.get("ev_heavy", 0), evs.get("ev_light", 0))
    return ev_optimal - ev_chosen

behavior["ev_loss"] = behavior.apply(compute_ev_loss, axis=1)

# Per-subject deviations
subj_devs = []
for s, sdf in behavior.groupby("subj"):
    heavy_rate = sdf["choice"].mean()
    survival_rate = (sdf["outcome"] == 0).mean()
    total_earnings = ((sdf["outcome"] == 0) * sdf["choice"].map({1: 5, 0: 1})).sum()

    # Separate overcaution into distance-driven and threat-driven
    # Distance-driven: overcautious at low T (choosing light when heavy is optimal at T=0.1)
    low_t = sdf[sdf["threat"].round(1) == 0.1]
    high_t = sdf[sdf["threat"].round(1) == 0.9]

    subj_devs.append({
        "subj": s,
        "optimality_rate": sdf["is_optimal"].mean(),
        "overcautious_rate": sdf["is_overcautious"].mean(),
        "overrisky_rate": sdf["is_overrisky"].mean(),
        "mean_ev_loss": sdf["ev_loss"].mean(),
        "heavy_rate": heavy_rate,
        "survival_rate": survival_rate,
        "total_earnings": total_earnings,
        "heavy_rate_lowT": low_t["choice"].mean() if len(low_t) > 0 else np.nan,
        "heavy_rate_highT": high_t["choice"].mean() if len(high_t) > 0 else np.nan,
        "threat_sensitivity_choice": (low_t["choice"].mean() - high_t["choice"].mean()) if len(low_t) > 0 and len(high_t) > 0 else np.nan,
    })

dev_df = pd.DataFrame(subj_devs)
print(f"\nPer-subject deviations (N={len(dev_df)}):")
print(f"  Overcautious rate: {dev_df['overcautious_rate'].mean():.3f} ± {dev_df['overcautious_rate'].std():.3f}")
print(f"  Overrisky rate: {dev_df['overrisky_rate'].mean():.3f} ± {dev_df['overrisky_rate'].std():.3f}")
print(f"  Threat sensitivity (ΔP(heavy) low-high T): {dev_df['threat_sensitivity_choice'].mean():.3f}")

# ══════════════════════════════════════════════════════════════════════════
# SECTION 2: TRIPLE DISSOCIATION — k, β, cd → specific deviations
# ══════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("SECTION 2: TRIPLE DISSOCIATION")
print("=" * 70)

df = dev_df.merge(params, on="subj")

# --- k → overcautious rate (effort-driven avoidance) ---
r_k_oc, p_k_oc = safe_pearsonr(df["log_k"].values, df["overcautious_rate"].values)
r_k_oc_part, p_k_oc_part = partial_corr(df["log_k"].values, df["overcautious_rate"].values,
                                          df[["log_beta", "log_cd"]].values)
print(f"\nk → overcautious rate:")
print(f"  Bivariate: r={r_k_oc:.3f}, p={p_k_oc:.2e}")
print(f"  Partial (|β,cd): r={r_k_oc_part:.3f}, p={p_k_oc_part:.2e}")
all_results.append({"test": "k_overcautious_bivariate", "r": r_k_oc, "p": p_k_oc})
all_results.append({"test": "k_overcautious_partial", "r": r_k_oc_part, "p": p_k_oc_part})

# --- β → threat sensitivity in choice ---
r_b_ts, p_b_ts = safe_pearsonr(df["log_beta"].values, df["threat_sensitivity_choice"].values)
r_b_ts_part, p_b_ts_part = partial_corr(df["log_beta"].values, df["threat_sensitivity_choice"].values,
                                          df[["log_k", "log_cd"]].values)
print(f"\nβ → threat sensitivity (ΔP(heavy) low-high T):")
print(f"  Bivariate: r={r_b_ts:.3f}, p={p_b_ts:.2e}")
print(f"  Partial (|k,cd): r={r_b_ts_part:.3f}, p={p_b_ts_part:.2e}")
all_results.append({"test": "beta_threat_sensitivity_bivariate", "r": r_b_ts, "p": p_b_ts})
all_results.append({"test": "beta_threat_sensitivity_partial", "r": r_b_ts_part, "p": p_b_ts_part})

# --- β → overcautious rate (should also contribute) ---
r_b_oc, p_b_oc = safe_pearsonr(df["log_beta"].values, df["overcautious_rate"].values)
r_b_oc_part, p_b_oc_part = partial_corr(df["log_beta"].values, df["overcautious_rate"].values,
                                          df[["log_k", "log_cd"]].values)
print(f"\nβ → overcautious rate:")
print(f"  Bivariate: r={r_b_oc:.3f}, p={p_b_oc:.2e}")
print(f"  Partial (|k,cd): r={r_b_oc_part:.3f}, p={p_b_oc_part:.2e}")
all_results.append({"test": "beta_overcautious_bivariate", "r": r_b_oc, "p": p_b_oc})
all_results.append({"test": "beta_overcautious_partial", "r": r_b_oc_part, "p": p_b_oc_part})

# --- β → heavy rate at high T specifically ---
r_b_ht, p_b_ht = safe_pearsonr(df["log_beta"].values, df["heavy_rate_highT"].values)
print(f"\nβ → P(heavy) at T=0.9: r={r_b_ht:.3f}, p={p_b_ht:.2e}")
all_results.append({"test": "beta_heavy_highT", "r": r_b_ht, "p": p_b_ht})

# --- k → heavy rate at low T ---
r_k_lt, p_k_lt = safe_pearsonr(df["log_k"].values, df["heavy_rate_lowT"].values)
print(f"k → P(heavy) at T=0.1: r={r_k_lt:.3f}, p={p_k_lt:.2e}")
all_results.append({"test": "k_heavy_lowT", "r": r_k_lt, "p": p_k_lt})

# --- cd → vigor gap ---
print("\nLoading vigor timeseries...")
vigor = pd.read_parquet(VIGOR_TS)
vigor_gap_subj = vigor.groupby("subj")["vigor_resid"].apply(lambda x: np.abs(x).mean()).reset_index()
vigor_gap_subj.columns = ["subj", "vigor_gap"]
df = df.merge(vigor_gap_subj, on="subj", how="left")

r_cd_vg, p_cd_vg = safe_pearsonr(df["log_cd"].values, df["vigor_gap"].values)
r_cd_vg_part, p_cd_vg_part = partial_corr(df["log_cd"].values, df["vigor_gap"].values,
                                            df[["log_k", "log_beta"]].values)
print(f"\ncd → vigor gap:")
print(f"  Bivariate: r={r_cd_vg:.3f}, p={p_cd_vg:.2e}")
print(f"  Partial (|k,β): r={r_cd_vg_part:.3f}, p={p_cd_vg_part:.2e}")
all_results.append({"test": "cd_vigor_gap_bivariate", "r": r_cd_vg, "p": p_cd_vg})
all_results.append({"test": "cd_vigor_gap_partial", "r": r_cd_vg_part, "p": p_cd_vg_part})

# --- Multiple regressions with 3 params ---
print("\n--- Multiple Regression: overcautious_rate ~ log(k) + log(β) + log(cd) ---")
X3 = sm.add_constant(df[["log_k", "log_beta", "log_cd"]].values)
model_oc3 = sm.OLS(df["overcautious_rate"].values, X3).fit()
print(f"  R² = {model_oc3.rsquared:.4f}")
print(f"  log(k):    β={model_oc3.params[1]:.4f}, t={model_oc3.tvalues[1]:.3f}, p={model_oc3.pvalues[1]:.2e}")
print(f"  log(beta): β={model_oc3.params[2]:.4f}, t={model_oc3.tvalues[2]:.3f}, p={model_oc3.pvalues[2]:.2e}")
print(f"  log(cd):   β={model_oc3.params[3]:.4f}, t={model_oc3.tvalues[3]:.3f}, p={model_oc3.pvalues[3]:.2e}")
all_results.append({"test": "overcautious_3param_R2", "r": model_oc3.rsquared, "p": model_oc3.f_pvalue})

# Unique R² for each
model_no_k = sm.OLS(df["overcautious_rate"].values, sm.add_constant(df[["log_beta", "log_cd"]].values)).fit()
model_no_b = sm.OLS(df["overcautious_rate"].values, sm.add_constant(df[["log_k", "log_cd"]].values)).fit()
model_no_cd = sm.OLS(df["overcautious_rate"].values, sm.add_constant(df[["log_k", "log_beta"]].values)).fit()
unique_k = model_oc3.rsquared - model_no_k.rsquared
unique_b = model_oc3.rsquared - model_no_b.rsquared
unique_cd = model_oc3.rsquared - model_no_cd.rsquared
print(f"  Unique R² — k: {unique_k:.4f}, β: {unique_b:.4f}, cd: {unique_cd:.4f}")
all_results.append({"test": "overcautious_unique_R2_k", "r": unique_k, "p": model_oc3.pvalues[1]})
all_results.append({"test": "overcautious_unique_R2_beta", "r": unique_b, "p": model_oc3.pvalues[2]})
all_results.append({"test": "overcautious_unique_R2_cd", "r": unique_cd, "p": model_oc3.pvalues[3]})

# ══════════════════════════════════════════════════════════════════════════
# SECTION 3: AFFECT — CALIBRATION & DISCREPANCY
# ══════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("SECTION 3: AFFECT — CALIBRATION & DISCREPANCY")
print("=" * 70)

# In the v2 model we have no gamma/epsilon. For affect, danger = T directly.
# S_affect = 1 - T (simple, no probability weighting)
anxiety_df = feelings[feelings["questionLabel"] == "anxiety"].copy()
anxiety_df["danger"] = anxiety_df["threat"]  # danger = T directly

# Population regression: anxiety ~ danger
pop_slope, pop_intercept = np.polyfit(anxiety_df["danger"].values, anxiety_df["response"].values, 1)
print(f"Population model: anxiety = {pop_intercept:.3f} + {pop_slope:.3f} * danger(T)")

calib_list = []
for s, sdf in anxiety_df.groupby("subj"):
    danger = sdf["danger"].values
    anxiety = sdf["response"].values
    if np.std(danger) > 0 and np.std(anxiety) > 0:
        r_cal, _ = pearsonr(anxiety, danger)
    else:
        r_cal = np.nan
    predicted = pop_slope * danger + pop_intercept
    disc = (anxiety - predicted).mean()
    calib_list.append({"subj": s, "calibration": r_cal, "discrepancy": disc})

calib_df = pd.DataFrame(calib_list).dropna(subset=["calibration", "discrepancy"])
print(f"Subjects with valid calibration/discrepancy: {len(calib_df)}")

r_cal_disc, p_cal_disc = safe_pearsonr(calib_df["calibration"].values, calib_df["discrepancy"].values)
print(f"r(calibration, discrepancy) = {r_cal_disc:.3f}, p = {p_cal_disc:.4f}")
print(f"  {'PASS: orthogonal' if abs(r_cal_disc) < 0.15 else 'NOTE: |r|>0.15'}")
all_results.append({"test": "calibration_discrepancy_orthogonality", "r": r_cal_disc, "p": p_cal_disc})

# Merge affect with main df
df = df.merge(calib_df, on="subj", how="inner")
print(f"Subjects with all data: {len(df)}")

# Residual overcaution after removing k + β + cd
X_base = sm.add_constant(df[["log_k", "log_beta", "log_cd"]].values)
model_base = sm.OLS(df["overcautious_rate"].values, X_base).fit()
df["residual_overcaution"] = model_base.resid
print(f"\nBase model (k+β+cd → overcaution): R² = {model_base.rsquared:.4f}")
all_results.append({"test": "base_overcaution_R2_3param", "r": model_base.rsquared, "p": model_base.f_pvalue})

# Discrepancy → residual overcaution
r_disc_resid, p_disc_resid = safe_pearsonr(df["discrepancy"].values, df["residual_overcaution"].values)
print(f"\nDiscrepancy → residual overcaution: r={r_disc_resid:.3f}, p={p_disc_resid:.4f}")
all_results.append({"test": "discrepancy_residual_overcaution", "r": r_disc_resid, "p": p_disc_resid})

# Hierarchical ΔR²
X_full_disc = sm.add_constant(df[["log_k", "log_beta", "log_cd", "discrepancy"]].values)
model_full_disc = sm.OLS(df["overcautious_rate"].values, X_full_disc).fit()
delta_R2_disc = model_full_disc.rsquared - model_base.rsquared
n = len(df)
f_change = (delta_R2_disc / 1) / ((1 - model_full_disc.rsquared) / (n - 5))
p_f_change = 1 - f_dist.cdf(f_change, 1, n - 5)
print(f"  Hierarchical ΔR² = {delta_R2_disc:.4f}, F={f_change:.3f}, p={p_f_change:.4f}")
all_results.append({"test": "hierarchical_discrepancy_deltaR2", "r": delta_R2_disc, "p": p_f_change})

# Policy alignment
subj_cond_rates = behavior.groupby(["subj", "threat", "distance_H"])["choice"].mean().reset_index()
subj_cond_rates.columns = ["subj", "threat", "distance_H", "obs_heavy_rate"]
policy_opt = policy_df[["threat", "distance_H", "optimal_choice"]].copy()
policy_opt["optimal_heavy"] = (policy_opt["optimal_choice"] == "heavy").astype(float)
subj_align = subj_cond_rates.merge(policy_opt[["threat", "distance_H", "optimal_heavy"]], on=["threat", "distance_H"])
subj_align["abs_dev"] = np.abs(subj_align["obs_heavy_rate"] - subj_align["optimal_heavy"])
pa_subj = subj_align.groupby("subj")["abs_dev"].mean().reset_index()
pa_subj.columns = ["subj", "policy_misalignment"]
pa_subj["policy_alignment"] = 1 - pa_subj["policy_misalignment"]
df = df.merge(pa_subj, on="subj", how="left")

# Calibration → policy alignment
r_cal_pa, p_cal_pa = safe_pearsonr(df["calibration"].values, df["policy_alignment"].values)
r_cal_pa_part, p_cal_pa_part = partial_corr(
    df["calibration"].values, df["policy_alignment"].values,
    df[["log_k", "log_beta", "log_cd"]].values)
print(f"\nCalibration → policy alignment:")
print(f"  Bivariate: r={r_cal_pa:.3f}, p={p_cal_pa:.4f}")
print(f"  Partial (|k,β,cd): r={r_cal_pa_part:.3f}, p={p_cal_pa_part:.4f}")
all_results.append({"test": "calibration_alignment_bivariate", "r": r_cal_pa, "p": p_cal_pa})
all_results.append({"test": "calibration_alignment_partial", "r": r_cal_pa_part, "p": p_cal_pa_part})

# Hierarchical ΔR² for calibration
model_base_pa = sm.OLS(df["policy_alignment"].values, X_base).fit()
X_full_cal = sm.add_constant(df[["log_k", "log_beta", "log_cd", "calibration"]].values)
model_full_cal = sm.OLS(df["policy_alignment"].values, X_full_cal).fit()
delta_R2_cal = model_full_cal.rsquared - model_base_pa.rsquared
f_change_cal = (delta_R2_cal / 1) / ((1 - model_full_cal.rsquared) / (n - 5))
p_f_change_cal = 1 - f_dist.cdf(f_change_cal, 1, n - 5)
print(f"  Hierarchical ΔR² = {delta_R2_cal:.4f}, F={f_change_cal:.3f}, p={p_f_change_cal:.4f}")
all_results.append({"test": "hierarchical_calibration_deltaR2", "r": delta_R2_cal, "p": p_f_change_cal})

# ══════════════════════════════════════════════════════════════════════════
# SECTION 4: ENCOUNTER DYNAMICS
# ══════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("SECTION 4: ENCOUNTER DYNAMICS")
print("=" * 70)

vigor_atk = vigor[vigor["isAttackTrial"] == 1].copy()
vigor_atk = vigor_atk.dropna(subset=["encounterTime"])
vigor_atk["t_enc"] = vigor_atk["t"] - vigor_atk["encounterTime"]

pre_vigor = vigor_atk[(vigor_atk["t_enc"] >= -2) & (vigor_atk["t_enc"] < 0)].groupby("subj")["vigor_resid"].mean()
post_vigor = vigor_atk[(vigor_atk["t_enc"] >= 0) & (vigor_atk["t_enc"] <= 2)].groupby("subj")["vigor_resid"].mean()
reactivity = (post_vigor - pre_vigor).reset_index()
reactivity.columns = ["subj", "encounter_reactivity"]
df = df.merge(reactivity, on="subj", how="left")

# Reactivity → cd
r_react_cd, p_react_cd = safe_pearsonr(df["encounter_reactivity"].values, df["log_cd"].values)
print(f"Reactivity → log(cd): r={r_react_cd:.3f}, p={p_react_cd:.2e}")
all_results.append({"test": "reactivity_cd", "r": r_react_cd, "p": p_react_cd})

# Reactivity → β (NEW: does encounter reactivity relate to threat sensitivity?)
r_react_b, p_react_b = safe_pearsonr(df["encounter_reactivity"].values, df["log_beta"].values)
print(f"Reactivity → log(β): r={r_react_b:.3f}, p={p_react_b:.2e}")
all_results.append({"test": "reactivity_beta", "r": r_react_b, "p": p_react_b})

# Reactivity → k (should be independent)
r_react_k, p_react_k = safe_pearsonr(df["encounter_reactivity"].values, df["log_k"].values)
print(f"Reactivity → log(k): r={r_react_k:.3f}, p={p_react_k:.2e}")
all_results.append({"test": "reactivity_k", "r": r_react_k, "p": p_react_k})

# ══════════════════════════════════════════════════════════════════════════
# SECTION 5: CLINICAL ASSOCIATIONS
# ══════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("SECTION 5: CLINICAL ASSOCIATIONS")
print("=" * 70)

clinical_cols = ["STAI_Trait", "STAI_State", "OASIS_Total", "DASS21_Anxiety",
                 "DASS21_Depression", "DASS21_Stress", "PHQ9_Total", "AMI_Total",
                 "MFIS_Total", "STICSA_Total"]
df = df.merge(psych[["subj"] + clinical_cols], on="subj", how="left")

# β → anxiety measures (NEW: β is now the threat sensitivity parameter!)
print("\nβ → Clinical (THREAT SENSITIVITY → ANXIETY?):")
for clin in clinical_cols:
    r, p = safe_pearsonr(df["log_beta"].values, df[clin].values)
    star = "***" if p < 0.001 else ("**" if p < 0.01 else ("*" if p < 0.05 else ""))
    print(f"  β → {clin:<20s}: r={r:+.3f}, p={p:.4f} {star}")
    all_results.append({"test": f"beta_{clin}", "r": r, "p": p})

# k → clinical
print("\nk → Clinical (EFFORT AVERSION → APATHY?):")
for clin in clinical_cols:
    r, p = safe_pearsonr(df["log_k"].values, df[clin].values)
    star = "***" if p < 0.001 else ("**" if p < 0.01 else ("*" if p < 0.05 else ""))
    print(f"  k → {clin:<20s}: r={r:+.3f}, p={p:.4f} {star}")
    all_results.append({"test": f"k_{clin}", "r": r, "p": p})

# cd → clinical
print("\ncd → Clinical:")
for clin in clinical_cols:
    r, p = safe_pearsonr(df["log_cd"].values, df[clin].values)
    star = "***" if p < 0.001 else ("**" if p < 0.01 else ("*" if p < 0.05 else ""))
    print(f"  cd → {clin:<20s}: r={r:+.3f}, p={p:.4f} {star}")
    all_results.append({"test": f"cd_{clin}", "r": r, "p": p})

# Discrepancy → clinical
print("\nDiscrepancy → Clinical:")
for clin in clinical_cols:
    r, p = safe_pearsonr(df["discrepancy"].values, df[clin].values)
    star = "***" if p < 0.001 else ("**" if p < 0.01 else ("*" if p < 0.05 else ""))
    print(f"  disc → {clin:<20s}: r={r:+.3f}, p={p:.4f} {star}")
    all_results.append({"test": f"discrepancy_{clin}", "r": r, "p": p})

# ══════════════════════════════════════════════════════════════════════════
# SECTION 6: BEHAVIORAL PROFILES (k × β × cd)
# ══════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("SECTION 6: BEHAVIORAL PROFILES")
print("=" * 70)

df["k_group"] = np.where(df["log_k"] >= df["log_k"].median(), "high_k", "low_k")
df["beta_group"] = np.where(df["log_beta"] >= df["log_beta"].median(), "high_β", "low_β")

profile_names = {
    "low_k_low_β": "Bold",
    "low_k_high_β": "Threat-sensitive",
    "high_k_low_β": "Effort-averse",
    "high_k_high_β": "Cautious",
}

print(f"\n{'Profile':<20} {'Label':<20} {'N':>5} {'Earnings':>10} {'P(heavy)':>10} {'Overcaut':>10} {'Threat_sens':>12}")
for prof_key, label in profile_names.items():
    k_g, b_g = prof_key.split("_", 1)[0] + "_k", prof_key.split("_")[1] + "_β"
    # Parse the key properly
    parts = prof_key.split("_")
    k_val = parts[0] + "_k"
    b_val = parts[1] + "_β"
    pf = df[(df["k_group"] == k_val) & (df["beta_group"] == b_val)]
    if len(pf) > 0:
        print(f"  {label:<20} {len(pf):>5} {pf['total_earnings'].mean():>10.1f} "
              f"{pf['heavy_rate'].mean():>10.3f} {pf['overcautious_rate'].mean():>10.3f} "
              f"{pf['threat_sensitivity_choice'].mean():>12.3f}")

# ══════════════════════════════════════════════════════════════════════════
# SECTION 7: FULL CORRELATION MATRIX (6 predictors × 6 outcomes)
# ══════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("SECTION 7: FULL CORRELATION MATRIX")
print("=" * 70)

predictors = ["log_k", "log_beta", "log_cd", "calibration", "discrepancy", "encounter_reactivity"]
outcomes = ["overcautious_rate", "threat_sensitivity_choice", "vigor_gap",
            "policy_alignment", "residual_overcaution", "total_earnings"]

print(f"\n{'':>25s}", end="")
for out in outcomes:
    print(f"  {out[:15]:>15s}", end="")
print()

for pred in predictors:
    print(f"  {pred:>23s}", end="")
    for out in outcomes:
        mask = df[pred].notna() & df[out].notna()
        if mask.sum() >= 10:
            r, p = pearsonr(df.loc[mask, pred], df.loc[mask, out])
            star = "***" if p < 0.001 else ("**" if p < 0.01 else ("*" if p < 0.05 else ""))
            print(f"  {r:>7.3f}{star:<4s}    ", end="")
        else:
            print(f"  {'NA':>11s}    ", end="")
    print()

# Save correlation matrix
routes_list = []
for pred in predictors:
    for out in outcomes:
        mask = df[pred].notna() & df[out].notna()
        if mask.sum() >= 10:
            r, p = pearsonr(df.loc[mask, pred], df.loc[mask, out])
        else:
            r, p = np.nan, np.nan
        routes_list.append({"predictor": pred, "outcome": out, "r": r, "p": p})
routes_df = pd.DataFrame(routes_list)

# ══════════════════════════════════════════════════════════════════════════
# SECTION 8: FIGURES
# ══════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("SECTION 8: FIGURES")
print("=" * 70)

fig = plt.figure(figsize=(18, 14))
gs = GridSpec(2, 3, hspace=0.4, wspace=0.35)

# Panel A: k → overcautious rate
ax_a = fig.add_subplot(gs[0, 0])
ax_a.scatter(df["log_k"], df["overcautious_rate"], alpha=0.3, s=15, c="steelblue")
m, b_val = np.polyfit(df["log_k"], df["overcautious_rate"], 1)
x_range = np.linspace(df["log_k"].min(), df["log_k"].max(), 100)
ax_a.plot(x_range, m * x_range + b_val, "r-", lw=2)
ax_a.set_xlabel("log(k) — effort cost")
ax_a.set_ylabel("Overcautious rate")
ax_a.set_title(f"A. Effort cost → overcaution\nr={r_k_oc:.3f}", fontweight="bold")

# Panel B: β → threat sensitivity
ax_b = fig.add_subplot(gs[0, 1])
ax_b.scatter(df["log_beta"], df["threat_sensitivity_choice"], alpha=0.3, s=15, c="firebrick")
m2, b2 = np.polyfit(df["log_beta"].values, df["threat_sensitivity_choice"].values, 1)
x_range2 = np.linspace(df["log_beta"].min(), df["log_beta"].max(), 100)
ax_b.plot(x_range2, m2 * x_range2 + b2, "r-", lw=2)
ax_b.set_xlabel("log(β) — threat aversion")
ax_b.set_ylabel("Threat sensitivity\n(ΔP(heavy) low→high T)")
ax_b.set_title(f"B. Threat aversion → threat sensitivity\nr={r_b_ts:.3f}", fontweight="bold")

# Panel C: cd → vigor gap
ax_c = fig.add_subplot(gs[0, 2])
mask_vg = df["vigor_gap"].notna()
ax_c.scatter(df.loc[mask_vg, "log_cd"], df.loc[mask_vg, "vigor_gap"], alpha=0.3, s=15, c="darkgreen")
m3, b3 = np.polyfit(df.loc[mask_vg, "log_cd"], df.loc[mask_vg, "vigor_gap"], 1)
x_range3 = np.linspace(df["log_cd"].min(), df["log_cd"].max(), 100)
ax_c.plot(x_range3, m3 * x_range3 + b3, "r-", lw=2)
ax_c.set_xlabel("log(cd) — capture aversion")
ax_c.set_ylabel("Vigor gap (|actual - optimal|)")
ax_c.set_title(f"C. Capture aversion → vigor gap\nr={r_cd_vg:.3f}", fontweight="bold")

# Panel D: Correlation heatmap
ax_d = fig.add_subplot(gs[1, 0])
r_matrix = np.zeros((len(predictors), len(outcomes)))
p_matrix = np.zeros((len(predictors), len(outcomes)))
for i, pred in enumerate(predictors):
    for j, out in enumerate(outcomes):
        mask = df[pred].notna() & df[out].notna()
        if mask.sum() >= 10:
            r_matrix[i, j], p_matrix[i, j] = pearsonr(df.loc[mask, pred], df.loc[mask, out])

im = ax_d.imshow(r_matrix, cmap="RdBu_r", vmin=-0.6, vmax=0.6, aspect="auto")
for i in range(len(predictors)):
    for j in range(len(outcomes)):
        star = "***" if p_matrix[i,j] < 0.001 else ("**" if p_matrix[i,j] < 0.01 else ("*" if p_matrix[i,j] < 0.05 else ""))
        color = "white" if abs(r_matrix[i,j]) > 0.3 else "black"
        ax_d.text(j, i, f"{r_matrix[i,j]:.2f}{star}", ha="center", va="center", fontsize=6, color=color)

pred_labels = ["k", "β", "cd", "calib", "disc", "react"]
out_labels = ["overcaut", "threat_sens", "vigor_gap", "alignment", "resid_OC", "earnings"]
ax_d.set_xticks(range(len(outcomes)))
ax_d.set_xticklabels(out_labels, fontsize=7, rotation=30, ha="right")
ax_d.set_yticks(range(len(predictors)))
ax_d.set_yticklabels(pred_labels, fontsize=8)
ax_d.set_title("D. Full correlation matrix", fontweight="bold")
plt.colorbar(im, ax=ax_d, label="Pearson r", shrink=0.8)

# Panel E: Hierarchical R²
ax_e = fig.add_subplot(gs[1, 1])
labels = ["k+β+cd", "+disc", "+calib", "k+β+cd\n(align)", "+calib"]
r2_vals = [model_base.rsquared, model_full_disc.rsquared,
           sm.OLS(df["overcautious_rate"].values, sm.add_constant(df[["log_k","log_beta","log_cd","calibration"]].values)).fit().rsquared,
           model_base_pa.rsquared, model_full_cal.rsquared]
colors = ["lightgray", "crimson", "steelblue", "lightgray", "steelblue"]
bars = ax_e.bar(range(len(labels)), r2_vals, color=colors, edgecolor="black", linewidth=0.5)
ax_e.set_xticks(range(len(labels)))
ax_e.set_xticklabels(labels, fontsize=7, rotation=30, ha="right")
ax_e.set_ylabel("R²")
ax_e.set_title("E. Hierarchical R²", fontweight="bold")
for bar, val in zip(bars, r2_vals):
    ax_e.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.002, f"{val:.3f}", ha="center", fontsize=7)
ax_e.axvline(2.5, color="gray", ls="--", lw=0.5)

# Panel F: Condition predictions
ax_f = fig.add_subplot(gs[1, 2])
obs = behavior.groupby(["threat", "distance_H"])["choice"].mean().reset_index()
obs.columns = ["threat", "dist", "obs"]
obs_matrix = np.zeros((3, 3))
for i, t in enumerate([0.1, 0.5, 0.9]):
    for j, d in enumerate([1, 2, 3]):
        row = obs[(obs["threat"] == t) & (obs["dist"] == d)]
        if len(row) > 0:
            obs_matrix[i, j] = row["obs"].values[0]
im2 = ax_f.imshow(obs_matrix, cmap="RdBu_r", vmin=0, vmax=1, aspect="auto")
for i in range(3):
    for j in range(3):
        opt = policy_df[(policy_df["threat"] == [0.1,0.5,0.9][i]) & (policy_df["distance_H"] == [1,2,3][j])]["optimal_choice"].values[0]
        color = "white" if abs(obs_matrix[i,j] - 0.5) > 0.3 else "black"
        ax_f.text(j, i, f"{obs_matrix[i,j]:.2f}\n({'H' if opt=='heavy' else 'L'})",
                  ha="center", va="center", fontsize=9, fontweight="bold", color=color)
ax_f.set_xticks([0,1,2]); ax_f.set_xticklabels(["D=1","D=2","D=3"])
ax_f.set_yticks([0,1,2]); ax_f.set_yticklabels(["T=0.1","T=0.5","T=0.9"])
ax_f.set_title("F. P(heavy) & EV-optimal", fontweight="bold")
plt.colorbar(im2, ax=ax_f, label="P(heavy)", shrink=0.8)

fig.savefig(FIG_DIR / "fig_3param_pipeline.png", dpi=200, bbox_inches="tight")
print(f"Saved: {FIG_DIR / 'fig_3param_pipeline.png'}")

# ── Save all results ──
results_df = pd.DataFrame(all_results)
results_df.to_csv(OUT_DIR / "3param_v2_pipeline_results.csv", index=False)
routes_df.to_csv(OUT_DIR / "3param_v2_routes.csv", index=False)
policy_df.to_csv(OUT_DIR / "3param_v2_optimal_policy.csv", index=False)
dev_df.to_csv(OUT_DIR / "3param_v2_deviations.csv", index=False)
print(f"Saved all results to {OUT_DIR}")

# ══════════════════════════════════════════════════════════════════════════
# SUMMARY
# ══════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("SUMMARY OF KEY RESULTS")
print("=" * 70)
print(f"\nTRIPLE DISSOCIATION:")
print(f"  k → overcautious rate:        r={r_k_oc:.3f} (partial: {r_k_oc_part:.3f})")
print(f"  β → threat sensitivity:        r={r_b_ts:.3f} (partial: {r_b_ts_part:.3f})")
print(f"  cd → vigor gap:                r={r_cd_vg:.3f} (partial: {r_cd_vg_part:.3f})")
print(f"\nAFFECT:")
print(f"  Calibration-Discrepancy |r|:   {abs(r_cal_disc):.3f} ({'orthogonal' if abs(r_cal_disc)<0.15 else 'correlated'})")
print(f"  Discrepancy → resid overcaution: r={r_disc_resid:.3f}, ΔR²={delta_R2_disc:.4f}")
print(f"  Calibration → alignment:       r={r_cal_pa:.3f}, ΔR²={delta_R2_cal:.4f}")
print(f"\nENCOUNTER DYNAMICS:")
print(f"  Reactivity → cd:  r={r_react_cd:.3f}")
print(f"  Reactivity → β:   r={r_react_b:.3f}")
print(f"  Reactivity → k:   r={r_react_k:.3f}")
print(f"\nDone.")
