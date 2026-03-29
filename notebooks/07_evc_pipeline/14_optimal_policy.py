#!/usr/bin/env python3
"""
14_optimal_policy.py — Optimal policy derivation and per-subject deviation classification
========================================================================================

For each of 9 conditions (3T x 3D), compute:
  1. EV-optimal choice using empirical conditional survival rates
  2. Subjective-optimal choice under gamma=0.209
  3. Per-subject: classify each trial as optimal/overcautious/overrisky

Output:
  results/stats/optimal_policy.csv        — per-condition optimal choices
  results/stats/per_subject_deviations.csv — per-subject deviation metrics
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from scipy import stats
from pathlib import Path

# ── Paths ──
DATA_DIR = Path("/workspace/data/exploratory_350/processed/stage5_filtered_data_20260320_191950")
BEHAVIOR_FILE = DATA_DIR / "behavior.csv"
BEHAVIOR_RICH_FILE = DATA_DIR / "behavior_rich.csv"
PARAMS_FILE = Path("/workspace/results/stats/oc_evc_final_params.csv")
POP_FILE = Path("/workspace/results/stats/oc_evc_final_81_population.csv")
OUT_POLICY = Path("/workspace/results/stats/optimal_policy.csv")
OUT_DEVIATIONS = Path("/workspace/results/stats/per_subject_deviations.csv")

# ── Load data ──
print("=" * 70)
print("14. OPTIMAL POLICY DERIVATION")
print("=" * 70)

behavior = pd.read_csv(BEHAVIOR_FILE)
behavior_rich = pd.read_csv(BEHAVIOR_RICH_FILE, low_memory=False)
params = pd.read_csv(PARAMS_FILE)
pop = pd.read_csv(POP_FILE)

GAMMA = float(pop["gamma"].iloc[0])
EPSILON = float(pop["epsilon"].iloc[0])
P_ESC = float(pop["p_esc"].iloc[0])

print(f"Behavior: {behavior.shape[0]} trials, {behavior['subj'].nunique()} subjects")
print(f"Population: gamma={GAMMA:.4f}, epsilon={EPSILON:.4f}, p_esc={P_ESC:.4f}")

# ── Encoding: choice=1 -> heavy (R=5), choice=0 -> light (R=1) ──
# outcome=0 -> survived, outcome=1 -> captured
# distance_H: 1,2,3 (heavy cookie distance level); distance_L: always 1
# threat: 0.1, 0.5, 0.9

# Compute survived flag from behavior_rich
behavior_rich["survived"] = (behavior_rich["trialEndState"] == "escaped").astype(int)

# ══════════════════════════════════════════════════════════════════════════
# 1. EMPIRICAL CONDITIONAL SURVIVAL RATES
# ══════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("1. EMPIRICAL CONDITIONAL SURVIVAL RATES")
print("=" * 70)

# For each T x D_H condition, compute survival rate for heavy vs light choosers
# Note: light choosers always go to D_L=1 regardless of D_H
# Heavy choosers go to D_H which varies

conditions = []
for t in [0.1, 0.5, 0.9]:
    for d in [1, 2, 3]:
        # Heavy choosers in this condition
        mask_h = (behavior_rich["threat"].round(1) == t) & \
                 (behavior_rich["choice"] == 1) & \
                 (behavior_rich["distance_H"] == d)
        # Light choosers in this condition
        mask_l = (behavior_rich["threat"].round(1) == t) & \
                 (behavior_rich["choice"] == 0) & \
                 (behavior_rich["distance_H"] == d)

        n_h = mask_h.sum()
        n_l = mask_l.sum()
        surv_h = behavior_rich.loc[mask_h, "survived"].mean() if n_h > 0 else np.nan
        surv_l = behavior_rich.loc[mask_l, "survived"].mean() if n_l > 0 else np.nan

        # EV computation: R=5 for heavy, R=1 for light, C=5 capture penalty
        # EV = S * R - (1-S) * C
        ev_h = surv_h * 5 - (1 - surv_h) * 5 if not np.isnan(surv_h) else np.nan
        ev_l = surv_l * 1 - (1 - surv_l) * 5 if not np.isnan(surv_l) else np.nan

        optimal = "heavy" if (ev_h is not None and ev_l is not None and ev_h > ev_l) else "light"

        conditions.append({
            "threat": t, "distance_H": d,
            "surv_heavy": surv_h, "surv_light": surv_l,
            "n_heavy": n_h, "n_light": n_l,
            "ev_heavy": ev_h, "ev_light": ev_l,
            "optimal_choice": optimal,
            "ev_advantage": ev_h - ev_l if not (np.isnan(ev_h) or np.isnan(ev_l)) else np.nan,
        })

policy_df = pd.DataFrame(conditions)

print("\nEmpirical EV-optimal policy:")
print("-" * 90)
print(f"{'Threat':>8} {'Dist_H':>8} {'Surv_H':>8} {'Surv_L':>8} {'EV_H':>8} {'EV_L':>8} {'Optimal':>10} {'EV_adv':>8}")
for _, row in policy_df.iterrows():
    print(f"{row['threat']:>8.1f} {row['distance_H']:>8d} {row['surv_heavy']:>8.3f} "
          f"{row['surv_light']:>8.3f} {row['ev_heavy']:>8.2f} {row['ev_light']:>8.2f} "
          f"{row['optimal_choice']:>10s} {row['ev_advantage']:>8.2f}")

# ══════════════════════════════════════════════════════════════════════════
# 2. SUBJECTIVE OPTIMAL UNDER gamma=0.209
# ══════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("2. SUBJECTIVE OPTIMAL UNDER gamma=0.209")
print("=" * 70)

# Under the model, perceived threat is T^gamma (compressed)
# S = (1 - T^gamma) + epsilon * T^gamma * p_esc
# This is the SAME for heavy and light (it doesn't depend on distance in the model)
# The model's choice is driven by S * R_H - (1-S) * C vs S * R_L - (1-S) * C
# Since C is the same and S is the same for both options at a given T,
# the optimal choice simplifies to: heavy if S * 5 > S * 1, which is always true.
#
# But that can't be right — distance matters because it affects survival.
# The survival rate IS distance-dependent empirically.
#
# For subjective optimal, we use the model's survival at each T:
# S_model(T) = (1 - T^gamma) + epsilon * T^gamma * p_esc
# Then compute EV using this model-based S but keeping the empirical
# RELATIVE survival difference between heavy/light.
#
# Actually, the correct approach: use empirical surv_heavy and surv_light
# but for the "objective" comparison use gamma=1, and for "subjective" use gamma=0.209.
# The key insight is that under gamma<1, low threat is perceived as higher
# (since T^0.209 > T for T<1), making people more cautious.
#
# Better approach: Under gamma=0.209, the PERCEIVED survival probabilities
# are different from objective ones. We can compute what a rational agent
# with gamma=0.209 perception would choose, using the model's S function.
#
# For the subjective optimal, the question is:
# Given perceived threat T_perceived = T^gamma, what should you choose?
# The agent doesn't know the true conditional survival rates.
# Instead, the agent uses S(T) as its proxy for survival.
# Under the model: S(T) is the same for any choice at threat T.
# So the choice reduces to comparing rewards: heavy always dominates.
#
# This means ALL overcaution is "irrational" under the model, which is the point:
# ce and cd explain why people deviate from always-choose-heavy.
#
# Let me reframe: compute objective optimal (using true survival rates)
# and subjective optimal (adjusting survival rates by gamma distortion).

# For subjective optimal: rescale the empirical survival rates by the
# ratio of perceived-to-true hazard
# Perceived hazard at threat T: h_perceived = T^gamma
# True hazard at threat T: h_true = T
# Subjective survival ≈ empirical_survival * (h_true / h_perceived)  ... not quite right
#
# More principled: the agent perceives T_eff = T^gamma as the "true" threat.
# Under T_eff, what would the objective survival rates be?
# We can interpolate: if empirical surv(T=0.1)=0.9 and surv(T=0.5)=0.7,
# then at T_eff=0.1^0.209=0.62, surv ≈ interpolated between T=0.5 and T=0.9.
#
# Actually, let's just compare:
# Objective optimal: uses empirical surv rates with gamma=1 (no distortion)
# Subjective optimal: uses empirical surv rates but the DECISION THRESHOLD changes
# because the agent overweights low-T threat → is more cautious at low T
# and underweights high-T threat → is less cautious at high T.
#
# Simplest correct approach:
# Under gamma=1: EV_h = surv_h_empirical * 5 - (1-surv_h) * 5
# Under gamma=0.209: The agent perceives the threat as T^0.209.
# At T=0.1: perceived threat = 0.618 (much higher!)
# At T=0.9: perceived threat = 0.978 (slightly higher)
#
# The agent then uses the survival rates that CORRESPOND to perceived threat level.
# So at T=0.1 (perceived=0.62), they act as if facing T≈0.62,
# using survival rates from around T=0.5-0.9.
# This means at low T, they think survival is worse → more cautious → light is "rational"
# At high T, they think survival is about the same → similar to objective

# Implementation: for each condition, the "subjective EV" uses survival rates
# looked up at perceived_T = T^gamma (interpolated from empirical data)

# Build interpolation of survival rates from empirical data
from scipy.interpolate import interp1d

surv_by_threat_heavy = {}
surv_by_threat_light = {}
for d in [1, 2, 3]:
    threats = []
    sh = []
    sl = []
    for t in [0.1, 0.5, 0.9]:
        row = policy_df[(policy_df["threat"] == t) & (policy_df["distance_H"] == d)].iloc[0]
        threats.append(t)
        sh.append(row["surv_heavy"])
        sl.append(row["surv_light"])
    surv_by_threat_heavy[d] = interp1d(threats, sh, kind='linear', fill_value='extrapolate')
    surv_by_threat_light[d] = interp1d(threats, sl, kind='linear', fill_value='extrapolate')

subj_conditions = []
for _, row in policy_df.iterrows():
    t = row["threat"]
    d = int(row["distance_H"])
    t_perceived = t ** GAMMA
    t_perceived_clipped = np.clip(t_perceived, 0.1, 0.9)

    surv_h_subj = float(surv_by_threat_heavy[d](t_perceived_clipped))
    surv_l_subj = float(surv_by_threat_light[d](t_perceived_clipped))

    ev_h_subj = surv_h_subj * 5 - (1 - surv_h_subj) * 5
    ev_l_subj = surv_l_subj * 1 - (1 - surv_l_subj) * 5

    subj_optimal = "heavy" if ev_h_subj > ev_l_subj else "light"

    subj_conditions.append({
        "threat": t, "distance_H": d,
        "perceived_threat": t_perceived,
        "surv_heavy_subjective": surv_h_subj,
        "surv_light_subjective": surv_l_subj,
        "ev_heavy_subjective": ev_h_subj,
        "ev_light_subjective": ev_l_subj,
        "subjective_optimal": subj_optimal,
    })

subj_policy_df = pd.DataFrame(subj_conditions)
policy_df = policy_df.merge(subj_policy_df, on=["threat", "distance_H"])

print("\nSubjective (gamma=0.209) vs Objective optimal:")
print("-" * 100)
print(f"{'Threat':>8} {'Dist_H':>8} {'T_perc':>8} {'Obj_opt':>10} {'Subj_opt':>10} {'EV_H_obj':>10} {'EV_L_obj':>10} {'EV_H_sub':>10} {'EV_L_sub':>10}")
for _, row in policy_df.iterrows():
    print(f"{row['threat']:>8.1f} {row['distance_H']:>8.0f} {row['perceived_threat']:>8.3f} "
          f"{row['optimal_choice']:>10s} {row['subjective_optimal']:>10s} "
          f"{row['ev_heavy']:>10.2f} {row['ev_light']:>10.2f} "
          f"{row['ev_heavy_subjective']:>10.2f} {row['ev_light_subjective']:>10.2f}")

n_obj_heavy = (policy_df["optimal_choice"] == "heavy").sum()
n_subj_heavy = (policy_df["subjective_optimal"] == "heavy").sum()
n_shift = ((policy_df["optimal_choice"] == "heavy") & (policy_df["subjective_optimal"] == "light")).sum()
print(f"\nObjective: {n_obj_heavy}/9 conditions favor heavy")
print(f"Subjective (gamma={GAMMA:.3f}): {n_subj_heavy}/9 conditions favor heavy")
print(f"Conditions that shift from heavy→light under gamma distortion: {n_shift}")

# ══════════════════════════════════════════════════════════════════════════
# 3. PER-SUBJECT TRIAL CLASSIFICATION
# ══════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("3. PER-SUBJECT TRIAL CLASSIFICATION")
print("=" * 70)

# Create a lookup for objective optimal choice per condition
optimal_lookup = {}
subj_optimal_lookup = {}
ev_lookup = {}
for _, row in policy_df.iterrows():
    key = (row["threat"], int(row["distance_H"]))
    optimal_lookup[key] = 1 if row["optimal_choice"] == "heavy" else 0
    subj_optimal_lookup[key] = 1 if row["subjective_optimal"] == "heavy" else 0
    ev_lookup[key] = {
        "ev_heavy": row["ev_heavy"],
        "ev_light": row["ev_light"],
    }

# Classify each trial
behavior = behavior.copy()
behavior["optimal_choice"] = behavior.apply(
    lambda r: optimal_lookup.get((r["threat"], r["distance_H"]), np.nan), axis=1
)
behavior["subj_optimal_choice"] = behavior.apply(
    lambda r: subj_optimal_lookup.get((r["threat"], r["distance_H"]), np.nan), axis=1
)

# Classification
behavior["is_optimal"] = (behavior["choice"] == behavior["optimal_choice"]).astype(int)
behavior["is_overcautious"] = ((behavior["choice"] == 0) & (behavior["optimal_choice"] == 1)).astype(int)
behavior["is_overrisky"] = ((behavior["choice"] == 1) & (behavior["optimal_choice"] == 0)).astype(int)

# Under subjective optimal
behavior["is_subj_optimal"] = (behavior["choice"] == behavior["subj_optimal_choice"]).astype(int)
behavior["is_subj_overcautious"] = ((behavior["choice"] == 0) & (behavior["subj_optimal_choice"] == 1)).astype(int)
behavior["is_subj_overrisky"] = ((behavior["choice"] == 1) & (behavior["subj_optimal_choice"] == 0)).astype(int)

# Earnings loss: EV(optimal) - EV(chosen)
def compute_ev_loss(row):
    key = (row["threat"], row["distance_H"])
    evs = ev_lookup.get(key, {})
    ev_chosen = evs.get("ev_heavy", 0) if row["choice"] == 1 else evs.get("ev_light", 0)
    ev_optimal = max(evs.get("ev_heavy", 0), evs.get("ev_light", 0))
    return ev_optimal - ev_chosen

behavior["ev_loss"] = behavior.apply(compute_ev_loss, axis=1)

# Per-subject aggregation
subj_devs = []
for s, sdf in behavior.groupby("subj"):
    n_trials = len(sdf)
    n_optimal = sdf["is_optimal"].sum()
    n_overcautious = sdf["is_overcautious"].sum()
    n_overrisky = sdf["is_overrisky"].sum()

    n_subj_optimal = sdf["is_subj_optimal"].sum()
    n_subj_overcautious = sdf["is_subj_overcautious"].sum()
    n_subj_overrisky = sdf["is_subj_overrisky"].sum()

    total_ev_loss = sdf["ev_loss"].sum()
    mean_ev_loss = sdf["ev_loss"].mean()

    # Also compute heavy choice rate and survival rate
    heavy_rate = sdf["choice"].mean()
    survival_rate = (sdf["outcome"] == 0).mean()
    total_earnings = ((sdf["outcome"] == 0) * sdf["choice"].map({1: 5, 0: 1})).sum()

    subj_devs.append({
        "subj": s,
        "n_trials": n_trials,
        "optimality_rate": n_optimal / n_trials,
        "overcautious_rate": n_overcautious / n_trials,
        "overrisky_rate": n_overrisky / n_trials,
        "subj_optimality_rate": n_subj_optimal / n_trials,
        "subj_overcautious_rate": n_subj_overcautious / n_trials,
        "subj_overrisky_rate": n_subj_overrisky / n_trials,
        "total_ev_loss": total_ev_loss,
        "mean_ev_loss": mean_ev_loss,
        "heavy_rate": heavy_rate,
        "survival_rate": survival_rate,
        "total_earnings": total_earnings,
    })

dev_df = pd.DataFrame(subj_devs)

print(f"\nPer-subject deviation summary (N={len(dev_df)}):")
print(f"  Optimality rate:      {dev_df['optimality_rate'].mean():.3f} ± {dev_df['optimality_rate'].std():.3f}")
print(f"  Overcautious rate:    {dev_df['overcautious_rate'].mean():.3f} ± {dev_df['overcautious_rate'].std():.3f}")
print(f"  Overrisky rate:       {dev_df['overrisky_rate'].mean():.3f} ± {dev_df['overrisky_rate'].std():.3f}")
print(f"  Subj optimality rate: {dev_df['subj_optimality_rate'].mean():.3f} ± {dev_df['subj_optimality_rate'].std():.3f}")
print(f"  Subj overcautious:    {dev_df['subj_overcautious_rate'].mean():.3f} ± {dev_df['subj_overcautious_rate'].std():.3f}")
print(f"  Subj overrisky:       {dev_df['subj_overrisky_rate'].mean():.3f} ± {dev_df['subj_overrisky_rate'].std():.3f}")
print(f"  Mean EV loss:         {dev_df['mean_ev_loss'].mean():.3f} ± {dev_df['mean_ev_loss'].std():.3f}")
print(f"  Total earnings:       {dev_df['total_earnings'].mean():.1f} ± {dev_df['total_earnings'].std():.1f}")

# What fraction of "suboptimality" disappears under gamma distortion?
obj_subopt = 1 - dev_df["optimality_rate"].mean()
subj_subopt = 1 - dev_df["subj_optimality_rate"].mean()
pct_explained = (obj_subopt - subj_subopt) / obj_subopt * 100 if obj_subopt > 0 else 0
print(f"\n  Objective suboptimality: {obj_subopt:.3f}")
print(f"  Subjective suboptimality: {subj_subopt:.3f}")
print(f"  % suboptimality explained by gamma distortion: {pct_explained:.1f}%")

# ── Save ──
policy_df.to_csv(OUT_POLICY, index=False)
dev_df.to_csv(OUT_DEVIATIONS, index=False)
print(f"\nSaved: {OUT_POLICY}")
print(f"Saved: {OUT_DEVIATIONS}")
print("\nDone.")
