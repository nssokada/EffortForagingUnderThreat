#!/usr/bin/env python3
"""
15_deviation_analysis.py — Parameters drive specific deviations from optimal
============================================================================

Tests:
  - ce → overcautious rate (partial r, controlling for cd)
  - cd → survival rate (partial r, controlling for ce)
  - Multiple regressions: unique R² for each parameter
  - Gamma distortion: what fraction of suboptimality disappears
  - Behavioral profiles (ce×cd quadrants)

Output:
  results/stats/deviation_param_associations.csv
  results/figs/paper/fig_deviations.png
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
PARAMS_FILE = Path("/workspace/results/stats/oc_evc_final_params.csv")
DEVIATIONS_FILE = Path("/workspace/results/stats/per_subject_deviations.csv")
POLICY_FILE = Path("/workspace/results/stats/optimal_policy.csv")
BEHAVIOR_FILE = Path("/workspace/data/exploratory_350/processed/stage5_filtered_data_20260320_191950/behavior.csv")
VIGOR_TS = Path("/workspace/data/exploratory_350/processed/vigor_processed/smoothed_vigor_ts.parquet")
OUT_STATS = Path("/workspace/results/stats/deviation_param_associations.csv")
OUT_FIG = Path("/workspace/results/figs/paper/fig_deviations.png")
OUT_FIG.parent.mkdir(parents=True, exist_ok=True)

def safe_pearsonr(x, y):
    mask = np.isfinite(x) & np.isfinite(y)
    if mask.sum() < 10:
        return np.nan, np.nan
    return pearsonr(x[mask], y[mask])

def partial_corr(x, y, z):
    """Partial correlation of x and y, controlling for z."""
    mask = np.isfinite(x) & np.isfinite(y) & np.isfinite(z)
    x, y, z = x[mask], y[mask], z[mask]
    # Residualize x and y on z
    rx = sm.OLS(x, sm.add_constant(z)).fit().resid
    ry = sm.OLS(y, sm.add_constant(z)).fit().resid
    return pearsonr(rx, ry)

# ── Load data ──
print("=" * 70)
print("15. DEVIATION ANALYSIS")
print("=" * 70)

params = pd.read_csv(PARAMS_FILE)
dev = pd.read_csv(DEVIATIONS_FILE)
policy = pd.read_csv(POLICY_FILE)
behavior = pd.read_csv(BEHAVIOR_FILE)

# Merge params with deviations
df = dev.merge(params, on="subj")
df["log_ce"] = np.log(df["c_effort"])
df["log_cd"] = np.log(df["c_death"])

print(f"Subjects: {len(df)}")
print(f"log(ce) range: [{df['log_ce'].min():.2f}, {df['log_ce'].max():.2f}]")
print(f"log(cd) range: [{df['log_cd'].min():.2f}, {df['log_cd'].max():.2f}]")

results = []

# ══════════════════════════════════════════════════════════════════════════
# 1. ce → OVERCAUTIOUS RATE
# ══════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("1. ce → OVERCAUTIOUS RATE")
print("=" * 70)

r_ce_oc, p_ce_oc = safe_pearsonr(df["log_ce"].values, df["overcautious_rate"].values)
print(f"Pearson r(log_ce, overcautious_rate) = {r_ce_oc:.3f}, p = {p_ce_oc:.2e}")
results.append({"test": "ce_overcautious_bivariate", "r": r_ce_oc, "p": p_ce_oc, "n": len(df)})

r_ce_oc_part, p_ce_oc_part = partial_corr(
    df["log_ce"].values, df["overcautious_rate"].values, df["log_cd"].values
)
print(f"Partial r(log_ce, overcautious_rate | log_cd) = {r_ce_oc_part:.3f}, p = {p_ce_oc_part:.2e}")
results.append({"test": "ce_overcautious_partial", "r": r_ce_oc_part, "p": p_ce_oc_part, "n": len(df)})

r_ce_or, p_ce_or = safe_pearsonr(df["log_ce"].values, df["overrisky_rate"].values)
print(f"Pearson r(log_ce, overrisky_rate) = {r_ce_or:.3f}, p = {p_ce_or:.2e}")
results.append({"test": "ce_overrisky_bivariate", "r": r_ce_or, "p": p_ce_or, "n": len(df)})

# ══════════════════════════════════════════════════════════════════════════
# 2. cd → SURVIVAL RATE
# ══════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("2. cd → SURVIVAL RATE")
print("=" * 70)

r_cd_sr, p_cd_sr = safe_pearsonr(df["log_cd"].values, df["survival_rate"].values)
print(f"Pearson r(log_cd, survival_rate) = {r_cd_sr:.3f}, p = {p_cd_sr:.2e}")
results.append({"test": "cd_survival_bivariate", "r": r_cd_sr, "p": p_cd_sr, "n": len(df)})

r_cd_sr_part, p_cd_sr_part = partial_corr(
    df["log_cd"].values, df["survival_rate"].values, df["log_ce"].values
)
print(f"Partial r(log_cd, survival_rate | log_ce) = {r_cd_sr_part:.3f}, p = {p_cd_sr_part:.2e}")
results.append({"test": "cd_survival_partial", "r": r_cd_sr_part, "p": p_cd_sr_part, "n": len(df)})

# ══════════════════════════════════════════════════════════════════════════
# 3. VIGOR GAP — cd → |actual - optimal| vigor
# ══════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("3. cd → VIGOR GAP")
print("=" * 70)

# Compute vigor gap per subject from vigor timeseries
# vigor_resid captures excess vigor (above/below cookie-appropriate mean)
# vigor_gap = mean absolute vigor_resid (how far from "optimal" each subject is)
print("Loading vigor timeseries...")
vigor = pd.read_parquet(VIGOR_TS)
vigor_gap_subj = vigor.groupby("subj")["vigor_resid"].apply(lambda x: np.abs(x).mean()).reset_index()
vigor_gap_subj.columns = ["subj", "vigor_gap"]
print(f"Vigor gap computed for {len(vigor_gap_subj)} subjects")

df = df.merge(vigor_gap_subj, on="subj", how="left")

r_cd_vg, p_cd_vg = safe_pearsonr(df["log_cd"].values, df["vigor_gap"].values)
print(f"Pearson r(log_cd, vigor_gap) = {r_cd_vg:.3f}, p = {p_cd_vg:.2e}")
results.append({"test": "cd_vigor_gap_bivariate", "r": r_cd_vg, "p": p_cd_vg, "n": df["vigor_gap"].notna().sum()})

r_cd_vg_part, p_cd_vg_part = partial_corr(
    df["log_cd"].values, df["vigor_gap"].values, df["log_ce"].values
)
print(f"Partial r(log_cd, vigor_gap | log_ce) = {r_cd_vg_part:.3f}, p = {p_cd_vg_part:.2e}")
results.append({"test": "cd_vigor_gap_partial", "r": r_cd_vg_part, "p": p_cd_vg_part, "n": df["vigor_gap"].notna().sum()})

# ══════════════════════════════════════════════════════════════════════════
# 4. MULTIPLE REGRESSIONS — unique R² for each parameter
# ══════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("4. MULTIPLE REGRESSIONS")
print("=" * 70)

# Overcautious rate ~ log(ce) + log(cd)
X = sm.add_constant(df[["log_ce", "log_cd"]].values)
y = df["overcautious_rate"].values
model_oc = sm.OLS(y, X).fit()
print("\n--- overcautious_rate ~ log(ce) + log(cd) ---")
print(f"R² = {model_oc.rsquared:.4f}, Adj R² = {model_oc.rsquared_adj:.4f}")
print(f"  log(ce): beta={model_oc.params[1]:.4f}, t={model_oc.tvalues[1]:.3f}, p={model_oc.pvalues[1]:.2e}")
print(f"  log(cd): beta={model_oc.params[2]:.4f}, t={model_oc.tvalues[2]:.3f}, p={model_oc.pvalues[2]:.2e}")
results.append({"test": "overcautious_multiple_R2", "r": model_oc.rsquared, "p": model_oc.f_pvalue, "n": len(df),
                "beta_ce": model_oc.params[1], "t_ce": model_oc.tvalues[1], "p_ce": model_oc.pvalues[1],
                "beta_cd": model_oc.params[2], "t_cd": model_oc.tvalues[2], "p_cd": model_oc.pvalues[2]})

# Unique R² for ce: R²(full) - R²(cd only)
model_cd_only = sm.OLS(y, sm.add_constant(df[["log_cd"]].values)).fit()
model_ce_only = sm.OLS(y, sm.add_constant(df[["log_ce"]].values)).fit()
unique_R2_ce = model_oc.rsquared - model_cd_only.rsquared
unique_R2_cd = model_oc.rsquared - model_ce_only.rsquared
print(f"  Unique R² for ce: {unique_R2_ce:.4f}")
print(f"  Unique R² for cd: {unique_R2_cd:.4f}")
results.append({"test": "overcautious_unique_R2_ce", "r": unique_R2_ce, "p": model_oc.pvalues[1], "n": len(df)})
results.append({"test": "overcautious_unique_R2_cd", "r": unique_R2_cd, "p": model_oc.pvalues[2], "n": len(df)})

# Survival rate ~ log(ce) + log(cd)
y_sr = df["survival_rate"].values
model_sr = sm.OLS(y_sr, X).fit()
print("\n--- survival_rate ~ log(ce) + log(cd) ---")
print(f"R² = {model_sr.rsquared:.4f}, Adj R² = {model_sr.rsquared_adj:.4f}")
print(f"  log(ce): beta={model_sr.params[1]:.4f}, t={model_sr.tvalues[1]:.3f}, p={model_sr.pvalues[1]:.2e}")
print(f"  log(cd): beta={model_sr.params[2]:.4f}, t={model_sr.tvalues[2]:.3f}, p={model_sr.pvalues[2]:.2e}")
results.append({"test": "survival_multiple_R2", "r": model_sr.rsquared, "p": model_sr.f_pvalue, "n": len(df),
                "beta_ce": model_sr.params[1], "t_ce": model_sr.tvalues[1], "p_ce": model_sr.pvalues[1],
                "beta_cd": model_sr.params[2], "t_cd": model_sr.tvalues[2], "p_cd": model_sr.pvalues[2]})

model_cd_only_sr = sm.OLS(y_sr, sm.add_constant(df[["log_cd"]].values)).fit()
model_ce_only_sr = sm.OLS(y_sr, sm.add_constant(df[["log_ce"]].values)).fit()
unique_R2_ce_sr = model_sr.rsquared - model_cd_only_sr.rsquared
unique_R2_cd_sr = model_sr.rsquared - model_ce_only_sr.rsquared
print(f"  Unique R² for ce: {unique_R2_ce_sr:.4f}")
print(f"  Unique R² for cd: {unique_R2_cd_sr:.4f}")
results.append({"test": "survival_unique_R2_ce", "r": unique_R2_ce_sr, "p": model_sr.pvalues[1], "n": len(df)})
results.append({"test": "survival_unique_R2_cd", "r": unique_R2_cd_sr, "p": model_sr.pvalues[2], "n": len(df)})

# ══════════════════════════════════════════════════════════════════════════
# 5. GAMMA DISTORTION COMPARISON
# ══════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("5. GAMMA=1 vs GAMMA=0.21 OPTIMAL SURFACES")
print("=" * 70)

obj_subopt = 1 - df["optimality_rate"].mean()
subj_subopt = 1 - df["subj_optimality_rate"].mean()
pct_explained = (obj_subopt - subj_subopt) / obj_subopt * 100 if obj_subopt > 0 else 0

print(f"Mean objective suboptimality:  {obj_subopt:.3f}")
print(f"Mean subjective suboptimality: {subj_subopt:.3f}")
print(f"% explained by gamma distortion: {pct_explained:.1f}%")
results.append({"test": "gamma_distortion_pct_explained", "r": pct_explained / 100, "p": np.nan, "n": len(df)})

# ══════════════════════════════════════════════════════════════════════════
# 6. BEHAVIORAL PROFILES (median split ce × cd)
# ══════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("6. BEHAVIORAL PROFILES (ce × cd quadrants)")
print("=" * 70)

df["ce_group"] = np.where(df["log_ce"] >= df["log_ce"].median(), "high_ce", "low_ce")
df["cd_group"] = np.where(df["log_cd"] >= df["log_cd"].median(), "high_cd", "low_cd")
df["profile"] = df["ce_group"] + "_" + df["cd_group"]

profile_names = {
    "low_ce_high_cd": "Vigilant",
    "high_ce_high_cd": "Cautious",
    "low_ce_low_cd": "Bold",
    "high_ce_low_cd": "Disengaged",
}

print(f"\n{'Profile':<15} {'Label':<12} {'N':>5} {'Earnings':>10} {'Overcaut':>10} {'Survival':>10} {'Vigor_gap':>10}")
for prof in ["low_ce_high_cd", "high_ce_high_cd", "low_ce_low_cd", "high_ce_low_cd"]:
    pf = df[df["profile"] == prof]
    label = profile_names.get(prof, prof)
    vg = pf["vigor_gap"].mean() if "vigor_gap" in pf.columns else np.nan
    print(f"{prof:<15} {label:<12} {len(pf):>5} {pf['total_earnings'].mean():>10.1f} "
          f"{pf['overcautious_rate'].mean():>10.3f} {pf['survival_rate'].mean():>10.3f} "
          f"{vg:>10.3f}")
    results.append({
        "test": f"profile_{label}",
        "r": pf["total_earnings"].mean(),
        "p": np.nan,
        "n": len(pf),
    })

# ══════════════════════════════════════════════════════════════════════════
# 7. FIGURE
# ══════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("7. GENERATING FIGURE")
print("=" * 70)

fig = plt.figure(figsize=(14, 12))
gs = GridSpec(2, 2, hspace=0.35, wspace=0.35)

# Panel A: Optimal vs observed choice surface (9 cells)
ax_a = fig.add_subplot(gs[0, 0])

# Observed heavy-choice rate per condition
obs_rates = behavior.groupby(["threat", "distance_H"])["choice"].mean().reset_index()
obs_rates.columns = ["threat", "distance_H", "obs_heavy_rate"]
policy_plot = policy.merge(obs_rates, on=["threat", "distance_H"])

# Create heatmap-like display
threats = [0.1, 0.5, 0.9]
dists = [1, 2, 3]
obs_matrix = np.zeros((3, 3))
opt_matrix = np.zeros((3, 3))
for i, t in enumerate(threats):
    for j, d in enumerate(dists):
        row = policy_plot[(policy_plot["threat"] == t) & (policy_plot["distance_H"] == d)]
        if len(row) > 0:
            obs_matrix[i, j] = row["obs_heavy_rate"].values[0]
            opt_matrix[i, j] = 1 if row["optimal_choice"].values[0] == "heavy" else 0

im = ax_a.imshow(obs_matrix, cmap="RdBu_r", vmin=0, vmax=1, aspect="auto")
# Overlay optimal choice boundaries
for i in range(3):
    for j in range(3):
        opt_text = "H" if opt_matrix[i, j] == 1 else "L"
        color = "white" if abs(obs_matrix[i, j] - 0.5) > 0.3 else "black"
        ax_a.text(j, i, f"{obs_matrix[i, j]:.2f}\n({opt_text})",
                  ha="center", va="center", fontsize=9, fontweight="bold", color=color)

ax_a.set_xticks([0, 1, 2])
ax_a.set_xticklabels(["D=1", "D=2", "D=3"])
ax_a.set_yticks([0, 1, 2])
ax_a.set_yticklabels(["T=0.1", "T=0.5", "T=0.9"])
ax_a.set_xlabel("Heavy cookie distance")
ax_a.set_ylabel("Threat probability")
ax_a.set_title("A. Observed P(heavy) vs EV-optimal", fontweight="bold")
plt.colorbar(im, ax=ax_a, label="P(heavy choice)", shrink=0.8)

# Panel B: log(ce) → overcautious rate scatter
ax_b = fig.add_subplot(gs[0, 1])
ax_b.scatter(df["log_ce"], df["overcautious_rate"], alpha=0.3, s=15, c="steelblue")
# Regression line
m, b_val = np.polyfit(df["log_ce"], df["overcautious_rate"], 1)
x_range = np.linspace(df["log_ce"].min(), df["log_ce"].max(), 100)
ax_b.plot(x_range, m * x_range + b_val, "r-", lw=2)
ax_b.set_xlabel("log(c_effort)")
ax_b.set_ylabel("Overcautious rate")
ax_b.set_title(f"B. Effort cost → overcaution\nr={r_ce_oc:.3f}, p={p_ce_oc:.2e}", fontweight="bold")

# Panel C: log(cd) → survival rate scatter
ax_c = fig.add_subplot(gs[1, 0])
ax_c.scatter(df["log_cd"], df["survival_rate"], alpha=0.3, s=15, c="firebrick")
m2, b2 = np.polyfit(df["log_cd"], df["survival_rate"], 1)
x_range2 = np.linspace(df["log_cd"].min(), df["log_cd"].max(), 100)
ax_c.plot(x_range2, m2 * x_range2 + b2, "r-", lw=2)
ax_c.set_xlabel("log(c_death)")
ax_c.set_ylabel("Survival rate")
ax_c.set_title(f"C. Death cost → survival\nr={r_cd_sr:.3f}, p={p_cd_sr:.2e}", fontweight="bold")

# Panel D: gamma=0.21 vs gamma=1 optimal surfaces
ax_d = fig.add_subplot(gs[1, 1])

# For gamma=1 vs gamma=0.21, show which conditions shift
# Bar chart: observed choice rate in conditions where optimal shifts
n_conditions = len(policy)
obj_heavy = (policy["optimal_choice"] == "heavy").sum()
subj_heavy = (policy["subjective_optimal"] == "heavy").sum()

# Show EV advantage for each condition under both gammas
ev_adv_obj = policy["ev_advantage"].values
ev_adv_subj = policy["ev_heavy_subjective"].values - policy["ev_light_subjective"].values
x_pos = np.arange(len(policy))
labels = [f"T={row['threat']}\nD={int(row['distance_H'])}" for _, row in policy.iterrows()]

width = 0.35
bars1 = ax_d.bar(x_pos - width/2, ev_adv_obj, width, label=f"γ=1 (objective)", color="navy", alpha=0.7)
bars2 = ax_d.bar(x_pos + width/2, ev_adv_subj, width, label=f"γ={0.209:.2f} (subjective)", color="orange", alpha=0.7)
ax_d.axhline(0, color="black", lw=0.5, ls="--")
ax_d.set_xticks(x_pos)
ax_d.set_xticklabels(labels, fontsize=7)
ax_d.set_ylabel("EV(heavy) - EV(light)")
ax_d.set_xlabel("Condition")
ax_d.set_title(f"D. EV advantage: γ=1 vs γ=0.21", fontweight="bold")
ax_d.legend(fontsize=8)

fig.savefig(OUT_FIG, dpi=200, bbox_inches="tight")
print(f"Saved figure: {OUT_FIG}")

# ── Save stats ──
results_df = pd.DataFrame(results)
results_df.to_csv(OUT_STATS, index=False)
print(f"Saved: {OUT_STATS}")
print("\nDone.")
