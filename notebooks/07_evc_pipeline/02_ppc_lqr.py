#!/usr/bin/env python3
"""
Posterior Predictive Check (PPC) for the EVC-LQR model.

Uses fitted per-subject c_death, epsilon plus population c_effort=0.007, gamma=0.318
to generate choice and vigor predictions.

Panels:
  A: Choice PPC by condition (9 cells: 3T x 3D)
  B: Vigor PPC by condition (6 cells: 3T x 2 cookie)
  C: Per-subject choice r²
  D: Per-subject vigor r²
  E: Choice residuals by condition
  F: Vigor residuals by condition

Output:
  results/figs/paper/fig_ppc_lqr.png
  results/stats/evc_lqr_ppc.csv
"""

import sys
sys.path.insert(0, '/workspace')

import numpy as np
import pandas as pd
import ast
from scipy.stats import pearsonr, sem
from scipy.special import expit
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

plt.rcParams.update({
    'font.sans-serif': ['DejaVu Sans', 'Arial', 'sans-serif'],
    'font.family': 'sans-serif',
    'figure.dpi': 150,
    'axes.spines.right': False,
    'axes.spines.top': False,
})

# ── Paths ─────────────────────────────────────────────────────────────────────
DATA_DIR = '/workspace/data/exploratory_350/processed/stage5_filtered_data_20260320_191950'
PARAMS_PATH = '/workspace/results/stats/oc_evc_lqr_final_params.csv'
POP_PATH = '/workspace/results/stats/oc_evc_lqr_final_population.csv'
FIG_PATH = '/workspace/results/figs/paper/fig_ppc_lqr.png'
STATS_PATH = '/workspace/results/stats/evc_lqr_ppc.csv'

# ── Population parameters ────────────────────────────────────────────────────
pop = pd.read_csv(POP_PATH)
CE = float(pop['c_effort'].iloc[0])
GAMMA = float(pop['gamma'].iloc[0])
P_ESC = 0.6
SIGMA_MOTOR = 0.15
TAU = 0.5

print(f"Population params: ce={CE:.4f}, gamma={GAMMA:.3f}, p_esc={P_ESC}, sigma_motor={SIGMA_MOTOR}")

# ══════════════════════════════════════════════════════════════════════════════
# 1. Load data and params
# ══════════════════════════════════════════════════════════════════════════════
print("Loading data...")
beh = pd.read_csv(f'{DATA_DIR}/behavior_rich.csv')
params = pd.read_csv(PARAMS_PATH)

beh_c = beh[beh['type'] == 1].copy()

# Compute median press rate
print("Computing median press rates...")
rates = []
for _, row in beh_c.iterrows():
    try:
        press_times = np.array(ast.literal_eval(row['alignedEffortRate']), dtype=float)
    except Exception:
        rates.append(np.nan)
        continue
    ipis = np.diff(press_times)
    ipis = ipis[ipis > 0.01]
    if len(ipis) < 5:
        rates.append(np.nan)
        continue
    rates.append(np.median((1.0 / ipis) / row['calibrationMax']))

beh_c['median_rate'] = rates
beh_c['req_rate'] = np.where(beh_c['trialCookie_weight'] == 3.0, 0.9, 0.4)
beh_c['excess'] = beh_c['median_rate'] - beh_c['req_rate']
beh_c = beh_c.dropna(subset=['excess']).copy()

heavy_mean = beh_c[beh_c['trialCookie_weight'] == 3.0]['excess'].mean()
light_mean = beh_c[beh_c['trialCookie_weight'] == 1.0]['excess'].mean()
beh_c['excess_cc'] = beh_c['excess'] - np.where(
    beh_c['trialCookie_weight'] == 3.0, heavy_mean, light_mean
)

print(f"  Heavy mean excess: {heavy_mean:.4f}")
print(f"  Light mean excess: {light_mean:.4f}")
print(f"  N trials: {len(beh_c)}")

# Merge params
beh_c = beh_c.merge(params, on='subj', how='inner')
print(f"  N trials after merge: {len(beh_c)}")
print(f"  N subjects: {beh_c['subj'].nunique()}")

# ══════════════════════════════════════════════════════════════════════════════
# 2. Generate predictions
# ══════════════════════════════════════════════════════════════════════════════
print("Generating predictions...")

T = beh_c['threat'].values
dist_H = beh_c['distance_H'].values
cd = beh_c['c_death'].values
eps = beh_c['epsilon'].values
choice = beh_c['choice'].values

T_w = T ** GAMMA

# ── Choice prediction ─────────────────────────────────────────────────────────
S_full = (1.0 - T_w) + eps * T_w * P_ESC
S_stop = 1.0 - T_w

# Heavy (req=0.9, R=5, R+C=10)
eu_H_full = S_full * 5 - (1 - S_full) * cd * 10 - CE * 0.81 * dist_H
eu_H_stop = S_stop * 5 - (1 - S_stop) * cd * 10
eu_H = np.maximum(eu_H_full, eu_H_stop)

# Light (req=0.4, R=1, R+C=6, dist=1)
eu_L_full = S_full * 1 - (1 - S_full) * cd * 6 - CE * 0.16
eu_L_stop = S_stop * 1 - (1 - S_stop) * cd * 6
eu_L = np.maximum(eu_L_full, eu_L_stop)

logit = np.clip((eu_H - eu_L) / TAU, -20, 20)
p_H_pred = expit(logit)
beh_c['p_H_pred'] = p_H_pred

# ── Vigor prediction (LQR deviation cost) ────────────────────────────────────
chosen_R = np.where(choice == 1, 5.0, 1.0)
chosen_req = np.where(choice == 1, 0.9, 0.4)
chosen_dist = np.where(choice == 1, dist_H, 1.0)
chosen_offset = np.where(choice == 1, heavy_mean, light_mean)

u_grid = np.linspace(0.1, 1.5, 30)
u_g = u_grid[np.newaxis, :]

S_u = ((1.0 - T_w[:, np.newaxis])
       + eps[:, np.newaxis] * T_w[:, np.newaxis] * P_ESC
       * expit((u_g - chosen_req[:, np.newaxis]) / SIGMA_MOTOR))

# LQR deviation cost: ce * (u - req)^2 * D
deviation = u_g - chosen_req[:, np.newaxis]
effort_vigor = CE * deviation ** 2 * chosen_dist[:, np.newaxis]

eu_grid = (S_u * chosen_R[:, np.newaxis]
           - (1.0 - S_u) * cd[:, np.newaxis] * (chosen_R[:, np.newaxis] + 5.0)
           - effort_vigor)

# Stable softmax
eu_scaled = eu_grid * 10.0
eu_max = eu_scaled.max(axis=1, keepdims=True)
weights = np.exp(eu_scaled - eu_max)
weights = weights / weights.sum(axis=1, keepdims=True)
# Replace any remaining NaN rows with argmax
nan_rows = np.isnan(weights).any(axis=1)
if nan_rows.any():
    best_idx = np.argmax(eu_grid[nan_rows], axis=1)
    for ri, bi in zip(np.where(nan_rows)[0], best_idx):
        weights[ri, :] = 0
        weights[ri, bi] = 1.0
u_star = (weights * u_g).sum(axis=1)
excess_pred = u_star - chosen_req - chosen_offset

beh_c['excess_pred'] = excess_pred

# ══════════════════════════════════════════════════════════════════════════════
# 3. Compute fit metrics
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("EVC-LQR PPC — Fit Quality Metrics")
print("=" * 60)

# Choice accuracy
choice_pred_binary = (p_H_pred >= 0.5).astype(int)
choice_accuracy = (choice_pred_binary == choice).mean()
from sklearn.metrics import roc_auc_score
try:
    auc = roc_auc_score(choice, p_H_pred)
except Exception:
    auc = np.nan

print(f"\nChoice:")
print(f"  Accuracy = {choice_accuracy:.3f}")
print(f"  AUC = {auc:.3f}")
print(f"  Mean predicted P(H) = {p_H_pred.mean():.3f}")
print(f"  Observed P(H) = {choice.mean():.3f}")

# Vigor
r_vigor, p_vigor = pearsonr(excess_pred, beh_c['excess_cc'].values)
print(f"\nVigor:")
print(f"  r = {r_vigor:.3f}, r^2 = {r_vigor**2:.3f}, p = {p_vigor:.2e}")

# ── Per-subject metrics ───────────────────────────────────────────────────────
subj_stats = []
for s, sdf in beh_c.groupby('subj'):
    obs_pH = sdf['choice'].mean()
    pred_pH = sdf['p_H_pred'].mean()
    obs_ex = sdf['excess_cc'].mean()
    pred_ex = sdf['excess_pred'].mean()

    # Per-subject r^2 for choice (binary -> use AUC or just correlation of means)
    # For vigor, trial-level correlation
    if sdf['excess_cc'].std() > 0 and sdf['excess_pred'].std() > 0:
        r_v_s, _ = pearsonr(sdf['excess_pred'], sdf['excess_cc'])
        r2_vigor_s = r_v_s ** 2
    else:
        r2_vigor_s = np.nan

    subj_stats.append({
        'subj': s, 'obs_pH': obs_pH, 'pred_pH': pred_pH,
        'obs_excess': obs_ex, 'pred_excess': pred_ex,
        'r2_vigor': r2_vigor_s, 'n_trials': len(sdf),
    })

subj_stats = pd.DataFrame(subj_stats)

r_choice_subj, _ = pearsonr(subj_stats['obs_pH'], subj_stats['pred_pH'])
r_vigor_subj, _ = pearsonr(subj_stats['obs_excess'], subj_stats['pred_excess'])

# Per-subject choice r^2: compute as (obs - pred)^2 explained
subj_stats['r2_choice'] = 1 - ((subj_stats['obs_pH'] - subj_stats['pred_pH'])**2) / (
    (subj_stats['obs_pH'] - subj_stats['obs_pH'].mean())**2 + 1e-10)

print(f"\nPer-subject Choice: r = {r_choice_subj:.3f}, r^2 = {r_choice_subj**2:.3f}")
print(f"Per-subject Vigor:  r = {r_vigor_subj:.3f}, r^2 = {r_vigor_subj**2:.3f}")
print(f"Median per-subject vigor r^2: {subj_stats['r2_vigor'].median():.3f}")

# ── Condition-level summary ───────────────────────────────────────────────────
beh_c['cookie_type'] = np.where(beh_c['trialCookie_weight'] == 3.0, 'Heavy', 'Light')

choice_cond = beh_c.groupby(['threat', 'distance_H']).agg(
    obs_pH=('choice', 'mean'),
    obs_pH_sem=('choice', sem),
    pred_pH=('p_H_pred', 'mean'),
    pred_pH_sem=('p_H_pred', sem),
    n=('choice', 'count'),
).reset_index()

vigor_cond = beh_c.groupby(['threat', 'cookie_type']).agg(
    obs_excess=('excess_cc', 'mean'),
    obs_excess_sem=('excess_cc', sem),
    pred_excess=('excess_pred', 'mean'),
    pred_excess_sem=('excess_pred', sem),
    n=('choice', 'count'),
).reset_index()

print("\nCondition-level Choice:")
for _, row in choice_cond.iterrows():
    print(f"  T={row['threat']:.1f}, D={row['distance_H']:.0f}: "
          f"obs={row['obs_pH']:.3f}+/-{row['obs_pH_sem']:.3f}, "
          f"pred={row['pred_pH']:.3f}")

print("\nCondition-level Vigor:")
for _, row in vigor_cond.iterrows():
    print(f"  T={row['threat']:.1f}, {row['cookie_type']}: "
          f"obs={row['obs_excess']:.4f}+/-{row['obs_excess_sem']:.4f}, "
          f"pred={row['pred_excess']:.4f}")

# ══════════════════════════════════════════════════════════════════════════════
# 4. Create 6-panel figure
# ══════════════════════════════════════════════════════════════════════════════
print("\nCreating PPC figure...")

fig = plt.figure(figsize=(16, 11))
gs = GridSpec(2, 3, figure=fig, hspace=0.35, wspace=0.35)

THREAT_COLORS = {0.1: '#80C55F', 0.5: '#FAA71C', 0.9: '#D4145A'}

def _style(ax):
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.tick_params(labelsize=10)

# Panel A: Choice PPC by condition
ax_a = fig.add_subplot(gs[0, 0])
threats = [0.1, 0.5, 0.9]
distances = sorted(beh_c['distance_H'].unique())
x_positions = np.arange(len(distances))
width = 0.25

for i, t in enumerate(threats):
    mask = choice_cond['threat'] == t
    obs = choice_cond.loc[mask, 'obs_pH'].values
    obs_se = choice_cond.loc[mask, 'obs_pH_sem'].values
    pred = choice_cond.loc[mask, 'pred_pH'].values

    x = x_positions + (i - 1) * width
    ax_a.bar(x, pred, width * 0.85, color=THREAT_COLORS[t], alpha=0.35, edgecolor='none')
    ax_a.errorbar(x, obs, yerr=obs_se, fmt='o', color=THREAT_COLORS[t],
                  markersize=6, capsize=3, markeredgecolor='white', markeredgewidth=0.5,
                  label=f'T={t}', zorder=5)

ax_a.set_xticks(x_positions)
ax_a.set_xticklabels([f'D={d:.0f}' for d in distances])
ax_a.set_ylabel('P(choose heavy)', fontsize=11)
ax_a.set_xlabel('Distance', fontsize=11)
ax_a.set_title('A  Choice PPC by Condition', fontsize=12, fontweight='bold', loc='left')
ax_a.legend(fontsize=8, frameon=False)
ax_a.set_ylim(0, 1)
_style(ax_a)

# Panel B: Vigor PPC by condition
ax_b = fig.add_subplot(gs[0, 1])
cookie_types = ['Heavy', 'Light']
x_v = np.arange(len(cookie_types))

for i, t in enumerate(threats):
    mask = vigor_cond['threat'] == t
    obs = vigor_cond.loc[mask, 'obs_excess'].values
    obs_se = vigor_cond.loc[mask, 'obs_excess_sem'].values
    pred = vigor_cond.loc[mask, 'pred_excess'].values

    x = x_v + (i - 1) * width
    ax_b.bar(x, pred, width * 0.85, color=THREAT_COLORS[t], alpha=0.35, edgecolor='none')
    ax_b.errorbar(x, obs, yerr=obs_se, fmt='o', color=THREAT_COLORS[t],
                  markersize=6, capsize=3, markeredgecolor='white', markeredgewidth=0.5,
                  label=f'T={t}', zorder=5)

ax_b.set_xticks(x_v)
ax_b.set_xticklabels(cookie_types)
ax_b.set_ylabel('Excess effort (centered)', fontsize=11)
ax_b.set_xlabel('Cookie type', fontsize=11)
ax_b.set_title('B  Vigor PPC by Condition', fontsize=12, fontweight='bold', loc='left')
ax_b.legend(fontsize=8, frameon=False)
ax_b.axhline(0, color='#D1D5DB', linewidth=0.8, zorder=0)
_style(ax_b)

# Panel C: Per-subject choice r^2
ax_c = fig.add_subplot(gs[0, 2])
ax_c.scatter(subj_stats['pred_pH'], subj_stats['obs_pH'],
             s=20, alpha=0.5, color='#457B9D', edgecolor='white', linewidth=0.3)
ax_c.plot([0, 1], [0, 1], '--', color='#D1D5DB', linewidth=1, zorder=0)
ax_c.set_xlim(0, 1)
ax_c.set_ylim(0, 1)
ax_c.set_xlabel('Predicted P(heavy)', fontsize=11)
ax_c.set_ylabel('Observed P(heavy)', fontsize=11)
ax_c.set_title(f'C  Per-Subject Choice (r={r_choice_subj:.3f})',
               fontsize=12, fontweight='bold', loc='left')
_style(ax_c)

# Panel D: Per-subject vigor r^2
ax_d = fig.add_subplot(gs[1, 0])
ax_d.scatter(subj_stats['pred_excess'], subj_stats['obs_excess'],
             s=20, alpha=0.5, color='#E63946', edgecolor='white', linewidth=0.3)
mn = min(subj_stats['pred_excess'].min(), subj_stats['obs_excess'].min()) - 0.02
mx = max(subj_stats['pred_excess'].max(), subj_stats['obs_excess'].max()) + 0.02
ax_d.plot([mn, mx], [mn, mx], '--', color='#D1D5DB', linewidth=1, zorder=0)
ax_d.set_xlabel('Predicted excess effort', fontsize=11)
ax_d.set_ylabel('Observed excess effort', fontsize=11)
ax_d.set_title(f'D  Per-Subject Vigor (r={r_vigor_subj:.3f})',
               fontsize=12, fontweight='bold', loc='left')
_style(ax_d)

# Panel E: Choice residuals by condition
ax_e = fig.add_subplot(gs[1, 1])
choice_cond['resid'] = choice_cond['obs_pH'] - choice_cond['pred_pH']
labels_c = [f'T={t:.1f}\nD={d:.0f}' for t, d in
            zip(choice_cond['threat'], choice_cond['distance_H'])]
colors_c = [THREAT_COLORS[t] for t in choice_cond['threat']]
ax_e.bar(range(len(choice_cond)), choice_cond['resid'], color=colors_c, alpha=0.7, edgecolor='none')
ax_e.axhline(0, color='#D1D5DB', linewidth=0.8)
ax_e.set_xticks(range(len(choice_cond)))
ax_e.set_xticklabels(labels_c, fontsize=7, rotation=45, ha='right')
ax_e.set_ylabel('Residual (obs - pred)', fontsize=11)
ax_e.set_title('E  Choice Residuals', fontsize=12, fontweight='bold', loc='left')
_style(ax_e)

# Panel F: Vigor residuals by condition
ax_f = fig.add_subplot(gs[1, 2])
vigor_cond['resid'] = vigor_cond['obs_excess'] - vigor_cond['pred_excess']
labels_v = [f'T={t:.1f}\n{c}' for t, c in
            zip(vigor_cond['threat'], vigor_cond['cookie_type'])]
colors_v = [THREAT_COLORS[t] for t in vigor_cond['threat']]
ax_f.bar(range(len(vigor_cond)), vigor_cond['resid'], color=colors_v, alpha=0.7, edgecolor='none')
ax_f.axhline(0, color='#D1D5DB', linewidth=0.8)
ax_f.set_xticks(range(len(vigor_cond)))
ax_f.set_xticklabels(labels_v, fontsize=7, rotation=45, ha='right')
ax_f.set_ylabel('Residual (obs - pred)', fontsize=11)
ax_f.set_title('F  Vigor Residuals', fontsize=12, fontweight='bold', loc='left')
_style(ax_f)

plt.savefig(FIG_PATH, dpi=150, bbox_inches='tight', facecolor='white')
print(f"\nSaved figure: {FIG_PATH}")

# ══════════════════════════════════════════════════════════════════════════════
# 5. Save summary CSV
# ══════════════════════════════════════════════════════════════════════════════
choice_cond_out = choice_cond.copy()
choice_cond_out['domain'] = 'choice'
choice_cond_out = choice_cond_out.rename(columns={
    'obs_pH': 'obs_mean', 'obs_pH_sem': 'obs_sem',
    'pred_pH': 'pred_mean', 'pred_pH_sem': 'pred_sem',
})
choice_cond_out['cookie_type'] = 'all'

vigor_cond_out = vigor_cond.copy()
vigor_cond_out['domain'] = 'vigor'
vigor_cond_out = vigor_cond_out.rename(columns={
    'obs_excess': 'obs_mean', 'obs_excess_sem': 'obs_sem',
    'pred_excess': 'pred_mean', 'pred_excess_sem': 'pred_sem',
})
vigor_cond_out['distance_H'] = 'all'

cols = ['domain', 'threat', 'distance_H', 'cookie_type', 'obs_mean', 'obs_sem',
        'pred_mean', 'pred_sem', 'resid', 'n']
summary = pd.concat([choice_cond_out[cols], vigor_cond_out[cols]], ignore_index=True)

# Add overall metrics row
overall = pd.DataFrame([{
    'domain': 'overall', 'threat': np.nan, 'distance_H': np.nan,
    'cookie_type': np.nan, 'obs_mean': np.nan, 'obs_sem': np.nan,
    'pred_mean': np.nan, 'pred_sem': np.nan, 'resid': np.nan, 'n': len(beh_c),
    'choice_accuracy': choice_accuracy, 'choice_auc': auc,
    'vigor_r': r_vigor, 'vigor_r2': r_vigor**2,
    'subj_choice_r': r_choice_subj, 'subj_vigor_r': r_vigor_subj,
}])
summary = pd.concat([summary, overall], ignore_index=True)

summary.to_csv(STATS_PATH, index=False, float_format='%.6f')
print(f"Saved stats: {STATS_PATH}")

# ══════════════════════════════════════════════════════════════════════════════
# 6. Final summary
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("EVC-LQR PPC SUMMARY")
print("=" * 60)
print(f"Choice accuracy:         {choice_accuracy:.3f}")
print(f"Choice AUC:              {auc:.3f}")
print(f"Per-subj choice r:       {r_choice_subj:.3f}")
print(f"Trial-level vigor r:     {r_vigor:.3f}")
print(f"Trial-level vigor r^2:   {r_vigor**2:.3f}")
print(f"Per-subj vigor r:        {r_vigor_subj:.3f}")
print(f"Max |choice residual|:   {choice_cond['resid'].abs().max():.4f}")
print(f"Max |vigor residual|:    {vigor_cond['resid'].abs().max():.4f}")
print(f"Population: ce={CE:.4f}, gamma={GAMMA:.3f}, tau={TAU}, p_esc={P_ESC}")
print("=" * 60)
