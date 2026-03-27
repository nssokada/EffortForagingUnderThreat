"""
Generate publication-quality figures for the optimal control paper.
Figures saved to results/figs/paper/ at 200 DPI.
"""
import sys
sys.path.insert(0, '.')
import jax
import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist
from numpyro.infer import SVI, Trace_ELBO, Predictive
from numpyro.infer.autoguide import AutoNormal
from jax import random
import numpy as np
import pandas as pd
import ast
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from scipy.stats import pearsonr, ttest_rel
import os
import warnings
warnings.filterwarnings('ignore')

jax.config.update('jax_enable_x64', True)

OUT_DIR = 'results/figs/paper'
os.makedirs(OUT_DIR, exist_ok=True)

# Use a clean style
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams.update({
    'font.size': 11,
    'axes.titlesize': 13,
    'axes.labelsize': 12,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'figure.titlesize': 14,
    'axes.spines.top': False,
    'axes.spines.right': False,
})

# Colors
C_LOW = '#4878CF'   # blue for low threat
C_HIGH = '#C44E52'  # red for high threat
C_HEAVY = '#555555'
C_LIGHT = '#AAAAAA'

# ---------------------------------------------------------------------------
# DATA PREP
# ---------------------------------------------------------------------------
print("Loading data...")
beh = pd.read_csv('data/exploratory_350/processed/stage5_filtered_data_20260320_191950/behavior_rich.csv')
beh_c = beh[beh['type'] == 1].copy()

# Compute median_rate
rates = []
for _, row in beh_c.iterrows():
    try:
        press_times = np.array(ast.literal_eval(row['alignedEffortRate']), dtype=float)
    except:
        rates.append(np.nan)
        continue
    ipis = np.diff(press_times)
    ipis = ipis[ipis > 0.01]
    if len(ipis) < 5:
        rates.append(np.nan)
        continue
    rates.append(np.median((1.0 / ipis) / row['calibrationMax']))
beh_c['median_rate'] = rates

subjects = sorted(beh_c['subj'].unique())
subj_to_idx = {s: i for i, s in enumerate(subjects)}
N_S = len(subjects)
beh_c2 = beh_c.dropna(subset=['median_rate']).copy()
N_T = len(beh_c2)

subj_idx = jnp.array([subj_to_idx[s] for s in beh_c2['subj']])
T = jnp.array(beh_c2['threat'].values)
dist_H = jnp.array(beh_c2['distance_H'].values, dtype=jnp.float64)
choice = jnp.array(beh_c2['choice'].values)
vigor_obs = np.array(beh_c2['median_rate'].values)
chosen_R_arr = np.where(np.array(choice) == 1, 5.0, 1.0)

print(f"N subjects = {N_S}, N trials = {N_T}")

# ---------------------------------------------------------------------------
# FIT CHOICE MODEL
# ---------------------------------------------------------------------------
print("Fitting choice model (25k SVI steps)...")

def choice_model(subj_idx, T, dist_H, choice=None):
    mu_ce = numpyro.sample('mu_ce', dist.Normal(0.0, 1.0))
    mu_cd = numpyro.sample('mu_cd', dist.Normal(0.0, 1.0))
    sigma_ce = numpyro.sample('sigma_ce', dist.HalfNormal(0.5))
    sigma_cd = numpyro.sample('sigma_cd', dist.HalfNormal(0.5))
    tau_raw = numpyro.sample('tau_raw', dist.Normal(-1.0, 1.0))
    tau = jnp.clip(jnp.exp(tau_raw), 0.01, 20.0)
    p_esc_raw = numpyro.sample('p_esc_raw', dist.Normal(0.0, 1.0))
    p_esc = jax.nn.sigmoid(p_esc_raw)
    with numpyro.plate('subjects', N_S):
        ce_raw = numpyro.sample('ce_raw', dist.Normal(0.0, 1.0))
        cd_raw = numpyro.sample('cd_raw', dist.Normal(0.0, 1.0))
    c_effort = jnp.exp(mu_ce + sigma_ce * ce_raw)
    c_death = jnp.exp(mu_cd + sigma_cd * cd_raw)
    numpyro.deterministic('c_effort', c_effort)
    numpyro.deterministic('c_death', c_death)
    numpyro.deterministic('p_esc', p_esc)
    ce_i = c_effort[subj_idx]
    cd_i = c_death[subj_idx]
    S_full = (1.0 - T) + T * p_esc
    S_stop = 1.0 - T
    eu_H_full = S_full * 5 - (1 - S_full) * cd_i * 10 - ce_i * 0.81 * dist_H
    eu_H_stop = S_stop * 5 - (1 - S_stop) * cd_i * 10
    eu_H = jnp.maximum(eu_H_full, eu_H_stop)
    eu_L_full = S_full * 1 - (1 - S_full) * cd_i * 6 - ce_i * 0.16
    eu_L_stop = S_stop * 1 - (1 - S_stop) * cd_i * 6
    eu_L = jnp.maximum(eu_L_full, eu_L_stop)
    logit = jnp.clip((eu_H - eu_L) / tau, -20, 20)
    p_H = jax.nn.sigmoid(logit)
    numpyro.deterministic('p_H', p_H)
    with numpyro.plate('trials', N_T):
        numpyro.sample('obs', dist.Bernoulli(probs=jnp.clip(p_H, 1e-6, 1 - 1e-6)), obs=choice)

ckwargs = dict(subj_idx=subj_idx, T=T, dist_H=dist_H, choice=choice)
guide = AutoNormal(choice_model)
svi = SVI(choice_model, guide, numpyro.optim.Adam(0.003), Trace_ELBO())
state = svi.init(random.PRNGKey(42), **ckwargs)
update_fn = jax.jit(svi.update)
for i in range(25000):
    state, loss = update_fn(state, **ckwargs)
    if (i + 1) % 5000 == 0:
        print(f"  Step {i+1}: loss = {loss:.1f}")

params = svi.get_params(state)
pred = Predictive(choice_model, guide=guide, params=params, num_samples=500)
samples = pred(random.PRNGKey(44), **{k: v for k, v in ckwargs.items() if k != 'choice'})

ce = np.array(samples['c_effort']).mean(0)
cd = np.array(samples['c_death']).mean(0)
p_esc_val = float(np.array(samples['p_esc']).mean())
p_H_pred = np.array(samples['p_H']).mean(0)

print(f"p_esc = {p_esc_val:.3f}")
print(f"c_effort: mean={ce.mean():.3f}, c_death: mean={cd.mean():.3f}")

# ---------------------------------------------------------------------------
# Attach predictions back to data
# ---------------------------------------------------------------------------
beh_c2 = beh_c2.copy()
beh_c2['p_H_pred'] = np.array(p_H_pred)
beh_c2['subj_idx'] = [subj_to_idx[s] for s in beh_c2['subj']]
beh_c2['ce'] = ce[beh_c2['subj_idx'].values]
beh_c2['cd'] = cd[beh_c2['subj_idx'].values]

# Threat categories
threat_vals = np.array(beh_c2['threat'].values)
threat_sorted = np.sort(np.unique(threat_vals))
print(f"Unique threat levels: {threat_sorted}")

# Create threat bins (low/med/high or use actual values)
if len(threat_sorted) == 3:
    threat_labels = {threat_sorted[0]: 'Low', threat_sorted[1]: 'Med', threat_sorted[2]: 'High'}
else:
    # Tercile split
    t1, t2 = np.percentile(threat_vals, [33.3, 66.7])
    beh_c2['threat_bin'] = pd.cut(threat_vals, bins=[-np.inf, t1, t2, np.inf], labels=['Low', 'Med', 'High'])

if len(threat_sorted) == 3:
    beh_c2['threat_bin'] = beh_c2['threat'].map(threat_labels)

# Binary threat: use lowest vs highest, excluding middle
# This sharpens the Simpson's paradox contrast
beh_c2['threat_hi'] = np.where(beh_c2['threat'] == threat_sorted[-1], 1,
                                np.where(beh_c2['threat'] == threat_sorted[0], 0, np.nan))
# Keep a copy with all three for other figures, filter for Simpson's
beh_c2_binary = beh_c2.dropna(subset=['threat_hi']).copy()
beh_c2_binary['threat_hi'] = beh_c2_binary['threat_hi'].astype(int)

# =========================================================================
# FIGURE 1: Simpson's Paradox
# =========================================================================
print("\n--- Figure 1: Simpson's Paradox ---")

fig, axes = plt.subplots(1, 3, figsize=(13, 4.5))

def subj_means(df, groupcol='threat_hi', valcol='median_rate'):
    """Compute subject-level means for each group level."""
    return df.groupby(['subj', groupcol])[valcol].mean().reset_index()

def paired_bar(ax, df, title, groupcol='threat_hi', valcol='median_rate',
               ylim_bottom=None):
    """Bar plot with subject-level means, paired t-test, error bars."""
    sm = subj_means(df, groupcol, valcol)
    # Align subjects (only those with both conditions)
    sm_wide = sm.pivot(index='subj', columns=groupcol, values=valcol).dropna()
    lo_paired = sm_wide[0].values
    hi_paired = sm_wide[1].values
    t_stat, p_val = ttest_rel(lo_paired, hi_paired)
    d_val = np.mean(hi_paired - lo_paired) / np.std(hi_paired - lo_paired, ddof=1)

    means = [np.mean(lo_paired), np.mean(hi_paired)]
    sems = [np.std(lo_paired, ddof=1) / np.sqrt(len(lo_paired)),
            np.std(hi_paired, ddof=1) / np.sqrt(len(hi_paired))]

    bars = ax.bar([0, 1], means, yerr=sems, color=[C_LOW, C_HIGH],
                  edgecolor='black', linewidth=0.7, capsize=5, width=0.55,
                  error_kw={'linewidth': 1.2})
    ax.set_xticks([0, 1])
    ax.set_xticklabels(['Low threat\n(10%)', 'High threat\n(90%)'])
    ax.set_ylabel('Vigor (normalized median rate)')
    ax.set_title(title, fontweight='bold', fontsize=11)

    # Set y-axis to zoom into the relevant range
    if ylim_bottom is not None:
        ax.set_ylim(bottom=ylim_bottom)

    # Annotate p-value and effect size
    ymax = max(means) + max(sems) + 0.005
    if p_val < 0.001:
        pstr = f'p < .001, d = {d_val:.2f}'
    elif p_val < 0.05:
        pstr = f'p = {p_val:.3f}, d = {d_val:.2f}'
    else:
        pstr = f'p = {p_val:.2f}, d = {d_val:.2f}'
    ax.annotate(pstr, xy=(0.5, ymax + 0.003), ha='center', fontsize=9.5,
                fontstyle='italic')
    # Bracket
    ax.plot([0, 0, 1, 1], [ymax - 0.001, ymax, ymax, ymax - 0.001],
            color='black', linewidth=1)
    return p_val, d_val

# Panel A: Marginal (all trials regardless of choice)
p_a, d_a = paired_bar(axes[0], beh_c2_binary,
                       'A. Marginal effect of threat', ylim_bottom=0.85)

# Panel B: Heavy cookie only (choice == 1)
heavy_df = beh_c2_binary[beh_c2_binary['choice'] == 1]
p_b, d_b = paired_bar(axes[1], heavy_df,
                       'B. Heavy cookie only', ylim_bottom=0.85)

# Panel C: Light cookie only (choice == 0)
light_df = beh_c2_binary[beh_c2_binary['choice'] == 0]
p_c, d_c = paired_bar(axes[2], light_df,
                       'C. Light cookie only', ylim_bottom=0.85)

fig.suptitle("Simpson's Paradox: Threat Effects on Vigor",
             fontsize=14, fontweight='bold', y=1.02)
fig.tight_layout()
fig.savefig(f'{OUT_DIR}/fig1_simpsons_paradox.png', dpi=200, bbox_inches='tight')
plt.close()
print(f"  Marginal: p={p_a:.4f}, d={d_a:.3f}")
print(f"  Heavy:    p={p_b:.4f}, d={d_b:.3f}")
print(f"  Light:    p={p_c:.4f}, d={d_c:.3f}")

# =========================================================================
# FIGURE 2: OC Model Choice Predictions
# =========================================================================
print("\n--- Figure 2: Choice Predictions ---")

fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))

# Panel A: Observed vs predicted P(choose heavy) by condition
# Conditions = threat x distance (3 threat x 3 distance = 9 conditions)
dist_sorted = np.sort(beh_c2['distance_H'].unique())
if len(dist_sorted) > 3:
    beh_c2['dist_bin'] = pd.qcut(beh_c2['distance_H'], 3, labels=['Near', 'Mid', 'Far'])
else:
    dist_labels = {dist_sorted[0]: 'Near', dist_sorted[1]: 'Mid', dist_sorted[2]: 'Far'}
    beh_c2['dist_bin'] = beh_c2['distance_H'].map(dist_labels)

cond_obs = beh_c2.groupby(['threat_bin', 'dist_bin'])['choice'].mean()
cond_pred = beh_c2.groupby(['threat_bin', 'dist_bin'])['p_H_pred'].mean()

obs_vals = []
pred_vals = []
labels = []
for tb in ['Low', 'Med', 'High']:
    for db in ['Near', 'Mid', 'Far']:
        if (tb, db) in cond_obs.index:
            obs_vals.append(cond_obs[(tb, db)])
            pred_vals.append(cond_pred[(tb, db)])
            labels.append(f'{tb}/{db}')

obs_vals = np.array(obs_vals)
pred_vals = np.array(pred_vals)

# Color by threat
colors_cond = []
for l in labels:
    if l.startswith('Low'):
        colors_cond.append(C_LOW)
    elif l.startswith('Med'):
        colors_cond.append('#8B8B4B')
    else:
        colors_cond.append(C_HIGH)

# Markers by distance
markers_cond = []
for l in labels:
    if 'Near' in l:
        markers_cond.append('o')
    elif 'Mid' in l:
        markers_cond.append('s')
    else:
        markers_cond.append('^')

ax = axes[0]
for i in range(len(obs_vals)):
    ax.scatter(pred_vals[i], obs_vals[i], color=colors_cond[i],
               marker=markers_cond[i], s=80, zorder=5, edgecolors='black', linewidth=0.5)

# Identity line
lims = [min(min(obs_vals), min(pred_vals)) - 0.05,
        max(max(obs_vals), max(pred_vals)) + 0.05]
ax.plot(lims, lims, '--', color='gray', linewidth=1, zorder=1)
r_val, _ = pearsonr(obs_vals, pred_vals)
ax.set_xlabel('Predicted P(choose heavy)')
ax.set_ylabel('Observed P(choose heavy)')
ax.set_title('A. Choice predictions by condition', fontweight='bold')
ax.text(0.05, 0.92, f'r = {r_val:.3f}', transform=ax.transAxes, fontsize=11)

# Legend
from matplotlib.lines import Line2D
leg_elements = [
    Line2D([0], [0], marker='o', color='w', markerfacecolor='gray', markersize=8, label='Near'),
    Line2D([0], [0], marker='s', color='w', markerfacecolor='gray', markersize=8, label='Mid'),
    Line2D([0], [0], marker='^', color='w', markerfacecolor='gray', markersize=8, label='Far'),
    Line2D([0], [0], marker='o', color='w', markerfacecolor=C_LOW, markersize=8, label='Low threat'),
    Line2D([0], [0], marker='o', color='w', markerfacecolor='#8B8B4B', markersize=8, label='Med threat'),
    Line2D([0], [0], marker='o', color='w', markerfacecolor=C_HIGH, markersize=8, label='High threat'),
]
ax.legend(handles=leg_elements, fontsize=8, loc='lower right', ncol=2)

# Panel B: Choice accuracy by threat level
ax = axes[1]
beh_c2['correct'] = ((beh_c2['p_H_pred'] >= 0.5).astype(int) == beh_c2['choice'].values).astype(int)
acc_by_threat = beh_c2.groupby('threat_bin')['correct'].mean()
threat_order = ['Low', 'Med', 'High']
acc_vals = [acc_by_threat[t] for t in threat_order if t in acc_by_threat.index]
colors_bar = [C_LOW, '#8B8B4B', C_HIGH]
bars = ax.bar(range(len(acc_vals)), acc_vals, color=colors_bar[:len(acc_vals)],
              edgecolor='black', linewidth=0.7, width=0.55)
ax.set_xticks(range(len(acc_vals)))
ax.set_xticklabels([t for t in threat_order if t in acc_by_threat.index])
ax.set_ylabel('Classification accuracy')
ax.set_xlabel('Threat level')
ax.set_title('B. Choice accuracy by threat', fontweight='bold')
ax.set_ylim([0.5, 1.0])
# Add value labels
for i, v in enumerate(acc_vals):
    ax.text(i, v + 0.01, f'{v:.2f}', ha='center', fontsize=10)

overall_acc = beh_c2['correct'].mean()
ax.axhline(overall_acc, color='black', linestyle='--', linewidth=0.8, alpha=0.5)
ax.text(len(acc_vals) - 0.5, overall_acc + 0.01, f'Overall: {overall_acc:.2f}',
        fontsize=9, ha='right')

fig.suptitle('Figure 2. Choice Model Predictions', fontsize=14, fontweight='bold', y=1.02)
fig.tight_layout()
fig.savefig(f'{OUT_DIR}/fig2_choice_predictions.png', dpi=200, bbox_inches='tight')
plt.close()
print(f"  Overall choice accuracy = {overall_acc:.3f}, r = {r_val:.3f}")

# =========================================================================
# FIGURE 3: Zero-Parameter Vigor Prediction
# =========================================================================
print("\n--- Figure 3: Vigor Prediction ---")

# Compute predicted vigor from choice-only params
# v = benefit / (benefit + cost), power-scaled
gamma = 0.32
ce_trial = ce[beh_c2['subj_idx'].values]
cd_trial = cd[beh_c2['subj_idx'].values]
T_np = np.array(beh_c2['threat'].values)
chosen_R = np.where(np.array(beh_c2['choice'].values) == 1, 5.0, 1.0)

benefit = cd_trial * T_np * p_esc_val * (chosen_R + 5)
cost = ce_trial
v_pred_raw = benefit / (benefit + cost + 1e-8)
v_pred = v_pred_raw ** gamma

beh_c2['v_pred'] = v_pred

fig, axes = plt.subplots(1, 3, figsize=(14, 5))

# Panel A: Predicted vs observed vigor by threat|choice (6 points)
ax = axes[0]
beh_c2['choice_label'] = beh_c2['choice'].map({1: 'Heavy', 0: 'Light'})

cond_means = beh_c2.groupby(['threat_bin', 'choice_label']).agg(
    obs_vigor=('median_rate', 'mean'),
    pred_vigor=('v_pred', 'mean'),
    obs_se=('median_rate', lambda x: x.std() / np.sqrt(len(x))),
    pred_se=('v_pred', lambda x: x.std() / np.sqrt(len(x))),
).reset_index()

threat_order_map = {'Low': 0, 'Med': 1, 'High': 2}
cond_means['t_order'] = cond_means['threat_bin'].map(threat_order_map)
cond_means = cond_means.sort_values(['choice_label', 't_order'])

for choice_type, color, marker, offset in [('Heavy', C_HEAVY, 'o', -0.05),
                                              ('Light', C_LIGHT, 's', 0.05)]:
    sub = cond_means[cond_means['choice_label'] == choice_type]
    x_pos = sub['t_order'].values + offset
    ax.errorbar(x_pos, sub['obs_vigor'], yerr=sub['obs_se'],
                fmt=marker, color=color, markersize=8, capsize=4,
                label=f'{choice_type} (obs)', zorder=5, markeredgecolor='black',
                linewidth=1.5, markeredgewidth=0.5)
    ax.plot(x_pos, sub['pred_vigor'], '--', color=color, linewidth=1.5, alpha=0.7)
    ax.scatter(x_pos, sub['pred_vigor'], marker=marker, facecolors='none',
               edgecolors=color, s=70, linewidths=1.5, zorder=4,
               label=f'{choice_type} (pred)')

ax.set_xticks([0, 1, 2])
ax.set_xticklabels(['Low', 'Med', 'High'])
ax.set_xlabel('Threat level')
ax.set_ylabel('Vigor (median rate)')
ax.set_title('A. Vigor by threat and choice', fontweight='bold')
ax.legend(fontsize=8, loc='upper left')

# Correlation between 6-point means
r_cond, p_cond = pearsonr(cond_means['obs_vigor'], cond_means['pred_vigor'])
ax.text(0.95, 0.05, f'r = {r_cond:.3f}', transform=ax.transAxes, ha='right', fontsize=10)

# Panel B: Subject-level scatter
ax = axes[1]
subj_summary = beh_c2.groupby('subj').agg(
    obs_mean=('median_rate', 'mean'),
    pred_mean=('v_pred', 'mean'),
).reset_index()

ax.scatter(subj_summary['pred_mean'], subj_summary['obs_mean'],
           alpha=0.4, s=20, color='#4878CF', edgecolors='none')
r_subj, p_subj = pearsonr(subj_summary['pred_mean'], subj_summary['obs_mean'])

# Fit line
z = np.polyfit(subj_summary['pred_mean'], subj_summary['obs_mean'], 1)
p_line = np.poly1d(z)
x_range = np.linspace(subj_summary['pred_mean'].min(), subj_summary['pred_mean'].max(), 100)
ax.plot(x_range, p_line(x_range), '--', color=C_HIGH, linewidth=1.5)

ax.set_xlabel('Predicted mean vigor')
ax.set_ylabel('Observed mean vigor')
ax.set_title(f'B. Subject-level prediction (N={len(subj_summary)})', fontweight='bold')
ax.text(0.05, 0.92, f'r = {r_subj:.3f}\np < .001' if p_subj < 0.001 else f'r = {r_subj:.3f}\np = {p_subj:.3f}',
        transform=ax.transAxes, fontsize=10)

# Panel C: Residual analysis
ax = axes[2]
subj_summary['residual'] = subj_summary['obs_mean'] - subj_summary['pred_mean']
subj_summary['abs_residual'] = np.abs(subj_summary['residual'])

# Color by residual magnitude
scatter = ax.scatter(subj_summary['pred_mean'], subj_summary['residual'],
                     c=subj_summary['abs_residual'], cmap='RdYlGn_r',
                     alpha=0.5, s=20, edgecolors='none')
ax.axhline(0, color='black', linestyle='-', linewidth=0.8)
ax.set_xlabel('Predicted mean vigor')
ax.set_ylabel('Residual (obs - pred)')
ax.set_title('C. Residual analysis', fontweight='bold')

# Annotate worst subjects
worst = subj_summary.nlargest(5, 'abs_residual')
for _, row in worst.iterrows():
    subj_str = str(row['subj'])
    ax.annotate(f"{subj_str[:6]}...", (row['pred_mean'], row['residual']),
                fontsize=6, alpha=0.7, xytext=(5, 5), textcoords='offset points')

cbar = fig.colorbar(scatter, ax=ax, shrink=0.8, pad=0.02)
cbar.set_label('|Residual|', fontsize=9)

fig.suptitle('Figure 3. Zero-Parameter Vigor Predictions from Choice Model',
             fontsize=14, fontweight='bold', y=1.02)
fig.tight_layout()
fig.savefig(f'{OUT_DIR}/fig3_vigor_prediction.png', dpi=200, bbox_inches='tight')
plt.close()
print(f"  Condition r = {r_cond:.3f}, Subject r = {r_subj:.3f}")

# =========================================================================
# FIGURE 4: Cost Parameter Space
# =========================================================================
print("\n--- Figure 4: Cost Parameter Space ---")

fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))

# Build subject-level summary with params and behavior
subj_params = pd.DataFrame({
    'subj': subjects,
    'c_effort': ce,
    'c_death': cd,
})

# Merge behavioral outcomes
subj_beh = beh_c2.groupby('subj').agg(
    mean_reward=('trialReward', 'mean'),
    p_heavy=('choice', 'mean'),
    mean_vigor=('median_rate', 'mean'),
    n_trials=('choice', 'count'),
).reset_index()

# Choice accuracy per subject
subj_acc = beh_c2.groupby('subj')['correct'].mean().reset_index()
subj_acc.columns = ['subj', 'choice_acc']

subj_params = subj_params.merge(subj_beh, on='subj', how='left')
subj_params = subj_params.merge(subj_acc, on='subj', how='left')

# Escape rate (fraction of attack trials survived)
# Check if outcome column has escape info
if 'outcome' in beh_c2.columns:
    # Compute escape rate from attack trials
    attack_trials = beh_c2[beh_c2['threat'] > 0]
    if len(attack_trials) > 0:
        subj_esc = attack_trials.groupby('subj').apply(
            lambda x: (x['outcome'] == 'escape').mean() if 'escape' in x['outcome'].values else np.nan
        ).reset_index()
        subj_esc.columns = ['subj', 'escape_rate']
        subj_params = subj_params.merge(subj_esc, on='subj', how='left')

# Panel A: c_effort vs c_death, colored by mean reward (earnings proxy)
ax = axes[0]
# Cap outliers for plotting
ce_plot = np.clip(subj_params['c_effort'], 0, np.percentile(subj_params['c_effort'], 99))
cd_plot = np.clip(subj_params['c_death'], 0, np.percentile(subj_params['c_death'], 99))

scatter_a = ax.scatter(ce_plot, cd_plot,
                       c=subj_params['mean_reward'], cmap='viridis',
                       alpha=0.5, s=20, edgecolors='none')
ax.set_xlabel('c_effort (effort sensitivity)')
ax.set_ylabel('c_death (death sensitivity)')
ax.set_title('A. Cost parameters colored by earnings', fontweight='bold')
cbar_a = fig.colorbar(scatter_a, ax=ax, shrink=0.8, pad=0.02)
cbar_a.set_label('Mean reward', fontsize=9)

# Add quadrant labels
ax.axhline(np.median(cd_plot), color='gray', linestyle=':', linewidth=0.8, alpha=0.5)
ax.axvline(np.median(ce_plot), color='gray', linestyle=':', linewidth=0.8, alpha=0.5)

# Panel B: Same, colored by choice accuracy
ax = axes[1]
scatter_b = ax.scatter(ce_plot, cd_plot,
                       c=subj_params['choice_acc'], cmap='RdYlGn',
                       alpha=0.5, s=20, edgecolors='none')
ax.set_xlabel('c_effort (effort sensitivity)')
ax.set_ylabel('c_death (death sensitivity)')
ax.set_title('B. Cost parameters colored by choice accuracy', fontweight='bold')
cbar_b = fig.colorbar(scatter_b, ax=ax, shrink=0.8, pad=0.02)
cbar_b.set_label('Choice accuracy', fontsize=9)

ax.axhline(np.median(cd_plot), color='gray', linestyle=':', linewidth=0.8, alpha=0.5)
ax.axvline(np.median(ce_plot), color='gray', linestyle=':', linewidth=0.8, alpha=0.5)

# Correlation annotations
r_ce_acc, p_ce_acc = pearsonr(subj_params['c_effort'].dropna(), subj_params['choice_acc'].dropna())
r_cd_acc, p_cd_acc = pearsonr(subj_params['c_death'].dropna(), subj_params['choice_acc'].dropna())
ax.text(0.05, 0.92, f'r(ce,acc)={r_ce_acc:.2f}\nr(cd,acc)={r_cd_acc:.2f}',
        transform=ax.transAxes, fontsize=9)

fig.suptitle('Figure 4. Individual Differences in Cost Parameters',
             fontsize=14, fontweight='bold', y=1.02)
fig.tight_layout()
fig.savefig(f'{OUT_DIR}/fig4_cost_parameters.png', dpi=200, bbox_inches='tight')
plt.close()

print("\nAll figures saved to results/figs/paper/")
print("Done!")
