"""
13_vigor_dynamics.py — Dynamical vigor model: encounter-evoked reactivity
==========================================================================
Approach 3 (trial-phase means): compute pre/post encounter excess vigor
for all subjects × attack trials. Test whether encounter change relates
to threat, c_death, clinical measures. Then Approach 2 (piecewise slopes).

Output:
  - results/stats/evc_vigor_dynamics.csv
  - results/figs/paper/fig_vigor_dynamics.png
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from scipy import stats
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# ── Paths ──────────────────────────────────────────────────────────────
DATA_DIR = Path('/workspace/data/exploratory_350/processed')
VIGOR_TS = DATA_DIR / 'vigor_processed/smoothed_vigor_ts.parquet'
PARAMS_FILE = Path('/workspace/results/stats/oc_evc_final_params.csv')
PSYCH_FILE = DATA_DIR / 'stage5_filtered_data_20260320_191950/psych.csv'
FACTOR_FILE = Path('/workspace/results/stats/psych_factor_scores.csv')
OUT_STATS = Path('/workspace/results/stats/evc_vigor_dynamics.csv')
OUT_FIG = Path('/workspace/results/figs/paper/fig_vigor_dynamics.png')
OUT_FIG.parent.mkdir(parents=True, exist_ok=True)

# ── 1. Load data ──────────────────────────────────────────────────────
print("Loading data...")
df = pd.read_parquet(VIGOR_TS)
params = pd.read_csv(PARAMS_FILE)
psych = pd.read_csv(PSYCH_FILE)
factors = pd.read_csv(FACTOR_FILE)

print(f"  Vigor TS: {df.shape[0]:,} rows, {df['subj'].nunique()} subjects")
print(f"  Params: {params.shape[0]} subjects")
print(f"  Psych: {psych.shape[0]} subjects")

# ── 2. Compute cookie-centered excess (vigor_resid already is this) ──
# vigor_resid = vigor_norm - mean(vigor_norm for same choice at same time t)
# This is already in the data. Verify:
print("\nvigor_resid stats:")
print(f"  mean = {df['vigor_resid'].mean():.4f}")
print(f"  std  = {df['vigor_resid'].std():.4f}")

# ── 3. Approach 3: Pre/post encounter means ──────────────────────────
print("\n=== APPROACH 3: Trial-phase means ===")

# Filter to attack trials only
atk = df[df['isAttackTrial'] == 1].copy()
print(f"Attack trials: {atk.shape[0]:,} rows, "
      f"{atk.drop_duplicates(['subj','trial']).shape[0]:,} trials")

# Drop rows with NaN encounterTime
atk = atk.dropna(subset=['encounterTime'])
# Also need encounterTime > 0 (some are exactly 0 which means encounter at start)
atk = atk[atk['encounterTime'] > 0.1]
print(f"After filtering encounterTime > 0.1: {atk.shape[0]:,} rows")

# For each row, compute whether it's pre or post encounter
atk['phase'] = np.where(atk['t'] < atk['encounterTime'], 'pre', 'post')

# Compute pre and post means per subject × trial
phase_means = atk.groupby(['subj', 'trial', 'phase', 'threat', 'startDistance'])['vigor_resid'].mean().reset_index()
phase_wide = phase_means.pivot_table(
    index=['subj', 'trial', 'threat', 'startDistance'],
    columns='phase',
    values='vigor_resid'
).reset_index()

# Some trials may not have both phases (very early or late encounters)
phase_wide = phase_wide.dropna(subset=['pre', 'post'])
phase_wide['change'] = phase_wide['post'] - phase_wide['pre']

print(f"\nTrials with both phases: {phase_wide.shape[0]:,}")
print(f"Subjects: {phase_wide['subj'].nunique()}")

# ── 3a. Mean encounter change by threat level ────────────────────────
print("\n--- Encounter change by threat ---")
threat_means = phase_wide.groupby('threat')['change'].agg(['mean', 'std', 'count'])
for t, row in threat_means.iterrows():
    se = row['std'] / np.sqrt(row['count'])
    print(f"  T={t}: mean={row['mean']:.4f}, SE={se:.4f}, n={int(row['count'])}")

# One-way ANOVA on encounter change by threat
groups = [g['change'].values for _, g in phase_wide.groupby('threat')]
f_stat, p_anova = stats.f_oneway(*groups)
print(f"  ANOVA: F={f_stat:.3f}, p={p_anova:.4e}")

# Post-hoc t-tests
threats = sorted(phase_wide['threat'].unique())
for i in range(len(threats)):
    for j in range(i+1, len(threats)):
        g1 = phase_wide[phase_wide['threat'] == threats[i]]['change']
        g2 = phase_wide[phase_wide['threat'] == threats[j]]['change']
        t_stat, p_val = stats.ttest_ind(g1, g2)
        d = (g1.mean() - g2.mean()) / np.sqrt((g1.std()**2 + g2.std()**2) / 2)
        print(f"  T={threats[i]} vs T={threats[j]}: t={t_stat:.3f}, p={p_val:.4e}, d={d:.3f}")

# ── 3b. Per-subject encounter reactivity ─────────────────────────────
print("\n--- Per-subject encounter reactivity ---")
subj_react = phase_wide.groupby('subj')['change'].mean().reset_index()
subj_react.columns = ['subj', 'reactivity']
print(f"  N = {subj_react.shape[0]}")
print(f"  Mean reactivity = {subj_react['reactivity'].mean():.4f}")
print(f"  SD = {subj_react['reactivity'].std():.4f}")
print(f"  Range = [{subj_react['reactivity'].min():.4f}, {subj_react['reactivity'].max():.4f}]")

# One-sample t-test: is reactivity significantly different from 0?
t_one, p_one = stats.ttest_1samp(subj_react['reactivity'], 0)
print(f"  One-sample t-test vs 0: t={t_one:.3f}, p={p_one:.4e}")

# ── 3c. Per-subject reactivity by threat ─────────────────────────────
print("\n--- Per-subject reactivity BY threat ---")
subj_threat_react = phase_wide.groupby(['subj', 'threat'])['change'].mean().reset_index()
subj_threat_react.columns = ['subj', 'threat', 'reactivity']

for t in threats:
    vals = subj_threat_react[subj_threat_react['threat'] == t]['reactivity']
    t_stat, p_val = stats.ttest_1samp(vals, 0)
    print(f"  T={t}: mean={vals.mean():.4f}, t={t_stat:.3f}, p={p_val:.4e}")

# ── 3d. Correlate reactivity with model params ──────────────────────
print("\n--- Reactivity × model params ---")
merged = subj_react.merge(params, on='subj')
merged['log_cd'] = np.log(merged['c_death'])
merged['log_ce'] = np.log(merged['c_effort'])

for col, label in [('log_cd', 'log(c_death)'), ('log_ce', 'log(c_effort)')]:
    r, p = stats.pearsonr(merged['reactivity'], merged[col])
    print(f"  reactivity × {label}: r={r:.3f}, p={p:.4e}")

# ── 3e. Correlate with clinical measures ─────────────────────────────
print("\n--- Reactivity × clinical measures ---")
clin_merged = subj_react.merge(psych[['subj', 'STAI_Trait', 'STAI_State',
                                       'OASIS_Total', 'DASS21_Anxiety',
                                       'DASS21_Depression', 'DASS21_Stress',
                                       'PHQ9_Total', 'AMI_Total', 'MFIS_Total',
                                       'STICSA_Total']], on='subj')

clin_results = []
for col in ['STAI_Trait', 'STAI_State', 'OASIS_Total', 'DASS21_Anxiety',
            'DASS21_Depression', 'DASS21_Stress', 'PHQ9_Total', 'AMI_Total',
            'MFIS_Total', 'STICSA_Total']:
    valid = clin_merged[['reactivity', col]].dropna()
    if len(valid) > 10:
        r, p = stats.pearsonr(valid['reactivity'], valid[col])
        clin_results.append({'measure': col, 'r': r, 'p': p, 'n': len(valid)})
        star = '*' if p < 0.05 else ''
        print(f"  reactivity × {col}: r={r:.3f}, p={p:.4f}{star}")

# Factor scores
print("\n--- Reactivity × factor scores ---")
fac_merged = subj_react.merge(factors, on='subj')
for f in ['F1', 'F2', 'F3']:
    r, p = stats.pearsonr(fac_merged['reactivity'], fac_merged[f])
    clin_results.append({'measure': f, 'r': r, 'p': p, 'n': len(fac_merged)})
    star = '*' if p < 0.05 else ''
    print(f"  reactivity × {f}: r={r:.3f}, p={p:.4f}{star}")

# ── 3f. Stability across blocks ─────────────────────────────────────
print("\n--- Stability across blocks ---")
# Assign blocks: trials 0-26 = block 1, 27-53 = block 2, 54-80 = block 3
phase_wide['block'] = (phase_wide['trial'] // 27).clip(0, 2) + 1
block_react = phase_wide.groupby(['subj', 'block'])['change'].mean().reset_index()
block_wide = block_react.pivot(index='subj', columns='block', values='change').dropna()

if block_wide.shape[1] >= 2:
    r12, p12 = stats.pearsonr(block_wide[1], block_wide[2])
    r13, p13 = stats.pearsonr(block_wide[1], block_wide[3])
    r23, p23 = stats.pearsonr(block_wide[2], block_wide[3])
    print(f"  Block 1-2: r={r12:.3f}, p={p12:.4e}")
    print(f"  Block 1-3: r={r13:.3f}, p={p13:.4e}")
    print(f"  Block 2-3: r={r23:.3f}, p={p23:.4e}")
    mean_r = np.mean([r12, r13, r23])
    print(f"  Mean cross-block r = {mean_r:.3f}")

# ── 3g. Incremental prediction ───────────────────────────────────────
print("\n--- Does reactivity predict ABOVE AND BEYOND static params? ---")
from sklearn.linear_model import LinearRegression

# Outcome: some clinical measure (try STAI_Trait as a key one)
full = subj_react.merge(params, on='subj').merge(
    psych[['subj', 'STAI_Trait', 'OASIS_Total', 'AMI_Total']], on='subj')
full['log_cd'] = np.log(full['c_death'])
full['log_ce'] = np.log(full['c_effort'])
full = full.dropna()

for outcome in ['STAI_Trait', 'OASIS_Total', 'AMI_Total']:
    y = full[outcome].values
    X_base = full[['log_cd', 'log_ce']].values
    X_full = full[['log_cd', 'log_ce', 'reactivity']].values

    # R² for base model
    lr_base = LinearRegression().fit(X_base, y)
    r2_base = lr_base.score(X_base, y)

    # R² for full model
    lr_full = LinearRegression().fit(X_full, y)
    r2_full = lr_full.score(X_full, y)

    delta_r2 = r2_full - r2_base
    # F-test for incremental R²
    n = len(y)
    k_full = X_full.shape[1]
    k_base = X_base.shape[1]
    f_inc = (delta_r2 / (k_full - k_base)) / ((1 - r2_full) / (n - k_full - 1))
    p_inc = 1 - stats.f.cdf(f_inc, k_full - k_base, n - k_full - 1)
    print(f"  {outcome}: R²_base={r2_base:.4f}, R²_full={r2_full:.4f}, "
          f"ΔR²={delta_r2:.4f}, F={f_inc:.3f}, p={p_inc:.4f}")

# ── 4. Approach 2: Encounter-aligned timeseries ─────────────────────
print("\n=== APPROACH 2: Encounter-aligned timeseries ===")

# For attack trials, align time to encounter
atk_full = df[df['isAttackTrial'] == 1].copy()
atk_full = atk_full.dropna(subset=['encounterTime'])
atk_full = atk_full[atk_full['encounterTime'] > 0.1]
atk_full['t_enc'] = atk_full['t'] - atk_full['encounterTime']

# Bin into 0.1s bins around encounter
atk_full['t_bin'] = (atk_full['t_enc'] * 10).round() / 10  # 0.1s bins

# Window: -3s to +3s around encounter
window = atk_full[(atk_full['t_bin'] >= -3.0) & (atk_full['t_bin'] <= 3.0)]

# Mean vigor_resid by threat × t_bin
enc_ts = window.groupby(['threat', 't_bin'])['vigor_resid'].agg(['mean', 'sem']).reset_index()
print(f"Encounter-aligned timeseries: {enc_ts.shape[0]} bins")

# ── 4a. Piecewise slopes ────────────────────────────────────────────
print("\n--- Piecewise slopes per subject ---")

def fit_piecewise_slopes(group):
    """Fit pre and post-encounter slopes."""
    pre = group[group['t_enc'] < 0]
    post = group[(group['t_enc'] >= 0) & (group['t_enc'] <= 3)]

    result = {'subj': group['subj'].iloc[0]}

    if len(pre) >= 5:
        slope, intercept, r, p, se = stats.linregress(pre['t_enc'], pre['vigor_resid'])
        result['pre_slope'] = slope
        result['pre_intercept'] = intercept
    else:
        result['pre_slope'] = np.nan
        result['pre_intercept'] = np.nan

    if len(post) >= 5:
        slope, intercept, r, p, se = stats.linregress(post['t_enc'], post['vigor_resid'])
        result['post_slope'] = slope
        result['post_intercept'] = intercept
    else:
        result['post_slope'] = np.nan
        result['post_intercept'] = np.nan

    return pd.Series(result)

# Aggregate across trials per subject first, then fit slopes
# Actually, fit slopes on pooled trials per subject
subj_slopes_list = []
for subj_id, sg in atk_full.groupby('subj'):
    sg = sg.copy()
    sg['t_enc'] = sg['t'] - sg['encounterTime']
    sg_win = sg[(sg['t_enc'] >= -3) & (sg['t_enc'] <= 3)]
    result = fit_piecewise_slopes(sg_win)
    subj_slopes_list.append(result)

subj_slopes = pd.DataFrame(subj_slopes_list)
subj_slopes['slope_change'] = subj_slopes['post_slope'] - subj_slopes['pre_slope']

print(f"  N = {subj_slopes.dropna().shape[0]}")
print(f"  Pre-encounter slope: mean={subj_slopes['pre_slope'].mean():.4f}, "
      f"SD={subj_slopes['pre_slope'].std():.4f}")
print(f"  Post-encounter slope: mean={subj_slopes['post_slope'].mean():.4f}, "
      f"SD={subj_slopes['post_slope'].std():.4f}")
print(f"  Slope change: mean={subj_slopes['slope_change'].mean():.4f}")
t_slope, p_slope = stats.ttest_1samp(subj_slopes['slope_change'].dropna(), 0)
print(f"  Slope change t-test: t={t_slope:.3f}, p={p_slope:.4e}")

# Correlate slope change with reactivity
merged_slopes = subj_slopes.merge(subj_react, on='subj')
r_sc, p_sc = stats.pearsonr(
    merged_slopes['slope_change'].dropna(),
    merged_slopes.loc[merged_slopes['slope_change'].notna(), 'reactivity']
)
print(f"  slope_change × reactivity: r={r_sc:.3f}, p={p_sc:.4e}")

# Correlate slope change with log(cd)
merged_slopes2 = merged_slopes.merge(params, on='subj')
merged_slopes2['log_cd'] = np.log(merged_slopes2['c_death'])
valid = merged_slopes2.dropna(subset=['slope_change'])
r_cd, p_cd = stats.pearsonr(valid['slope_change'], valid['log_cd'])
print(f"  slope_change × log(c_death): r={r_cd:.3f}, p={p_cd:.4e}")

# ── 5. Save results ─────────────────────────────────────────────────
print("\n=== Saving results ===")

results = []

# Encounter change by threat
for t in threats:
    vals = phase_wide[phase_wide['threat'] == t]['change']
    results.append({
        'analysis': 'encounter_change_by_threat',
        'variable': f'T={t}',
        'mean': vals.mean(),
        'sd': vals.std(),
        'se': vals.std() / np.sqrt(len(vals)),
        'n': len(vals),
        'test': 'one-sample t',
        't_stat': stats.ttest_1samp(vals, 0)[0],
        'p_value': stats.ttest_1samp(vals, 0)[1],
    })

# ANOVA
results.append({
    'analysis': 'encounter_change_anova',
    'variable': 'threat',
    'test': 'one-way ANOVA',
    't_stat': f_stat,
    'p_value': p_anova,
    'n': phase_wide.shape[0],
})

# Overall reactivity
results.append({
    'analysis': 'overall_reactivity',
    'variable': 'encounter_change',
    'mean': subj_react['reactivity'].mean(),
    'sd': subj_react['reactivity'].std(),
    'n': len(subj_react),
    'test': 'one-sample t',
    't_stat': t_one,
    'p_value': p_one,
})

# Correlations with params
for col, label in [('log_cd', 'log(c_death)'), ('log_ce', 'log(c_effort)')]:
    r, p = stats.pearsonr(merged['reactivity'], merged[col])
    results.append({
        'analysis': 'reactivity_x_param',
        'variable': label,
        'mean': r,
        'test': 'Pearson r',
        'p_value': p,
        'n': len(merged),
    })

# Clinical correlations
for cr in clin_results:
    results.append({
        'analysis': 'reactivity_x_clinical',
        'variable': cr['measure'],
        'mean': cr['r'],
        'test': 'Pearson r',
        'p_value': cr['p'],
        'n': cr['n'],
    })

# Block stability
if block_wide.shape[1] >= 2:
    results.append({
        'analysis': 'block_stability',
        'variable': 'mean_cross_block_r',
        'mean': mean_r,
        'test': 'Pearson r (average)',
        'p_value': np.nan,
        'n': len(block_wide),
    })

# Piecewise slopes
results.append({
    'analysis': 'piecewise_slope_change',
    'variable': 'post_slope - pre_slope',
    'mean': subj_slopes['slope_change'].mean(),
    'sd': subj_slopes['slope_change'].std(),
    'test': 'one-sample t',
    't_stat': t_slope,
    'p_value': p_slope,
    'n': subj_slopes['slope_change'].notna().sum(),
})

results.append({
    'analysis': 'slope_change_x_log_cd',
    'variable': 'slope_change × log(c_death)',
    'mean': r_cd,
    'test': 'Pearson r',
    'p_value': p_cd,
    'n': len(valid),
})

results_df = pd.DataFrame(results)
results_df.to_csv(OUT_STATS, index=False)
print(f"Saved: {OUT_STATS}")

# ── 6. Robust stats for reporting ────────────────────────────────────
# Spearman and outlier-free Pearson
r_spearman, p_spearman = stats.spearmanr(merged['reactivity'], merged['log_cd'])
clean = merged[merged['reactivity'] < 2]  # remove extreme outlier
r_clean, p_clean = stats.pearsonr(clean['reactivity'], clean['log_cd'])
print(f"  Robust: Spearman r={r_spearman:.3f}, p={p_spearman:.4e}")
print(f"  Robust: Pearson (no outlier) r={r_clean:.3f}, p={p_clean:.4e}")

# Add robust stats to results
results.append({
    'analysis': 'reactivity_x_param_robust',
    'variable': 'log(c_death) Spearman',
    'mean': r_spearman,
    'test': 'Spearman rho',
    'p_value': p_spearman,
    'n': len(merged),
})
results.append({
    'analysis': 'reactivity_x_param_robust',
    'variable': 'log(c_death) no outlier',
    'mean': r_clean,
    'test': 'Pearson r',
    'p_value': p_clean,
    'n': len(clean),
})

# Re-save with robust results
results_df = pd.DataFrame(results)
results_df.to_csv(OUT_STATS, index=False)

# ── 7. Figure ────────────────────────────────────────────────────────
print("\n=== Creating figure ===")

fig = plt.figure(figsize=(14, 10))
gs = GridSpec(2, 2, hspace=0.35, wspace=0.3)

# Color palette for threat levels
threat_colors = {0.1: '#2ca02c', 0.5: '#ff7f0e', 0.9: '#d62728'}

# Panel A: Encounter-aligned excess vigor by threat
ax_a = fig.add_subplot(gs[0, 0])
for t_level in sorted(enc_ts['threat'].unique()):
    sub = enc_ts[enc_ts['threat'] == t_level]
    ax_a.plot(sub['t_bin'], sub['mean'], color=threat_colors[t_level],
              label=f'T={t_level}', lw=2)
    ax_a.fill_between(sub['t_bin'],
                       sub['mean'] - sub['sem'],
                       sub['mean'] + sub['sem'],
                       color=threat_colors[t_level], alpha=0.15)
ax_a.axvline(0, color='k', ls='--', lw=1, alpha=0.5)
ax_a.axhline(0, color='gray', ls=':', lw=0.5)
ax_a.set_xlabel('Time relative to encounter (s)')
ax_a.set_ylabel('Excess vigor (cookie-centered)')
ax_a.set_title('A. Encounter-aligned vigor by threat')
ax_a.legend(frameon=False)
ax_a.set_xlim(-3, 3)

# Panel B: Per-subject encounter reactivity distribution
ax_b = fig.add_subplot(gs[0, 1])
ax_b.hist(subj_react['reactivity'], bins=40, color='steelblue',
          edgecolor='white', alpha=0.8)
ax_b.axvline(0, color='k', ls='--', lw=1)
ax_b.axvline(subj_react['reactivity'].mean(), color='red', ls='-', lw=2,
             label=f'Mean={subj_react["reactivity"].mean():.3f}')
ax_b.set_xlabel('Encounter reactivity (post - pre)')
ax_b.set_ylabel('Count')
ax_b.set_title('B. Distribution of encounter reactivity')
ax_b.legend(frameon=False)

# Panel C: Reactivity × log(c_death) scatter (use clean data)
ax_c = fig.add_subplot(gs[1, 0])
ax_c.scatter(clean['log_cd'], clean['reactivity'], alpha=0.3, s=15,
             color='steelblue', edgecolors='none')
# Mark outlier separately
outliers = merged[merged['reactivity'] >= 2]
if len(outliers) > 0:
    ax_c.scatter(outliers['log_cd'], outliers['reactivity'], alpha=0.5, s=30,
                 color='red', edgecolors='darkred', marker='x', zorder=5)
# Add regression line (on clean data)
z_fit = np.polyfit(clean['log_cd'], clean['reactivity'], 1)
x_line = np.linspace(clean['log_cd'].min(), clean['log_cd'].max(), 100)
ax_c.plot(x_line, np.polyval(z_fit, x_line), 'r-', lw=2)
ax_c.set_xlabel('log(c_death)')
ax_c.set_ylabel('Encounter reactivity')
ax_c.set_title(f'C. Reactivity x log(c_death)\nr={r_clean:.3f} (rho={r_spearman:.3f})')

# Panel D: Key correlates bar chart
ax_d = fig.add_subplot(gs[1, 1])
# Show bar chart of key correlations
bar_data = [
    ('log(cd)', stats.pearsonr(merged['reactivity'], merged['log_cd'])[0],
     stats.pearsonr(merged['reactivity'], merged['log_cd'])[1]),
    ('log(ce)', stats.pearsonr(merged['reactivity'], merged['log_ce'])[0],
     stats.pearsonr(merged['reactivity'], merged['log_ce'])[1]),
    ('AMI', stats.pearsonr(clin_merged['reactivity'], clin_merged['AMI_Total'])[0],
     stats.pearsonr(clin_merged['reactivity'], clin_merged['AMI_Total'])[1]),
    ('F3\n(apathy)', stats.pearsonr(fac_merged['reactivity'], fac_merged['F3'])[0],
     stats.pearsonr(fac_merged['reactivity'], fac_merged['F3'])[1]),
    ('STAI-T', stats.pearsonr(clin_merged['reactivity'], clin_merged['STAI_Trait'])[0],
     stats.pearsonr(clin_merged['reactivity'], clin_merged['STAI_Trait'])[1]),
    ('OASIS', stats.pearsonr(clin_merged['reactivity'], clin_merged['OASIS_Total'])[0],
     stats.pearsonr(clin_merged['reactivity'], clin_merged['OASIS_Total'])[1]),
]
labels = [d[0] for d in bar_data]
r_vals = [d[1] for d in bar_data]
p_vals = [d[2] for d in bar_data]
colors = ['firebrick' if abs(r) > 0.1 and p < 0.05 else 'gray' for r, p in zip(r_vals, p_vals)]

bars = ax_d.barh(range(len(labels)), r_vals, color=colors, alpha=0.7, edgecolor='white')
ax_d.set_yticks(range(len(labels)))
ax_d.set_yticklabels(labels)
ax_d.axvline(0, color='k', lw=0.5)
ax_d.set_xlabel('Pearson r')
ax_d.set_title('D. Correlates of encounter reactivity')
# Add significance stars
for i, (r_v, p_v) in enumerate(zip(r_vals, p_vals)):
    if p_v < 0.001:
        ax_d.text(r_v + 0.02 * np.sign(r_v), i, '***', va='center', fontsize=10)
    elif p_v < 0.01:
        ax_d.text(r_v + 0.02 * np.sign(r_v), i, '**', va='center', fontsize=10)
    elif p_v < 0.05:
        ax_d.text(r_v + 0.02 * np.sign(r_v), i, '*', va='center', fontsize=10)
ax_d.invert_yaxis()

fig.savefig(OUT_FIG, dpi=200, bbox_inches='tight', facecolor='white')
print(f"Saved: {OUT_FIG}")
plt.close()

# ── 7. Summary ───────────────────────────────────────────────────────
print("\n" + "="*60)
print("SUMMARY")
print("="*60)
print(f"\n1. Overall encounter reactivity: mean={subj_react['reactivity'].mean():.4f}, "
      f"t={t_one:.3f}, p={p_one:.4e}")
print(f"   {'SIGNIFICANT' if p_one < 0.05 else 'NOT SIGNIFICANT'} — "
      f"vigor {'increases' if subj_react['reactivity'].mean() > 0 else 'decreases'} "
      f"after predator encounter")

print(f"\n2. Threat modulates encounter response: F={f_stat:.3f}, p={p_anova:.4e}")
print(f"   {'SIGNIFICANT' if p_anova < 0.05 else 'NOT SIGNIFICANT'}")
for t in threats:
    vals = phase_wide[phase_wide['threat'] == t]['change']
    print(f"   T={t}: Δ = {vals.mean():.4f}")

r_cd_val, p_cd_val = stats.pearsonr(merged['reactivity'], merged['log_cd'])
print(f"\n3. Reactivity × log(c_death): r={r_cd_val:.3f}, p={p_cd_val:.4e}")
print(f"   Spearman rho={r_spearman:.3f}, Pearson (no outlier) r={r_clean:.3f}")
print(f"   {'SIGNIFICANT' if p_cd_val < 0.05 else 'NOT SIGNIFICANT'}")

print(f"\n4. Block stability: mean r = {mean_r:.3f}")
print(f"   {'STABLE (trait-like)' if mean_r > 0.3 else 'UNSTABLE (state-dependent)' if mean_r < 0.1 else 'MODERATE STABILITY'}")

print(f"\n5. Piecewise slope change: t={t_slope:.3f}, p={p_slope:.4e}")

sig_clin = [cr for cr in clin_results if cr['p'] < 0.05]
if sig_clin:
    print(f"\n6. Significant clinical correlations (uncorrected p<.05):")
    for cr in sig_clin:
        print(f"   reactivity × {cr['measure']}: r={cr['r']:.3f}, p={cr['p']:.4f}")
else:
    print(f"\n6. No significant clinical correlations (all p>.05)")

print("\nDone.")
