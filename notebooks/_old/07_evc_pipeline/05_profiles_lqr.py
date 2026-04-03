#!/usr/bin/env python3
"""
Behavioral Profiles & Quadrant Analysis for EVC-LQR model.

Analyses:
  1. cd x eps quadrant analysis (median split)
  2. Compare quadrants on: P(heavy), survival, earnings, OASIS, AMI, PHQ-9
  3. Parameter -> outcome correlations and regressions

Output:
  results/stats/evc_lqr_profiles.csv
  results/figs/paper/fig_lqr_quadrants.png
"""

import sys
sys.path.insert(0, '/workspace')

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, f_oneway, ttest_ind, spearmanr
from statsmodels.stats.multicomp import pairwise_tukeyhsd
import statsmodels.api as sm
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import warnings
warnings.filterwarnings('ignore')

plt.rcParams.update({
    'font.sans-serif': ['DejaVu Sans', 'Arial', 'sans-serif'],
    'font.family': 'sans-serif',
    'figure.dpi': 150,
    'axes.spines.right': False,
    'axes.spines.top': False,
})

# ── Paths ─────────────────────────────────────────────────────────────────────
DATA_DIR   = '/workspace/data/exploratory_350/processed/stage5_filtered_data_20260320_191950'
PARAM_FILE = '/workspace/results/stats/oc_evc_lqr_final_params.csv'
POP_FILE   = '/workspace/results/stats/oc_evc_lqr_final_population.csv'
FIG_OUT    = '/workspace/results/figs/paper/fig_lqr_quadrants.png'
STATS_OUT  = '/workspace/results/stats/evc_lqr_profiles.csv'

# ── Load data ─────────────────────────────────────────────────────────────────
params = pd.read_csv(PARAM_FILE)
behav  = pd.read_csv(f'{DATA_DIR}/behavior.csv')
psych  = pd.read_csv(f'{DATA_DIR}/psych.csv')

pop = pd.read_csv(POP_FILE)
CE = float(pop['c_effort'].iloc[0])
GAMMA = float(pop['gamma'].iloc[0])

print(f"Params: {len(params)} subjects")
print(f"Behavior: {len(behav)} trials, {behav['subj'].nunique()} subjects")
print(f"Psych: {len(psych)} subjects")

# ── Compute subject-level behavioral summaries ────────────────────────────────
subj_behav = behav.groupby('subj').agg(
    p_heavy    = ('choice', 'mean'),
    survival   = ('outcome', lambda x: 1 - x.mean()),
    n_trials   = ('choice', 'count'),
).reset_index()

# Earnings (approximate: R=5 for heavy, R=1 for light; -5 for capture)
beh_earn = behav.copy()
beh_earn['reward_earned'] = np.where(beh_earn['choice'] == 1, 5.0, 1.0) * (1 - beh_earn['outcome']) - 5.0 * beh_earn['outcome']
subj_earn = beh_earn.groupby('subj')['reward_earned'].sum().reset_index()
subj_earn.columns = ['subj', 'total_earnings']

# Merge everything
df = params.merge(subj_behav, on='subj').merge(subj_earn, on='subj').merge(psych, on='subj')
df['log_cd'] = np.log(df['c_death'])
df['log_eps'] = np.log(df['epsilon'])

print(f"\nMerged: {len(df)} subjects")
print(f"  P(heavy): {df['p_heavy'].mean():.3f} +/- {df['p_heavy'].std():.3f}")
print(f"  Survival: {df['survival'].mean():.3f} +/- {df['survival'].std():.3f}")
print(f"  Earnings: {df['total_earnings'].mean():.1f} +/- {df['total_earnings'].std():.1f}")

# ══════════════════════════════════════════════════════════════════════════════
# 1. Parameter-behavior correlations (log space)
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("1. PARAMETER-BEHAVIOR CORRELATIONS")
print("=" * 60)

outcomes = ['p_heavy', 'survival', 'total_earnings', 'OASIS_Total', 'AMI_Total', 'PHQ9_Total', 'STAI_Trait']
corr_results = []

for param in ['log_cd', 'log_eps']:
    for outcome in outcomes:
        r, p = pearsonr(df[param], df[outcome])
        print(f"  {param:8s} -> {outcome:18s}: r={r:+.3f}, p={p:.4f}")
        corr_results.append({
            'parameter': param, 'outcome': outcome, 'r': r, 'p': p,
        })

# ══════════════════════════════════════════════════════════════════════════════
# 2. cd x eps quadrant analysis (median split)
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("2. QUADRANT ANALYSIS (median split)")
print("=" * 60)

cd_med = df['c_death'].median()
eps_med = df['epsilon'].median()
print(f"Median c_death: {cd_med:.3f}")
print(f"Median epsilon: {eps_med:.3f}")

df['cd_group'] = np.where(df['c_death'] >= cd_med, 'high_cd', 'low_cd')
df['eps_group'] = np.where(df['epsilon'] >= eps_med, 'high_eps', 'low_eps')
df['quadrant'] = df['cd_group'] + '_' + df['eps_group']

# Quadrant labels
quadrant_labels = {
    'high_cd_high_eps': 'Vigilant\n(hi cd, hi eps)',
    'high_cd_low_eps':  'Helpless\n(hi cd, lo eps)',
    'low_cd_high_eps':  'Reckless\n(lo cd, hi eps)',
    'low_cd_low_eps':   'Disengaged\n(lo cd, lo eps)',
}

quadrant_colors = {
    'high_cd_high_eps': '#E63946',
    'high_cd_low_eps':  '#457B9D',
    'low_cd_high_eps':  '#2A9D8F',
    'low_cd_low_eps':   '#E9C46A',
}

print("\nQuadrant sizes:")
for q in ['high_cd_high_eps', 'high_cd_low_eps', 'low_cd_high_eps', 'low_cd_low_eps']:
    n = (df['quadrant'] == q).sum()
    print(f"  {quadrant_labels[q].split(chr(10))[0]:12s}: n={n}")

# Compare quadrants on outcomes
quad_results = []
print("\nQuadrant comparisons:")
for outcome in outcomes:
    groups = [df.loc[df['quadrant'] == q, outcome].values for q in
              ['high_cd_high_eps', 'high_cd_low_eps', 'low_cd_high_eps', 'low_cd_low_eps']]
    F, p = f_oneway(*groups)
    means = [g.mean() for g in groups]
    print(f"\n  {outcome}:")
    print(f"    F({3},{len(df)-4}) = {F:.2f}, p = {p:.4f}")
    for q, m in zip(['Vigilant', 'Helpless', 'Reckless', 'Disengaged'], means):
        print(f"      {q:12s}: {m:.3f}")

    quad_results.append({
        'outcome': outcome, 'F': F, 'p': p,
        'mean_vigilant': means[0], 'mean_helpless': means[1],
        'mean_reckless': means[2], 'mean_disengaged': means[3],
    })

# ══════════════════════════════════════════════════════════════════════════════
# 3. Multiple regression: params -> task outcomes
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("3. MULTIPLE REGRESSION")
print("=" * 60)

reg_results = []
for outcome in ['p_heavy', 'survival', 'total_earnings']:
    X = df[['log_cd', 'log_eps']].copy()
    X['interaction'] = X['log_cd'] * X['log_eps']
    X = sm.add_constant(X)
    model = sm.OLS(df[outcome], X).fit()

    print(f"\n  {outcome} ~ log_cd + log_eps + log_cd:log_eps")
    print(f"    R2={model.rsquared:.3f}, R2_adj={model.rsquared_adj:.3f}")
    print(f"    b_cd={model.params['log_cd']:+.3f} (p={model.pvalues['log_cd']:.4f})")
    print(f"    b_eps={model.params['log_eps']:+.3f} (p={model.pvalues['log_eps']:.4f})")
    print(f"    b_int={model.params['interaction']:+.3f} (p={model.pvalues['interaction']:.4f})")

    reg_results.append({
        'outcome': outcome, 'R2': model.rsquared, 'R2_adj': model.rsquared_adj,
        'b_cd': model.params['log_cd'], 'p_cd': model.pvalues['log_cd'],
        'b_eps': model.params['log_eps'], 'p_eps': model.pvalues['log_eps'],
        'b_int': model.params['interaction'], 'p_int': model.pvalues['interaction'],
    })

# ══════════════════════════════════════════════════════════════════════════════
# Save results
# ══════════════════════════════════════════════════════════════════════════════
all_results = []
for r in corr_results:
    r['analysis'] = 'correlation'
    all_results.append(r)
for r in quad_results:
    r['analysis'] = 'quadrant_anova'
    all_results.append(r)
for r in reg_results:
    r['analysis'] = 'regression'
    all_results.append(r)

pd.DataFrame(all_results).to_csv(STATS_OUT, index=False)
print(f"\nSaved: {STATS_OUT}")

# ══════════════════════════════════════════════════════════════════════════════
# 4. Publication figure (4 panels)
# ══════════════════════════════════════════════════════════════════════════════
print("\nCreating figure...")

fig = plt.figure(figsize=(14, 12))
gs = GridSpec(2, 2, figure=fig, hspace=0.35, wspace=0.35)

# ── Panel A: Parameter space scatter ──
ax_a = fig.add_subplot(gs[0, 0])

for q in ['high_cd_high_eps', 'high_cd_low_eps', 'low_cd_high_eps', 'low_cd_low_eps']:
    mask = df['quadrant'] == q
    ax_a.scatter(df.loc[mask, 'log_cd'], df.loc[mask, 'log_eps'],
                 s=25, alpha=0.5, color=quadrant_colors[q],
                 label=quadrant_labels[q].split('\n')[0],
                 edgecolor='white', linewidth=0.3)

ax_a.axvline(np.log(cd_med), color='#D1D5DB', linewidth=1, linestyle='--')
ax_a.axhline(np.log(eps_med), color='#D1D5DB', linewidth=1, linestyle='--')
ax_a.set_xlabel('log(c_death)', fontsize=12)
ax_a.set_ylabel('log(epsilon)', fontsize=12)
ax_a.set_title('A  Parameter Space', fontsize=13, fontweight='bold', loc='left')
ax_a.legend(fontsize=8, frameon=False, loc='upper left')
ax_a.spines['top'].set_visible(False)
ax_a.spines['right'].set_visible(False)

# ── Panel B: P(heavy) by quadrant ──
ax_b = fig.add_subplot(gs[0, 1])

quads = ['high_cd_high_eps', 'high_cd_low_eps', 'low_cd_high_eps', 'low_cd_low_eps']
quad_short = ['Vigilant', 'Helpless', 'Reckless', 'Disengaged']
x_pos = np.arange(4)

for outcome_name, ax_plot, title, panel in [('p_heavy', ax_b, 'P(choose heavy)', 'B')]:
    means = [df.loc[df['quadrant'] == q, outcome_name].mean() for q in quads]
    sems = [df.loc[df['quadrant'] == q, outcome_name].sem() for q in quads]
    colors = [quadrant_colors[q] for q in quads]
    ax_plot.bar(x_pos, means, yerr=sems, color=colors, alpha=0.7, edgecolor='none', capsize=4)
    ax_plot.set_xticks(x_pos)
    ax_plot.set_xticklabels(quad_short, fontsize=9, rotation=20, ha='right')
    ax_plot.set_ylabel(title, fontsize=11)
    ax_plot.set_title(f'{panel}  {title} by Quadrant', fontsize=13, fontweight='bold', loc='left')
    ax_plot.spines['top'].set_visible(False)
    ax_plot.spines['right'].set_visible(False)

# ── Panel C: Survival by quadrant ──
ax_c = fig.add_subplot(gs[1, 0])
means = [df.loc[df['quadrant'] == q, 'survival'].mean() for q in quads]
sems = [df.loc[df['quadrant'] == q, 'survival'].sem() for q in quads]
colors = [quadrant_colors[q] for q in quads]
ax_c.bar(x_pos, means, yerr=sems, color=colors, alpha=0.7, edgecolor='none', capsize=4)
ax_c.set_xticks(x_pos)
ax_c.set_xticklabels(quad_short, fontsize=9, rotation=20, ha='right')
ax_c.set_ylabel('Survival rate', fontsize=11)
ax_c.set_title('C  Survival by Quadrant', fontsize=13, fontweight='bold', loc='left')
ax_c.spines['top'].set_visible(False)
ax_c.spines['right'].set_visible(False)

# ── Panel D: Clinical by quadrant (OASIS) ──
ax_d = fig.add_subplot(gs[1, 1])

clinical_outcomes = ['OASIS_Total', 'PHQ9_Total', 'AMI_Total']
clinical_short = ['OASIS', 'PHQ-9', 'AMI']
width_clin = 0.22

for i, (clin, clin_label) in enumerate(zip(clinical_outcomes, clinical_short)):
    means = [df.loc[df['quadrant'] == q, clin].mean() for q in quads]
    sems = [df.loc[df['quadrant'] == q, clin].sem() for q in quads]
    ax_d.bar(x_pos + (i - 1) * width_clin, means, width_clin * 0.9, yerr=sems,
             label=clin_label, alpha=0.7, capsize=2, edgecolor='none')

ax_d.set_xticks(x_pos)
ax_d.set_xticklabels(quad_short, fontsize=9, rotation=20, ha='right')
ax_d.set_ylabel('Score', fontsize=11)
ax_d.set_title('D  Clinical Measures by Quadrant', fontsize=13, fontweight='bold', loc='left')
ax_d.legend(fontsize=8, frameon=False)
ax_d.spines['top'].set_visible(False)
ax_d.spines['right'].set_visible(False)

plt.savefig(FIG_OUT, dpi=150, bbox_inches='tight', facecolor='white')
print(f"Saved: {FIG_OUT}")

print("\nDone!")
