#!/usr/bin/env python3
"""
Behavioral Profiles & Quadrant Analysis — EVC+gamma model parameters.

Analyses:
  1. Parameter-behavior correlations (log space)
  2. cd × ε quadrant analysis (median split)
  3. Multiple regression: params → task outcomes
  4. Publication figure (4 panels)

Outputs:
  - results/figs/paper/fig_quadrants.png (150 DPI)
  - results/stats/evc_quadrant_analysis.csv
"""

import sys
sys.path.insert(0, '/workspace')
sys.path.insert(0, '/workspace/scripts')

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

# ── Paths ─────────────────────────────────────────────────────────────────────
DATA_DIR  = '/workspace/data/exploratory_350/processed/stage5_filtered_data_20260320_191950'
PARAM_FILE = '/workspace/results/stats/oc_evc_gamma_params.csv'
FIG_OUT   = '/workspace/results/figs/paper/fig_quadrants.png'
STATS_OUT = '/workspace/results/stats/evc_quadrant_analysis.csv'

# ── Load data ─────────────────────────────────────────────────────────────────
params = pd.read_csv(PARAM_FILE)
behav  = pd.read_csv(f'{DATA_DIR}/behavior.csv')
psych  = pd.read_csv(f'{DATA_DIR}/psych.csv')

print(f"Params: {len(params)} subjects")
print(f"Behavior: {len(behav)} trials, {behav['subj'].nunique()} subjects")
print(f"Psych: {len(psych)} subjects")

# ── Compute subject-level behavioral summaries ────────────────────────────────
# choice=1 means chose H (heavy/high-effort option)
subj_behav = behav.groupby('subj').agg(
    p_heavy    = ('choice', 'mean'),
    survival   = ('outcome', lambda x: 1 - x.mean()),  # outcome=1 means captured
    n_trials   = ('choice', 'count'),
).reset_index()

# Total earnings: +5 if chose H and survived, +1 if chose L and survived, -5 if captured
# R_H = 5, R_L = 1, capture penalty = -5
def compute_earnings(group):
    reward = np.where(group['choice'] == 1, 5.0, 1.0)
    earnings = np.where(group['outcome'] == 1, -5.0, reward)
    return earnings.sum()

earnings = behav.groupby('subj').apply(compute_earnings).reset_index()
earnings.columns = ['subj', 'total_earnings']
subj_behav = subj_behav.merge(earnings, on='subj')

# ── Merge all subject-level data ──────────────────────────────────────────────
df = params.merge(subj_behav, on='subj').merge(psych[['subj', 'OASIS_Total', 'STAI_State',
                                                        'PHQ9_Total', 'AMI_Emotional',
                                                        'STAI_Trait', 'DASS21_Anxiety']], on='subj')

# Log-transform parameters
df['log_ce']  = np.log(df['c_effort'])
df['log_cd']  = np.log(df['c_death'])
df['log_eps'] = np.log(df['epsilon'])

print(f"\nMerged dataset: {len(df)} subjects")
print(f"\nParameter summary (raw):")
print(df[['c_effort', 'c_death', 'epsilon']].describe().round(4))
print(f"\nParameter summary (log):")
print(df[['log_ce', 'log_cd', 'log_eps']].describe().round(4))

# ══════════════════════════════════════════════════════════════════════════════
# 1. PARAMETER-BEHAVIOR CORRELATIONS (log space)
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*70)
print("1. PARAMETER-BEHAVIOR CORRELATIONS (log space)")
print("="*70)

log_params = ['log_ce', 'log_cd', 'log_eps']
outcomes   = ['p_heavy', 'survival', 'total_earnings']
param_labels = {'log_ce': 'log(c_effort)', 'log_cd': 'log(c_death)', 'log_eps': 'log(ε)'}
outcome_labels = {'p_heavy': 'P(choose heavy)', 'survival': 'Survival rate', 'total_earnings': 'Total earnings'}

corr_rows = []
for p in log_params:
    for o in outcomes:
        r, pval = pearsonr(df[p], df[o])
        rho, pval_s = spearmanr(df[p], df[o])
        corr_rows.append({
            'parameter': param_labels[p],
            'outcome': outcome_labels[o],
            'pearson_r': r,
            'pearson_p': pval,
            'spearman_rho': rho,
            'spearman_p': pval_s,
            'n': len(df)
        })
        sig = '***' if pval < 0.001 else '**' if pval < 0.01 else '*' if pval < 0.05 else ''
        print(f"  {param_labels[p]:15s} × {outcome_labels[o]:20s}: r={r:+.3f} (p={pval:.4f}) {sig}")

corr_df = pd.DataFrame(corr_rows)

# ══════════════════════════════════════════════════════════════════════════════
# 2. cd × ε QUADRANT ANALYSIS
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*70)
print("2. cd × ε QUADRANT ANALYSIS (median split)")
print("="*70)

med_cd  = df['log_cd'].median()
med_eps = df['log_eps'].median()
print(f"  Median log(cd) = {med_cd:.3f},  Median log(ε) = {med_eps:.3f}")

def assign_quadrant(row):
    hi_cd  = row['log_cd']  >= med_cd
    hi_eps = row['log_eps'] >= med_eps
    if hi_cd and hi_eps:
        return 'Adaptive\nVigilance'
    elif hi_cd and not hi_eps:
        return 'Anxious\nHelplessness'
    elif not hi_cd and hi_eps:
        return 'Bold\nVigorous'
    else:
        return 'Disengaged'

df['quadrant'] = df.apply(assign_quadrant, axis=1)

# Short names for stats output
quad_short = {
    'Adaptive\nVigilance': 'Adaptive Vigilance',
    'Anxious\nHelplessness': 'Anxious Helplessness',
    'Bold\nVigorous': 'Bold Vigorous',
    'Disengaged': 'Disengaged'
}

quad_order = ['Adaptive\nVigilance', 'Anxious\nHelplessness', 'Bold\nVigorous', 'Disengaged']

print(f"\n  Quadrant sizes:")
for q in quad_order:
    n = (df['quadrant'] == q).sum()
    print(f"    {quad_short[q]:25s}: n={n}")

# Behavioral & clinical measures by quadrant
behav_vars = ['p_heavy', 'survival', 'total_earnings']
clin_vars  = ['OASIS_Total', 'STAI_State', 'PHQ9_Total', 'AMI_Emotional']

all_vars = behav_vars + clin_vars
var_labels = {
    'p_heavy': 'P(heavy)', 'survival': 'Survival', 'total_earnings': 'Earnings',
    'OASIS_Total': 'OASIS', 'STAI_State': 'STAI-State',
    'PHQ9_Total': 'PHQ-9', 'AMI_Emotional': 'AMI-Emotional'
}

quad_stats_rows = []
print(f"\n  {'Variable':20s} {'F':>8s} {'p':>10s}  Quad means")
print("  " + "-"*80)

for var in all_vars:
    groups = [df.loc[df['quadrant'] == q, var].dropna() for q in quad_order]
    F, p = f_oneway(*groups)
    means = [g.mean() for g in groups]
    sds   = [g.std() for g in groups]
    sig = '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else ''
    print(f"  {var_labels[var]:20s} F={F:6.2f} p={p:.4f}{sig:4s}  "
          f"AV={means[0]:.2f} AH={means[1]:.2f} BV={means[2]:.2f} DI={means[3]:.2f}")

    # Tukey post-hoc
    all_vals = pd.concat(groups, ignore_index=True)
    all_labels = []
    for i, q in enumerate(quad_order):
        all_labels.extend([quad_short[q]] * len(groups[i]))
    tukey = pairwise_tukeyhsd(all_vals, all_labels, alpha=0.05)

    for i, q in enumerate(quad_order):
        quad_stats_rows.append({
            'variable': var_labels[var],
            'quadrant': quad_short[q],
            'mean': means[i],
            'sd': sds[i],
            'n': len(groups[i]),
            'F_stat': F,
            'F_pval': p
        })

# ══════════════════════════════════════════════════════════════════════════════
# 3. MULTIPLE REGRESSION: params → task outcomes
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*70)
print("3. MULTIPLE REGRESSION: log(params) → task outcomes")
print("="*70)

X = df[['log_ce', 'log_cd', 'log_eps']]
X = sm.add_constant(X)

reg_rows = []
for outcome in ['total_earnings', 'survival', 'p_heavy']:
    y = df[outcome]
    model = sm.OLS(y, X).fit()
    print(f"\n  {outcome_labels.get(outcome, outcome)}")
    print(f"    R² = {model.rsquared:.3f}, Adj R² = {model.rsquared_adj:.3f}, F = {model.fvalue:.2f}, p = {model.f_pvalue:.2e}")
    for pred in ['log_ce', 'log_cd', 'log_eps']:
        b = model.params[pred]
        se = model.bse[pred]
        t = model.tvalues[pred]
        p = model.pvalues[pred]
        sig = '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else ''
        print(f"    {param_labels[pred]:15s}: β={b:+.4f} (SE={se:.4f}), t={t:.2f}, p={p:.4f} {sig}")
        reg_rows.append({
            'outcome': outcome_labels.get(outcome, outcome),
            'predictor': param_labels[pred],
            'beta': b,
            'se': se,
            't': t,
            'p': p,
            'R2': model.rsquared
        })

reg_df = pd.DataFrame(reg_rows)

# ══════════════════════════════════════════════════════════════════════════════
# 4. SAVE STATS
# ══════════════════════════════════════════════════════════════════════════════
quad_stats_df = pd.DataFrame(quad_stats_rows)

# Combine all stats into one file with section labels
with open(STATS_OUT, 'w') as f:
    f.write("# EVC Quadrant Analysis — Behavioral Profiles\n")
    f.write(f"# N = {len(df)} subjects\n")
    f.write(f"# Median log(c_death) = {med_cd:.4f}, Median log(epsilon) = {med_eps:.4f}\n\n")

    f.write("## Section 1: Parameter-Behavior Correlations\n")
    corr_df.to_csv(f, index=False)

    f.write("\n## Section 2: Quadrant Summary Statistics\n")
    quad_stats_df.to_csv(f, index=False)

    f.write("\n## Section 3: Regression Results\n")
    reg_df.to_csv(f, index=False)

print(f"\n  Stats saved to {STATS_OUT}")

# ══════════════════════════════════════════════════════════════════════════════
# 5. FIGURE
# ══════════════════════════════════════════════════════════════════════════════
print("\n  Generating figure...")

# Style
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.size': 9,
    'axes.linewidth': 0.8,
    'axes.spines.top': False,
    'axes.spines.right': False,
    'xtick.major.width': 0.8,
    'ytick.major.width': 0.8,
})

quad_colors = {
    'Adaptive\nVigilance':    '#2166ac',  # blue
    'Anxious\nHelplessness':  '#b2182b',  # red
    'Bold\nVigorous':         '#1b7837',  # green
    'Disengaged':             '#878787',  # gray
}

fig = plt.figure(figsize=(12, 10))
gs = GridSpec(2, 2, hspace=0.35, wspace=0.30, left=0.08, right=0.95, top=0.94, bottom=0.06)

# ── Panel A: 2D scatter of log(cd) vs log(ε), colored by survival ─────────
ax_a = fig.add_subplot(gs[0, 0])

sc = ax_a.scatter(df['log_cd'], df['log_eps'], c=df['survival'],
                  cmap='RdYlGn', s=18, alpha=0.7, edgecolors='none', vmin=0.5, vmax=1.0)
cb = plt.colorbar(sc, ax=ax_a, shrink=0.8, pad=0.02)
cb.set_label('Survival rate', fontsize=8)
cb.ax.tick_params(labelsize=7)

# Quadrant lines
ax_a.axhline(med_eps, color='k', ls='--', lw=0.8, alpha=0.5)
ax_a.axvline(med_cd,  color='k', ls='--', lw=0.8, alpha=0.5)

# Quadrant labels
xlim = ax_a.get_xlim()
ylim = ax_a.get_ylim()
label_kw = dict(fontsize=7.5, fontweight='bold', alpha=0.7, ha='center', va='center')
ax_a.text(np.mean([med_cd, xlim[1]]), np.mean([med_eps, ylim[1]]),
          'Adaptive\nVigilance', color=quad_colors['Adaptive\nVigilance'], **label_kw)
ax_a.text(np.mean([med_cd, xlim[1]]), np.mean([med_eps, ylim[0]]),
          'Anxious\nHelplessness', color=quad_colors['Anxious\nHelplessness'], **label_kw)
ax_a.text(np.mean([med_cd, xlim[0]]), np.mean([med_eps, ylim[1]]),
          'Bold\nVigorous', color=quad_colors['Bold\nVigorous'], **label_kw)
ax_a.text(np.mean([med_cd, xlim[0]]), np.mean([med_eps, ylim[0]]),
          'Disengaged', color=quad_colors['Disengaged'], **label_kw)

ax_a.set_xlabel('log(c_death)')
ax_a.set_ylabel('log(ε)')
ax_a.set_title('A   Parameter Space × Survival', fontsize=10, fontweight='bold', loc='left')

# ── Panel B: Grouped bar chart — 4 quadrants × 3 behavioral outcomes ──────
ax_b = fig.add_subplot(gs[0, 1])

behav_display = ['p_heavy', 'survival', 'total_earnings']
behav_bar_labels = ['P(heavy)', 'Survival', 'Earnings (z)']

# Z-score earnings for comparable scale
df['earnings_z'] = (df['total_earnings'] - df['total_earnings'].mean()) / df['total_earnings'].std()

bar_data = []
for q in quad_order:
    qd = df[df['quadrant'] == q]
    bar_data.append([qd['p_heavy'].mean(), qd['survival'].mean(), qd['earnings_z'].mean()])

bar_data = np.array(bar_data)  # (4 quadrants, 3 outcomes)
bar_err = []
for q in quad_order:
    qd = df[df['quadrant'] == q]
    bar_err.append([qd['p_heavy'].sem(), qd['survival'].sem(), qd['earnings_z'].sem()])
bar_err = np.array(bar_err)

x = np.arange(3)
width = 0.18
for i, q in enumerate(quad_order):
    ax_b.bar(x + (i - 1.5) * width, bar_data[i], width,
             yerr=bar_err[i], capsize=2, color=quad_colors[q],
             label=quad_short[q], alpha=0.85, edgecolor='white', linewidth=0.5)

ax_b.set_xticks(x)
ax_b.set_xticklabels(behav_bar_labels)
ax_b.legend(fontsize=6.5, ncol=2, loc='upper left', framealpha=0.8)
ax_b.set_ylabel('Value (prob or z-score)')
ax_b.set_title('B   Behavioral Outcomes by Quadrant', fontsize=10, fontweight='bold', loc='left')
ax_b.axhline(0, color='k', lw=0.5, alpha=0.3)

# ── Panel C: Clinical measures by quadrant ─────────────────────────────────
ax_c = fig.add_subplot(gs[1, 0])

clin_display = ['OASIS_Total', 'AMI_Emotional', 'PHQ9_Total']
clin_bar_labels = ['OASIS', 'AMI-Emo', 'PHQ-9']

# Z-score clinical measures for comparability
for cv in clin_display:
    df[f'{cv}_z'] = (df[cv] - df[cv].mean()) / df[cv].std()

clin_bar_data = []
clin_bar_err = []
for q in quad_order:
    qd = df[df['quadrant'] == q]
    clin_bar_data.append([qd[f'{cv}_z'].mean() for cv in clin_display])
    clin_bar_err.append([qd[f'{cv}_z'].sem() for cv in clin_display])

clin_bar_data = np.array(clin_bar_data)
clin_bar_err = np.array(clin_bar_err)

x = np.arange(len(clin_display))
for i, q in enumerate(quad_order):
    ax_c.bar(x + (i - 1.5) * width, clin_bar_data[i], width,
             yerr=clin_bar_err[i], capsize=2, color=quad_colors[q],
             label=quad_short[q], alpha=0.85, edgecolor='white', linewidth=0.5)

ax_c.set_xticks(x)
ax_c.set_xticklabels(clin_bar_labels)
ax_c.legend(fontsize=6.5, ncol=2, loc='upper left', framealpha=0.8)
ax_c.set_ylabel('Z-scored clinical measure')
ax_c.set_title('C   Clinical Profiles by Quadrant', fontsize=10, fontweight='bold', loc='left')
ax_c.axhline(0, color='k', lw=0.5, alpha=0.3)

# ── Panel D: ε vs OASIS, colored by c_death ───────────────────────────────
ax_d = fig.add_subplot(gs[1, 1])

hi_cd = df['log_cd'] >= med_cd
colors_d = np.where(hi_cd, '#b2182b', '#2166ac')
labels_d = np.where(hi_cd, 'High c_death', 'Low c_death')

for label, color, mask in [('High c_death', '#b2182b', hi_cd), ('Low c_death', '#2166ac', ~hi_cd)]:
    sub = df[mask]
    ax_d.scatter(sub['log_eps'], sub['OASIS_Total'], c=color, s=18, alpha=0.5,
                 edgecolors='none', label=label)
    # Regression line per group
    slope, intercept = np.polyfit(sub['log_eps'], sub['OASIS_Total'], 1)
    x_line = np.linspace(sub['log_eps'].min(), sub['log_eps'].max(), 50)
    ax_d.plot(x_line, slope * x_line + intercept, color=color, lw=1.5, alpha=0.8)

# Overall correlation
r, p = pearsonr(df['log_eps'], df['OASIS_Total'])
ax_d.text(0.02, 0.97, f'r = {r:+.3f}, p = {p:.3f}', transform=ax_d.transAxes,
          fontsize=8, va='top', ha='left',
          bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor='gray', alpha=0.8))

ax_d.set_xlabel('log(ε)')
ax_d.set_ylabel('OASIS (anxiety)')
ax_d.legend(fontsize=7, loc='upper right', framealpha=0.8)
ax_d.set_title('D   Effort Noise × Anxiety by Threat Sensitivity', fontsize=10, fontweight='bold', loc='left')

# ── Save ──────────────────────────────────────────────────────────────────────
fig.savefig(FIG_OUT, dpi=150, bbox_inches='tight', facecolor='white')
plt.close(fig)
print(f"  Figure saved to {FIG_OUT}")
print("\nDone.")
