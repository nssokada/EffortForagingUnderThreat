"""
EVC+gamma model: Log-space clinical correlation analysis.

Analyses:
  5.1 Individual log-space correlations (FDR-corrected)
  5.2 c_death × epsilon interaction on clinical measures
  5.3 Factor score prediction
  5.4 Joint prediction (multiple regression)

Output:
  results/stats/evc_clinical_log.csv
  results/stats/evc_clinical_interactions.csv
  results/stats/evc_clinical_joint.csv
  results/figs/paper/fig_s_clinical.png
"""

import sys
sys.path.insert(0, '/workspace')

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, zscore
from statsmodels.stats.multitest import multipletests
import statsmodels.api as sm
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# ── Load data ──────────────────────────────────────────────────────────

params = pd.read_csv('/workspace/results/stats/oc_evc_gamma_params.csv')
psych = pd.read_csv('/workspace/data/exploratory_350/processed/stage5_filtered_data_20260320_191950/psych.csv')
factors = pd.read_csv('/workspace/results/stats/psych_factor_scores.csv')
loadings = pd.read_csv('/workspace/results/stats/psych_factor_loadings.csv')

# Merge on subj
df = params.merge(psych, on='subj').merge(factors, on='subj')
print(f"Merged N = {len(df)}")

# Log-transform parameters
df['log_ce'] = np.log(df['c_effort'])
df['log_cd'] = np.log(df['c_death'])
df['log_eps'] = np.log(df['epsilon'])

# Clinical subscales to analyze (exclude RT columns and IDs)
clinical_cols = [c for c in psych.columns
                 if c not in ('participantID', 'subj') and '_RT' not in c
                 and '_Total' not in c.replace('OASIS_Total','X').replace('PHQ9_Total','X').replace('STICSA_Total','X')]
# Simpler: just pick the meaningful subscales
clinical_cols = [
    'DASS21_Stress', 'DASS21_Anxiety', 'DASS21_Depression',
    'AMI_Behavioural', 'AMI_Social', 'AMI_Emotional',
    'MFIS_Physical', 'MFIS_Cognitive', 'MFIS_Psychosocial',
    'OASIS_Total', 'PHQ9_Total', 'STICSA_Total', 'STAI_Trait',
]

# Z-score clinical measures
for col in clinical_cols:
    df[col + '_z'] = zscore(df[col], nan_policy='omit')

clinical_z = [c + '_z' for c in clinical_cols]
log_params = ['log_ce', 'log_cd', 'log_eps']
param_labels = ['log(c_effort)', 'log(c_death)', 'log(ε)']

# ══════════════════════════════════════════════════════════════════════
# 5.1  Individual Log-Space Correlations
# ══════════════════════════════════════════════════════════════════════

print("\n" + "="*70)
print("5.1  INDIVIDUAL LOG-SPACE CORRELATIONS")
print("="*70)

rows = []
for lp, lp_label in zip(log_params, param_labels):
    for cz, c_raw in zip(clinical_z, clinical_cols):
        mask = df[[lp, cz]].notna().all(axis=1)
        x, y = df.loc[mask, lp].values, df.loc[mask, cz].values
        r, p = pearsonr(x, y)
        n = mask.sum()
        # 95% CI for r via Fisher z-transform
        z_r = np.arctanh(r)
        se = 1.0 / np.sqrt(n - 3)
        ci_lo = np.tanh(z_r - 1.96 * se)
        ci_hi = np.tanh(z_r + 1.96 * se)
        rows.append({
            'parameter': lp_label,
            'clinical_measure': c_raw,
            'r': r, 'p_uncorrected': p, 'n': n,
            'ci_lo': ci_lo, 'ci_hi': ci_hi,
        })

corr_df = pd.DataFrame(rows)

# FDR correction within each parameter
for lp_label in param_labels:
    mask = corr_df['parameter'] == lp_label
    pvals = corr_df.loc[mask, 'p_uncorrected'].values
    _, p_fdr, _, _ = multipletests(pvals, method='fdr_bh')
    corr_df.loc[mask, 'p_FDR'] = p_fdr

corr_df.to_csv('/workspace/results/stats/evc_clinical_log.csv', index=False)

# Print results
for lp_label in param_labels:
    sub = corr_df[corr_df['parameter'] == lp_label].sort_values('p_uncorrected')
    print(f"\n── {lp_label} ──")
    for _, row in sub.iterrows():
        sig = '***' if row['p_FDR'] < .001 else '**' if row['p_FDR'] < .01 else '*' if row['p_FDR'] < .05 else '†' if row['p_uncorrected'] < .05 else ''
        print(f"  {row['clinical_measure']:25s}  r={row['r']:+.3f}  [{row['ci_lo']:+.3f}, {row['ci_hi']:+.3f}]  "
              f"p={row['p_uncorrected']:.4f}  p_FDR={row['p_FDR']:.4f}  {sig}")

n_sig_fdr = (corr_df['p_FDR'] < .05).sum()
n_sig_unc = (corr_df['p_uncorrected'] < .05).sum()
print(f"\nSummary: {n_sig_fdr} FDR-significant, {n_sig_unc} uncorrected-significant out of {len(corr_df)} tests")

# ══════════════════════════════════════════════════════════════════════
# 5.2  c_death × epsilon Interaction
# ══════════════════════════════════════════════════════════════════════

print("\n" + "="*70)
print("5.2  c_death × epsilon INTERACTION")
print("="*70)

interaction_rows = []
for cz, c_raw in zip(clinical_z, clinical_cols):
    mask = df[['log_cd', 'log_eps', cz]].notna().all(axis=1)
    y = df.loc[mask, cz].values
    X = df.loc[mask, ['log_cd', 'log_eps']].copy()
    X['cd_x_eps'] = X['log_cd'] * X['log_eps']
    X = sm.add_constant(X)
    try:
        model = sm.OLS(y, X).fit()
        interaction_rows.append({
            'clinical_measure': c_raw,
            'beta_cd': model.params['log_cd'],
            'beta_eps': model.params['log_eps'],
            'beta_interaction': model.params['cd_x_eps'],
            't_interaction': model.tvalues['cd_x_eps'],
            'p_interaction': model.pvalues['cd_x_eps'],
            'R2': model.rsquared,
            'R2_adj': model.rsquared_adj,
        })
    except Exception as e:
        print(f"  {c_raw}: FAILED — {e}")

int_df = pd.DataFrame(interaction_rows)
int_df.to_csv('/workspace/results/stats/evc_clinical_interactions.csv', index=False)

print(f"\n{'Measure':25s} {'β_cd':>8s} {'β_eps':>8s} {'β_int':>8s} {'t_int':>8s} {'p_int':>8s} {'R²':>6s}")
print("-" * 85)
for _, row in int_df.sort_values('p_interaction').iterrows():
    sig = '*' if row['p_interaction'] < .05 else '†' if row['p_interaction'] < .10 else ''
    print(f"  {row['clinical_measure']:25s} {row['beta_cd']:+.4f} {row['beta_eps']:+.4f} "
          f"{row['beta_interaction']:+.4f} {row['t_interaction']:+.3f} {row['p_interaction']:.4f} {row['R2']:.3f} {sig}")

# ══════════════════════════════════════════════════════════════════════
# 5.3  Factor Score Prediction
# ══════════════════════════════════════════════════════════════════════

print("\n" + "="*70)
print("5.3  FACTOR SCORE PREDICTION")
print("="*70)

factor_labels = {
    'F1': 'General distress',
    'F2': 'Fatigue/somatic',
    'F3': 'Apathy/amotivation',
}

for f_col in ['F1', 'F2', 'F3']:
    print(f"\n── {f_col} ({factor_labels[f_col]}) ──")
    mask = df[['log_ce', 'log_cd', 'log_eps', f_col]].notna().all(axis=1)
    y = df.loc[mask, f_col].values

    # Model 1: main effects only
    X1 = sm.add_constant(df.loc[mask, ['log_ce', 'log_cd', 'log_eps']])
    m1 = sm.OLS(y, X1).fit()
    print(f"  Model 1 (main effects): R²={m1.rsquared:.4f}, F={m1.fvalue:.2f}, p={m1.f_pvalue:.4f}")
    for var in ['log_ce', 'log_cd', 'log_eps']:
        print(f"    {var:10s}  β={m1.params[var]:+.4f}  t={m1.tvalues[var]:+.3f}  p={m1.pvalues[var]:.4f}")

    # Model 2: with cd×eps interaction
    X2 = df.loc[mask, ['log_ce', 'log_cd', 'log_eps']].copy()
    X2['cd_x_eps'] = X2['log_cd'] * X2['log_eps']
    X2 = sm.add_constant(X2)
    m2 = sm.OLS(y, X2).fit()
    print(f"  Model 2 (+interaction): R²={m2.rsquared:.4f}, F={m2.fvalue:.2f}, p={m2.f_pvalue:.4f}")
    print(f"    cd×eps interaction: β={m2.params['cd_x_eps']:+.4f}  t={m2.tvalues['cd_x_eps']:+.3f}  p={m2.pvalues['cd_x_eps']:.4f}")

# ══════════════════════════════════════════════════════════════════════
# 5.4  Joint Prediction (Multiple Regression)
# ══════════════════════════════════════════════════════════════════════

print("\n" + "="*70)
print("5.4  JOINT PREDICTION")
print("="*70)

joint_rows = []
for cz, c_raw in zip(clinical_z, clinical_cols):
    mask = df[['log_ce', 'log_cd', 'log_eps', cz]].notna().all(axis=1)
    y = df.loc[mask, cz].values
    X = sm.add_constant(df.loc[mask, ['log_ce', 'log_cd', 'log_eps']])
    model = sm.OLS(y, X).fit()

    # Partial R² for each predictor
    partial_r2 = {}
    for var in ['log_ce', 'log_cd', 'log_eps']:
        t = model.tvalues[var]
        n_obs = len(y)
        k = 3  # number of predictors
        partial_r2[var] = t**2 / (t**2 + n_obs - k - 1)

    joint_rows.append({
        'clinical_measure': c_raw,
        'R2': model.rsquared,
        'R2_adj': model.rsquared_adj,
        'F_stat': model.fvalue,
        'F_pvalue': model.f_pvalue,
        'beta_ce': model.params['log_ce'],
        't_ce': model.tvalues['log_ce'],
        'p_ce': model.pvalues['log_ce'],
        'partial_r2_ce': partial_r2['log_ce'],
        'beta_cd': model.params['log_cd'],
        't_cd': model.tvalues['log_cd'],
        'p_cd': model.pvalues['log_cd'],
        'partial_r2_cd': partial_r2['log_cd'],
        'beta_eps': model.params['log_eps'],
        't_eps': model.tvalues['log_eps'],
        'p_eps': model.pvalues['log_eps'],
        'partial_r2_eps': partial_r2['log_eps'],
    })

joint_df = pd.DataFrame(joint_rows)
joint_df.to_csv('/workspace/results/stats/evc_clinical_joint.csv', index=False)

print(f"\n{'Measure':25s} {'R²':>6s} {'R²adj':>6s} {'F':>7s} {'p(F)':>8s}  "
      f"{'β_ce':>7s} {'p_ce':>7s}  {'β_cd':>7s} {'p_cd':>7s}  {'β_eps':>7s} {'p_eps':>7s}")
print("-" * 120)
for _, row in joint_df.sort_values('F_pvalue').iterrows():
    sig = '*' if row['F_pvalue'] < .05 else ''
    print(f"  {row['clinical_measure']:25s} {row['R2']:.4f} {row['R2_adj']:.4f} {row['F_stat']:7.2f} {row['F_pvalue']:.4f}  "
          f"{row['beta_ce']:+.4f} {row['p_ce']:.4f}  {row['beta_cd']:+.4f} {row['p_cd']:.4f}  "
          f"{row['beta_eps']:+.4f} {row['p_eps']:.4f} {sig}")

# ══════════════════════════════════════════════════════════════════════
# FOREST PLOT
# ══════════════════════════════════════════════════════════════════════

print("\n" + "="*70)
print("GENERATING FOREST PLOT")
print("="*70)

fig, axes = plt.subplots(1, 3, figsize=(14, 5), sharey=True)

# Clean measure names for display
name_map = {
    'DASS21_Stress': 'DASS Stress',
    'DASS21_Anxiety': 'DASS Anxiety',
    'DASS21_Depression': 'DASS Depression',
    'AMI_Behavioural': 'AMI Behavioural',
    'AMI_Social': 'AMI Social',
    'AMI_Emotional': 'AMI Emotional',
    'MFIS_Physical': 'MFIS Physical',
    'MFIS_Cognitive': 'MFIS Cognitive',
    'MFIS_Psychosocial': 'MFIS Psychosocial',
    'OASIS_Total': 'OASIS',
    'PHQ9_Total': 'PHQ-9',
    'STICSA_Total': 'STICSA',
    'STAI_Trait': 'STAI Trait',
}

for ax, lp_label in zip(axes, param_labels):
    sub = corr_df[corr_df['parameter'] == lp_label].copy()
    sub = sub.sort_values('r')
    sub['display'] = sub['clinical_measure'].map(name_map)

    y_pos = np.arange(len(sub))
    colors = []
    for _, row in sub.iterrows():
        if row['p_FDR'] < .05:
            colors.append('#1b4f72')   # dark blue — FDR significant
        elif row['p_uncorrected'] < .05:
            colors.append('#5dade2')   # medium blue — uncorrected
        else:
            colors.append('#d5dbdb')   # light grey — ns

    # Error bars
    xerr_lo = sub['r'].values - sub['ci_lo'].values
    xerr_hi = sub['ci_hi'].values - sub['r'].values

    ax.barh(y_pos, sub['r'].values, xerr=[xerr_lo, xerr_hi],
            color=colors, edgecolor='none', height=0.6, capsize=3,
            error_kw={'elinewidth': 1, 'color': '#555555'})
    ax.axvline(0, color='black', linewidth=0.8, linestyle='-')
    ax.set_yticks(y_pos)
    ax.set_yticklabels(sub['display'].values, fontsize=9)
    ax.set_xlabel('Pearson r', fontsize=10)
    ax.set_title(lp_label, fontsize=11, fontweight='bold')
    ax.set_xlim(-0.25, 0.25)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

# Legend
from matplotlib.patches import Patch
legend_elements = [
    Patch(facecolor='#1b4f72', label='FDR p < .05'),
    Patch(facecolor='#5dade2', label='Uncorrected p < .05'),
    Patch(facecolor='#d5dbdb', label='n.s.'),
]
axes[2].legend(handles=legend_elements, loc='lower right', fontsize=8, frameon=False)

plt.tight_layout()
plt.savefig('/workspace/results/figs/paper/fig_s_clinical.png', dpi=300, bbox_inches='tight')
print("Saved: results/figs/paper/fig_s_clinical.png")

# ══════════════════════════════════════════════════════════════════════
# SIMPLE SLOPES PLOTS FOR SIGNIFICANT INTERACTIONS
# ══════════════════════════════════════════════════════════════════════

sig_interactions = int_df[int_df['p_interaction'] < .05]
if len(sig_interactions) > 0:
    print(f"\nGenerating simple slopes plots for {len(sig_interactions)} significant interactions...")
    n_plots = len(sig_interactions)
    fig_ss, axes_ss = plt.subplots(1, max(n_plots, 1), figsize=(5 * max(n_plots, 1), 4))
    if n_plots == 1:
        axes_ss = [axes_ss]

    for ax, (_, row) in zip(axes_ss, sig_interactions.iterrows()):
        c_raw = row['clinical_measure']
        cz = c_raw + '_z'
        mask = df[['log_cd', 'log_eps', cz]].notna().all(axis=1)
        subdf = df.loc[mask].copy()

        # Median split on log_cd
        med = subdf['log_cd'].median()
        lo = subdf[subdf['log_cd'] <= med]
        hi = subdf[subdf['log_cd'] > med]

        for group, label, color in [(lo, 'Low c_death', '#3498db'), (hi, 'High c_death', '#e74c3c')]:
            # Simple slope via OLS
            X = sm.add_constant(group['log_eps'])
            m = sm.OLS(group[cz].values, X).fit()
            eps_range = np.linspace(group['log_eps'].min(), group['log_eps'].max(), 100)
            y_pred = m.params['const'] + m.params['log_eps'] * eps_range
            ax.scatter(group['log_eps'], group[cz], alpha=0.2, s=10, color=color)
            ax.plot(eps_range, y_pred, color=color, linewidth=2,
                    label=f'{label} (β={m.params["log_eps"]:.3f}, p={m.pvalues["log_eps"]:.3f})')

        ax.set_xlabel('log(ε)', fontsize=10)
        ax.set_ylabel(f'{name_map.get(c_raw, c_raw)} (z)', fontsize=10)
        ax.set_title(f'{name_map.get(c_raw, c_raw)}: c_death × ε interaction', fontsize=10)
        ax.legend(fontsize=8, frameon=False)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

    plt.tight_layout()
    plt.savefig('/workspace/results/figs/paper/fig_s_clinical_interactions.png', dpi=300, bbox_inches='tight')
    print("Saved: results/figs/paper/fig_s_clinical_interactions.png")
else:
    print("\nNo significant cd × eps interactions — skipping simple slopes plots.")

# ══════════════════════════════════════════════════════════════════════
# SUMMARY
# ══════════════════════════════════════════════════════════════════════

print("\n" + "="*70)
print("SUMMARY")
print("="*70)

print(f"\nN = {len(df)} subjects")
print(f"Parameters: c_effort, c_death, epsilon (all log-transformed)")
print(f"Clinical measures: {len(clinical_cols)} subscales")
print(f"\nIndividual correlations: {n_sig_fdr} FDR-significant, {n_sig_unc} uncorrected-significant / {len(corr_df)} tests")
print(f"Interactions (cd×eps, p<.05): {(int_df['p_interaction'] < .05).sum()} / {len(int_df)}")

# Best joint models
best_joint = joint_df.sort_values('F_pvalue').head(3)
print(f"\nTop 3 joint models by F-test:")
for _, row in best_joint.iterrows():
    print(f"  {row['clinical_measure']:25s}  R²={row['R2']:.4f}  F={row['F_stat']:.2f}  p={row['F_pvalue']:.4f}")

print("\nNote: Effect sizes of r=0.10-0.15 are typical for computational")
print("psychiatry studies in healthy community samples (Prolific).")
print("\nDone.")
