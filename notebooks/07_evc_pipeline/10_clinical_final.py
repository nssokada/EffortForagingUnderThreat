#!/usr/bin/env python3
"""
Clinical correlations for the EVC 2+2 model.

Analyses:
  1. Log-space correlations: log(ce), log(cd) with all psychiatric subscales (FDR)
  2. ce x cd interaction for each clinical measure
  3. Factor score prediction (F1, F2, F3)

Output:
  results/stats/evc_final_clinical.csv
  results/figs/paper/fig_s_final_clinical.png (forest plot)
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

plt.rcParams.update({
    'font.sans-serif': ['DejaVu Sans', 'Arial', 'sans-serif'],
    'font.family': 'sans-serif',
    'figure.dpi': 150,
    'axes.spines.right': False,
    'axes.spines.top': False,
})

# ── Load data ──
params = pd.read_csv('/workspace/results/stats/oc_evc_final_params.csv')
psych = pd.read_csv('/workspace/data/exploratory_350/processed/stage5_filtered_data_20260320_191950/psych.csv')
factors = pd.read_csv('/workspace/results/stats/psych_factor_scores.csv')

df = params.merge(psych, on='subj').merge(factors, on='subj')
print(f"Merged N = {len(df)}")

# Log-transform parameters
df['log_ce'] = np.log(df['c_effort'])
df['log_cd'] = np.log(df['c_death'])

# Clinical subscales
clinical_cols = [c for c in psych.columns
                 if c not in ('participantID', 'subj') and '_RT' not in c]
clinical_cols = [c for c in clinical_cols if
                 any(c.startswith(p) for p in ['DASS21_', 'AMI_', 'MFIS_', 'OASIS_',
                                                'PHQ9_', 'STICSA_', 'STAI_'])]

print(f"Clinical measures: {len(clinical_cols)}")

# ══════════════════════════════════════════════════════════════════════════════
# 1. Individual log-space correlations (FDR-corrected)
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("1. LOG-SPACE CORRELATIONS (FDR-corrected)")
print("=" * 60)

results = []
for param in ['log_ce', 'log_cd']:
    for col in clinical_cols:
        r, p = pearsonr(df[param], df[col])
        results.append({
            'parameter': param.replace('log_', ''),
            'clinical_measure': col,
            'r': r, 'p_uncorrected': p,
            'n': len(df),
        })

results_df = pd.DataFrame(results)

_, p_fdr, _, _ = multipletests(results_df['p_uncorrected'], method='fdr_bh')
results_df['p_fdr'] = p_fdr
results_df['sig_fdr'] = results_df['p_fdr'] < 0.05

print("\nSignificant after FDR correction:")
sig = results_df[results_df['sig_fdr']].sort_values('p_fdr')
if len(sig) > 0:
    for _, row in sig.iterrows():
        print(f"  {row['parameter']:8s} -> {row['clinical_measure']:25s}: "
              f"r={row['r']:+.3f}, p_fdr={row['p_fdr']:.4f}")
else:
    print("  None survived FDR correction")

print("\nUncorrected p < 0.05:")
unc = results_df[results_df['p_uncorrected'] < 0.05].sort_values('p_uncorrected')
for _, row in unc.iterrows():
    fdr_note = " *FDR*" if row['sig_fdr'] else ""
    print(f"  {row['parameter']:8s} -> {row['clinical_measure']:25s}: "
          f"r={row['r']:+.3f}, p={row['p_uncorrected']:.4f}{fdr_note}")

# ══════════════════════════════════════════════════════════════════════════════
# 2. ce x cd interaction
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("2. c_effort x c_death INTERACTION")
print("=" * 60)

interaction_results = []
for col in clinical_cols:
    X = df[['log_ce', 'log_cd']].copy()
    X['interaction'] = X['log_ce'] * X['log_cd']
    X = sm.add_constant(X)
    y = df[col]

    model = sm.OLS(y, X).fit()
    interaction_results.append({
        'clinical_measure': col,
        'b_ce': model.params['log_ce'], 'p_ce': model.pvalues['log_ce'],
        'b_cd': model.params['log_cd'], 'p_cd': model.pvalues['log_cd'],
        'b_interaction': model.params['interaction'], 'p_interaction': model.pvalues['interaction'],
        'R2': model.rsquared, 'R2_adj': model.rsquared_adj,
        'f_stat': model.fvalue, 'f_p': model.f_pvalue,
    })

interaction_df = pd.DataFrame(interaction_results)

print("\nSignificant interactions (p < 0.05):")
sig_int = interaction_df[interaction_df['p_interaction'] < 0.05].sort_values('p_interaction')
if len(sig_int) > 0:
    for _, row in sig_int.iterrows():
        print(f"  {row['clinical_measure']:25s}: b_int={row['b_interaction']:+.3f}, "
              f"p={row['p_interaction']:.4f}, R2={row['R2']:.3f}")
else:
    print("  None")

print("\nOverall model fits:")
for _, row in interaction_df.sort_values('f_p').head(10).iterrows():
    print(f"  {row['clinical_measure']:25s}: R2={row['R2']:.3f}, F-p={row['f_p']:.4f}")

# ══════════════════════════════════════════════════════════════════════════════
# 3. Factor score prediction
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("3. FACTOR SCORE PREDICTION")
print("=" * 60)

factor_results = []
for f in ['F1', 'F2', 'F3']:
    r_ce, p_ce = pearsonr(df['log_ce'], df[f])
    r_cd, p_cd = pearsonr(df['log_cd'], df[f])
    print(f"\n{f}:")
    print(f"  log(ce)  -> {f}: r={r_ce:+.3f}, p={p_ce:.4f}")
    print(f"  log(cd)  -> {f}: r={r_cd:+.3f}, p={p_cd:.4f}")

    X = df[['log_ce', 'log_cd']].copy()
    X['interaction'] = X['log_ce'] * X['log_cd']
    X = sm.add_constant(X)
    model = sm.OLS(df[f], X).fit()

    print(f"  Joint model R2={model.rsquared:.3f}, R2_adj={model.rsquared_adj:.3f}")
    print(f"    b_ce={model.params['log_ce']:+.3f} (p={model.pvalues['log_ce']:.4f})")
    print(f"    b_cd={model.params['log_cd']:+.3f} (p={model.pvalues['log_cd']:.4f})")
    print(f"    b_int={model.params['interaction']:+.3f} (p={model.pvalues['interaction']:.4f})")

    factor_results.append({
        'factor': f,
        'r_ce': r_ce, 'p_ce': p_ce,
        'r_cd': r_cd, 'p_cd': p_cd,
        'R2_joint': model.rsquared, 'R2_adj': model.rsquared_adj,
        'b_ce': model.params['log_ce'], 'bp_ce': model.pvalues['log_ce'],
        'b_cd': model.params['log_cd'], 'bp_cd': model.pvalues['log_cd'],
        'b_int': model.params['interaction'], 'bp_int': model.pvalues['interaction'],
    })

factor_df = pd.DataFrame(factor_results)

# ══════════════════════════════════════════════════════════════════════════════
# Save all results
# ══════════════════════════════════════════════════════════════════════════════
results_df.to_csv('/workspace/results/stats/evc_final_clinical.csv', index=False)
interaction_df.to_csv('/workspace/results/stats/evc_final_clinical_interactions.csv', index=False)
factor_df.to_csv('/workspace/results/stats/evc_final_clinical_factors.csv', index=False)
print(f"\nSaved: results/stats/evc_final_clinical.csv")
print(f"Saved: results/stats/evc_final_clinical_interactions.csv")
print(f"Saved: results/stats/evc_final_clinical_factors.csv")

# ══════════════════════════════════════════════════════════════════════════════
# Forest plot figure
# ══════════════════════════════════════════════════════════════════════════════
print("\nCreating forest plot...")

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 8), sharey=True)

ce_results = results_df[results_df['parameter'] == 'ce'].copy()
cd_results = results_df[results_df['parameter'] == 'cd'].copy()

# Order by ce r
ce_results = ce_results.sort_values('r')
order = ce_results['clinical_measure'].values
cd_results = cd_results.set_index('clinical_measure').loc[order].reset_index()

y_pos = np.arange(len(order))

# Panel 1: c_effort
colors_ce = ['#2A9D8F' if row['sig_fdr'] else ('#8CC5B5' if row['p_uncorrected'] < 0.05 else '#D1D5DB')
             for _, row in ce_results.iterrows()]
ax1.barh(y_pos, ce_results['r'].values, color=colors_ce, edgecolor='none', height=0.7)
ax1.set_yticks(y_pos)
ax1.set_yticklabels(order, fontsize=9)
ax1.axvline(0, color='black', linewidth=0.5)
ax1.set_xlabel('Pearson r', fontsize=11)
ax1.set_title('c_effort (effort cost)', fontsize=12, fontweight='bold')
ax1.spines['right'].set_visible(False)
ax1.spines['top'].set_visible(False)

for i, (_, row) in enumerate(ce_results.iterrows()):
    if row['sig_fdr']:
        ax1.text(row['r'] + 0.01 * np.sign(row['r']), i, '**', va='center', fontsize=10, fontweight='bold')
    elif row['p_uncorrected'] < 0.05:
        ax1.text(row['r'] + 0.01 * np.sign(row['r']), i, '*', va='center', fontsize=10)

# Panel 2: c_death
colors_cd = ['#457B9D' if row['sig_fdr'] else ('#A0C4D8' if row['p_uncorrected'] < 0.05 else '#D1D5DB')
             for _, row in cd_results.iterrows()]
ax2.barh(y_pos, cd_results['r'].values, color=colors_cd, edgecolor='none', height=0.7)
ax2.axvline(0, color='black', linewidth=0.5)
ax2.set_xlabel('Pearson r', fontsize=11)
ax2.set_title('c_death (capture aversion)', fontsize=12, fontweight='bold')
ax2.spines['right'].set_visible(False)
ax2.spines['top'].set_visible(False)

for i, (_, row) in enumerate(cd_results.iterrows()):
    if row['sig_fdr']:
        ax2.text(row['r'] + 0.01 * np.sign(row['r']), i, '**', va='center', fontsize=10, fontweight='bold')
    elif row['p_uncorrected'] < 0.05:
        ax2.text(row['r'] + 0.01 * np.sign(row['r']), i, '*', va='center', fontsize=10)

from matplotlib.patches import Patch
legend_elements = [Patch(facecolor='#2A9D8F', label='FDR sig'),
                   Patch(facecolor='#8CC5B5', label='p<.05 uncorrected'),
                   Patch(facecolor='#D1D5DB', label='n.s.')]
ax1.legend(handles=legend_elements, loc='lower right', fontsize=8, frameon=False)

plt.tight_layout()
plt.savefig('/workspace/results/figs/paper/fig_s_final_clinical.png',
            dpi=150, bbox_inches='tight', facecolor='white')
print(f"Saved: /workspace/results/figs/paper/fig_s_final_clinical.png")

print("\nDone!")
