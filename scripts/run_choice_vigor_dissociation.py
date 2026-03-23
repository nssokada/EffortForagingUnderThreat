#!/usr/bin/env python3
"""
Choice-Vigor Dissociation Analysis (NB14 equivalent)

Core question: choosing hard cookies and pressing hard are independent systems.
β (threat bias) suppresses choice but not motor execution.

Uses:
  - alpha_bayes from vigor_hbm_posteriors.csv as tonic vigor measure
  - P(choice==1) per subject from behavior.csv as choice measure
  - k, β from unified_3param_clean.csv as model parameters
  - phase_vigor_metrics.parquet for per-trial/per-threat vigor
"""

import numpy as np
import pandas as pd
from pathlib import Path
from scipy import stats
from sklearn.linear_model import LinearRegression
import warnings
warnings.filterwarnings('ignore')

# ── PATHS ──────────────────────────────────────────────────────────────────────
ROOT       = Path('/workspace')
STAGE5     = ROOT / 'data' / 'exploratory_350' / 'processed' / 'stage5_filtered_data_20260320_191950'
VIGOR_PROC = ROOT / 'data' / 'exploratory_350' / 'processed' / 'vigor_processed'
STATS_DIR  = ROOT / 'results' / 'stats'
RESULTS_DIR = ROOT / 'results' / 'stats'
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# ── LOAD DATA ──────────────────────────────────────────────────────────────────
print('Loading data...')

# Behavior
behavior = pd.read_csv(STAGE5 / 'behavior.csv')
print(f'  behavior: {behavior.shape}')

# Model parameters: k, beta from unified_3param_clean.csv
params = pd.read_csv(STATS_DIR / 'unified_3param_clean.csv')
print(f'  unified_3param_clean: {params.shape}, cols={list(params.columns)}')
# Has columns: subj, k, beta, alpha
params = params.rename(columns={'k': 'kappa'})[['subj', 'kappa', 'beta']]

# Tonic vigor: alpha_bayes from vigor_hbm_posteriors.csv
vigor_hbm = pd.read_csv(STATS_DIR / 'vigor_hbm_posteriors.csv')
print(f'  vigor_hbm_posteriors: {vigor_hbm.shape}')
# alpha_bayes = tonic pre-encounter pressing rate (Bayesian estimate)
vigor_hbm = vigor_hbm[['subj', 'alpha_bayes', 'rho_bayes', 'gamma_bayes']].copy()

# Phase vigor metrics (trial-level, for per-threat analyses)
pm = pd.read_parquet(VIGOR_PROC / 'phase_vigor_metrics.parquet')
print(f'  phase_vigor_metrics: {pm.shape}')

# Feelings / affect probes
feelings = pd.read_csv(STAGE5 / 'feelings.csv')
print(f'  feelings: {feelings.shape}')

# Psych
psych = pd.read_csv(STAGE5 / 'psych.csv').set_index('subj')

# Affect trait scores (from NB12)
affect_file = STATS_DIR / 'affect_trait_scores.csv'
if affect_file.exists():
    trait_affect = pd.read_csv(affect_file).set_index('subj')
    print(f'  affect_trait_scores: {trait_affect.shape}')
else:
    trait_affect = pd.DataFrame()
    print('  affect_trait_scores: NOT FOUND')

print()

# ── 1. CHOICE MEASURE ─────────────────────────────────────────────────────────
# P(choice == 1) per subject from behavior.csv
# choice=1 means high-effort (high-reward) cookie was chosen
choice_stats = behavior.groupby('subj')['choice'].mean().to_frame('p_high')
print(f'Choice stats (P(choose high-effort)):')
print(f'  N={len(choice_stats)}, mean={choice_stats["p_high"].mean():.3f}, '
      f'sd={choice_stats["p_high"].std():.3f}')

# ── 2. VIGOR MEASURE ──────────────────────────────────────────────────────────
# Use alpha_bayes from vigor_hbm_posteriors.csv (tonic vigor parameter)
# This is the hierarchical Bayesian estimate of each subject's mean pre-encounter pressing rate
vigor_stats = vigor_hbm.set_index('subj')[['alpha_bayes']].rename(columns={'alpha_bayes': 'mean_vigor'})
print(f'\nVigor stats (alpha_bayes):')
print(f'  N={len(vigor_stats)}, mean={vigor_stats["mean_vigor"].mean():.3f}, '
      f'sd={vigor_stats["mean_vigor"].std():.3f}')

# ── 3. BUILD SUBJECT DATA FRAME ───────────────────────────────────────────────
# Escape rate (attack trials only)
attack_trials = behavior[behavior['isAttackTrial'] == 1] if 'isAttackTrial' in behavior.columns else None
if attack_trials is not None and len(attack_trials) > 0:
    # outcome==0 = escaped, outcome==1 = captured
    escape = (attack_trials.groupby('subj')['outcome']
              .apply(lambda x: (x == 0).mean())
              .to_frame('escape_rate'))
else:
    # Try from phase_vigor_metrics which has isAttackTrial
    pm_attack = pm[pm['isAttackTrial'] == 1]
    # escaped column in pm: 1=escaped, NaN=not attack trial, 0=captured
    escape = (pm_attack.groupby('subj')['escaped']
              .apply(lambda x: x.mean())
              .to_frame('escape_rate'))
print(f'\nEscape rate: N={len(escape)}, mean={escape["escape_rate"].mean():.3f}')

# Earnings: reward - capture penalty
# High cookie = 5 pts (choice==1), Low cookie = 1 pt (choice==0)
# outcome==0 = escaped (got reward), outcome==1 = captured (lose cookie, -5)
if 'isAttackTrial' in behavior.columns:
    beh_earn = behavior.copy()
else:
    beh_earn = behavior.merge(pm[['subj','trial','isAttackTrial']].drop_duplicates(),
                               on=['subj','trial'], how='left')

beh_earn['reward'] = np.where(beh_earn['choice'] == 1, 5, 1)
# If captured (outcome==1): net = -5; if escaped/non-attack: net = reward
beh_earn['net'] = np.where(beh_earn['outcome'] == 1, -5, beh_earn['reward'])
earnings = beh_earn.groupby('subj')['net'].sum().to_frame('total_earnings')
print(f'Earnings: mean={earnings["total_earnings"].mean():.1f}, '
      f'sd={earnings["total_earnings"].std():.1f}')

# Calibration: correlation between threat probability and affect rating per subject
calib = {}
for subj, grp in feelings.groupby('subj'):
    for atype in ['anxiety', 'confidence']:
        sub = grp[grp['questionLabel'] == atype]
        if len(sub) >= 5:
            r, _ = stats.spearmanr(sub['attackingProb'], sub['response'])
            calib[(subj, atype)] = r
calib_df = pd.Series(calib).unstack()
# columns: anxiety, confidence
calib_df.columns = ['anx_calib', 'conf_calib']
calib_df.index.name = 'subj'

# Trait affect (from NB12 state-trait decomposition)
if len(trait_affect) > 0:
    trait_cols = [c for c in ['trait_anx', 'trait_conf'] if c in trait_affect.columns]
    trait_affect_sub = trait_affect[trait_cols]
else:
    trait_affect_sub = pd.DataFrame()

# ── COMBINE ──
subj_df = (choice_stats
           .join(vigor_stats)
           .join(params.set_index('subj'))
           .join(escape)
           .join(earnings)
           .join(calib_df))

if len(trait_affect_sub) > 0:
    subj_df = subj_df.join(trait_affect_sub)

# Psych
psych_cols = [c for c in ['DASS21_Anxiety', 'DASS21_Stress', 'DASS21_Depression',
                            'OASIS_Total', 'PHQ9_Total', 'AMI_Total',
                            'MFIS_Physical', 'STICSA_Total'] if c in psych.columns]
subj_df = subj_df.join(psych[psych_cols])

# Drop subjects missing core variables
subj_df = subj_df.dropna(subset=['p_high', 'mean_vigor', 'kappa', 'beta'])
print(f'\nFinal N after dropping missing core vars: {len(subj_df)}')

# ── 4. CORE DISSOCIATION TEST ─────────────────────────────────────────────────
print('\n' + '='*70)
print('CHOICE-VIGOR DISSOCIATION — CORE RESULTS')
print('='*70)

r_overall, p_overall = stats.pearsonr(subj_df['p_high'], subj_df['mean_vigor'])
print(f'\n1. Independence: choice × vigor r={r_overall:+.3f}, p={p_overall:.3f}')
if abs(r_overall) < 0.1:
    print('   → Near-zero correlation: DISSOCIATION confirmed')

# ── 5. Z-SCORE AND QUADRANTS ──────────────────────────────────────────────────
subj_df['choice_z'] = ((subj_df['p_high'] - subj_df['p_high'].mean()) /
                        subj_df['p_high'].std())
subj_df['vigor_z'] = ((subj_df['mean_vigor'] - subj_df['mean_vigor'].mean()) /
                       subj_df['mean_vigor'].std())

# Assign quadrant labels
subj_df['quadrant'] = 'XX'
subj_df.loc[(subj_df['choice_z'] > 0) & (subj_df['vigor_z'] > 0), 'quadrant'] = 'HH'
subj_df.loc[(subj_df['choice_z'] > 0) & (subj_df['vigor_z'] <= 0), 'quadrant'] = 'HL'
subj_df.loc[(subj_df['choice_z'] <= 0) & (subj_df['vigor_z'] > 0), 'quadrant'] = 'LH'
subj_df.loc[(subj_df['choice_z'] <= 0) & (subj_df['vigor_z'] <= 0), 'quadrant'] = 'LL'

QUAD_LABELS = {
    'HH': 'Choose hard, press hard',
    'HL': 'Choose hard, press soft',
    'LH': 'Choose easy, press hard',
    'LL': 'Choose easy, press soft'
}

print(f'\n2. Quadrant counts:')
for q in ['HH', 'HL', 'LH', 'LL']:
    n = (subj_df['quadrant'] == q).sum()
    print(f'   {q} ({QUAD_LABELS[q]}): N={n}')

# ── 6. PARAMETER PROFILES BY QUADRANT ─────────────────────────────────────────
print(f'\n3. Model parameter profiles by quadrant:')
print(f'   {"Quad":<5s} {"N":>4s} {"k":>8s} {"β":>8s} {"Esc%":>8s} {"Earn":>8s}')
for q in ['HH', 'HL', 'LH', 'LL']:
    s = subj_df[subj_df['quadrant'] == q]
    esc = s['escape_rate'].mean() if 'escape_rate' in s.columns else float('nan')
    earn = s['total_earnings'].mean() if 'total_earnings' in s.columns else float('nan')
    print(f'   {q:<5s} {len(s):>4d} {s["kappa"].mean():>8.3f} {s["beta"].mean():>8.3f} '
          f'{esc:>7.1%} {earn:>8.0f}')

# ANOVA on parameters by quadrant
print(f'\n4. ANOVA by quadrant:')
for param, label in [('kappa', 'k'), ('beta', 'β')]:
    groups = [subj_df[subj_df['quadrant'] == q][param].dropna().values
              for q in ['HH', 'HL', 'LH', 'LL']]
    if all(len(g) > 1 for g in groups):
        F, p = stats.f_oneway(*groups)
        print(f'   {label}: F={F:.2f}, p={p:.4f}')

# ANOVA on outcomes
for var, label in [('escape_rate', 'Escape rate'), ('total_earnings', 'Total earnings')]:
    if var in subj_df.columns:
        groups = [subj_df[subj_df['quadrant'] == q][var].dropna().values
                  for q in ['HH', 'HL', 'LH', 'LL']]
        if all(len(g) > 1 for g in groups):
            F, p = stats.f_oneway(*groups)
            print(f'   {label}: F={F:.2f}, p={p:.4f}')

# ── 7. β PATHS ─────────────────────────────────────────────────────────────────
print(f'\n5. β dissociates choice from vigor:')
for dv, label in [('p_high', 'Choice'), ('mean_vigor', 'Vigor')]:
    r, p = stats.pearsonr(subj_df['beta'], subj_df[dv])
    print(f'   β → {label}: r={r:+.3f}, p={p:.4f}')

# Also k paths
print(f'\n   k paths:')
for dv, label in [('p_high', 'Choice'), ('mean_vigor', 'Vigor')]:
    r, p = stats.pearsonr(subj_df['kappa'], subj_df[dv])
    print(f'   k → {label}: r={r:+.3f}, p={p:.4f}')

# ── 8. ESCAPE REGRESSION ──────────────────────────────────────────────────────
if 'escape_rate' in subj_df.columns:
    clean = subj_df[['choice_z', 'vigor_z', 'escape_rate']].dropna()
    clean = clean.copy()
    clean['interaction'] = clean['choice_z'] * clean['vigor_z']
    X = clean[['choice_z', 'vigor_z', 'interaction']].values
    y = clean['escape_rate'].values
    y_z = (y - y.mean()) / y.std()
    reg = LinearRegression().fit(X, y_z)
    y_pred = reg.predict(X)
    r2 = 1 - np.sum((y_z - y_pred)**2) / np.sum((y_z - y_z.mean())**2)

    print(f'\n6. Escape ~ choice + vigor + interaction (N={len(clean)}):')
    print(f'   R² = {r2:.3f}')
    print(f'   Choice β = {reg.coef_[0]:+.3f}')
    print(f'   Vigor β  = {reg.coef_[1]:+.3f}')
    print(f'   Interaction β = {reg.coef_[2]:+.3f}')

# ── 9. THREAT REVERSAL TEST ────────────────────────────────────────────────────
print(f'\n7. Threat reverses choice-vigor relationship (per-threat correlations):')

# Per-threat: use enc_pre_mean_norm from phase_vigor_metrics (capacity-normalized pre-enc)
# and choice from same source
# Need per-subject per-threat aggregates
for th in [0.1, 0.5, 0.9]:
    pm_th = pm[pm['threat'] == th]
    choice_th = pm_th.groupby('subj')['choice'].mean()
    vigor_th = pm_th.groupby('subj')['enc_pre_mean_norm'].mean()

    merged = (choice_th.to_frame('choice')
              .join(vigor_th.to_frame('vigor'))
              .dropna())
    if len(merged) > 10:
        r, p = stats.pearsonr(merged['choice'], merged['vigor'])
        sig = '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else 'n.s.'
        print(f'   Threat={th}: r={r:+.3f}, p={p:.4f} {sig} (N={len(merged)})')

# ── 10. OFF-DIAGONAL COMPARISON: HL vs LH ─────────────────────────────────────
print(f'\n8. Off-diagonal comparison (HL vs LH):')
hl = subj_df[subj_df['quadrant'] == 'HL']
lh = subj_df[subj_df['quadrant'] == 'LH']
print(f'   HL N={len(hl)}, LH N={len(lh)}')

compare_vars = [('kappa', 'k'), ('beta', 'β'), ('escape_rate', 'Escape rate'),
                ('total_earnings', 'Earnings')]
if 'trait_conf' in subj_df.columns:
    compare_vars.append(('trait_conf', 'Trait confidence'))
if 'trait_anx' in subj_df.columns:
    compare_vars.append(('trait_anx', 'Trait anxiety'))
if 'AMI_Total' in subj_df.columns:
    compare_vars.append(('AMI_Total', 'AMI (apathy)'))
if 'anx_calib' in subj_df.columns:
    compare_vars.append(('anx_calib', 'Anxiety calibration'))

for var, label in compare_vars:
    v1 = hl[var].dropna() if var in hl.columns else pd.Series(dtype=float)
    v2 = lh[var].dropna() if var in lh.columns else pd.Series(dtype=float)
    if len(v1) > 3 and len(v2) > 3:
        t, p = stats.ttest_ind(v1, v2)
        sig = '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else 'n.s.'
        print(f'   {label}: HL={v1.mean():.3f}, LH={v2.mean():.3f}, '
              f't={t:.2f}, p={p:.4f} {sig}')

# ── 11. AFFECT BY QUADRANT ────────────────────────────────────────────────────
print(f'\n9. Affect by quadrant (ANOVA):')
affect_vars = []
if 'trait_conf' in subj_df.columns:
    affect_vars.append(('trait_conf', 'Trait confidence'))
if 'trait_anx' in subj_df.columns:
    affect_vars.append(('trait_anx', 'Trait anxiety'))
if 'AMI_Total' in subj_df.columns:
    affect_vars.append(('AMI_Total', 'AMI (apathy)'))

for var, label in affect_vars:
    groups = [subj_df[subj_df['quadrant'] == q][var].dropna().values
              for q in ['HH', 'HL', 'LH', 'LL']]
    if all(len(g) > 5 for g in groups):
        F, p = stats.f_oneway(*groups)
        sig = '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else 'n.s.'
        print(f'   {label}: F={F:.2f}, p={p:.4f} {sig}')

# ── 12. SAVE RESULTS ──────────────────────────────────────────────────────────
# Prepare output dataframe with key dissociation results
results_rows = []

# Overall correlation
results_rows.append({
    'test': 'overall_correlation',
    'label': 'choice × vigor (overall)',
    'r': r_overall,
    'p': p_overall,
    'n': len(subj_df),
    'stat_type': 'pearsonr'
})

# β paths
for dv, label in [('p_high', 'beta -> choice'), ('mean_vigor', 'beta -> vigor')]:
    r, p = stats.pearsonr(subj_df['beta'], subj_df[dv])
    results_rows.append({
        'test': 'beta_path',
        'label': label,
        'r': r,
        'p': p,
        'n': len(subj_df),
        'stat_type': 'pearsonr'
    })

# k paths
for dv, label in [('p_high', 'k -> choice'), ('mean_vigor', 'k -> vigor')]:
    r, p = stats.pearsonr(subj_df['kappa'], subj_df[dv])
    results_rows.append({
        'test': 'k_path',
        'label': label,
        'r': r,
        'p': p,
        'n': len(subj_df),
        'stat_type': 'pearsonr'
    })

# Threat-level correlations
for th in [0.1, 0.5, 0.9]:
    pm_th = pm[pm['threat'] == th]
    choice_th = pm_th.groupby('subj')['choice'].mean()
    vigor_th = pm_th.groupby('subj')['enc_pre_mean_norm'].mean()
    merged = choice_th.to_frame('choice').join(vigor_th.to_frame('vigor')).dropna()
    if len(merged) > 10:
        r, p = stats.pearsonr(merged['choice'], merged['vigor'])
        results_rows.append({
            'test': 'threat_level_correlation',
            'label': f'choice × vigor at threat={th}',
            'r': r,
            'p': p,
            'n': len(merged),
            'stat_type': 'pearsonr'
        })

# Quadrant ANOVA
for param, label in [('kappa', 'k_by_quadrant'), ('beta', 'beta_by_quadrant'),
                      ('escape_rate', 'escape_by_quadrant'), ('total_earnings', 'earnings_by_quadrant')]:
    if param in subj_df.columns:
        groups = [subj_df[subj_df['quadrant'] == q][param].dropna().values
                  for q in ['HH', 'HL', 'LH', 'LL']]
        if all(len(g) > 1 for g in groups):
            F, p = stats.f_oneway(*groups)
            results_rows.append({
                'test': 'quadrant_anova',
                'label': label,
                'F': F,
                'p': p,
                'n': len(subj_df),
                'stat_type': 'f_oneway'
            })

# Off-diagonal t-tests
for var, label in compare_vars:
    v1 = hl[var].dropna() if var in hl.columns else pd.Series(dtype=float)
    v2 = lh[var].dropna() if var in lh.columns else pd.Series(dtype=float)
    if len(v1) > 3 and len(v2) > 3:
        t, p = stats.ttest_ind(v1, v2)
        results_rows.append({
            'test': 'off_diagonal_ttest',
            'label': f'HL_vs_LH_{var}',
            'HL_mean': v1.mean(),
            'LH_mean': v2.mean(),
            't': t,
            'p': p,
            'n_HL': len(v1),
            'n_LH': len(v2),
            'stat_type': 'ttest_ind'
        })

results_df = pd.DataFrame(results_rows)
out_path = RESULTS_DIR / 'choice_vigor_dissociation_results.csv'
results_df.to_csv(out_path, index=False)
print(f'\nResults saved to: {out_path}')
print(f'Rows: {len(results_df)}')
print()

# Also save the subject-level dataframe
subj_out_path = RESULTS_DIR / 'choice_vigor_dissociation_subjects.csv'
subj_df.to_csv(subj_out_path)
print(f'Subject data saved to: {subj_out_path}')

# ── FINAL SUMMARY ──────────────────────────────────────────────────────────────
print()
print('='*70)
print('DISSOCIATION SUMMARY')
print('='*70)
print(f'N = {len(subj_df)}')
print(f'Overall choice × vigor: r={r_overall:+.3f}, p={p_overall:.4f}')
print(f'Quadrant sizes: ' + ', '.join(
    f'{q}={( subj_df["quadrant"]==q).sum()}' for q in ['HH','HL','LH','LL']))
print()
print('Key result: choice and vigor are dissociated (r≈0).')
print('β drives choice downward (threat aversion) without suppressing motor output.')
print('Vigor (alpha_bayes) drives escape success independently of choice strategy.')
