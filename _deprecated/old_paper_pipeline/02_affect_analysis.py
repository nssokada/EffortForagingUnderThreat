"""
02_affect_analysis.py — S → Affect validation
===============================================
Tests whether the survival signal S predicts subjective anxiety and confidence
at probe trials (forced-choice trials with affect ratings).

Analyses:
  1. Compute S_probe per probe trial (using L3_add, λ fixed from model fit)
  2. LMMs: anxiety ~ S_z, confidence ~ S_z  (within-subject)
  3. k × S interaction on anxiety (does effort sensitivity modulate the affect signal?)
  4. Cross-domain threat sensitivity correlations (choice vs. affect)

Outputs:
    results/stats/paper/affect_results.csv

Usage:
    export PATH="$HOME/.local/bin:$PATH"
    python3 notebooks/06_paper_pipeline/02_affect_analysis.py
"""

import sys
import numpy as np
import pandas as pd
from pathlib import Path
from scipy import stats

# ── Paths ──────────────────────────────────────────────────────────────────────
ROOT     = Path('/workspace')
DATA_DIR = ROOT / 'data/exploratory_350/processed/stage5_filtered_data_20260320_191950'
STAT_DIR = ROOT / 'results/stats'
OUT_DIR  = ROOT / 'results/stats/paper'
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ── Load data ──────────────────────────────────────────────────────────────────
print('=' * 70)
print('STEP 1: Loading data')
print('=' * 70)

feelings = pd.read_csv(DATA_DIR / 'feelings.csv')
behavior = pd.read_csv(DATA_DIR / 'behavior.csv')

# Load choice parameters (from 01_choice_model.py or fallback to unified_3param_clean.csv)
params_path_new = OUT_DIR / 'choice_params.csv'
params_path_old = STAT_DIR / 'unified_3param_clean.csv'
if params_path_new.exists():
    params = pd.read_csv(params_path_new)
    print(f'Loaded choice params from paper pipeline: {params_path_new}')
else:
    params = pd.read_csv(params_path_old)
    print(f'Loaded choice params from unified_3param_clean.csv (fallback)')

print(f'Feelings: {len(feelings)} rows, {feelings["subj"].nunique()} subjects')
print(f'Feelings columns: {feelings.columns.tolist()}')
print(f'Question types: {feelings["questionLabel"].unique() if "questionLabel" in feelings.columns else feelings["category"].unique() if "category" in feelings.columns else "unknown"}')

# Normalize column names for questionLabel/category
if 'questionLabel' in feelings.columns:
    label_col = 'questionLabel'
elif 'category' in feelings.columns:
    label_col = 'category'
else:
    raise ValueError('Cannot find label column in feelings.csv')

print(f'Using label column: {label_col}')

# ── Compute S_probe ───────────────────────────────────────────────────────────
print('\n' + '=' * 70)
print('STEP 2: Compute S_probe per probe trial')
print('=' * 70)

# Use λ=2.0 (approximate L3_add population estimate; consistent with affect_survival.ipynb)
# This can be updated if exact SVI estimate is available from choice_params
LAM = 2.0
print(f'Using λ = {LAM} (L3_add population estimate)')

# feelings.csv has: threat, distance, trialNumber, response, subj, questionLabel/category
# S_probe = (1-T) + T/(1+λ·D) where D is the distance field
dist_col = None
for col in ['distance', 'distanceFromSafety']:
    if col in feelings.columns:
        dist_col = col
        break
if dist_col is None:
    raise ValueError('Cannot find distance column in feelings.csv')
print(f'Using distance column: {dist_col}')

# Map distance (metres) to discrete D ∈ {1,2,3} if needed
# distanceFromSafety is 4.0, 9.0 game units → map to D ∈ {1,2,3}
# From task_design: D=1→5 units, D=2→7 units, D=3→9 units
# distanceFromSafety appears to be raw game units
# Use as continuous variable for S, then also compute discrete version
feelings = feelings.copy()
feelings['S_probe'] = (1 - feelings['threat']) + feelings['threat'] / (1 + LAM * feelings[dist_col])

# Z-score S_probe within subject for LMM
subj_s_mean = feelings.groupby('subj')['S_probe'].transform('mean')
subj_s_std  = feelings.groupby('subj')['S_probe'].transform('std')
feelings['S_z'] = (feelings['S_probe'] - subj_s_mean) / subj_s_std.clip(lower=1e-6)

print(f'S_probe range: [{feelings["S_probe"].min():.3f}, {feelings["S_probe"].max():.3f}]')
print(f'S_probe mean: {feelings["S_probe"].mean():.3f}, SD: {feelings["S_probe"].std():.3f}')

# ── LMMs: anxiety ~ S_z, confidence ~ S_z ─────────────────────────────────────
print('\n' + '=' * 70)
print('STEP 3: LMMs — S_probe predicts anxiety and confidence')
print('=' * 70)

# Filter by question type
anxiety_mask    = feelings[label_col].str.lower().str.contains('anxiety')
confidence_mask = feelings[label_col].str.lower().str.contains('confidence')

df_anx  = feelings[anxiety_mask].copy()
df_conf = feelings[confidence_mask].copy()

print(f'Anxiety probes:    {len(df_anx)} rows, {df_anx["subj"].nunique()} subjects')
print(f'Confidence probes: {len(df_conf)} rows, {df_conf["subj"].nunique()} subjects')


def run_lmm_manual(df, dv='response', predictor='S_z', label=''):
    """
    Simplified within-subject LMM via per-subject OLS then random-effects meta-analysis.
    Returns population-level β, SE, t, p.
    """
    slopes = []
    intercepts = []
    subjs = df['subj'].unique()
    for s in subjs:
        sub = df[df['subj'] == s].dropna(subset=[predictor, dv])
        if len(sub) < 4:
            continue
        x = sub[predictor].values
        y = sub[dv].values
        if x.std() < 1e-6:
            continue
        slope, intercept, *_ = stats.linregress(x, y)
        slopes.append(slope)
        intercepts.append(intercept)

    slopes = np.array(slopes)
    pop_beta = slopes.mean()
    pop_se   = slopes.std() / np.sqrt(len(slopes))
    t_val    = pop_beta / pop_se if pop_se > 0 else np.nan
    df_      = len(slopes) - 1
    p_val    = 2 * stats.t.sf(abs(t_val), df=df_) if not np.isnan(t_val) else np.nan
    ci95_lo  = pop_beta - 1.96 * pop_se
    ci95_hi  = pop_beta + 1.96 * pop_se
    sig      = '*' if p_val < 0.05 else 'n.s.'
    print(f'  [{label}] β={pop_beta:+.4f}, SE={pop_se:.4f}, t({df_})={t_val:+.3f}, '
          f'p={p_val:.4f} {sig}, 95%CI=[{ci95_lo:.4f},{ci95_hi:.4f}]  n_subj={len(slopes)}')
    return {
        'outcome':   label,
        'predictor': predictor,
        'beta':      pop_beta,
        'se':        pop_se,
        'df':        df_,
        't':         t_val,
        'p':         p_val,
        'ci95_lo':   ci95_lo,
        'ci95_hi':   ci95_hi,
        'n_subj':    len(slopes),
    }


print('\nMain effects (S_z → affect rating):')
res_anx  = run_lmm_manual(df_anx,  dv='response', predictor='S_z', label='anxiety ~ S_z')
res_conf = run_lmm_manual(df_conf, dv='response', predictor='S_z', label='confidence ~ S_z')

# ── k × S interaction ─────────────────────────────────────────────────────────
print('\n' + '=' * 70)
print('STEP 4: k × S interaction on anxiety')
print('=' * 70)

# Merge per-subject k into anxiety data
df_anx2 = df_anx.merge(params[['subj', 'k']], on='subj', how='left')
df_anx2 = df_anx2.dropna(subset=['k'])

# Log-transform k (it's right-skewed) then z-score
df_anx2['logk']   = np.log(df_anx2['k'])
subj_k_mean       = df_anx2.groupby('subj')['logk'].transform('mean')
df_anx2['k_z']    = (df_anx2['logk'] - df_anx2['logk'].mean()) / df_anx2['logk'].std()

# Add interaction term
df_anx2['kxS']    = df_anx2['k_z'] * df_anx2['S_z']

print('k × S_z interaction on anxiety:')
res_kxS = run_lmm_manual(df_anx2, dv='response', predictor='kxS', label='anxiety ~ k×S_z')

# ── Cross-domain threat sensitivity ──────────────────────────────────────────
print('\n' + '=' * 70)
print('STEP 5: Cross-domain threat sensitivity correlations')
print('=' * 70)

# Per-subject: slope of anxiety ~ threat (model-free threat sensitivity in affect)
affect_slopes = []
for s in df_anx['subj'].unique():
    sub = df_anx[df_anx['subj'] == s].dropna(subset=['threat', 'response'])
    if len(sub) < 4:
        continue
    slope, *_ = stats.linregress(sub['threat'].values, sub['response'].values)
    affect_slopes.append({'subj': s, 'affect_threat_slope': slope})
affect_slopes_df = pd.DataFrame(affect_slopes)

# Merge with choice k (effort discounting) and beta (threat bias)
cross = affect_slopes_df.merge(params[['subj', 'k', 'beta']], on='subj', how='inner')
cross['logk']   = np.log(cross['k'])
cross['logbeta'] = np.log(cross['beta'].clip(lower=1e-6))

print(f'N subjects for cross-domain analysis: {len(cross)}')
for param_col, label in [('logk', 'log-k'), ('logbeta', 'log-β')]:
    x = cross[param_col].values
    y = cross['affect_threat_slope'].values
    r_val, p_val = stats.pearsonr(x, y)
    sig = '*' if p_val < 0.05 else 'n.s.'
    print(f'  affect_threat_slope ~ {label}: r={r_val:+.4f}, p={p_val:.4f} {sig}')

# ── Save outputs ───────────────────────────────────────────────────────────────
print('\n' + '=' * 70)
print('STEP 6: Saving outputs')
print('=' * 70)

results_rows = [res_anx, res_conf, res_kxS]
results_df   = pd.DataFrame(results_rows)

# Add cross-domain correlations as additional rows
for param_col, label in [('logk', 'affect_slope~logk'), ('logbeta', 'affect_slope~logbeta')]:
    x = cross[param_col].values
    y = cross['affect_threat_slope'].values
    r_val, p_val = stats.pearsonr(x, y)
    results_df = pd.concat([results_df, pd.DataFrame([{
        'outcome':   'affect_threat_slope',
        'predictor': label,
        'beta':      r_val,
        'se':        np.nan,
        'df':        len(cross) - 2,
        't':         np.nan,
        'p':         p_val,
        'ci95_lo':   np.nan,
        'ci95_hi':   np.nan,
        'n_subj':    len(cross),
    }])], ignore_index=True)

results_df.to_csv(OUT_DIR / 'affect_results.csv', index=False)
print(f'Saved: {OUT_DIR}/affect_results.csv')

print('\n' + '=' * 70)
print('SUMMARY')
print('=' * 70)
print(f'Anxiety  ~ S_z: β={res_anx["beta"]:+.4f}, p={res_anx["p"]:.4f}')
print(f'Confidence ~ S_z: β={res_conf["beta"]:+.4f}, p={res_conf["p"]:.4f}')
print(f'Anxiety ~ k×S_z: β={res_kxS["beta"]:+.4f}, p={res_kxS["p"]:.4f}')
print('\nDone.')
