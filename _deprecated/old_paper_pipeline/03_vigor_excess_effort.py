"""
03_vigor_excess_effort.py — Excess effort analysis (H3)
========================================================
Tests whether subjects exert excess motor effort (vigor above trial demand)
that is modulated by danger (S).

Key constructs:
  - demand:       effort_H (chosen option's effort level) as proxy for motor demand
  - mean_vigor:   mean vigor_norm per trial (from kernel-smoothed timeseries)
  - excess_effort: residual vigor after regressing out demand (per-subject)
  - danger:       1 - S_trial, where S_trial = (1-T) + T/(1+λ·D_H)
  - α_i:          per-subject baseline vigor intercept
  - δ_i:          per-subject excess-effort sensitivity to danger (slope)

Analyses:
  1. Compute trial-level mean vigor and demand per subject
  2. Compute S_trial and danger per trial
  3. Per-subject OLS: excess ~ danger → α_i, δ_i
  4. Split-half reliability of δ (odd/even trials)
  5. Within-choice tests: does danger predict excess within each choice level?
  6. T×D interaction on excess effort
  7. Save vigor_params.csv

Outputs:
    results/stats/paper/vigor_params.csv

Usage:
    export PATH="$HOME/.local/bin:$PATH"
    python3 notebooks/06_paper_pipeline/03_vigor_excess_effort.py
"""

import numpy as np
import pandas as pd
from pathlib import Path
from scipy import stats

# ── Paths ──────────────────────────────────────────────────────────────────────
ROOT      = Path('/workspace')
DATA_DIR  = ROOT / 'data/exploratory_350/processed/stage5_filtered_data_20260320_191950'
VIGOR_DIR = ROOT / 'data/exploratory_350/processed/vigor_processed'
OUT_DIR   = ROOT / 'results/stats/paper'
OUT_DIR.mkdir(parents=True, exist_ok=True)

R_H = 5.0
R_L = 1.0
LAM = 2.0  # λ from L3_add model

# ── Load data ──────────────────────────────────────────────────────────────────
print('=' * 70)
print('STEP 1: Loading data')
print('=' * 70)

behavior = pd.read_csv(DATA_DIR / 'behavior.csv')
print(f'Loaded behavior.csv: {len(behavior)} trials, {behavior["subj"].nunique()} subjects')

print('Loading smoothed_vigor_ts.parquet...')
vigor_ts = pd.read_parquet(VIGOR_DIR / 'smoothed_vigor_ts.parquet')
print(f'Loaded vigor timeseries: {len(vigor_ts)} rows')
print(f'Columns: {vigor_ts.columns.tolist()}')

# ── Compute trial-level mean vigor ─────────────────────────────────────────────
print('\n' + '=' * 70)
print('STEP 2: Compute trial-level mean vigor')
print('=' * 70)

# Average vigor_norm over all timepoints within each (subj, trial)
trial_vigor = (
    vigor_ts
    .groupby(['subj', 'trial'])['vigor_norm']
    .mean()
    .reset_index()
    .rename(columns={'vigor_norm': 'mean_vigor'})
)
print(f'Trial-vigor table: {len(trial_vigor)} rows')

# Also compute pre-encounter vigor (first half of trial up to encounterTime)
# Use trials where encounterTime is finite
vigor_ts_clean = vigor_ts[vigor_ts['encounterTime'].notna()].copy()
pre_enc = (
    vigor_ts_clean[vigor_ts_clean['t'] < vigor_ts_clean['encounterTime']]
    .groupby(['subj', 'trial'])['vigor_norm']
    .mean()
    .reset_index()
    .rename(columns={'vigor_norm': 'pre_enc_vigor'})
)
print(f'Pre-encounter vigor: {len(pre_enc)} rows')

# ── Merge with behavior ────────────────────────────────────────────────────────
print('\n' + '=' * 70)
print('STEP 3: Merge and compute demand, S, danger')
print('=' * 70)

# behavior.csv trial = 1-45 (behavioral trials only, sequential rank within subject)
# vigor trial = 0-80 (global event index including probe trials)
# These do NOT align directly. We need to rank-match within each subject.
# Approach: for each subject, rank the vigor trials that match behavioral conditions,
# then merge by rank. Simplest correct approach: merge on subj only and use
# trial_vigor which already has mean vigor per (subj, global_trial).
# Since behavior trial 1-45 are the non-probe events, we match by computing
# the vigor for each global event and then joining.
#
# Actually, the simplest approach is to just use the vigor timeseries which
# contains threat/choice columns and group those to get behavioral trial vigor.

# Recompute trial-level vigor directly from the timeseries, filtering to
# trials that have choice data (non-probe)
vigor_beh = vigor_ts[vigor_ts['choice'].notna()].copy()
trial_vigor_beh = (
    vigor_beh
    .groupby(['subj', 'trial'])
    .agg(mean_vigor=('vigor_norm', 'mean'),
         threat=('threat', 'first'),
         choice=('choice', 'first'),
         distance_H=('distance_H', 'first'),
         effort_H=('effort_H', 'first'))
    .reset_index()
)
print(f'Trial vigor (behavioral only): {len(trial_vigor_beh)} rows')

# Now merge with behavior.csv by matching conditions (subj + threat + choice + effort_H + distance_H)
# or more simply, just use the vigor-derived data since it has all the columns we need
merged = trial_vigor_beh.copy()
# Add outcome from behavior (need to align)
# Match by subj and rank order
merged = merged.sort_values(['subj', 'trial'])
merged['beh_rank'] = merged.groupby('subj').cumcount() + 1
behavior_ranked = behavior.sort_values(['subj', 'trial']).copy()
behavior_ranked['beh_rank'] = behavior_ranked.groupby('subj').cumcount() + 1
merged = merged.merge(
    behavior_ranked[['subj', 'beh_rank', 'outcome']],
    on=['subj', 'beh_rank'],
    how='left'
)

print(f'Merged: {len(merged)} trials, NaN vigor: {merged["mean_vigor"].isna().sum()}')

# Demand: effort for the CHOSEN option (0.40 if chose low, effort_H if chose high)
merged['demand'] = np.where(merged['choice'] == 0, 0.4, merged['effort_H'])

# S_trial: computed from chosen option's threat and distance
merged['S_trial'] = (1 - merged['threat']) + merged['threat'] / (1 + LAM * merged['distance_H'])
merged['danger']  = 1 - merged['S_trial']

print(f'S_trial range: [{merged["S_trial"].min():.3f}, {merged["S_trial"].max():.3f}]')
print(f'danger range:  [{merged["danger"].min():.3f}, {merged["danger"].max():.3f}]')

# Drop rows with missing vigor
merged_clean = merged.dropna(subset=['mean_vigor', 'demand', 'danger']).copy()
print(f'After dropping NaN: {len(merged_clean)} trials')

# ── Per-subject OLS: excess ~ danger ──────────────────────────────────────────
print('\n' + '=' * 70)
print('STEP 4: Per-subject OLS — excess effort ~ danger')
print('=' * 70)

# Compute excess effort as vigor minus demand (simple subtraction, not regression)
# Demand is the minimum pressing rate for the chosen cookie:
#   choice=0 → demand=0.40 (low option always)
#   choice=1 → demand=effort_H (0.6, 0.8, or 1.0)
merged_clean['demand'] = np.where(merged_clean['choice'] == 0, 0.4, merged_clean['effort_H'])
merged_clean['excess'] = merged_clean['mean_vigor'] - merged_clean['demand']
merged_excess = merged_clean.copy()
print(f'Excess effort computed for {merged_excess["subj"].nunique()} subjects, '
      f'{len(merged_excess)} trials')
print(f'excess range: [{merged_excess["excess"].min():.4f}, {merged_excess["excess"].max():.4f}]')

# Now per-subject OLS: excess ~ danger → α_i (intercept), δ_i (slope)
vigor_param_rows = []
for s in merged_excess['subj'].unique():
    sub = merged_excess[merged_excess['subj'] == s].dropna(subset=['excess', 'danger'])
    if len(sub) < 5:
        continue
    x = sub['danger'].values
    y = sub['excess'].values
    if x.std() < 1e-6:
        continue
    slope, intercept, r_val, p_val, se = stats.linregress(x, y)
    vigor_param_rows.append({
        'subj':    s,
        'alpha_v': intercept,   # baseline excess vigor
        'delta':   slope,       # excess vigor sensitivity to danger
        'r':       r_val,
        'p':       p_val,
        'n_trials': len(sub),
    })

vigor_params_df = pd.DataFrame(vigor_param_rows)
print(f'\nVigor parameters (N={len(vigor_params_df)} subjects):')
print(f'  α_v (intercept): mean={vigor_params_df["alpha_v"].mean():.4f}, '
      f'SD={vigor_params_df["alpha_v"].std():.4f}')
print(f'  δ (danger slope): mean={vigor_params_df["delta"].mean():.4f}, '
      f'SD={vigor_params_df["delta"].std():.4f}')

# Test δ > 0 (one-sample t-test)
t_delta, p_delta = stats.ttest_1samp(vigor_params_df['delta'], 0)
print(f'\n  H3 test δ > 0: t({len(vigor_params_df)-1})={t_delta:.3f}, p={p_delta:.4f}'
      + (' *' if p_delta < 0.05 else ' n.s.'))

# ── Split-half reliability of δ ───────────────────────────────────────────────
print('\n' + '=' * 70)
print('STEP 5: Split-half reliability of δ (odd/even trials)')
print('=' * 70)

merged_excess['trial_idx'] = merged_excess.groupby('subj').cumcount()
half_rows = []
for s in merged_excess['subj'].unique():
    sub = merged_excess[merged_excess['subj'] == s]
    for half, mask in [('odd', sub['trial_idx'] % 2 == 0),
                       ('even', sub['trial_idx'] % 2 == 1)]:
        sub_h = sub[mask].dropna(subset=['excess', 'danger'])
        if len(sub_h) < 4 or sub_h['danger'].std() < 1e-6:
            continue
        slope, *_ = stats.linregress(sub_h['danger'].values, sub_h['excess'].values)
        half_rows.append({'subj': s, 'half': half, 'delta': slope})

half_df = pd.DataFrame(half_rows).pivot(index='subj', columns='half', values='delta').dropna()
r_sh, p_sh = stats.pearsonr(half_df['odd'].values, half_df['even'].values)
# Spearman-Brown correction
r_corrected = 2 * r_sh / (1 + r_sh)
print(f'Split-half δ reliability: r={r_sh:.4f}, Spearman-Brown={r_corrected:.4f}, '
      f'p={p_sh:.4f}, N={len(half_df)}')

# ── Within-choice tests ───────────────────────────────────────────────────────
print('\n' + '=' * 70)
print('STEP 6: Within-choice tests (danger → excess within choice=0 and choice=1)')
print('=' * 70)

for choice_val, label in [(0, 'chose low-effort'), (1, 'chose high-effort')]:
    sub_df = merged_excess[merged_excess['choice'] == choice_val]
    slopes = []
    for s in sub_df['subj'].unique():
        s_sub = sub_df[sub_df['subj'] == s].dropna(subset=['excess', 'danger'])
        if len(s_sub) < 3 or s_sub['danger'].std() < 1e-6:
            continue
        slope, *_ = stats.linregress(s_sub['danger'].values, s_sub['excess'].values)
        slopes.append(slope)
    if slopes:
        t_val, p_val = stats.ttest_1samp(slopes, 0)
        print(f'  [{label}] δ: mean={np.mean(slopes):.4f}, '
              f't({len(slopes)-1})={t_val:.3f}, p={p_val:.4f}'
              + (' *' if p_val < 0.05 else ' n.s.'))

# ── T×D interaction on excess effort ──────────────────────────────────────────
print('\n' + '=' * 70)
print('STEP 7: T×D interaction on excess effort')
print('=' * 70)

# Use per-subject slopes of excess on T and D and their interaction
# Compute TxD interaction: excess ~ T + D + T*D
from itertools import product

cell_means = (
    merged_excess
    .groupby(['subj', 'threat', 'distance_H'])['excess']
    .mean()
    .reset_index()
)

# ANOVA-style: compare corners
high_t = cell_means['threat'] == 0.9
low_t  = cell_means['threat'] == 0.1
high_d = cell_means['distance_H'] == 3
low_d  = cell_means['distance_H'] == 1

# T main effect (collapsing D)
t_main_slopes = []
for s in cell_means['subj'].unique():
    sub = cell_means[cell_means['subj'] == s]
    if len(sub) < 6:
        continue
    x = sub['threat'].values
    y = sub['excess'].values
    if x.std() < 1e-6:
        continue
    slope, *_ = stats.linregress(x, y)
    t_main_slopes.append(slope)
t_t, p_t = stats.ttest_1samp(t_main_slopes, 0)
print(f'  T main effect on excess: mean δ={np.mean(t_main_slopes):.4f}, '
      f't({len(t_main_slopes)-1})={t_t:.3f}, p={p_t:.4f}'
      + (' *' if p_t < 0.05 else ' n.s.'))

# D main effect
d_main_slopes = []
for s in cell_means['subj'].unique():
    sub = cell_means[cell_means['subj'] == s]
    if len(sub) < 6:
        continue
    x = sub['distance_H'].values
    y = sub['excess'].values
    if x.std() < 1e-6:
        continue
    slope, *_ = stats.linregress(x, y)
    d_main_slopes.append(slope)
t_d, p_d = stats.ttest_1samp(d_main_slopes, 0)
print(f'  D main effect on excess: mean δ={np.mean(d_main_slopes):.4f}, '
      f't({len(d_main_slopes)-1})={t_d:.3f}, p={p_d:.4f}'
      + (' *' if p_d < 0.05 else ' n.s.'))

# T×D interaction via cell means subtraction
# δ_interaction = (excess[hi_T,hi_D] - excess[lo_T,hi_D]) - (excess[hi_T,lo_D] - excess[lo_T,lo_D])
interaction_vals = []
for s in cell_means['subj'].unique():
    sub = cell_means[cell_means['subj'] == s]
    try:
        hh = sub[(sub['threat'] == 0.9) & (sub['distance_H'] == 3)]['excess'].values
        lh = sub[(sub['threat'] == 0.1) & (sub['distance_H'] == 3)]['excess'].values
        hl = sub[(sub['threat'] == 0.9) & (sub['distance_H'] == 1)]['excess'].values
        ll = sub[(sub['threat'] == 0.1) & (sub['distance_H'] == 1)]['excess'].values
        if len(hh) > 0 and len(lh) > 0 and len(hl) > 0 and len(ll) > 0:
            delta_int = (hh.mean() - lh.mean()) - (hl.mean() - ll.mean())
            interaction_vals.append(delta_int)
    except Exception:
        pass

if interaction_vals:
    t_int, p_int = stats.ttest_1samp(interaction_vals, 0)
    print(f'  T×D interaction on excess: mean={np.mean(interaction_vals):.4f}, '
          f't({len(interaction_vals)-1})={t_int:.3f}, p={p_int:.4f}'
          + (' *' if p_int < 0.05 else ' n.s.'))

# ── Save outputs ───────────────────────────────────────────────────────────────
print('\n' + '=' * 70)
print('STEP 8: Saving outputs')
print('=' * 70)

# Add split-half reliability and population-level test to output
vigor_params_df['delta_sh_r'] = r_sh
vigor_params_df['delta_sh_r_corrected'] = r_corrected

vigor_params_df.to_csv(OUT_DIR / 'vigor_params.csv', index=False)
print(f'Saved: {OUT_DIR}/vigor_params.csv')

print('\n' + '=' * 70)
print('SUMMARY')
print('=' * 70)
print(f'N subjects: {len(vigor_params_df)}')
print(f'δ (excess vigor sensitivity to danger): mean={vigor_params_df["delta"].mean():.4f} ± {vigor_params_df["delta"].std():.4f}')
print(f'H3 test δ > 0: t={t_delta:.3f}, p={p_delta:.4f}')
print(f'Split-half reliability: r={r_sh:.4f}, SB-corrected={r_corrected:.4f}')
print('\nDone.')
