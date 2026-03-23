#!/usr/bin/env python3
"""
Phase Extraction — reproduces NB04 (04_phase_extraction.ipynb)

Inputs:
  - smoothed_vigor_ts.parquet   (vigor_processed/)
  - trial_events.parquet        (vigor_prep/)

Outputs (to vigor_processed/):
  - phase_trial_metrics.parquet  — trial-level DVs (resid only)
  - phase_vigor_metrics.parquet  — trial-level DVs (resid + norm variants)
  - encounter_phase_ts.parquet   — encounter window time series for downstream
  - terminal_phase_ts.parquet    — terminal window time series for downstream
"""

import numpy as np
import pandas as pd
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# ── Paths ──────────────────────────────────────────────────────────────────────
ROOT       = Path('/workspace')
VIGOR_PROC = ROOT / 'data' / 'exploratory_350' / 'processed' / 'vigor_processed'
VIGOR_PREP = ROOT / 'data' / 'exploratory_350' / 'processed' / 'vigor_prep'

# ── Constants ──────────────────────────────────────────────────────────────────
ONSET_WINDOW    = 2.0   # seconds from effort onset
ENC_HALF_WINDOW = 1.0   # seconds on each side of encounter
TERM_WINDOW     = 2.0   # seconds before trial resolution

# ── Load data ──────────────────────────────────────────────────────────────────
print("Loading smoothed_vigor_ts.parquet ...")
sv = pd.read_parquet(VIGOR_PROC / 'smoothed_vigor_ts.parquet')
print(f"  {sv.shape[0]:,} rows, {sv['subj'].nunique()} subjects, "
      f"{sv.groupby(['subj', 'trial']).ngroups:,} trials")
print(f"  Columns: {list(sv.columns)}")

# ── Validation ─────────────────────────────────────────────────────────────────
trial_check = sv.groupby(['subj', 'trial']).agg(
    encounterTime=('encounterTime', 'first'),
    isAttackTrial=('isAttackTrial', 'first'),
).reset_index()

n_enc_null = trial_check['encounterTime'].isna().sum()
n_attack   = (trial_check['isAttackTrial'] == 1).sum()
n_nonattack = (trial_check['isAttackTrial'] == 0).sum()
print(f"\nValidation:")
print(f"  Total trials: {len(trial_check):,}")
print(f"  Attack: {n_attack:,}, Non-attack: {n_nonattack:,}")
print(f"  encounterTime null: {n_enc_null} (should be 0)")

# ── Build trial-level index ────────────────────────────────────────────────────
print("\nBuilding trial-level index ...")
trial_info = (sv.groupby(['subj', 'trial'])
    .agg({
        'participantID': 'first',
        'trialName': 'first',
        'threat': 'first',
        'choice': 'first',
        'outcome': 'first',
        'isAttackTrial': 'first',
        'encounterTime': 'first',
        'trialEscapeTime': 'first',
        'trialCaptureTime': 'first',
        'trialEndTime': 'first',
        'startDistance': 'first',
        'effort_H': 'first',
        'distance_H': 'first',
    })
    .reset_index())

# Resolution time: escape > capture > trialEnd
trial_info['resolution_time'] = np.where(
    trial_info['trialEscapeTime'] > 0,
    trial_info['trialEscapeTime'],
    np.where(trial_info['trialCaptureTime'] > 0,
             trial_info['trialCaptureTime'],
             trial_info['trialEndTime']))

trial_info['attack']   = trial_info['isAttackTrial'].astype(int)
trial_info['threat_c'] = trial_info['threat'] - 0.5
trial_info['escaped']  = np.where(
    trial_info['attack'] == 1,
    (trial_info['trialEscapeTime'] > 0).astype(int),
    np.nan)

print(f"  Trial info: {trial_info.shape}")
print(f"  Resolution time range: [{trial_info['resolution_time'].min():.2f}, "
      f"{trial_info['resolution_time'].max():.2f}]")

# ==============================================================================
# ONSET PHASE [0, 2s]
# ==============================================================================
print("\n── Onset phase [0, 2s] ──")
onset_ts = sv[(sv['t'] >= 0) & (sv['t'] <= ONSET_WINDOW)].copy()
print(f"  Onset time series: {len(onset_ts):,} rows")

def onset_trial_dvs(grp):
    t = grp['t'].values
    v = grp['vigor_resid'].values
    if len(t) < 3 or t[-1] <= t[0]:
        return pd.Series({'onset_slope': np.nan, 'onset_mean': np.nan})
    return pd.Series({
        'onset_slope': np.polyfit(t, v, 1)[0],
        'onset_mean': np.mean(v),
    })

onset_dvs = (onset_ts
    .groupby(['subj', 'trial'])
    .apply(onset_trial_dvs, include_groups=False)
    .reset_index())

print(f"  Onset DVs: {onset_dvs.shape}")
print(f"    onset_slope: M={onset_dvs['onset_slope'].mean():.4f}, "
      f"SD={onset_dvs['onset_slope'].std():.4f}, "
      f"N valid={onset_dvs['onset_slope'].notna().sum()}")
print(f"    onset_mean:  M={onset_dvs['onset_mean'].mean():.4f}, "
      f"SD={onset_dvs['onset_mean'].std():.4f}, "
      f"N valid={onset_dvs['onset_mean'].notna().sum()}")

# ==============================================================================
# ENCOUNTER PHASE [enc-1s, enc+1s]
# ==============================================================================
print("\n── Encounter phase [enc±1s] ──")
sv_enc = sv.merge(trial_info[['subj', 'trial', 'attack']], on=['subj', 'trial'])
sv_enc['t_enc'] = sv_enc['t'] - sv_enc['encounterTime']

encounter_ts = sv_enc[
    (sv_enc['t_enc'] >= -ENC_HALF_WINDOW) &
    (sv_enc['t_enc'] <= ENC_HALF_WINDOW)
].copy()
encounter_ts['post'] = (encounter_ts['t_enc'] > 0).astype(int)

print(f"  Encounter time series (before quality filter): {len(encounter_ts):,} rows")

# Quality filter: require full window coverage
trial_window = (encounter_ts.groupby(['subj', 'trial'])
    .agg(t_min=('t_enc', 'min'), t_max=('t_enc', 'max'), n_pts=('t_enc', 'size'))
    .reset_index())

valid_enc = trial_window[
    (trial_window['t_min'] <= -ENC_HALF_WINDOW + 0.15) &
    (trial_window['t_max'] >= ENC_HALF_WINDOW - 0.15) &
    (trial_window['n_pts'] >= 10)
][['subj', 'trial']]

encounter_ts = encounter_ts.merge(valid_enc, on=['subj', 'trial'])
n_enc_trials = encounter_ts.groupby(['subj', 'trial']).ngroups
print(f"  Encounter time series (after filter): {len(encounter_ts):,} rows, "
      f"{n_enc_trials:,} trials")
print(f"  Dropped {len(trial_window) - len(valid_enc)} trials with incomplete windows")

def encounter_trial_dvs(grp):
    pre  = grp.loc[grp['post'] == 0, 'vigor_resid']
    post = grp.loc[grp['post'] == 1, 'vigor_resid']
    if len(pre) < 2 or len(post) < 2:
        return pd.Series({'enc_pre_mean': np.nan, 'enc_post_mean': np.nan,
                          'enc_spike': np.nan})
    return pd.Series({
        'enc_pre_mean': pre.mean(),
        'enc_post_mean': post.mean(),
        'enc_spike': post.mean() - pre.mean(),
    })

encounter_dvs = (encounter_ts
    .groupby(['subj', 'trial'])
    .apply(encounter_trial_dvs, include_groups=False)
    .reset_index())

print(f"  Encounter DVs: {encounter_dvs.shape}")
print(f"    enc_spike (all): M={encounter_dvs['enc_spike'].mean():.4f}, "
      f"SD={encounter_dvs['enc_spike'].std():.4f}")

enc_with_attack = encounter_dvs.merge(trial_info[['subj', 'trial', 'attack']], on=['subj', 'trial'])
for atk in [0, 1]:
    sub = enc_with_attack[enc_with_attack['attack'] == atk]['enc_spike']
    label = 'Attack' if atk else 'Non-attack'
    print(f"    enc_spike ({label:10s}): M={sub.mean():.4f}, SD={sub.std():.4f}, N={len(sub)}")

# ==============================================================================
# TERMINAL PHASE [resolution-2s, resolution]
# ==============================================================================
print("\n── Terminal phase [res-2s, res] ──")
sv_term = sv.merge(trial_info[['subj', 'trial', 'resolution_time', 'attack']],
                   on=['subj', 'trial'])
sv_term['t_term'] = sv_term['t'] - sv_term['resolution_time']

terminal_ts = sv_term[
    (sv_term['t_term'] >= -TERM_WINDOW) &
    (sv_term['t_term'] <= 0)
].copy()

print(f"  Terminal time series (before quality filter): {len(terminal_ts):,} rows")

term_counts = terminal_ts.groupby(['subj', 'trial']).size().reset_index(name='n_pts')
valid_term  = term_counts[term_counts['n_pts'] >= 10][['subj', 'trial']]
terminal_ts = terminal_ts.merge(valid_term, on=['subj', 'trial'])

n_term_trials = terminal_ts.groupby(['subj', 'trial']).ngroups
print(f"  Terminal time series (after filter): {len(terminal_ts):,} rows, "
      f"{n_term_trials:,} trials")
print(f"  Dropped {len(term_counts) - len(valid_term)} trials with < 10 timepoints")

def terminal_trial_dvs(grp):
    t = grp['t_term'].values
    v = grp['vigor_resid'].values
    if len(t) < 3 or t[-1] <= t[0]:
        return pd.Series({'term_mean': np.nan, 'term_slope': np.nan})
    return pd.Series({
        'term_mean': np.mean(v),
        'term_slope': np.polyfit(t, v, 1)[0],
    })

terminal_dvs = (terminal_ts
    .groupby(['subj', 'trial'])
    .apply(terminal_trial_dvs, include_groups=False)
    .reset_index())

print(f"  Terminal DVs: {terminal_dvs.shape}")
print(f"    term_mean: M={terminal_dvs['term_mean'].mean():.4f}, "
      f"SD={terminal_dvs['term_mean'].std():.4f}")

# ==============================================================================
# MERGE → phase_trial_metrics.parquet
# ==============================================================================
print("\n── Merging → phase_trial_metrics ──")
df = trial_info.copy()
df = df.merge(onset_dvs,    on=['subj', 'trial'], how='left')
df = df.merge(encounter_dvs, on=['subj', 'trial'], how='left')
df = df.merge(terminal_dvs,  on=['subj', 'trial'], how='left')

print(f"  Phase trial metrics: {df.shape}")
print("  DV coverage:")
for col in ['onset_slope', 'onset_mean', 'enc_spike', 'enc_pre_mean',
            'enc_post_mean', 'term_mean', 'term_slope']:
    n_valid = df[col].notna().sum()
    print(f"    {col:20s}: {n_valid:,} / {len(df):,} ({100*n_valid/len(df):.1f}%)")

df.to_parquet(VIGOR_PROC / 'phase_trial_metrics.parquet', index=False)
print(f"  Saved phase_trial_metrics.parquet: {df.shape}")

# ==============================================================================
# SAVE PHASE TIME SERIES (for downstream notebooks)
# ==============================================================================
print("\n── Saving phase time series ──")
enc_save_cols  = ['subj', 'trial', 't_enc', 'vigor_resid', 'post', 'attack']
term_save_cols = ['subj', 'trial', 't_term', 'vigor_resid', 'attack']

encounter_ts[enc_save_cols].to_parquet(VIGOR_PROC / 'encounter_phase_ts.parquet', index=False)
print(f"  Saved encounter_phase_ts.parquet: {encounter_ts[enc_save_cols].shape}")

terminal_ts[term_save_cols].to_parquet(VIGOR_PROC / 'terminal_phase_ts.parquet', index=False)
print(f"  Saved terminal_phase_ts.parquet: {terminal_ts[term_save_cols].shape}")

# ==============================================================================
# NORM VARIANTS → phase_vigor_metrics.parquet
# ==============================================================================
print("\n── Computing norm variants → phase_vigor_metrics ──")

def onset_trial_dvs_norm(grp):
    t = grp['t'].values
    v = grp['vigor_norm'].values
    if len(t) < 3 or t[-1] <= t[0]:
        return pd.Series({'onset_slope_norm': np.nan, 'onset_mean_norm': np.nan})
    return pd.Series({
        'onset_slope_norm': np.polyfit(t, v, 1)[0],
        'onset_mean_norm': np.mean(v),
    })

def encounter_trial_dvs_norm(grp):
    pre  = grp.loc[grp['post'] == 0, 'vigor_norm']
    post = grp.loc[grp['post'] == 1, 'vigor_norm']
    if len(pre) < 2 or len(post) < 2:
        return pd.Series({'enc_pre_mean_norm': np.nan, 'enc_post_mean_norm': np.nan,
                          'enc_spike_norm': np.nan})
    return pd.Series({
        'enc_pre_mean_norm': pre.mean(),
        'enc_post_mean_norm': post.mean(),
        'enc_spike_norm': post.mean() - pre.mean(),
    })

def terminal_trial_dvs_norm(grp):
    t = grp['t_term'].values
    v = grp['vigor_norm'].values
    if len(t) < 3 or t[-1] <= t[0]:
        return pd.Series({'term_mean_norm': np.nan, 'term_slope_norm': np.nan})
    return pd.Series({
        'term_mean_norm': np.mean(v),
        'term_slope_norm': np.polyfit(t, v, 1)[0],
    })

# Onset norm
onset_ts_norm   = sv[(sv['t'] >= 0) & (sv['t'] <= ONSET_WINDOW)].copy()
onset_dvs_norm  = (onset_ts_norm
    .groupby(['subj', 'trial'])
    .apply(onset_trial_dvs_norm, include_groups=False)
    .reset_index())

# Encounter norm (re-compute window on vigor_norm)
sv_enc_norm         = sv.merge(trial_info[['subj', 'trial', 'attack']], on=['subj', 'trial'])
sv_enc_norm['t_enc'] = sv_enc_norm['t'] - sv_enc_norm['encounterTime']
enc_norm            = sv_enc_norm[
    (sv_enc_norm['t_enc'] >= -ENC_HALF_WINDOW) &
    (sv_enc_norm['t_enc'] <= ENC_HALF_WINDOW)
].copy()
enc_norm['post']    = (enc_norm['t_enc'] > 0).astype(int)
encounter_dvs_norm  = (enc_norm
    .groupby(['subj', 'trial'])
    .apply(encounter_trial_dvs_norm, include_groups=False)
    .reset_index())

# Terminal norm (re-compute window on vigor_norm)
sv_term_norm          = sv.merge(trial_info[['subj', 'trial', 'resolution_time', 'attack']],
                                 on=['subj', 'trial'])
sv_term_norm['t_term'] = sv_term_norm['t'] - sv_term_norm['resolution_time']
term_norm             = sv_term_norm[
    (sv_term_norm['t_term'] >= -TERM_WINDOW) &
    (sv_term_norm['t_term'] <= 0)
].copy()
terminal_dvs_norm     = (term_norm
    .groupby(['subj', 'trial'])
    .apply(terminal_trial_dvs_norm, include_groups=False)
    .reset_index())

# Build phase_vigor_metrics: rename resid columns, merge norm columns
pm = df.rename(columns={
    'onset_slope':    'onset_slope_resid',
    'onset_mean':     'onset_mean_resid',
    'enc_pre_mean':   'enc_pre_mean_resid',
    'enc_post_mean':  'enc_post_mean_resid',
    'enc_spike':      'enc_spike_resid',
    'term_mean':      'term_mean_resid',
    'term_slope':     'term_slope_resid',
})
pm = pm.merge(onset_dvs_norm,    on=['subj', 'trial'], how='left')
pm = pm.merge(encounter_dvs_norm, on=['subj', 'trial'], how='left')
pm = pm.merge(terminal_dvs_norm,  on=['subj', 'trial'], how='left')

pm.to_parquet(VIGOR_PROC / 'phase_vigor_metrics.parquet', index=False)
print(f"  Saved phase_vigor_metrics.parquet: {pm.shape}")
print(f"  Columns: {pm.columns.tolist()}")

# ==============================================================================
# FINAL SUMMARY
# ==============================================================================
print("\n══ DONE ══")
print(f"Outputs in {VIGOR_PROC}:")
for fname in ['phase_trial_metrics.parquet', 'phase_vigor_metrics.parquet',
              'encounter_phase_ts.parquet', 'terminal_phase_ts.parquet']:
    fpath = VIGOR_PROC / fname
    if fpath.exists():
        size_mb = fpath.stat().st_size / 1e6
        print(f"  {fname:<35s}  {size_mb:.1f} MB")
    else:
        print(f"  {fname:<35s}  MISSING")
