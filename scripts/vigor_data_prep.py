"""
Vigor Data Prep: convert stage2 processed_trials.pkl into the parquet files
expected by the kernel-smoothing notebook (02_kernel_smoothing.ipynb).

Outputs (all in VIGOR_PREP directory):
  keypress_events.parquet  — one row per keypress, cols: participantID, trial, t
  trial_events.parquet     — one row per trial, all trial-level metadata
  effort_ts.parquet        — participantID + calibrationMax (for normalization)
"""

import sys, glob, pickle
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]

# ── Find stage2 output ────────────────────────────────────────────────────────
stage2_dirs = sorted(glob.glob(str(
    ROOT / 'data' / 'exploratory_350' / 'processed' / 'stage2_trial_processing_*'
)))
stage5_dirs = sorted(glob.glob(str(
    ROOT / 'data' / 'exploratory_350' / 'processed' / 'stage5_filtered_data_*'
)))

STAGE2_DIR = Path(stage2_dirs[-1])
STAGE5_DIR = Path(stage5_dirs[-1])
VIGOR_PREP = ROOT / 'data' / 'exploratory_350' / 'processed' / 'vigor_prep'
VIGOR_PREP.mkdir(parents=True, exist_ok=True)

print(f"Stage2: {STAGE2_DIR}")
print(f"Stage5: {STAGE5_DIR}")
print(f"Output: {VIGOR_PREP}")

# ── Load stage2 processed trials ─────────────────────────────────────────────
print("\nLoading processed_trials.pkl...")
with open(STAGE2_DIR / 'processed_trials.pkl', 'rb') as f:
    trials = pickle.load(f)
print(f"  Loaded: {len(trials):,} rows, {trials['participantID'].nunique()} participants")

# ── Load stage5 subject mapping to filter to final 293 subjects ───────────────
print("Loading subject mapping...")
subj_map = pd.read_csv(STAGE5_DIR / 'subject_mapping.csv')
valid_pids = set(subj_map['participantID'])
print(f"  Valid participants: {len(valid_pids)}")

trials = trials[trials['participantID'].isin(valid_pids)].copy()
print(f"  After filtering: {len(trials):,} rows, {trials['participantID'].nunique()} participants")

# ── Merge subj integer ID ─────────────────────────────────────────────────────
trials = trials.merge(subj_map[['participantID', 'subj']], on='participantID', how='left')

# ── keypress_events.parquet ───────────────────────────────────────────────────
# alignedEffortRate: list of keypress timestamps (seconds from effort onset)
print("\nBuilding keypress_events...")
kp_rows = []
for _, row in trials.iterrows():
    kp_times = row['alignedEffortRate']
    if not isinstance(kp_times, (list, np.ndarray)) or len(kp_times) == 0:
        continue
    for t in kp_times:
        kp_rows.append({
            'participantID': row['participantID'],
            'subj': row['subj'],
            'trialName': row['trialName'],
            'trial': row['trial'],
            't': float(t),
        })

kp_df = pd.DataFrame(kp_rows)
print(f"  keypress_events: {len(kp_df):,} rows")
kp_df.to_parquet(VIGOR_PREP / 'keypress_events.parquet', index=False)
print(f"  Saved: keypress_events.parquet")

# ── trial_events.parquet ──────────────────────────────────────────────────────
print("\nBuilding trial_events...")

# Compute trialEndTime: last keypress time + small buffer, or use raw end
def get_trial_end(row):
    kp = row['alignedEffortRate']
    if isinstance(kp, (list, np.ndarray)) and len(kp) > 0:
        return float(max(kp)) + 0.5
    return np.nan

trials['trialEndTime_effort'] = trials.apply(get_trial_end, axis=1)

te_cols = [
    'participantID', 'subj', 'trialName', 'trial', 'threat',
    'choice', 'outcome', 'isAttackTrial',
    'encounterTime', 'trialEscapeTime', 'trialCaptureTime',
    'startDistance', 'effort_H', 'effort_L', 'distance_H', 'distance_L',
    'calibrationMax',
]
# Use the computed trial end time
te_df = trials[te_cols].copy()

# encounterTime is in absolute time; we need it relative to effort onset
# alignedEffortRate times are already effort-onset-relative
# encounterTime in processed_trials appears to be absolute (ms or s from task start)
# We need it relative to effort onset (playerEffortStartTime)
# Compute: encounterTime_relative = encounterTime - playerEffortStartTime
# But playerEffortStartTime is absolute. Let's check if encounterTime is in the
# same frame as alignedEffortRate.
# alignedEffortRate: time since effort onset (from stage2 processing)
# encounterTime: absolute time? Let's compare with firstEffortTime
# firstEffortTime is the effort onset in absolute terms.
# encounterTime_relative = encounterTime - firstEffortTime
if 'firstEffortTime' in trials.columns:
    te_df = te_df.copy()
    te_df['encounterTime'] = trials['encounterTime'] - trials['firstEffortTime']
    te_df['trialEscapeTime'] = trials['trialEscapeTime'] - trials['firstEffortTime']
    te_df['trialCaptureTime'] = trials['trialCaptureTime'] - trials['firstEffortTime']
    # Clip negatives to 0 (non-attack trials)
    te_df['encounterTime'] = te_df['encounterTime'].clip(lower=0)
    te_df['trialEscapeTime'] = te_df['trialEscapeTime'].where(te_df['trialEscapeTime'] > 0, np.nan)
    te_df['trialCaptureTime'] = te_df['trialCaptureTime'].where(te_df['trialCaptureTime'] > 0, np.nan)

te_df['trialEndTime'] = trials['trialEndTime_effort']

print(f"  trial_events: {len(te_df):,} rows")
te_df.to_parquet(VIGOR_PREP / 'trial_events.parquet', index=False)
print(f"  Saved: trial_events.parquet")

# ── effort_ts.parquet (calibrationMax per participant) ───────────────────────
print("\nBuilding effort_ts (calibrationMax)...")
cal_df = (trials.groupby('participantID')['calibrationMax']
          .first()
          .reset_index())
cal_df.to_parquet(VIGOR_PREP / 'effort_ts.parquet', index=False)
print(f"  effort_ts: {len(cal_df)} participants")
print(f"  Saved: effort_ts.parquet")

# ── subject_mapping.csv ───────────────────────────────────────────────────────
subj_map.to_csv(VIGOR_PREP / 'subject_mapping.csv', index=False)
print(f"\nSaved: subject_mapping.csv ({len(subj_map)} rows)")

print("\nDone. Vigor prep complete.")
print(f"Output directory: {VIGOR_PREP}")
for f in sorted(VIGOR_PREP.iterdir()):
    size_mb = f.stat().st_size / 1e6
    print(f"  {f.name}: {size_mb:.1f} MB")
