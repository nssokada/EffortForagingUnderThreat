"""
Precompute per-trial vigor (median press rate / calibrationMax) from behavior_rich.csv.
Saves a lightweight CSV that the model comparison pipeline can load instantly.

Output: data/exploratory_350/processed/stage5_filtered_data_20260320_191950/trial_vigor.csv
Columns: subj, trial, type, threat, startDistance, distance_H,
         trialCookie_weight, calibrationMax, median_rate, actual_req, excess
"""

import numpy as np
import pandas as pd
import ast
import time
from pathlib import Path

EXCLUDE = [154, 197, 208]
DATA_DIR = Path("data/exploratory_350/processed/stage5_filtered_data_20260320_191950")


def compute_median_rate(effort_rate_str, calibration_max):
    """Compute median(1/IPI) / calibrationMax from alignedEffortRate string."""
    try:
        pt = np.array(ast.literal_eval(str(effort_rate_str)), dtype=float)
        ipis = np.diff(pt)
        ipis = ipis[ipis > 0.01]
        if len(ipis) >= 5 and calibration_max > 0:
            return np.median(1.0 / ipis) / calibration_max
        return np.nan
    except Exception:
        return np.nan


if __name__ == '__main__':
    t0 = time.time()
    print("Loading behavior_rich.csv...")
    beh = pd.read_csv(DATA_DIR / "behavior_rich.csv", low_memory=False)
    beh = beh[~beh['subj'].isin(EXCLUDE)].copy()
    print(f"  {len(beh)} trials, {beh['subj'].nunique()} subjects")

    print("Computing median press rates...")
    beh['median_rate'] = beh.apply(
        lambda row: compute_median_rate(row['alignedEffortRate'], row['calibrationMax']),
        axis=1
    )

    # Derived columns
    beh['actual_req'] = np.where(beh['trialCookie_weight'] == 3.0, 0.9, 0.4)
    beh['excess'] = beh['median_rate'] - beh['actual_req']
    beh['actual_dist'] = beh['startDistance'].map({5: 1, 7: 2, 9: 3})
    beh['actual_R'] = np.where(beh['trialCookie_weight'] == 3.0, 5.0, 1.0)
    beh['is_heavy'] = (beh['trialCookie_weight'] == 3.0).astype(int)

    # Keep only what the model needs
    keep_cols = ['subj', 'trial', 'type', 'threat', 'startDistance', 'actual_dist',
                 'distance_H', 'trialCookie_weight', 'calibrationMax',
                 'median_rate', 'actual_req', 'actual_R', 'is_heavy', 'excess']
    out = beh[keep_cols].copy()

    valid = out['median_rate'].notna().sum()
    missing = out['median_rate'].isna().sum()
    print(f"  Valid: {valid}, Missing: {missing}")

    out_path = DATA_DIR / "trial_vigor.csv"
    out.to_csv(out_path, index=False)
    print(f"  Saved to {out_path} ({out_path.stat().st_size / 1e6:.1f} MB)")
    print(f"  Done in {time.time() - t0:.0f}s")
