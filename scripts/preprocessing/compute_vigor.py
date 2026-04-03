"""
Vigor computation from Stage 2 processed_trials.pkl.

Reads from the pickle (which preserves list-type columns like effortRate)
rather than behavior_rich.csv (which loses them to string serialization).

Produces three files needed by the analysis notebooks:
  1. trial_vigor.csv     — per-trial normalized press rate (H1)
  2. vigor_metrics.csv   — per-trial × per-epoch vigor metrics (H2)
  3. cell_means.csv      — per-subject condition cell means (H3, H4, H5)

Usage:
  python scripts/preprocessing/compute_vigor.py --stage2_dir <path> --stage5_dir <path>
  python scripts/preprocessing/compute_vigor.py  # auto-detect
"""

import argparse
import numpy as np
import pandas as pd
from pathlib import Path


def find_latest_dir(base, prefix):
    candidates = sorted(base.glob(f"{prefix}*"))
    return candidates[-1] if candidates else None


def compute_ipi_metrics(timestamps, calibration_max, req=None):
    """Compute IPI-based vigor metrics from a list of press timestamps."""
    result = {'norm_rate': np.nan, 'n_presses': 0,
              'median_ipi': np.nan, 'press_sd': np.nan,
              'press_cv': np.nan, 'frac_full': np.nan,
              'relative_vigor': np.nan, 'pause_freq': np.nan}

    if not isinstance(timestamps, (list, np.ndarray)) or len(timestamps) < 2:
        result['n_presses'] = len(timestamps) if isinstance(timestamps, (list, np.ndarray)) else 0
        return result

    if calibration_max <= 0:
        calibration_max = 1.0

    ipis = np.diff(timestamps)
    ipis = ipis[ipis >= 0.01]  # remove <10ms artifacts
    if len(ipis) == 0:
        result['n_presses'] = len(timestamps)
        return result

    inst_rate = 1.0 / ipis
    norm_rate = inst_rate / calibration_max
    median_ipi = np.median(ipis)
    med_norm = float(np.median(norm_rate))
    sd_norm = float(np.std(norm_rate))
    mean_norm = float(np.mean(norm_rate))

    result.update({
        'norm_rate': med_norm,
        'median_ipi': float(median_ipi),
        'n_presses': len(timestamps),
        'press_sd': sd_norm,
        'press_cv': sd_norm / mean_norm if mean_norm > 0 else np.nan,
        'pause_freq': float(np.mean(ipis > max(2 * median_ipi, 0.5))),
    })

    if req and req > 0:
        result['relative_vigor'] = med_norm / req
        result['frac_full'] = float(np.mean(norm_rate >= req))

    return result


def extract_epoch(timestamps, t_start, t_end):
    """Extract timestamps within a time window."""
    if not isinstance(timestamps, (list, np.ndarray)):
        return []
    arr = np.array(timestamps)
    return arr[(arr >= t_start) & (arr <= t_end)].tolist()


def process_trial(row, subj_col='subj'):
    """Process a single trial into vigor metrics for all epochs."""
    # Get raw effort timestamps
    effort = row.get('alignedEffortRate') or row.get('effortRate')
    if not isinstance(effort, (list, np.ndarray)):
        effort = []

    cal = row.get('calibrationMax', 1.0)
    if not cal or cal <= 0 or np.isnan(cal):
        cal = 1.0

    weight = row.get('trialCookie_weight', 1)
    req = 0.9 if weight == 3 else 0.4
    is_heavy = 1 if weight == 3 else 0

    # Timing
    enc_t = row.get('encounterTime', np.nan)
    trial_end = row.get('trialEndTime', np.nan)
    strike_t = row.get('strikeTime', row.get('strike_time', np.nan))

    try:
        enc_t = float(enc_t) if enc_t is not None else np.nan
    except (ValueError, TypeError):
        enc_t = np.nan
    try:
        trial_end = float(trial_end) if trial_end is not None else np.nan
    except (ValueError, TypeError):
        trial_end = np.nan
    try:
        strike_t = float(strike_t) if strike_t is not None else np.nan
    except (ValueError, TypeError):
        strike_t = np.nan

    trial_start = min(effort) if effort else 0

    # Base trial info
    threat = row.get('threat', row.get('attackingProb', 0))
    base = {
        'subj': row.get(subj_col, row.get('participantID', '')),
        'trial': row.get('trial', row.name if hasattr(row, 'name') else 0),
        'T_round': round(float(threat), 1),
        'distance': row.get('distance_H', row.get('distance', 0)),
        'cookie': is_heavy,
        'is_attack': int(row.get('isAttackTrial', 0)),
        'type': int(row.get('type', 0)),
    }

    epochs = []

    # Full trial
    end = trial_end if not np.isnan(trial_end) else (max(effort) + 1 if effort else 999)
    ts = extract_epoch(effort, trial_start, end)
    m = compute_ipi_metrics(ts, cal, req)
    epochs.append({**base, 'epoch': 'full', **m})

    if not np.isnan(enc_t) and effort:
        # Onset: trial start to encounter
        ts = extract_epoch(effort, trial_start, enc_t)
        m = compute_ipi_metrics(ts, cal, req)
        epochs.append({**base, 'epoch': 'onset', **m})

        # Anticipatory: 1s before encounter
        ts = extract_epoch(effort, max(enc_t - 1.0, trial_start), enc_t)
        m = compute_ipi_metrics(ts, cal, req)
        epochs.append({**base, 'epoch': 'anticipatory', **m})

        # Reactive: encounter to encounter + 2s
        ts = extract_epoch(effort, enc_t, enc_t + 2.0)
        m = compute_ipi_metrics(ts, cal, req)
        epochs.append({**base, 'epoch': 'reactive', **m})

    if not np.isnan(strike_t) and row.get('isAttackTrial', 0) and effort:
        # Terminal: 2s before strike
        ts = extract_epoch(effort, max(strike_t - 2.0, trial_start), strike_t)
        m = compute_ipi_metrics(ts, cal, req)
        epochs.append({**base, 'epoch': 'terminal', **m})

    return epochs


def compute_cell_means(vigor_df):
    """Compute per-subject condition cell means (for H3/H4/H5 model fitting)."""
    full = vigor_df[(vigor_df['epoch'] == 'full') & (vigor_df['type'] == 1)].copy()
    full = full[full['norm_rate'].notna()]
    full['is_heavy'] = full['cookie']
    full['actual_dist'] = full['distance'] + 1  # 0-indexed to 1-indexed
    full['actual_R'] = np.where(full['is_heavy'] == 1, 5.0, 1.0)
    full['actual_req'] = np.where(full['is_heavy'] == 1, 0.9, 0.4)

    cells = full.groupby(['subj', 'T_round', 'actual_dist', 'is_heavy']).agg(
        mean_rate=('norm_rate', 'mean'),
        n_trials=('norm_rate', 'count'),
        sd_rate=('norm_rate', 'std'),
    ).reset_index()

    cells['sem'] = cells['sd_rate'] / np.sqrt(cells['n_trials'])
    cells['actual_R'] = np.where(cells['is_heavy'] == 1, 5.0, 1.0)
    cells['actual_req'] = np.where(cells['is_heavy'] == 1, 0.9, 0.4)

    subj_cookie_mean = cells.groupby(['subj', 'is_heavy'])['mean_rate'].transform('mean')
    cells['subj_cookie_mean'] = subj_cookie_mean
    cells['rel_rate'] = cells['mean_rate'] / cells['subj_cookie_mean']
    cells['rel_rate_cc'] = cells['rel_rate']

    return cells


def main():
    parser = argparse.ArgumentParser(description="Compute vigor metrics")
    parser.add_argument("--stage2_dir", type=str, default=None,
                        help="Stage 2 output dir (contains processed_trials.pkl)")
    parser.add_argument("--stage5_dir", type=str, default=None,
                        help="Stage 5 output dir (for saving trial_vigor.csv)")
    parser.add_argument("--output_dir", type=str, default="results/stats/vigor_analysis")
    parser.add_argument("--exclude", type=int, nargs="*", default=[])
    args = parser.parse_args()

    # Find directories
    for sample in ["exploratory_350", "confirmatory_350"]:
        base = Path(f"data/{sample}/processed")
        if base.exists():
            if not args.stage2_dir:
                args.stage2_dir = str(find_latest_dir(base, "stage2_"))
            if not args.stage5_dir:
                args.stage5_dir = str(find_latest_dir(base, "stage5_"))
            break

    if not args.stage2_dir:
        print("ERROR: No stage2 output found.")
        return

    stage2_dir = Path(args.stage2_dir)
    stage5_dir = Path(args.stage5_dir) if args.stage5_dir else stage2_dir
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load from pickle (preserves list columns)
    pkl_path = stage2_dir / "processed_trials.pkl"
    print(f"Loading {pkl_path}...")
    td = pd.read_pickle(pkl_path)

    # Add subj mapping from stage5 if available
    mapping_path = stage5_dir / "subject_mapping.csv"
    if mapping_path.exists():
        mapping = pd.read_csv(mapping_path)
        if 'participantID' in mapping.columns and 'subj' in mapping.columns:
            pid_to_subj = dict(zip(mapping['participantID'], mapping['subj']))
            td['subj'] = td['participantID'].map(pid_to_subj)
            td = td[td['subj'].notna()]
            td['subj'] = td['subj'].astype(int)

    if args.exclude:
        td = td[~td['subj'].isin(args.exclude)]

    print(f"  {len(td)} trials, {td['subj'].nunique()} subjects")

    # Check key columns
    has_aligned = 'alignedEffortRate' in td.columns and td['alignedEffortRate'].notna().any()
    has_raw = 'effortRate' in td.columns and td['effortRate'].notna().any()
    print(f"  alignedEffortRate: {'YES' if has_aligned else 'NO'}")
    print(f"  effortRate (raw): {'YES' if has_raw else 'NO'}")

    # Process all trials
    print("\nComputing vigor metrics...")
    all_epochs = []
    n_with_data = 0
    for idx, row in td.iterrows():
        epochs = process_trial(row)
        all_epochs.extend(epochs)
        if epochs and epochs[0].get('n_presses', 0) > 0:
            n_with_data += 1

    epoch_df = pd.DataFrame(all_epochs)
    print(f"  {len(epoch_df)} epoch rows, {n_with_data}/{len(td)} trials with keypress data")

    # Trial-level vigor (full epoch only, for H1)
    trial_vigor = epoch_df[epoch_df['epoch'] == 'full'].copy()
    trial_vigor.to_csv(stage5_dir / "trial_vigor.csv", index=False)
    print(f"  Saved trial_vigor.csv: {len(trial_vigor)} trials")

    # All epoch metrics (for H2)
    epoch_df.to_csv(out_dir / "vigor_metrics.csv", index=False)
    print(f"  Saved vigor_metrics.csv: {len(epoch_df)} rows")

    # Cell means (for H3/H4/H5)
    cell_means = compute_cell_means(epoch_df)
    cell_means.to_csv(out_dir / "cell_means.csv", index=False)
    print(f"  Saved cell_means.csv: {len(cell_means)} cells")

    print("\nDone.")


if __name__ == "__main__":
    main()
