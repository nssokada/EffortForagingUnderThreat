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
    strike_t = row.get('strikeTime', row.get('strike_time', row.get('circaStrikeTime', np.nan)))

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
    # Trial number — must be the per-subject trial index (0-80), NOT the
    # global DataFrame row index. Never fall back to row.name.
    trial_num = row.get('trial', np.nan)
    # Distance — `distance_H` from stage2 is already 1-indexed (1/2/3).
    # `distance` (when present, e.g. in older processed_trials.pkl) is
    # 0-indexed. Normalize both to 1-indexed here.
    if 'distance_H' in row.index:
        actual_dist = int(row['distance_H'])
    elif 'distance' in row.index and not pd.isna(row.get('distance')):
        actual_dist = int(row['distance']) + 1
    else:
        actual_dist = np.nan
    base = {
        'subj': row.get(subj_col, row.get('participantID', '')),
        'trial': int(trial_num) if not pd.isna(trial_num) else np.nan,
        'T_round': round(float(threat), 1),
        'distance': actual_dist,
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
    # `distance` in process_trial is now always 1-indexed (1/2/3).
    full['actual_dist'] = full['distance'].astype(int)
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


def _load_trial_source(stage5_dir: Path, stage2_dir: Path) -> pd.DataFrame:
    """Load the per-trial source DataFrame for vigor computation.

    Prefer the Stage 5 ``behavior_rich.pkl`` because (a) it is the cleanest
    filtered table and is small enough to load in low-memory environments,
    (b) it already has the per-subject ``trial`` index (0–80), and (c) it
    has ``distance_H`` (1-indexed) which we want to use directly.

    Falls back to Stage 2 ``processed_trials.pkl`` only if the Stage 5
    pickle is missing.
    """
    s5_pkl = stage5_dir / "behavior_rich.pkl"
    if s5_pkl.exists():
        print(f"Loading {s5_pkl}...")
        return pd.read_pickle(s5_pkl)
    s2_pkl = stage2_dir / "processed_trials.pkl"
    print(f"Loading {s2_pkl}...")
    return pd.read_pickle(s2_pkl)


def _ensure_subj(td: pd.DataFrame, stage5_dir: Path) -> pd.DataFrame:
    """Make sure the DataFrame has an integer ``subj`` column."""
    if 'subj' in td.columns and td['subj'].notna().all():
        td = td.copy()
        td['subj'] = td['subj'].astype(int)
        return td
    mapping_path = stage5_dir / "subject_mapping.csv"
    if mapping_path.exists():
        mapping = pd.read_csv(mapping_path)
        if 'participantID' in mapping.columns and 'subj' in mapping.columns:
            pid_to_subj = dict(zip(mapping['participantID'], mapping['subj']))
            td = td.copy()
            td['subj'] = td['participantID'].map(pid_to_subj)
            td = td[td['subj'].notna()]
            td['subj'] = td['subj'].astype(int)
    return td


def compute_vigor_outputs(stage2_dir: Path, stage5_dir: Path, out_dir: Path,
                          exclude=None):
    """Compute vigor metrics for one sample and write all outputs."""
    stage2_dir = Path(stage2_dir)
    stage5_dir = Path(stage5_dir)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    td = _load_trial_source(stage5_dir, stage2_dir)
    td = _ensure_subj(td, stage5_dir)

    if exclude:
        td = td[~td['subj'].isin(exclude)]

    print(f"  {len(td)} trials, {td['subj'].nunique()} subjects")

    has_aligned = 'alignedEffortRate' in td.columns and td['alignedEffortRate'].notna().any()
    has_raw = 'effortRate' in td.columns and td['effortRate'].notna().any()
    print(f"  alignedEffortRate: {'YES' if has_aligned else 'NO'}")
    print(f"  effortRate (raw): {'YES' if has_raw else 'NO'}")

    print("  Computing vigor metrics...")
    all_epochs = []
    n_with_data = 0
    for _, row in td.iterrows():
        epochs = process_trial(row)
        all_epochs.extend(epochs)
        if epochs and epochs[0].get('n_presses', 0) > 0:
            n_with_data += 1

    epoch_df = pd.DataFrame(all_epochs)
    print(f"  {len(epoch_df)} epoch rows, {n_with_data}/{len(td)} trials with keypress data")

    # Trial-level vigor (full epoch only, for H1)
    trial_vigor = epoch_df[epoch_df['epoch'] == 'full'].copy()
    trial_vigor.to_csv(stage5_dir / "trial_vigor.csv", index=False)
    print(f"  Saved {stage5_dir / 'trial_vigor.csv'} ({len(trial_vigor)} trials)")

    # All epoch metrics (for H2)
    epoch_df.to_csv(out_dir / "vigor_metrics.csv", index=False)
    print(f"  Saved {out_dir / 'vigor_metrics.csv'} ({len(epoch_df)} rows)")

    # Cell means (for H3/H4/H5)
    cell_means = compute_cell_means(epoch_df)
    cell_means.to_csv(out_dir / "cell_means.csv", index=False)
    print(f"  Saved {out_dir / 'cell_means.csv'} ({len(cell_means)} cells)")

    return {
        'trial_vigor': stage5_dir / "trial_vigor.csv",
        'vigor_metrics': out_dir / "vigor_metrics.csv",
        'cell_means': out_dir / "cell_means.csv",
    }


def main():
    parser = argparse.ArgumentParser(description="Compute vigor metrics")
    parser.add_argument("--stage2_dir", type=str, default=None,
                        help="Stage 2 output dir (contains processed_trials.pkl)")
    parser.add_argument("--stage5_dir", type=str, default=None,
                        help="Stage 5 output dir (for behavior_rich.pkl + trial_vigor.csv)")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Output dir for vigor_metrics.csv and cell_means.csv. "
                             "If omitted, defaults to results/stats/vigor_analysis/<sample>/")
    parser.add_argument("--sample", type=str, default=None,
                        choices=["exploratory_350", "confirmatory_350"],
                        help="Restrict to a single sample. If omitted, runs both.")
    parser.add_argument("--exclude", type=int, nargs="*", default=[])
    args = parser.parse_args()

    # If user passed explicit dirs, run once
    if args.stage2_dir or args.stage5_dir:
        stage2_dir = Path(args.stage2_dir) if args.stage2_dir else None
        stage5_dir = Path(args.stage5_dir) if args.stage5_dir else stage2_dir
        if stage2_dir is None:
            stage2_dir = stage5_dir
        out_dir = Path(args.output_dir) if args.output_dir else Path("results/stats/vigor_analysis")
        compute_vigor_outputs(stage2_dir, stage5_dir, out_dir, exclude=args.exclude)
        print("\nDone.")
        return

    # Otherwise auto-detect samples and process each
    samples = [args.sample] if args.sample else ["exploratory_350", "confirmatory_350"]
    for sample in samples:
        base = Path(f"data/{sample}/processed")
        if not base.exists():
            print(f"[skip] {sample}: no data dir")
            continue
        stage2_dir = find_latest_dir(base, "stage2_")
        stage5_dir = find_latest_dir(base, "stage5_")
        if stage5_dir is None:
            print(f"[skip] {sample}: no stage5 dir")
            continue
        # Sample-specific output dir (avoids exploratory/confirmatory overwrite)
        sample_short = sample.replace("_350", "")
        out_dir = (Path(args.output_dir) if args.output_dir
                   else Path("results/stats/vigor_analysis") / sample_short)
        print("\n" + "=" * 60)
        print(f"Sample: {sample}")
        print(f"  stage2: {stage2_dir}")
        print(f"  stage5: {stage5_dir}")
        print(f"  output: {out_dir}")
        print("=" * 60)
        compute_vigor_outputs(stage2_dir or stage5_dir, stage5_dir, out_dir,
                              exclude=args.exclude)

    print("\nDone.")


if __name__ == "__main__":
    main()
