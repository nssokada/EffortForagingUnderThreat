"""
LIMA Preprocessing Pipeline - Stage 2: Trial Data Processing
=============================================================
Processes trial data to compute:
- Aligned effort rates and positions
- Effort metrics (binned counts, relative effort)
- Strike times and encounter times
- Choice variables

Outputs:
- behavior.csv / behavior.pkl: Core behavioral data
- behavior_rich.csv / behavior_rich.pkl: Extended behavioral metrics
- processed_trials.pkl: Full processed trial data
- window_analysis.csv / window_analysis.pkl: Window analysis results
"""

import ast
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from config import PipelineConfig, default_config
from utils import PipelineLogger, load_pickle, save_outputs, create_processing_report


# =============================================================================
# Constants
# =============================================================================

DEFAULT_WINDOW = 0.1
VALID_ATTACK_PROBS = [0.1, 0.5, 0.9]


# =============================================================================
# Alignment Functions
# =============================================================================

def correct_attacking_prob(prob: float) -> float:
    """Round attacking probability to nearest valid value (0.1, 0.5, 0.9)."""
    return min(VALID_ATTACK_PROBS, key=lambda x: abs(x - prob))


def align_to_trial(
    effort_rate: List,
    start_time: float,
    end_time: float
) -> Optional[List]:
    """Align effort rate times relative to trial start time."""
    if not isinstance(effort_rate, list):
        return None
    return [t - start_time for t in effort_rate if start_time <= t <= end_time]


def align_positions(
    positions: List[Dict],
    start_time: float,
    end_time: float
) -> Optional[List[Dict]]:
    """Align position times relative to trial start time."""
    if not isinstance(positions, list):
        return None

    aligned = []
    for pos in positions:
        if 'time' not in pos:
            continue
        if start_time <= pos['time'] <= end_time:
            new_pos = pos.copy()
            new_pos['time'] = pos['time'] - start_time
            aligned.append(new_pos)
    return aligned


# =============================================================================
# Window Calculation Functions
# =============================================================================

def calculate_optimal_windows(
    data: pd.DataFrame,
    logger: Optional[PipelineLogger] = None
) -> Tuple[pd.DataFrame, float]:
    """
    Calculate optimal window sizes based on inter-press intervals.

    Returns:
        Tuple of (participant window DataFrame, group window size)
    """
    grouped_data = data.groupby("participantID")["alignedEffortRate"].apply(list)

    results = []
    for participant_id, trial_list in grouped_data.items():
        trial_medians = []
        for trial in trial_list:
            if trial is not None and len(trial) > 1:
                trial_arr = np.sort(np.array(trial))
                trial_ipi = np.diff(trial_arr)
                trial_medians.append(np.median(trial_ipi))

        if trial_medians:
            median_ipi = np.median(trial_medians)
            window = np.round(median_ipi, 3)
        else:
            median_ipi = None
            window = None

        results.append({
            'participantID': participant_id,
            'median_ipi': median_ipi,
            'suggested_window': window
        })

    df_results = pd.DataFrame(results)
    valid_windows = [w for w in df_results['suggested_window'] if w is not None]
    group_window = np.median(valid_windows) if valid_windows else DEFAULT_WINDOW

    if logger:
        logger.log(f"Calculated windows for {len(df_results)} participants")
        logger.log(f"Group window size: {group_window:.4f}s")

    return df_results, group_window


# =============================================================================
# Effort Computation Functions
# =============================================================================

def compute_bin_counts(timestamps: List, bin_edges: np.ndarray) -> np.ndarray:
    """Compute histogram counts for timestamps."""
    if timestamps is None or not timestamps:
        return np.zeros(len(bin_edges) - 1, dtype=int)
    return np.histogram(timestamps, bins=bin_edges)[0]


def calculate_relative_effort(
    binned_counts: np.ndarray,
    press_count: float,
    window: float
) -> np.ndarray:
    """Calculate relative effort as proportion of calibrated maximum."""
    if press_count == 0 or window == 0:
        return np.zeros_like(binned_counts, dtype=float)
    calibrated_max = press_count * window
    return binned_counts / calibrated_max


def calculate_effort_metrics(
    aligned_effort_rate: List
) -> Tuple[List, List, Optional[float]]:
    """
    Calculate effort IPI and inverse latency metrics.

    Returns:
        Tuple of (IPI tuples, inverse latency tuples, mean IPI)
    """
    if not isinstance(aligned_effort_rate, list) or len(aligned_effort_rate) == 0:
        return [], [], None

    press_times = sorted(aligned_effort_rate)
    ipi_tuples = []
    inverse_latency_tuples = []

    if len(press_times) > 0:
        first_ipi = press_times[0]
        ipi_tuples.append((first_ipi, press_times[0]))
        if first_ipi > 0:
            inverse_latency_tuples.append((1.0 / first_ipi, press_times[0]))

    ipi_values = []
    for i in range(1, len(press_times)):
        ipi = press_times[i] - press_times[i - 1]
        ipi_tuples.append((ipi, press_times[i]))
        ipi_values.append(ipi)
        if ipi > 0:
            inverse_latency_tuples.append((1.0 / ipi, press_times[i]))

    mean_ipi = np.mean(ipi_values) if ipi_values else None

    return ipi_tuples, inverse_latency_tuples, mean_ipi


# =============================================================================
# Position & Time Functions
# =============================================================================

def get_predator_spawn_time(aligned_positions: List[Dict]) -> float:
    """Get predator spawn time from aligned positions."""
    if not isinstance(aligned_positions, list) or not aligned_positions:
        return -1
    return aligned_positions[0].get('time', -1)


def get_player_start_time(player_position: List[Dict]) -> float:
    """Get player start time from positions."""
    if not isinstance(player_position, list) or not player_position:
        return -1
    return player_position[0].get('time', -1)


def calculate_encounter_time(distance: float) -> float:
    """Calculate encounter time based on start distance."""
    if distance <= 5:
        return 2.5
    elif distance == 9:
        return 5.0
    return 3.5


def calculate_strike_time(
    positions: List[Dict],
    start_time: float,
    end_time: float,
    capture_time: float,
    threshold: float = 1.5,
    T: float = 0.75,
    min_dt: float = 0.5,
    max_dt: float = 1.0
) -> Dict[str, float]:
    """
    Calculate predator strike time using velocity-based detection.

    Returns:
        Dictionary with strike_time and circaStrikeTime
    """
    if capture_time == -1:
        return {'strike_time': -1, 'circaStrikeTime': -1}

    aligned = align_positions(positions, start_time, end_time)
    if aligned is None or len(aligned) == 0:
        return {'strike_time': -1, 'circaStrikeTime': -1}

    df = pd.DataFrame(aligned)
    circa_strike_time = df['time'].iloc[0]

    for i in range(1, len(df)):
        current_time = df.loc[i, 'time']
        if current_time < min_dt:
            continue

        best_candidate = None
        best_diff = float('inf')

        for j in range(i):
            dt = current_time - df.loc[j, 'time']
            if min_dt <= dt <= max_dt:
                diff = abs(dt - T)
                if diff < best_diff:
                    best_diff = diff
                    best_candidate = j

        if best_candidate is not None:
            dt = current_time - df.loc[best_candidate, 'time']
            displacement = np.sqrt(
                (df.loc[i, 'x'] - df.loc[best_candidate, 'x']) ** 2 +
                (df.loc[i, 'y'] - df.loc[best_candidate, 'y']) ** 2
            )
            speed = displacement / dt if dt != 0 else 0

            if speed > threshold:
                return {'strike_time': current_time, 'circaStrikeTime': circa_strike_time}

    # Fallback
    if capture_time is not None:
        aligned_capture = capture_time - start_time
        return {'strike_time': aligned_capture - 1.0, 'circaStrikeTime': circa_strike_time}

    return {'strike_time': -1, 'circaStrikeTime': -1}


def calculate_distance_H(row) -> float:
    """Calculate high-effort distance based on effort_H value."""
    effort_h = row.get('effort_H')
    if effort_h == 0.6:
        return 1
    elif effort_h == 0.8:
        return 2
    elif effort_h == 1.0:
        return 3
    return np.nan


# =============================================================================
# Advanced Effort Metrics Functions
# =============================================================================

def parse_effort_data(effort_cell) -> List:
    """Convert effort data from string to list if necessary."""
    if isinstance(effort_cell, str):
        try:
            return ast.literal_eval(effort_cell)
        except (ValueError, SyntaxError):
            return []
    if isinstance(effort_cell, np.ndarray):
        return effort_cell.tolist()
    return effort_cell if isinstance(effort_cell, list) else []


def get_time_window_indices(
    time_points: np.ndarray,
    start: float,
    end: float,
    reference_time: Optional[float] = None
) -> np.ndarray:
    """Get indices for data points within a specified time window."""
    if reference_time is not None:
        lower_bound = reference_time + start
        upper_bound = reference_time + end
    else:
        lower_bound = start
        upper_bound = end

    return np.where((time_points >= lower_bound) & (time_points < upper_bound))[0]


def compute_mean_effort(
    effort_cell,
    window: float,
    start: float,
    end: float,
    reference_time: Optional[float] = None
) -> float:
    """Compute mean effort over a specified time window."""
    effort = parse_effort_data(effort_cell)
    if not effort:
        return np.nan

    n_bins = len(effort)
    time_points = np.arange(n_bins) * window
    indices = get_time_window_indices(time_points, start, end, reference_time)

    valid_efforts = [val for val in np.array(effort)[indices] if val is not None]
    return np.mean(valid_efforts) if valid_efforts else np.nan


def compute_peak_effort(
    effort_cell,
    window: float,
    start: float,
    end: float,
    reference_time: Optional[float] = None
) -> float:
    """Compute peak (maximum) effort over a specified time window."""
    effort = parse_effort_data(effort_cell)
    if not effort:
        return np.nan

    n_bins = len(effort)
    time_points = np.arange(n_bins) * window
    indices = get_time_window_indices(time_points, start, end, reference_time)

    valid_efforts = [val for val in np.array(effort)[indices] if val is not None]
    return np.max(valid_efforts) if valid_efforts else np.nan


def compute_auc_effort(
    effort_cell,
    window: float,
    start: float,
    end: float,
    reference_time: Optional[float] = None
) -> float:
    """Compute AUC for effort values over a specified time window."""
    effort = parse_effort_data(effort_cell)
    if not effort:
        return np.nan

    n_bins = len(effort)
    time_points = np.arange(n_bins) * window
    indices = get_time_window_indices(time_points, start, end, reference_time)

    valid_efforts = []
    valid_times = []
    for idx in indices:
        if idx < len(effort) and effort[idx] is not None:
            valid_efforts.append(effort[idx])
            valid_times.append(time_points[idx])

    if len(valid_efforts) < 2:
        return np.nan

    return np.trapezoid(valid_efforts, valid_times)


def calculate_all_effort_metrics(
    df: pd.DataFrame,
    logger: Optional[PipelineLogger] = None
) -> pd.DataFrame:
    """
    Calculate all effort metrics for various trial phases.

    Phases:
    - preEncounter: -1.5s to 0s relative to circaStrikeTime
    - postEncounter: 0s to 1.5s relative to circaStrikeTime
    - onset: 0s to 1.5s absolute
    - circaStrike: -1.5s to 0s relative to strike_time
    - strike: 0s to 1.5s relative to strike_time
    - trial: 0s to trialEndTime
    """
    metric_configs = [
        # Pre-encounter metrics
        ('mean', 'preEncounter', 'relativeEffort_individual', 'suggested_window_size', -1.5, 0, 'circaStrikeTime'),
        ('peak', 'preEncounter', 'relativeEffort_adaptive', 'groupWindow', -1.5, 0, 'circaStrikeTime'),
        ('auc', 'preEncounter', 'relativeEffort_adaptive', 'groupWindow', -1.5, 0, 'circaStrikeTime'),
        # Post-encounter metrics
        ('mean', 'postEncounter', 'relativeEffort_individual', 'suggested_window_size', 0, 1.5, 'circaStrikeTime'),
        ('peak', 'postEncounter', 'relativeEffort_adaptive', 'groupWindow', 0, 1.5, 'circaStrikeTime'),
        ('auc', 'postEncounter', 'relativeEffort_adaptive', 'groupWindow', 0, 1.5, 'circaStrikeTime'),
        # Onset metrics (absolute time)
        ('mean', 'onset', 'relativeEffort_individual', 'suggested_window_size', 0, 1.5, None),
        ('peak', 'onset', 'relativeEffort_adaptive', 'groupWindow', 0, 1.5, None),
        ('auc', 'onset', 'relativeEffort_adaptive', 'groupWindow', 0, 1.5, None),
        # Circa-strike metrics
        ('mean', 'circaStrike', 'relativeEffort_individual', 'suggested_window_size', -1.5, 0, 'strike_time'),
        ('peak', 'circaStrike', 'relativeEffort_adaptive', 'groupWindow', -1.5, 0, 'strike_time'),
        ('auc', 'circaStrike', 'relativeEffort_adaptive', 'groupWindow', -1.5, 0, 'strike_time'),
        # Strike metrics
        ('mean', 'strike', 'relativeEffort_individual', 'suggested_window_size', 0, 1.5, 'strike_time'),
        ('peak', 'strike', 'relativeEffort_adaptive', 'groupWindow', 0, 1.5, 'strike_time'),
        ('auc', 'strike', 'relativeEffort_adaptive', 'groupWindow', 0, 1.5, 'strike_time'),
    ]

    metric_functions = {
        'mean': compute_mean_effort,
        'peak': compute_peak_effort,
        'auc': compute_auc_effort
    }

    # Calculate metrics with reference time
    for metric_type, event_type, effort_col, window_col, start, end, ref_time_col in metric_configs:
        column_name = f"{metric_type}_{event_type}_effort"
        func = metric_functions[metric_type]

        # Check required columns
        required_cols = [effort_col, window_col]
        if ref_time_col:
            required_cols.append(ref_time_col)

        if not all(col in df.columns for col in required_cols):
            continue

        if ref_time_col:
            df[column_name] = df.apply(
                lambda row, f=func, ec=effort_col, wc=window_col, s=start, e=end, rtc=ref_time_col: f(
                    row[ec], row[wc], s, e, row[rtc]
                ),
                axis=1
            )
        else:
            df[column_name] = df.apply(
                lambda row, f=func, ec=effort_col, wc=window_col, s=start, e=end: f(
                    row[ec], row[wc], s, e
                ),
                axis=1
            )

    # Calculate trial-level metrics
    if all(col in df.columns for col in ['relativeEffort_individual', 'suggested_window_size', 'trialEndTime']):
        df['mean_trial_effort'] = df.apply(
            lambda row: compute_mean_effort(
                row['relativeEffort_individual'],
                row['suggested_window_size'],
                0,
                row['trialEndTime']
            ),
            axis=1
        )

    if all(col in df.columns for col in ['relativeEffort_adaptive', 'groupWindow', 'trialEndTime']):
        df['peak_trial_effort'] = df.apply(
            lambda row: compute_peak_effort(
                row['relativeEffort_adaptive'],
                row['groupWindow'],
                0,
                row['trialEndTime']
            ),
            axis=1
        )

        df['auc_trial_effort'] = df.apply(
            lambda row: compute_auc_effort(
                row['relativeEffort_adaptive'],
                row['groupWindow'],
                0,
                row['trialEndTime']
            ),
            axis=1
        )

    if logger:
        logger.log("Calculated all effort metrics (mean, peak, AUC) for all phases")

    return df


# =============================================================================
# Processing Pipeline Steps
# =============================================================================

def step_correct_attacking_prob(
    trial_data: pd.DataFrame,
    config: PipelineConfig,
    logger: PipelineLogger,
    output_dir: Path
) -> pd.DataFrame:
    """Step 1: Correct attacking probability and filter invalid participants."""
    if 'attackingProb' not in trial_data.columns:
        logger.log("Column 'attackingProb' not found, skipping correction", "warning")
        return trial_data

    # Identify participants with invalid attackingProb (> max threshold)
    bad_participants = trial_data.loc[
        trial_data['attackingProb'] > config.max_attacking_prob, 'participantID'
    ].unique()

    if len(bad_participants) > 0:
        logger.log(f"Found {len(bad_participants)} participants with attackingProb > {config.max_attacking_prob}")

        # Save removed participants to log directory
        removed_file = output_dir / "participants_removed_attackingProb.txt"
        with open(removed_file, "w") as f:
            for pid in bad_participants:
                f.write(f"{pid}\n")
        logger.log(f"Saved removed participant IDs to {removed_file}")

        # Drop those participants
        trial_data = trial_data.loc[
            ~trial_data['participantID'].isin(bad_participants)
        ].reset_index(drop=True)
        logger.log(f"Removed {len(bad_participants)} participants, {len(trial_data)} rows remaining")

    # Snap remaining values to valid threat levels
    trial_data['attackingProb'] = trial_data['attackingProb'].apply(correct_attacking_prob)
    logger.log("Corrected attackingProb values to nearest valid level")

    # Rename to threat
    trial_data = trial_data.rename(columns={'attackingProb': 'threat'})
    logger.log("Renamed 'attackingProb' to 'threat'")

    return trial_data


def step_merge_calibration(
    trial_data: pd.DataFrame,
    experiment_info: pd.DataFrame,
    logger: PipelineLogger
) -> pd.DataFrame:
    """Step 2: Merge with experiment info for calibration data."""
    if experiment_info.empty:
        logger.log("Experiment info is empty, skipping calibration merge", "warning")
        return trial_data

    calibration_cols = ['participantID', 'effortPressCount', 'effortPressLatency']
    available_cols = [c for c in calibration_cols if c in experiment_info.columns]

    if len(available_cols) < 2:
        logger.log("Required calibration columns not found in experiment info", "warning")
        return trial_data

    calibration_data = experiment_info[available_cols].copy()
    calibration_data = calibration_data.drop_duplicates(subset=['participantID'])
    trial_data = trial_data.merge(calibration_data, on='participantID', how='left')
    logger.log(f"Merged calibration data ({len(calibration_data)} participants)")

    return trial_data


def step_filter_by_press_count(
    trial_data: pd.DataFrame,
    config: PipelineConfig,
    logger: PipelineLogger
) -> pd.DataFrame:
    """Step 3: Filter participants by press count threshold."""
    if 'effortPressCount' not in trial_data.columns:
        logger.log("Column 'effortPressCount' not found, skipping filter", "warning")
        return trial_data

    initial_participants = trial_data['participantID'].nunique()
    valid_participants = trial_data[
        trial_data['effortPressCount'] <= config.press_count_threshold
    ]['participantID'].unique()

    trial_data = trial_data[trial_data['participantID'].isin(valid_participants)]
    removed = initial_participants - len(valid_participants)

    logger.log(f"Press count filter (threshold={config.press_count_threshold}): "
               f"kept {len(valid_participants)}, removed {removed} participants")
    logger.log(f"Rows remaining: {len(trial_data)}")

    return trial_data


def step_align_effort_rates(
    trial_data: pd.DataFrame,
    logger: PipelineLogger
) -> pd.DataFrame:
    """Step 4-5: Extract start times and align effort rates."""
    # Get player effort start time
    if 'playerPosition' in trial_data.columns:
        trial_data['playerEffortStartTime'] = trial_data['playerPosition'].apply(get_player_start_time)
        logger.log("Extracted player effort start times")

    # Align effort rates
    required_cols = ['effortRate', 'playerEffortStartTime', 'trialEndTime']
    if all(col in trial_data.columns for col in required_cols):
        trial_data['alignedEffortRate'] = trial_data.apply(
            lambda row: align_to_trial(
                row['effortRate'],
                row['playerEffortStartTime'],
                row['trialEndTime']
            ),
            axis=1
        )
        trial_data['firstEffortTime'] = trial_data['alignedEffortRate'].apply(
            lambda x: min(x) if x and len(x) > 0 else None
        )
        logger.log("Aligned effort rates to trial windows")

    return trial_data


def step_calculate_windows_and_metrics(
    trial_data: pd.DataFrame,
    logger: PipelineLogger
) -> Tuple[pd.DataFrame, pd.DataFrame, float, Dict]:
    """Step 6: Calculate optimal windows and effort metrics."""
    if 'alignedEffortRate' not in trial_data.columns:
        logger.log("alignedEffortRate not found, using default window", "warning")
        return trial_data, pd.DataFrame(), DEFAULT_WINDOW, {}

    df_windows, group_window = calculate_optimal_windows(trial_data, logger)
    participant_windows = dict(zip(df_windows['participantID'], df_windows['suggested_window']))

    # Calculate IPI metrics
    trial_data[['effortIPI', 'effortInverseLatency', 'meanIPI']] = trial_data['alignedEffortRate'].apply(
        lambda x: pd.Series(calculate_effort_metrics(x))
    )
    logger.log("Calculated effort IPI and inverse latency metrics")

    return trial_data, df_windows, group_window, participant_windows


def step_calculate_calibration_max(
    trial_data: pd.DataFrame,
    logger: PipelineLogger
) -> pd.DataFrame:
    """Step 7: Calculate calibration max (presses per second)."""
    if 'effortPressCount' not in trial_data.columns:
        logger.log("effortPressCount not found, skipping calibrationMax", "warning")
        return trial_data

    # Calculate mean press count per participant and convert to rate (10s calibration)
    calibration_max = trial_data.groupby('participantID')['effortPressCount'].mean() / 10
    calibration_max = calibration_max.reset_index()
    calibration_max.columns = ['participantID', 'calibrationMax']

    trial_data = trial_data.merge(calibration_max, on='participantID', how='left')
    mean_cal = trial_data['calibrationMax'].mean()
    logger.log(f"Calculated calibrationMax (press rate/sec): mean={mean_cal:.2f}")

    return trial_data


def step_compute_binned_effort(
    trial_data: pd.DataFrame,
    group_window: float,
    participant_windows: Dict,
    logger: PipelineLogger
) -> pd.DataFrame:
    """Step: Compute binned counts and relative effort."""
    required_cols = ['alignedEffortRate', 'trialEndTime']
    if not all(col in trial_data.columns for col in required_cols):
        logger.log("Required columns for binned effort not found", "warning")
        return trial_data

    def get_participant_window(pid):
        w = participant_windows.get(pid)
        if w is None or pd.isna(w) or w <= 0:
            return group_window
        return w

    def compute_bins_for_row(row, window):
        if window is None or pd.isna(window) or window <= 0:
            window = DEFAULT_WINDOW

        max_time = row['trialEndTime'] - row.get('playerEffortStartTime', 0)
        if pd.isna(max_time) or max_time <= 0:
            return np.array([])
        bin_edges = np.arange(0, max_time + window, window)
        return compute_bin_counts(row['alignedEffortRate'], bin_edges)

    # Adaptive (group window)
    trial_data['binnedCounts_adaptive'] = trial_data.apply(
        lambda row: compute_bins_for_row(row, group_window), axis=1
    )

    # Individual windows
    trial_data['binnedCounts_individual'] = trial_data.apply(
        lambda row: compute_bins_for_row(row, get_participant_window(row['participantID'])),
        axis=1
    )

    # Calculate relative effort
    trial_data['relativeEffort_adaptive'] = trial_data.apply(
        lambda row: calculate_relative_effort(
            row['binnedCounts_adaptive'],
            row.get('calibrationMax', 0),
            group_window
        ),
        axis=1
    )

    trial_data['relativeEffort_individual'] = trial_data.apply(
        lambda row: calculate_relative_effort(
            row['binnedCounts_individual'],
            row.get('calibrationMax', 0),
            get_participant_window(row['participantID'])
        ),
        axis=1
    )

    logger.log("Computed binned counts and relative effort (adaptive & individual)")

    return trial_data


def step_align_positions(
    trial_data: pd.DataFrame,
    logger: PipelineLogger
) -> pd.DataFrame:
    """Step 8: Align predator and player positions."""
    required = ['playerEffortStartTime', 'trialEndTime']

    if all(col in trial_data.columns for col in required + ['predatorPosition']):
        trial_data['alignedPredatorPosition'] = trial_data.apply(
            lambda row: align_positions(
                row['predatorPosition'],
                row['playerEffortStartTime'],
                row['trialEndTime']
            ),
            axis=1
        )
        logger.log("Aligned predator positions")

    if all(col in trial_data.columns for col in required + ['playerPosition']):
        trial_data['alignedPlayerPosition'] = trial_data.apply(
            lambda row: align_positions(
                row['playerPosition'],
                row['playerEffortStartTime'],
                row['trialEndTime']
            ),
            axis=1
        )
        logger.log("Aligned player positions")

    return trial_data


def step_calculate_strike_times(
    trial_data: pd.DataFrame,
    config: PipelineConfig,
    logger: PipelineLogger
) -> pd.DataFrame:
    """Step 9: Calculate strike times."""
    required = ['predatorPosition', 'playerEffortStartTime', 'trialEndTime', 'trialCaptureTime']
    if not all(col in trial_data.columns for col in required):
        logger.log("Required columns for strike time not found", "warning")
        return trial_data

    strike_results = trial_data.apply(
        lambda row: calculate_strike_time(
            row['predatorPosition'],
            row['playerEffortStartTime'],
            row['trialEndTime'],
            row.get('trialCaptureTime', -1),
            threshold=config.strike_threshold,
            T=config.strike_T,
            min_dt=config.strike_min_dt,
            max_dt=config.strike_max_dt
        ),
        axis=1
    )
    trial_data['strike_time'] = strike_results.apply(lambda x: x['strike_time'])
    trial_data['circaStrikeTime'] = strike_results.apply(lambda x: x['circaStrikeTime'])
    logger.log("Calculated strike times")

    return trial_data


def step_process_start_distance(
    trial_data: pd.DataFrame,
    logger: PipelineLogger
) -> pd.DataFrame:
    """Step 10: Process start distance and predator times."""
    if 'trialCookie_weight' not in trial_data.columns or 'cookie1PosX' not in trial_data.columns:
        logger.log("Cookie columns not found, skipping start distance", "warning")
        return trial_data

    mapping = {0: 5, 1: 7, 2: 9}
    trial_data['startDistance'] = np.where(
        trial_data['trialCookie_weight'] == 3,
        trial_data['cookie1PosX'].replace(mapping),
        trial_data['cookie2PosX'].replace(mapping)
    )

    if 'alignedPredatorPosition' in trial_data.columns:
        trial_data['predatorSpawnTime'] = trial_data['alignedPredatorPosition'].apply(
            get_predator_spawn_time
        )

    trial_data['encounterTime'] = trial_data['startDistance'].apply(calculate_encounter_time)
    trial_data['predatorAttackTime'] = trial_data['startDistance'].apply(
        lambda d: 2 * calculate_encounter_time(d)
    )

    # Fill missing strike times
    if 'circaStrikeTime' in trial_data.columns:
        mask = trial_data['circaStrikeTime'] == -1
        trial_data.loc[mask, 'circaStrikeTime'] = trial_data.loc[mask, 'encounterTime']

    if 'strike_time' in trial_data.columns:
        mask = trial_data['strike_time'] == -1
        trial_data.loc[mask, 'strike_time'] = trial_data.loc[mask, 'predatorAttackTime']

    logger.log("Processed start distance and predator timing")

    return trial_data


def step_add_choice_variables(
    trial_data: pd.DataFrame,
    logger: PipelineLogger
) -> pd.DataFrame:
    """Step 11: Add choice variables."""
    if 'trialCookie_weight' not in trial_data.columns:
        logger.log("trialCookie_weight not found, skipping choice variables", "warning")
        return trial_data

    trial_data['choice'] = np.where(trial_data['trialCookie_weight'] == 3, 1, 0)

    if 'cookie1PosX' in trial_data.columns and 'cookie2PosX' in trial_data.columns:
        trial_data['choice_distDifference'] = trial_data['cookie1PosX'] - trial_data['cookie2PosX']

    logger.log("Added choice variables")

    return trial_data


def step_adjust_time_variables(
    trial_data: pd.DataFrame,
    logger: PipelineLogger
) -> pd.DataFrame:
    """Step 12: Adjust time variables relative to effort start."""
    time_cols = ['trialCaptureTime', 'trialEndTime', 'trialEscapeTime']

    for time_col in time_cols:
        if time_col in trial_data.columns and 'playerEffortStartTime' in trial_data.columns:
            trial_data[time_col] = trial_data.apply(
                lambda row, tc=time_col: (
                    row[tc] - row['playerEffortStartTime']
                    if pd.notnull(row[tc]) and row[tc] != -1 else -1
                ),
                axis=1
            )

    if 'trialCookie_time' in trial_data.columns and 'trialStartTime' in trial_data.columns:
        trial_data['RTTime_choice'] = trial_data['trialCookie_time'] - trial_data['trialStartTime']

    logger.log("Adjusted time variables relative to effort start")

    return trial_data


def step_add_window_info(
    trial_data: pd.DataFrame,
    df_windows: pd.DataFrame,
    group_window: float,
    logger: PipelineLogger
) -> pd.DataFrame:
    """Step 13: Add window size information."""
    if not df_windows.empty:
        trial_data = trial_data.merge(
            df_windows[['participantID', 'suggested_window']],
            on='participantID',
            how='left'
        )
        trial_data.rename(columns={'suggested_window': 'suggested_window_size'}, inplace=True)

    trial_data['groupWindow'] = group_window
    logger.log(f"Added window info (group={group_window:.4f})")

    return trial_data


def step_compute_derived_columns(
    trial_data: pd.DataFrame,
    logger: PipelineLogger
) -> pd.DataFrame:
    """Compute effort_level, effort_diff, trial number, and outcome."""
    # Effort level and diff
    if 'choice' in trial_data.columns and 'choice_distDifference' in trial_data.columns:
        trial_data['effort_level'] = np.where(
            trial_data['choice'] == 0,
            0.4,
            0.6 + 0.2 * trial_data['choice_distDifference']
        )
        trial_data['effort_diff'] = np.where(
            trial_data['choice_distDifference'] == 0,
            0.2,
            0.2 + 0.2 * trial_data['choice_distDifference']
        )
        logger.log("Computed effort_level and effort_diff")

    # Trial number
    if 'trialName' in trial_data.columns:
        trial_data['trial'] = trial_data['trialName'].str.extract(r'(\d+)').astype(int)
        logger.log("Extracted trial numbers")

    # Outcome encoding
    if 'trialEndState' in trial_data.columns:
        categorical_outcome = pd.Categorical(trial_data['trialEndState'])
        trial_data['outcome'] = categorical_outcome.codes
        # CAPTURE/FAIL=1, ESCAPE=0
        trial_data['outcome'] = np.where(trial_data['outcome'] <= 0, 1, 0)
        logger.log("Encoded outcome (CAPTURE/FAIL=1, ESCAPE=0)")

    # Effort L/H reconstruction
    if all(c in trial_data.columns for c in ['effort_diff', 'effort_level', 'choice']):
        diff = trial_data['effort_diff'].astype(float).abs()
        base = trial_data['effort_level'].astype(float)
        choice = trial_data['choice'].astype(int)

        trial_data['effort_H'] = np.where(choice == 1, base, base + diff).round(1)
        trial_data['effort_L'] = np.where(choice == 1, base - diff, base).round(1)
        logger.log("Reconstructed effort_L and effort_H")

    # Distance H/L
    if 'effort_H' in trial_data.columns:
        trial_data['distance_H'] = trial_data.apply(calculate_distance_H, axis=1)
        trial_data['distance_L'] = 1
        logger.log("Computed distance_H and distance_L")

    return trial_data


def create_behavior_dataframe(
    trial_data: pd.DataFrame,
    logger: PipelineLogger
) -> pd.DataFrame:
    """Create the core behavior DataFrame."""
    cols = [
        'participantID', 'type', 'trial', 'threat',
        'effort_L', 'effort_H', 'distance_L', 'distance_H',
        'choice', 'outcome'
    ]
    available_cols = [c for c in cols if c in trial_data.columns]

    behavior = trial_data[available_cols].copy()

    # Filter for type==1 trials only
    if 'type' in behavior.columns:
        behavior = behavior.loc[behavior['type'].eq(1)].copy()
        behavior = behavior.drop(columns=['type'])

    # Sort and renumber trials per participant
    behavior = behavior.sort_values(['participantID', 'trial'])
    behavior['trial'] = behavior.groupby('participantID').cumcount() + 1

    logger.log(f"Created behavior DataFrame: {len(behavior)} rows, "
               f"{behavior['participantID'].nunique()} participants")

    return behavior


def create_behavior_rich_dataframe(
    trial_data: pd.DataFrame,
    logger: PipelineLogger
) -> pd.DataFrame:
    """Create the extended behavior_rich DataFrame."""
    cols_rich = [
        # Identity + condition
        'participantID', 'trial', 'threat',
        'effort_L', 'effort_H', 'distance_L', 'distance_H',
        'choice', 'outcome', 'type', 'trialEndState', 'isAttackTrial',
        # Cookie properties
        'trialCookie_weight', 'trialCookie_rewardValue', 'trialReward',
        # Timing
        'trialEndTime', 'encounterTime', 'strikeTime', 'circaStrikeTime',
        # Calibration + keypresses
        'effortPressCount', 'effortPressLatency', 'firstEffortTime', 'RTTime_choice',
        'calibrationMax', 'alignedEffortRate',
        # Epoch effort metrics
        'mean_preEncounter_effort', 'peak_preEncounter_effort', 'auc_preEncounter_effort',
        'mean_postEncounter_effort', 'peak_postEncounter_effort', 'auc_postEncounter_effort',
        'mean_onset_effort', 'peak_onset_effort', 'auc_onset_effort',
        'mean_circaStrike_effort', 'peak_circaStrike_effort', 'auc_circaStrike_effort',
        'mean_strike_effort', 'peak_strike_effort', 'auc_strike_effort',
        'mean_trial_effort', 'peak_trial_effort', 'auc_trial_effort',
    ]
    available_cols = [c for c in cols_rich if c in trial_data.columns]

    behavior_rich = trial_data[available_cols].copy()

    # Rename columns for clarity
    rename_map = {
        'effortPressCount': 'press_count_calibration',
        'effortPressLatency': 'press_latency_calibration',
        'firstEffortTime': 'first_press_trial',
        'RTTime_choice': 'choice_RT',
    }
    behavior_rich = behavior_rich.rename(columns={
        k: v for k, v in rename_map.items() if k in behavior_rich.columns
    })

    logger.log(f"Created behavior_rich DataFrame: {len(behavior_rich)} rows, "
               f"{len(available_cols)} columns")

    return behavior_rich


# =============================================================================
# Main Processing Function
# =============================================================================

def run_stage2(
    trial_data_path: Path = None,
    experiment_info_path: Path = None,
    output_dir: Path = None,
    config: PipelineConfig = None
) -> Dict[str, Dict[str, Path]]:
    """
    Run Stage 2: Trial data processing.

    Parameters:
        trial_data_path: Path to trial data pickle
        experiment_info_path: Path to experiment info pickle
        output_dir: Output directory
        config: Pipeline configuration

    Returns:
        Dictionary of output file paths
    """
    config = config or default_config
    output_dir = Path(output_dir or config.stage2_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger = PipelineLogger("Stage2_TrialProcessing", output_dir)
    logger.log("=" * 60)
    logger.log("Starting Stage 2: Trial Data Processing")
    logger.log("=" * 60)

    # Load data
    trial_data = load_pickle(trial_data_path, logger)
    experiment_info = load_pickle(experiment_info_path, logger)

    initial_rows = len(trial_data)
    initial_participants = trial_data['participantID'].nunique()
    logger.log(f"Loaded {initial_rows} trial rows from {initial_participants} participants")

    # ==========================================================================
    # Processing Steps
    # ==========================================================================

    # Step 1: Correct attacking probability
    trial_data = step_correct_attacking_prob(trial_data, config, logger, output_dir)

    # Step 2: Merge calibration data
    trial_data = step_merge_calibration(trial_data, experiment_info, logger)

    # Step 3: Filter by press count
    trial_data = step_filter_by_press_count(trial_data, config, logger)

    # Step 4-5: Align effort rates
    trial_data = step_align_effort_rates(trial_data, logger)

    # Step 6: Calculate windows and IPI metrics
    trial_data, df_windows, group_window, participant_windows = step_calculate_windows_and_metrics(
        trial_data, logger
    )

    # Step 7: Calculate calibration max
    trial_data = step_calculate_calibration_max(trial_data, logger)

    # Compute binned effort
    trial_data = step_compute_binned_effort(trial_data, group_window, participant_windows, logger)

    # Step 8: Align positions
    trial_data = step_align_positions(trial_data, logger)

    # Step 9: Calculate strike times
    trial_data = step_calculate_strike_times(trial_data, config, logger)

    # Step 10: Process start distance
    trial_data = step_process_start_distance(trial_data, logger)

    # Step 11: Add choice variables
    trial_data = step_add_choice_variables(trial_data, logger)

    # Step 12: Adjust time variables
    trial_data = step_adjust_time_variables(trial_data, logger)

    # Step 13: Add window info
    trial_data = step_add_window_info(trial_data, df_windows, group_window, logger)

    # Compute derived columns
    trial_data = step_compute_derived_columns(trial_data, logger)

    # Calculate all effort metrics
    trial_data = calculate_all_effort_metrics(trial_data, logger)

    # ==========================================================================
    # Create Output DataFrames
    # ==========================================================================

    logger.log("-" * 40)
    logger.log("Creating output DataFrames")

    # Create behavior DataFrame
    behavior = create_behavior_dataframe(trial_data, logger)

    # Create behavior_rich DataFrame
    behavior_rich = create_behavior_rich_dataframe(trial_data, logger)

    # ==========================================================================
    # Save Outputs
    # ==========================================================================

    logger.log("-" * 40)
    logger.log("Saving outputs")

    outputs = {}

    # Save behavior (core behavioral data)
    outputs['behavior'] = save_outputs(behavior, output_dir, 'behavior', logger=logger)

    # Save behavior_rich (extended metrics)
    outputs['behavior_rich'] = save_outputs(behavior_rich, output_dir, 'behavior_rich', logger=logger)

    # Save full processed data (pickle only for internal use)
    outputs['processed'] = save_outputs(
        trial_data, output_dir, 'processed_trials',
        save_csv=False, save_pickle=True, logger=logger
    )

    # Save window analysis
    if not df_windows.empty:
        df_windows['group_window'] = group_window
        outputs['windows'] = save_outputs(df_windows, output_dir, 'window_analysis', logger=logger)

    # ==========================================================================
    # Create Processing Report
    # ==========================================================================

    final_participants = trial_data['participantID'].nunique()

    input_info = {
        'trial_data_path': str(trial_data_path),
        'experiment_info_path': str(experiment_info_path),
        'initial_rows': initial_rows,
        'initial_participants': initial_participants
    }

    output_info = {
        'final_rows': len(trial_data),
        'final_participants': final_participants,
        'behavior_rows': len(behavior),
        'behavior_rich_rows': len(behavior_rich),
        'output_directory': str(output_dir)
    }

    processing_stats = {
        'group_window_size': group_window,
        'participants_removed': initial_participants - final_participants,
        'threat_values': sorted(trial_data['threat'].unique().tolist()) if 'threat' in trial_data.columns else [],
        'trial_end_states': trial_data['trialEndState'].value_counts().to_dict() if 'trialEndState' in trial_data.columns else {}
    }

    create_processing_report(output_dir, "stage2", input_info, output_info, processing_stats, logger)

    # Save log
    logger.save_log()

    logger.log("=" * 60)
    logger.log("Stage 2 complete!")
    logger.log(f"Outputs: behavior.csv, behavior_rich.csv + pickles")
    logger.log("=" * 60)

    return outputs


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run Stage 2: Trial Data Processing")
    parser.add_argument("--trial-data", "-t", required=True, help="Path to trial data pickle")
    parser.add_argument("--experiment-info", "-e", required=True, help="Path to experiment info pickle")
    parser.add_argument("--output", "-o", help="Path to output directory")

    args = parser.parse_args()

    config = PipelineConfig()
    run_stage2(Path(args.trial_data), Path(args.experiment_info), args.output, config)