"""
Vigor computation from behavior_rich.csv (Stage 5 output).

Produces three files needed by the analysis notebooks:
  1. trial_vigor.csv     — per-trial normalized press rate (H1)
  2. vigor_metrics.csv   — per-trial × per-epoch vigor metrics (H2)
  3. cell_means.csv      — per-subject condition cell means (H3, H4, H5)

Usage:
  python scripts/preprocessing/compute_vigor.py --stage5_dir <path>
  python scripts/preprocessing/compute_vigor.py  # auto-detect
"""

import argparse
import numpy as np
import pandas as pd
import ast
from pathlib import Path


def find_stage5_dir():
    """Auto-detect latest stage5 output."""
    base = Path("data/exploratory_350/processed")
    candidates = sorted(base.glob("stage5_*"))
    return candidates[-1] if candidates else None


def parse_effort_rates(val):
    """Parse alignedEffortRate from string to list of floats."""
    if isinstance(val, list):
        return val
    if isinstance(val, str):
        try:
            return ast.literal_eval(val)
        except (ValueError, SyntaxError):
            return []
    return []


def compute_ipi_metrics(rates, calibration_max):
    """Compute IPI-based vigor metrics from a list of press timestamps."""
    if len(rates) < 2 or calibration_max <= 0:
        return {'norm_rate': np.nan, 'n_presses': len(rates),
                'median_ipi': np.nan, 'press_sd': np.nan,
                'press_cv': np.nan, 'frac_full': np.nan,
                'relative_vigor': np.nan, 'pause_freq': np.nan}

    ipis = np.diff(rates)
    # Remove artifacts < 10ms
    ipis = ipis[ipis >= 0.01]
    if len(ipis) == 0:
        return {'norm_rate': np.nan, 'n_presses': len(rates),
                'median_ipi': np.nan, 'press_sd': np.nan,
                'press_cv': np.nan, 'frac_full': np.nan,
                'relative_vigor': np.nan, 'pause_freq': np.nan}

    inst_rate = 1.0 / ipis
    norm_rate = inst_rate / calibration_max
    median_ipi = np.median(ipis)
    med_norm = np.median(norm_rate)
    sd_norm = np.std(norm_rate)

    return {
        'norm_rate': med_norm,
        'median_ipi': median_ipi,
        'n_presses': len(rates),
        'press_sd': sd_norm,
        'press_cv': sd_norm / np.mean(norm_rate) if np.mean(norm_rate) > 0 else np.nan,
        'frac_full': np.nan,  # filled per-epoch with req
        'relative_vigor': np.nan,
        'pause_freq': np.mean(ipis > max(2 * median_ipi, 0.5)),
    }


def compute_epoch_metrics(rates, calibration_max, t_start, t_end, req=None):
    """Compute vigor metrics for a time epoch."""
    if not rates or len(rates) < 2:
        return compute_ipi_metrics([], calibration_max)

    rates_arr = np.array(rates)
    in_epoch = rates_arr[(rates_arr >= t_start) & (rates_arr <= t_end)]
    result = compute_ipi_metrics(in_epoch.tolist(), calibration_max)

    if req and result['norm_rate'] is not np.nan and not np.isnan(result.get('norm_rate', np.nan)):
        result['relative_vigor'] = result['norm_rate'] / req if req > 0 else np.nan
        ipis = np.diff(in_epoch)
        ipis = ipis[ipis >= 0.01]
        if len(ipis) > 0:
            inst_norm = (1.0 / ipis) / calibration_max
            result['frac_full'] = np.mean(inst_norm >= req)

    return result


def process_trial_vigor(beh):
    """Compute per-trial full-trial vigor (for H1)."""
    rows = []
    for _, trial in beh.iterrows():
        rates = parse_effort_rates(trial.get('alignedEffortRate', []))
        cal = trial.get('calibrationMax', 1.0)
        if cal <= 0:
            cal = 1.0

        m = compute_ipi_metrics(rates, cal)
        req = 0.9 if trial.get('trialCookie_weight', 1) == 3 else 0.4
        if not np.isnan(m['norm_rate']):
            m['relative_vigor'] = m['norm_rate'] / req
            ipis = np.diff([r for r in np.array(rates) if r >= 0])
            ipis = ipis[ipis >= 0.01]
            if len(ipis) > 0:
                inst_norm = (1.0 / ipis) / cal
                m['frac_full'] = float(np.mean(inst_norm >= req))

        rows.append({
            'subj': trial['subj'],
            'trial': trial.get('trialNumber', trial.name),
            'T_round': round(trial.get('threat', trial.get('attackingProb', 0)), 1),
            'distance': trial.get('distance', 0),
            'cookie': 1 if trial.get('trialCookie_weight', 1) == 3 else 0,
            'is_attack': int(trial.get('isAttackTrial', 0)),
            'type': int(trial.get('type', 0)),
            **m,
        })
    return pd.DataFrame(rows)


def process_epoch_metrics(beh):
    """Compute per-trial × per-epoch vigor metrics (for H2)."""
    rows = []
    for _, trial in beh.iterrows():
        rates = parse_effort_rates(trial.get('alignedEffortRate', []))
        cal = trial.get('calibrationMax', 1.0)
        if cal <= 0:
            cal = 1.0

        req = 0.9 if trial.get('trialCookie_weight', 1) == 3 else 0.4
        enc_t = pd.to_numeric(trial.get('encounterTime', np.nan), errors='coerce')
        trial_end = pd.to_numeric(trial.get('trialEndTime', np.nan), errors='coerce')
        strike_t = pd.to_numeric(trial.get('strikeTime', np.nan), errors='coerce')
        trial_start = min(rates) if rates else 0

        base_info = {
            'subj': trial['subj'],
            'trial': trial.get('trialNumber', trial.name),
            'T_round': round(trial.get('threat', trial.get('attackingProb', 0)), 1),
            'distance': trial.get('distance', 0),
            'cookie': 1 if trial.get('trialCookie_weight', 1) == 3 else 0,
            'is_attack': int(trial.get('isAttackTrial', 0)),
            'type': int(trial.get('type', 0)),
        }

        # Full trial
        m = compute_epoch_metrics(rates, cal, trial_start, trial_end if not np.isnan(trial_end) else 999, req)
        rows.append({**base_info, 'epoch': 'full', **m})

        if not np.isnan(enc_t):
            # Onset: trial start to encounter
            m = compute_epoch_metrics(rates, cal, trial_start, enc_t, req)
            rows.append({**base_info, 'epoch': 'onset', **m})

            # Anticipatory: 1s before encounter
            m = compute_epoch_metrics(rates, cal, max(enc_t - 1.0, trial_start), enc_t, req)
            rows.append({**base_info, 'epoch': 'anticipatory', **m})

            # Reactive: encounter to encounter + 2s
            m = compute_epoch_metrics(rates, cal, enc_t, enc_t + 2.0, req)
            rows.append({**base_info, 'epoch': 'reactive', **m})

        if not np.isnan(strike_t) and trial.get('isAttackTrial', 0):
            # Terminal: 2s before strike
            m = compute_epoch_metrics(rates, cal, max(strike_t - 2.0, trial_start), strike_t, req)
            rows.append({**base_info, 'epoch': 'terminal', **m})

    return pd.DataFrame(rows)


def compute_cell_means(vigor_df):
    """Compute per-subject condition cell means (for H3/H4/H5 model fitting)."""
    # Use full-epoch, choice trials only
    full = vigor_df[(vigor_df['epoch'] == 'full') & (vigor_df['type'] == 1)].copy()
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

    # Relative rate and cookie-centered rate
    subj_cookie_mean = cells.groupby(['subj', 'is_heavy'])['mean_rate'].transform('mean')
    cells['subj_cookie_mean'] = subj_cookie_mean
    cells['rel_rate'] = cells['mean_rate'] / cells['subj_cookie_mean']
    cells['rel_rate_cc'] = cells['rel_rate']  # cookie-centered

    return cells


def main():
    parser = argparse.ArgumentParser(description="Compute vigor metrics from Stage 5 output")
    parser.add_argument("--stage5_dir", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default="results/stats/vigor_analysis")
    parser.add_argument("--exclude", type=int, nargs="*", default=[154, 197, 208])
    args = parser.parse_args()

    stage5_dir = Path(args.stage5_dir) if args.stage5_dir else find_stage5_dir()
    if stage5_dir is None:
        print("ERROR: No stage5 output found. Run preprocessing first.")
        return

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading behavior_rich.csv from {stage5_dir}")
    beh = pd.read_csv(stage5_dir / "behavior_rich.csv", low_memory=False)
    beh = beh[~beh['subj'].isin(args.exclude)]
    print(f"  {len(beh)} trials, {beh['subj'].nunique()} subjects")

    # 1. Trial-level vigor (H1)
    print("\nComputing trial-level vigor...")
    trial_vigor = process_trial_vigor(beh)
    trial_vigor.to_csv(stage5_dir / "trial_vigor.csv", index=False)
    print(f"  Saved {stage5_dir / 'trial_vigor.csv'} ({len(trial_vigor)} trials)")

    # 2. Epoch metrics (H2)
    print("\nComputing epoch metrics...")
    epoch_metrics = process_epoch_metrics(beh)
    epoch_metrics.to_csv(out_dir / "vigor_metrics.csv", index=False)
    print(f"  Saved {out_dir / 'vigor_metrics.csv'} ({len(epoch_metrics)} rows)")

    # 3. Cell means (H3/H4/H5)
    print("\nComputing cell means...")
    cell_means = compute_cell_means(epoch_metrics)
    cell_means.to_csv(out_dir / "cell_means.csv", index=False)
    print(f"  Saved {out_dir / 'cell_means.csv'} ({len(cell_means)} cells)")

    print("\nDone.")


if __name__ == "__main__":
    main()
