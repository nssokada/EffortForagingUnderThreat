#!/usr/bin/env python3
"""
Compute all 7 vigor metrics per trial per epoch from raw keypresses.
Saves one big DataFrame that all downstream analyses use.
"""

import warnings; warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import ast
from pathlib import Path

DATA_DIR = Path("/workspace/data/exploratory_350/processed/stage5_filtered_data_20260320_191950")
OUT_DIR = Path("/workspace/results/stats/vigor_analysis")
OUT_DIR.mkdir(parents=True, exist_ok=True)
EXCLUDE = [154, 197, 208]


def compute_epoch_metrics(timestamps, cal_max, req, epoch_start, epoch_end):
    """Compute all 7 metrics for one epoch of one trial."""
    pts = timestamps[(timestamps >= epoch_start) & (timestamps < epoch_end)]
    if len(pts) < 3:
        return {k: np.nan for k in ['median_ipi', 'norm_rate', 'relative_vigor',
                                     'frac_full', 'press_sd', 'press_cv',
                                     'pause_freq', 'n_presses']}

    ipis = np.diff(pts)
    ipis = ipis[ipis > 0.01]  # filter artifacts
    if len(ipis) < 2:
        return {k: np.nan for k in ['median_ipi', 'norm_rate', 'relative_vigor',
                                     'frac_full', 'press_sd', 'press_cv',
                                     'pause_freq', 'n_presses']}

    rates = (1.0 / ipis) / cal_max  # normalized instantaneous rates

    median_ipi = np.median(ipis)
    norm_rate = np.median(rates)
    relative_vigor = norm_rate / req if req > 0 else np.nan
    frac_full = np.mean(rates >= req)
    press_sd = np.std(rates)
    press_cv = press_sd / np.mean(rates) if np.mean(rates) > 0 else np.nan

    # Pause frequency: IPIs > 2× median or > 0.5s
    threshold = max(2 * median_ipi, 0.5)
    pause_freq = np.mean(ipis > threshold)

    return {
        'median_ipi': median_ipi,
        'norm_rate': norm_rate,
        'relative_vigor': relative_vigor,
        'frac_full': frac_full,
        'press_sd': press_sd,
        'press_cv': press_cv,
        'pause_freq': pause_freq,
        'n_presses': len(pts),
    }


def run():
    print("Loading data...")
    beh = pd.read_csv(DATA_DIR / "behavior_rich.csv", low_memory=False)
    beh = beh[~beh['subj'].isin(EXCLUDE)].copy()
    beh['T_round'] = beh['threat'].round(1)
    beh['actual_dist'] = beh['startDistance'].map({5: 1, 7: 2, 9: 3})
    beh['is_heavy'] = (beh['trialCookie_weight'] == 3.0).astype(int)
    beh['actual_req'] = np.where(beh['is_heavy'] == 1, 0.9, 0.4)
    beh['enc_t'] = pd.to_numeric(beh['encounterTime'], errors='coerce')
    beh['strike_t'] = pd.to_numeric(beh['strike_time'], errors='coerce')
    beh['is_attack'] = beh['isAttackTrial'].astype(int)

    print(f"Trials: {len(beh)}, Subjects: {beh['subj'].nunique()}")

    # Compute metrics for each epoch
    records = []
    for idx, row in beh.iterrows():
        try:
            pt = np.array(ast.literal_eval(str(row['alignedEffortRate'])), dtype=float)
            if len(pt) < 5:
                continue

            cal = row['calibrationMax']
            req = row['actual_req']
            enc = row['enc_t']
            strike = row['strike_t']

            if cal <= 0 or pd.isna(enc):
                continue

            base = {
                'subj': row['subj'], 'trial': row['trial'],
                'T_round': row['T_round'], 'distance': row['actual_dist'],
                'cookie': row['is_heavy'], 'is_attack': row['is_attack'],
                'type': row['type'],
            }

            # Full trial
            full = compute_epoch_metrics(pt, cal, req, pt[0], pt[-1] + 0.01)
            r_full = {**base, 'epoch': 'full'}
            r_full.update({f'{k}': v for k, v in full.items()})
            records.append(r_full)

            # Onset (trial start to encounterTime)
            onset = compute_epoch_metrics(pt, cal, req, pt[0], enc)
            r_onset = {**base, 'epoch': 'onset'}
            r_onset.update(onset)
            records.append(r_onset)

            # Anticipatory (1s before encounter)
            antic = compute_epoch_metrics(pt, cal, req, max(enc - 1, pt[0]), enc)
            r_antic = {**base, 'epoch': 'anticipatory'}
            r_antic.update(antic)
            records.append(r_antic)

            # Reactive (encounter to encounter + 2s)
            react = compute_epoch_metrics(pt, cal, req, enc, enc + 2)
            r_react = {**base, 'epoch': 'reactive'}
            r_react.update(react)
            records.append(r_react)

            # Terminal (strike - 2s to strike, attack trials only)
            if row['is_attack'] == 1 and not pd.isna(strike) and strike > enc + 2:
                term = compute_epoch_metrics(pt, cal, req, strike - 2, strike)
                r_term = {**base, 'epoch': 'terminal'}
                r_term.update(term)
                records.append(r_term)

        except Exception:
            pass

        if (idx + 1) % 5000 == 0:
            print(f"  Processed {idx + 1}/{len(beh)} trials...")

    df = pd.DataFrame(records)
    print(f"\nTotal records: {len(df)}")
    print(f"By epoch: {df['epoch'].value_counts().to_dict()}")

    # Save
    out_path = OUT_DIR / "vigor_metrics.csv"
    df.to_csv(out_path, index=False)
    print(f"Saved: {out_path}")

    return df


if __name__ == '__main__':
    run()
