"""
H2 Results Figures — Vigor Dynamics and the Predatory Imminence Continuum.

Generates:
  h2a_encounter_spike.png  — Attack vs non-attack vigor (both samples)
  h2b_timecourse.png       — Vigor timecourse by encounter and threat (exploratory)

Usage:
  python scripts/plotting/plot_h2.py
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'notebooks', 'analysis'))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from scipy.stats import ttest_1samp
from pathlib import Path

from plotter import Colors, set_plot_style, style_axis

set_plot_style()
plt.rcParams.update({'savefig.dpi': 300, 'figure.dpi': 300})

OUT_DIR = Path('results/figs/h2')
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ============================================================
# Load data
# ============================================================

def load_sample(name):
    base = Path(f'data/{name}_350/processed')
    s5 = sorted(base.glob('stage5_*'))[-1]

    tv = pd.read_csv(s5 / 'trial_vigor.csv')
    beh = pd.read_csv(s5 / 'behavior_rich.csv', low_memory=False)

    exclude = [154, 197, 208] if name == 'exploratory' else []
    tv = tv[~tv['subj'].isin(exclude)]
    beh = beh[~beh['subj'].isin(exclude)]

    # Check for smoothed timeseries
    ts_path = Path(f'data/{name}_350/processed/vigor_processed/smoothed_vigor_ts.parquet')
    ts = pd.read_parquet(ts_path) if ts_path.exists() else None
    if ts is not None:
        ts = ts[~ts['subj'].isin(exclude)]

    return {
        'vigor': tv, 'beh': beh, 'timeseries': ts,
        'label': f'{"Exploratory" if name == "exploratory" else "Confirmatory"} (N={tv["subj"].nunique()})',
        'name': name,
    }

samples = {}
for name in ['exploratory', 'confirmatory']:
    try:
        samples[name] = load_sample(name)
    except Exception as e:
        print(f'Skipping {name}: {e}')

labels = [s["label"] for s in samples.values()]
print(f'Loaded: {", ".join(labels)}')


# ============================================================
# H2a: Encounter spike — attack vs non-attack
# ============================================================

def plot_h2a():
    n_samples = len(samples)
    fig, axes = plt.subplots(1, n_samples, figsize=(4.0 * n_samples, 3.5), sharey=True)
    if n_samples == 1:
        axes = [axes]

    for ax, (name, s) in zip(axes, samples.items()):
        tv = s['vigor']
        tv_valid = tv[(tv['type'] == 1) & tv['norm_rate'].notna()].copy()

        # Per-subject mean by attack status
        attack_means = tv_valid[tv_valid['is_attack'] == 1].groupby('subj')['norm_rate'].mean()
        nonattack_means = tv_valid[tv_valid['is_attack'] == 0].groupby('subj')['norm_rate'].mean()
        shared = sorted(set(attack_means.index) & set(nonattack_means.index))
        spike = attack_means.loc[shared] - nonattack_means.loc[shared]

        t_val, p_val = ttest_1samp(spike.dropna(), 0)
        d_val = spike.mean() / spike.std()

        # Bar: non-attack, attack (within-subject demeaned)
        subj_grand = tv_valid.groupby('subj')['norm_rate'].mean()
        na_demeaned = (nonattack_means.loc[shared] - subj_grand.loc[shared])
        a_demeaned = (attack_means.loc[shared] - subj_grand.loc[shared])

        means = [na_demeaned.mean(), a_demeaned.mean()]
        sems = [na_demeaned.sem(), a_demeaned.sem()]

        bars = ax.bar([0, 1], means, width=0.6,
                     yerr=[se * 1.96 for se in sems], capsize=4,
                     color=[Colors.CERULEAN2, Colors.RUBY1],
                     edgecolor='white', linewidth=0.5, alpha=0.85, zorder=3)

        ax.axhline(0, color=Colors.INK, lw=0.8, ls='--', alpha=0.4, zorder=1)

        # Annotate effect size
        ax.text(0.5, max(means) + max(sems) * 2.5,
               f'd = {d_val:.2f}', ha='center', fontsize=9, color=Colors.INK,
               fontstyle='italic')

        style_axis(ax, xlabel=None,
                  ylabel='Vigor (within-subject Δ)' if ax == axes[0] else None)
        ax.set_facecolor('#FCFCFD')
        ax.set_xticks([0, 1])
        ax.set_xticklabels(['No attack', 'Attack'], fontsize=10)
        ax.set_title(s['label'], fontsize=11, color=Colors.DARK_GREY, pad=10)

    plt.tight_layout()
    fig.savefig(OUT_DIR / 'h2a_encounter_spike.png', dpi=300, bbox_inches='tight', facecolor='white')
    fig.savefig(OUT_DIR / 'h2a_encounter_spike.pdf', bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print('  Saved h2a_encounter_spike')


# ============================================================
# H2b: Vigor timecourse by encounter and threat
# ============================================================

def _compute_encounter_aligned_rate(beh, window=2.5, bin_size=0.2):
    """Compute instantaneous press rate aligned to encounter time from raw timestamps."""
    import ast

    rows = []
    for _, trial in beh[beh['encounterTime'].notna()].iterrows():
        try:
            timestamps = ast.literal_eval(trial['alignedEffortRate'])
        except (ValueError, SyntaxError):
            continue
        if len(timestamps) < 3:
            continue

        enc_t = trial['encounterTime']
        subj = trial['subj']
        is_attack = int(trial['isAttackTrial'])
        ts = np.array(timestamps)

        # Compute instantaneous rate at each IPI midpoint
        ipis = np.diff(ts)
        valid = (ipis > 0.02) & (ipis < 2.0)  # filter impossibly fast / pauses
        ipis = ipis[valid]
        if len(ipis) < 2:
            continue
        midpoints = (ts[:-1][valid] + ts[1:][valid]) / 2
        rates = 1.0 / ipis

        # Align to encounter time
        t_enc = midpoints - enc_t

        # Filter to window
        mask = (t_enc >= -window) & (t_enc <= window)
        t_enc = t_enc[mask]
        rates = rates[mask]

        for t, r in zip(t_enc, rates):
            t_bin = round(t / bin_size) * bin_size
            rows.append({'subj': subj, 'is_attack': is_attack, 't_bin': t_bin, 'rate': r})

    return pd.DataFrame(rows)


def plot_h2b():
    """Vigor timecourse aligned to encounter (t=0), attack minus non-attack, both samples."""
    n_samples = len(samples)
    fig, axes = plt.subplots(1, n_samples, figsize=(4.5 * n_samples, 3.5), sharey=True)
    if n_samples == 1:
        axes = [axes]

    window_plot = 2.0  # display window

    for ax, (name, s) in zip(axes, samples.items()):
        beh = s['beh']
        print(f'    Computing encounter-aligned rates for {name}...')
        df = _compute_encounter_aligned_rate(beh, window=2.5)

        # Per-subject mean rate per bin per condition
        subj_bin = df.groupby(['subj', 'is_attack', 't_bin'])['rate'].mean().reset_index()

        a_binned = subj_bin[subj_bin['is_attack'] == 1][['subj', 't_bin', 'rate']]
        n_binned = subj_bin[subj_bin['is_attack'] == 0][['subj', 't_bin', 'rate']]

        merged = a_binned.merge(n_binned, on=['subj', 't_bin'], suffixes=('_attack', '_nonattack'))
        merged['diff'] = merged['rate_attack'] - merged['rate_nonattack']
        grand = merged.groupby('t_bin')['diff'].agg(['mean', 'sem']).reset_index()
        grand = grand.sort_values('t_bin')

        # Smooth with rolling average (3-bin window)
        grand['mean_smooth'] = grand['mean'].rolling(3, center=True, min_periods=1).mean()
        grand['sem_smooth'] = grand['sem'].rolling(3, center=True, min_periods=1).mean()

        # Trim to display window
        grand = grand[(grand['t_bin'] >= -window_plot) & (grand['t_bin'] <= window_plot)]

        ax.plot(grand['t_bin'], grand['mean_smooth'], color=Colors.RUBY1, lw=2.0, zorder=3)
        ax.fill_between(grand['t_bin'],
                      grand['mean_smooth'] - grand['sem_smooth'] * 1.96,
                      grand['mean_smooth'] + grand['sem_smooth'] * 1.96,
                      color=Colors.RUBY1, alpha=0.15, zorder=2)

        ax.axhline(0, color=Colors.INK, lw=0.8, ls='--', alpha=0.4, zorder=1)
        ax.axvline(0, color=Colors.INK, lw=1, ls='--', alpha=0.5, zorder=1)

        style_axis(ax, xlabel='Time from encounter (s)',
                  ylabel='Δ Vigor' if ax == axes[0] else None)
        ax.set_facecolor('#FCFCFD')
        ax.set_title(s['label'], fontsize=11, color=Colors.DARK_GREY, pad=10)

    plt.tight_layout()
    fig.savefig(OUT_DIR / 'h2b_timecourse.png', dpi=300, bbox_inches='tight', facecolor='white')
    fig.savefig(OUT_DIR / 'h2b_timecourse.pdf', bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print('  Saved h2b_timecourse')


# ============================================================
# H2b-pre: Anticipatory vigor by threat (first 2s of trial)
# ============================================================

def _compute_trial_aligned_rate(beh, window=2.5, bin_size=0.05):
    """Compute instantaneous press rate aligned to trial start, split by threat."""
    import ast

    rows = []
    for _, trial in beh.iterrows():
        try:
            timestamps = ast.literal_eval(trial['alignedEffortRate'])
        except (ValueError, SyntaxError):
            continue
        if len(timestamps) < 3:
            continue

        subj = trial['subj']
        threat = round(float(trial['threat']), 1) if pd.notna(trial.get('threat')) else None
        if threat is None:
            continue
        ts = np.array(timestamps)

        ipis = np.diff(ts)
        valid = (ipis > 0.02) & (ipis < 2.0)
        ipis = ipis[valid]
        if len(ipis) < 2:
            continue
        midpoints = (ts[:-1][valid] + ts[1:][valid]) / 2
        rates = 1.0 / ipis

        mask = (midpoints >= 0) & (midpoints <= window)
        midpoints = midpoints[mask]
        rates = rates[mask]

        for t, r in zip(midpoints, rates):
            t_bin = round(t / bin_size) * bin_size
            rows.append({'subj': subj, 'T_round': threat, 't_bin': t_bin, 'rate': r})

    return pd.DataFrame(rows)


def _kernel_smooth_trial(timestamps, t_grid, sigma=0.3):
    """Gaussian kernel density estimate of press rate — matches parquet pipeline (σ=0.3s, 20Hz)."""
    ts = np.asarray(timestamps)
    if len(ts) < 2:
        return np.full_like(t_grid, np.nan, dtype=float)
    diff = t_grid[:, None] - ts[None, :]
    kernel = np.exp(-0.5 * (diff / sigma) ** 2) / (sigma * np.sqrt(2 * np.pi))
    return kernel.sum(axis=1)


def _get_onset_timecourse(s, window=2.5, dt=0.05):
    """Kernel-smoothed vigor timecourse with Cousineau-Morey within-subject centering.

    Removes between-subject offset so the ribbon reflects within-subject variability
    across threat levels, rather than being dominated by baseline vigor differences.
    """
    import ast
    beh = s['beh']
    beh_valid = beh[beh['type'] == 1]

    t_grid = np.arange(0, window + dt, dt)
    rows = []

    for _, trial in beh_valid.iterrows():
        try:
            timestamps = ast.literal_eval(trial['alignedEffortRate'])
        except (ValueError, SyntaxError):
            continue
        if len(timestamps) < 3:
            continue

        subj = trial['subj']
        threat = round(float(trial['threat']), 1) if pd.notna(trial.get('threat')) else None
        if threat is None:
            continue

        # Normalize by calibration max if available
        cal_max = trial.get('calibrationMax', None)
        if pd.isna(cal_max) or cal_max is None or cal_max <= 0:
            cal_max = 1.0

        rate = _kernel_smooth_trial(timestamps, t_grid, sigma=0.3)
        vigor_norm = rate / cal_max

        for i, t in enumerate(t_grid):
            rows.append({'subj': subj, 'T_round': threat,
                         't_bin': round(t, 3), 'vigor_norm': vigor_norm[i]})

    df = pd.DataFrame(rows)

    # Per-subject × threat × t_bin means (across trials)
    subj_df = df.groupby(['subj', 'T_round', 't_bin'])['vigor_norm'].mean().reset_index()

    # Cousineau-Morey within-subject centering: remove subject-level offset,
    # add grand mean to preserve interpretable units.
    subj_grand = subj_df.groupby('subj')['vigor_norm'].transform('mean')
    grand_mean = subj_df['vigor_norm'].mean()
    subj_df['vigor_resid'] = subj_df['vigor_norm'] - subj_grand + grand_mean

    return subj_df


def plot_h2b_pre():
    """Onset effort by threat — first 2.5s of trial, within-subject centered."""
    n_samples = len(samples)
    fig, axes = plt.subplots(1, n_samples, figsize=(4.5 * n_samples, 3.8), sharey=True)
    if n_samples == 1:
        axes = [axes]

    threat_colors = {0.1: Colors.CERULEAN2, 0.5: Colors.SLATE, 0.9: Colors.RUBY1}
    plateau = (1.5, 2.5)  # window for effect-size annotation

    for ax, (name, s) in zip(axes, samples.items()):
        print(f'    Computing onset timecourse for {name}...')
        subj_bin = _get_onset_timecourse(s, window=2.5)

        for T, color in threat_colors.items():
            sub = subj_bin[subj_bin['T_round'] == T]
            grand = sub.groupby('t_bin')['vigor_resid'].agg(['mean', 'sem']).reset_index()
            grand = grand.sort_values('t_bin')

            ax.plot(grand['t_bin'], grand['mean'], color=color, lw=1.8,
                   label=f'T = {T}', zorder=3)
            ax.fill_between(grand['t_bin'],
                          grand['mean'] - grand['sem'],
                          grand['mean'] + grand['sem'],
                          color=color, alpha=0.18, zorder=2)

        # Plateau contrast: per-subject mean(T=0.9) − mean(T=0.1) in [1.5, 2.5]s
        plat = subj_bin[(subj_bin['t_bin'] >= plateau[0]) & (subj_bin['t_bin'] <= plateau[1])]
        subj_plat = plat.groupby(['subj', 'T_round'])['vigor_resid'].mean().unstack('T_round')
        if 0.9 in subj_plat.columns and 0.1 in subj_plat.columns:
            delta = (subj_plat[0.9] - subj_plat[0.1]).dropna()
            t_val, p_val = ttest_1samp(delta, 0)
            d_val = delta.mean() / delta.std()
            ann = f'Δ(0.9−0.1) = {delta.mean():+.3f}\nd = {d_val:.2f}'
            ax.text(0.04, 0.96, ann, transform=ax.transAxes,
                   fontsize=9, color=Colors.INK, va='top', ha='left',
                   fontstyle='italic',
                   bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                            edgecolor='#E5E7EB', linewidth=0.6))

        style_axis(ax, xlabel='Time from trial start (s)',
                  ylabel='Onset effort (within-subject)' if ax == axes[0] else None)
        ax.set_facecolor('#FCFCFD')
        ax.set_title(s['label'], fontsize=11, color=Colors.DARK_GREY, pad=10)
        ax.set_xlim(0, 2.5)

        if ax == axes[-1]:
            leg = ax.legend(fontsize=9, frameon=True, labelcolor=Colors.INK,
                           loc='lower right', title='Threat')
            leg.get_frame().set_facecolor('white')
            leg.get_frame().set_edgecolor('#E5E7EB')
            leg.get_frame().set_linewidth(0.8)
            leg.get_title().set_color(Colors.INK)
            leg.get_title().set_fontsize(9)

    plt.tight_layout()
    fig.savefig(OUT_DIR / 'h2b_anticipatory.png', dpi=300, bbox_inches='tight', facecolor='white')
    fig.savefig(OUT_DIR / 'h2b_anticipatory.pdf', bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print('  Saved h2b_anticipatory')


# ============================================================
# H2 Combined: Imminence continuum panel
# ============================================================

def _get_encounter_by_threat(s, window=2.5, dt=0.05):
    """Kernel-smoothed encounter-aligned timecourse, split by threat, attack trials only."""
    import ast
    beh = s['beh']
    attack = beh[(beh['isAttackTrial'] == 1) & beh['encounterTime'].notna()]

    t_grid = np.arange(-window, window + dt, dt)
    rows = []

    for _, trial in attack.iterrows():
        try:
            timestamps = ast.literal_eval(trial['alignedEffortRate'])
        except (ValueError, SyntaxError):
            continue
        if len(timestamps) < 3:
            continue

        subj = trial['subj']
        threat = round(float(trial['threat']), 1) if pd.notna(trial.get('threat')) else None
        if threat is None:
            continue

        enc_t = trial['encounterTime']
        # Shift timestamps relative to encounter
        ts_shifted = np.array(timestamps) - enc_t
        rate = _kernel_smooth_trial(ts_shifted, t_grid, sigma=0.3)
        for i, t in enumerate(t_grid):
            rows.append({'subj': subj, 'T_round': threat, 't_bin': round(t, 3), 'rate': rate[i]})

    df = pd.DataFrame(rows)
    return df.groupby(['subj', 'T_round', 't_bin'])['rate'].mean().reset_index()


def plot_h2_combined():
    """Combined imminence panel: A) anticipatory by threat, B) encounter spike by threat."""
    n_samples = len(samples)
    fig, axes = plt.subplots(2, n_samples, figsize=(4.5 * n_samples, 7), sharey='row')
    if n_samples == 1:
        axes = axes.reshape(2, 1)

    threat_colors = {0.1: Colors.CERULEAN2, 0.5: Colors.SLATE, 0.9: Colors.RUBY1}

    for col, (name, s) in enumerate(samples.items()):
        # ── Row 1: Anticipatory (onset, by threat) ──
        ax = axes[0, col]
        print(f'    Computing onset for {name}...')
        subj_bin = _get_onset_timecourse(s, window=2.5)

        for T, color in threat_colors.items():
            sub = subj_bin[subj_bin['T_round'] == T]
            grand = sub.groupby('t_bin')['vigor_resid'].agg(['mean', 'sem']).reset_index()
            grand = grand.sort_values('t_bin')
            ax.plot(grand['t_bin'], grand['mean'], color=color, lw=1.8,
                   label=f'T = {T}', zorder=3)
            ax.fill_between(grand['t_bin'],
                          grand['mean'] - grand['sem'],
                          grand['mean'] + grand['sem'],
                          color=color, alpha=0.18, zorder=2)

        style_axis(ax, xlabel='Time from trial start (s)',
                  ylabel='Onset effort (within-subject)' if col == 0 else None)
        ax.set_facecolor('#FCFCFD')
        ax.set_title(s['label'], fontsize=11, color=Colors.DARK_GREY, pad=8)
        ax.set_xlim(0, 2.5)
        if col == 0:
            leg = ax.legend(fontsize=8, frameon=True, labelcolor=Colors.INK,
                           loc='lower right', title='Threat')
            leg.get_frame().set_facecolor('white')
            leg.get_frame().set_edgecolor('#E5E7EB')
            leg.get_frame().set_linewidth(0.8)
            leg.get_title().set_color(Colors.INK)
            leg.get_title().set_fontsize(8)

        # ── Row 2: Encounter-aligned by threat (attack trials) ──
        ax = axes[1, col]
        print(f'    Computing encounter-aligned by threat for {name}...')
        enc_data = _get_encounter_by_threat(s, window=2.0)

        for T, color in threat_colors.items():
            sub = enc_data[enc_data['T_round'] == T]
            grand = sub.groupby('t_bin')['rate'].agg(['mean', 'sem']).reset_index()
            grand = grand.sort_values('t_bin')
            ax.plot(grand['t_bin'], grand['mean'], color=color, lw=1.5,
                   label=f'T = {T}', zorder=3)
            ax.fill_between(grand['t_bin'],
                          grand['mean'] - grand['sem'] * 1.96,
                          grand['mean'] + grand['sem'] * 1.96,
                          color=color, alpha=0.12, zorder=2)

        ax.axvline(0, color=Colors.INK, lw=1, ls='--', alpha=0.5, zorder=1)
        style_axis(ax, xlabel='Time from encounter (s)',
                  ylabel='Smoothed press rate' if col == 0 else None)
        ax.set_facecolor('#FCFCFD')

    plt.tight_layout()
    fig.savefig(OUT_DIR / 'h2_combined.png', dpi=300, bbox_inches='tight', facecolor='white')
    fig.savefig(OUT_DIR / 'h2_combined.pdf', bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print('  Saved h2_combined')


# ============================================================
# Run all
# ============================================================

if __name__ == '__main__':
    print('Generating H2 figures...')
    plot_h2a()
    plot_h2b()
    plot_h2b_pre()
    plot_h2_combined()
    print(f'Done. Saved to {OUT_DIR}/')
