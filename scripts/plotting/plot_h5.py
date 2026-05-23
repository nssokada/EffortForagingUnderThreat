"""
H5 Results Figures — Metacognitive Monitoring: Appraisal Dissociation.

Generates:
  h5c_appraisal_dissociation.png  — omega → confidence (neg) vs omega → anxiety (null), both samples
  h5d_error_type.png              — confidence → overcautious (neg) vs confidence → reckless (pos), confirmatory
  h5b_slope_shift.png             — anxiety slope → choice shift, both samples
  h5_combined.png                 — Full H5 panel for paper (confirmatory only)

Usage:
  python scripts/plotting/plot_h5.py
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'notebooks', 'analysis'))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from scipy.stats import zscore, linregress
from pathlib import Path

from plotter import Colors, set_plot_style, style_axis

# Apply global style
set_plot_style()
plt.rcParams.update({'savefig.dpi': 300, 'figure.dpi': 300})

OUT_DIR = Path('results/figs/h5')
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ============================================================
# Load data
# ============================================================

def load_sample(name):
    """Load a sample's data for H5 figures."""
    base = Path(f'data/{name}_350/processed')
    s5 = sorted(base.glob('stage5_*'))[-1]

    beh = pd.read_csv(s5 / 'behavior_rich.csv', low_memory=False)
    feel = pd.read_csv(s5 / 'feelings.csv')
    psych_path = s5 / 'psych.csv'
    psych = pd.read_csv(psych_path) if psych_path.exists() else None

    # Model parameters
    params = pd.read_csv(f'results/stats/joint_optimal/{name}/mcmc_m4_params.csv')

    # Exclusions
    exclude = [154, 197, 208] if name == 'exploratory' else []
    beh = beh[~beh['subj'].isin(exclude)]
    feel = feel[~feel['subj'].isin(exclude)]
    params = params[~params['subj'].isin(exclude)]
    if psych is not None:
        psych = psych[~psych['subj'].isin(exclude)]

    return {
        'beh': beh, 'feelings': feel, 'psych': psych, 'params': params,
        'label': f'{"Exploratory" if name == "exploratory" else "Confirmatory"} (N={beh["subj"].nunique()})',
        'name': name,
    }


def _compute_per_subject_anxiety_slope(feel):
    """Per-subject OLS slope of anxiety ~ threat, computed from raw feelings data."""
    anx = feel[feel['questionLabel'] == 'anxiety'].copy()
    anx['T_round'] = anx['threat'].round(1)

    slopes = {}
    for subj, g in anx.groupby('subj'):
        if len(g) < 3:
            continue
        x = g['T_round'].values.astype(float)
        y = g['response'].values.astype(float)
        mask = np.isfinite(x) & np.isfinite(y)
        if mask.sum() < 3:
            continue
        res = linregress(x[mask], y[mask])
        slopes[subj] = res.slope
    return pd.Series(slopes, name='anxiety_slope')


def _compute_calibration(feel):
    """Per-subject correlation between anxiety ratings and actual threat level."""
    anx = feel[feel['questionLabel'] == 'anxiety'].copy()

    calibrations = {}
    for subj, g in anx.groupby('subj'):
        x = g['threat'].values.astype(float)
        y = g['response'].values.astype(float)
        mask = np.isfinite(x) & np.isfinite(y)
        if mask.sum() < 4:
            continue
        r = np.corrcoef(x[mask], y[mask])[0, 1]
        calibrations[subj] = r
    return pd.Series(calibrations, name='calibration')


def compute_derived(s):
    """Compute all derived per-subject variables for H5 from raw data."""
    feel = s['feelings']
    beh = s['beh']
    params = s['params']

    # --- Mean affect ---
    mean_anx = feel[feel['questionLabel'] == 'anxiety'].groupby('subj')['response'].mean()
    mean_conf = feel[feel['questionLabel'] == 'confidence'].groupby('subj')['response'].mean()

    # --- Per-subject anxiety slope (computed from THIS sample's data) ---
    anxiety_slope = _compute_per_subject_anxiety_slope(feel)
    calibration = _compute_calibration(feel)

    # --- Choice shift: P(heavy | T=0.1) - P(heavy | T=0.9) ---
    beh_choice = beh[beh['type'] == 1].copy()
    beh_choice['T_round'] = beh_choice['threat'].round(1)
    p_heavy_low = beh_choice[beh_choice['T_round'] == 0.1].groupby('subj')['choice'].mean()
    p_heavy_high = beh_choice[beh_choice['T_round'] == 0.9].groupby('subj')['choice'].mean()
    choice_shift = p_heavy_low - p_heavy_high

    # --- Error counts ---
    # Overcautious: number of light choices (choice==0)
    n_overcautious = beh_choice.groupby('subj')['choice'].apply(
        lambda x: (x == 0).sum()).rename('n_overcautious')

    # Reckless: chose heavy at high threat & far distance
    reckless_cond = beh_choice[(beh_choice['T_round'] == 0.9) & (beh_choice['distance_H'] == 3)]
    n_reckless = reckless_cond.groupby('subj')['choice'].sum().rename('n_reckless')

    # --- Earnings (sum of trial rewards on free-choice trials) ---
    earnings = beh_choice.groupby('subj')['trialReward'].sum().rename('earnings')

    # --- Escape rate on attack trials ---
    attack = beh[beh['isAttackTrial'] == 1]
    escape_rate = attack.groupby('subj').apply(
        lambda g: (g['trialEndState'] == 'escaped').mean(),
        include_groups=False,
    ).rename('escape_rate')

    # --- pct_optimal using simple EV with S = exp(-T*D) ---
    R_H, R_L, C_PEN = 5.0, 1.0, 5.0
    bch = beh_choice.copy()
    bch['S_H'] = np.exp(-bch['threat'] * bch['distance_H'])
    bch['S_L'] = np.exp(-bch['threat'] * 1.0)
    bch['EV_H'] = R_H * bch['S_H'] - (1 - bch['S_H']) * C_PEN
    bch['EV_L'] = R_L * bch['S_L'] - (1 - bch['S_L']) * C_PEN
    bch['optimal_high'] = (bch['EV_H'] > bch['EV_L']).astype(int)
    bch['is_optimal'] = (bch['choice'] == bch['optimal_high']).astype(int)
    pct_opt = bch.groupby('subj')['is_optimal'].mean().rename('pct_opt')

    # --- Assemble subject-level DataFrame ---
    df = params[['subj', 'omega', 'kappa']].copy()
    df = df.merge(mean_anx.rename('mean_anxiety'), on='subj', how='left')
    df = df.merge(mean_conf.rename('mean_confidence'), on='subj', how='left')
    df = df.merge(anxiety_slope.rename('anxiety_slope'), left_on='subj', right_index=True, how='left')
    df = df.merge(calibration.rename('calibration'), left_on='subj', right_index=True, how='left')
    df = df.merge(choice_shift.rename('choice_shift'), on='subj', how='left')
    df = df.merge(n_overcautious.reset_index(), on='subj', how='left')
    df = df.merge(n_reckless.reset_index(), on='subj', how='left')
    df = df.merge(earnings.reset_index(), on='subj', how='left')
    df = df.merge(escape_rate.rename('escape_rate').reset_index(), on='subj', how='left')
    df = df.merge(pct_opt.reset_index(), on='subj', how='left')

    # Log-transform then z-score omega/kappa (heavy-tailed)
    df['log_omega'] = np.log(df['omega'].clip(lower=1e-6))
    valid = df['log_omega'].notna()
    if valid.sum() > 2:
        df.loc[valid, 'omega_z'] = zscore(df.loc[valid, 'log_omega'])

    # Z-score remaining variables
    for col in ['kappa', 'mean_anxiety', 'mean_confidence',
                'anxiety_slope', 'calibration', 'choice_shift',
                'n_overcautious', 'n_reckless',
                'earnings', 'escape_rate', 'pct_opt']:
        if col in df.columns:
            valid = df[col].notna()
            if valid.sum() > 2:
                df.loc[valid, f'{col}_z'] = zscore(df.loc[valid, col])

    return df


samples = {}
sample_dfs = {}
for name in ['exploratory', 'confirmatory']:
    try:
        s = load_sample(name)
        samples[name] = s
        sample_dfs[name] = compute_derived(s)
    except Exception as e:
        print(f'Skipping {name}: {e}')

labels = [s["label"] for s in samples.values()]
print(f'Loaded: {", ".join(labels)}')


# ============================================================
# Helper: scatter + regression
# ============================================================

def scatter_regression(ax, x, y, color, xlabel=None, ylabel=None, annotate=True):
    """Scatter with OLS regression line and 95% CI band."""
    mask = np.isfinite(x) & np.isfinite(y)
    x_clean, y_clean = x[mask], y[mask]
    if len(x_clean) < 5:
        return

    # Scatter
    ax.scatter(x_clean, y_clean, s=15, alpha=0.4, color=color, edgecolors='none', zorder=2)

    # OLS
    slope, intercept, r_value, p_value, std_err = linregress(x_clean, y_clean)

    # Regression line
    x_line = np.linspace(x_clean.min(), x_clean.max(), 200)
    y_line = intercept + slope * x_line
    ax.plot(x_line, y_line, color=color, lw=2, zorder=3)

    # 95% CI band
    n = len(x_clean)
    x_mean = x_clean.mean()
    ss_x = np.sum((x_clean - x_mean) ** 2)
    residuals = y_clean - (intercept + slope * x_clean)
    mse = np.sum(residuals ** 2) / (n - 2)
    se_line = np.sqrt(mse * (1.0 / n + (x_line - x_mean) ** 2 / ss_x))
    from scipy.stats import t as t_dist
    t_crit = t_dist.ppf(0.975, n - 2)
    ax.fill_between(x_line, y_line - t_crit * se_line, y_line + t_crit * se_line,
                    color=color, alpha=0.15, zorder=1)

    # Annotation
    if annotate:
        ci_lo = slope - t_crit * std_err
        ci_hi = slope + t_crit * std_err
        sig = '*' if p_value < 0.05 else ''
        label = f'\u03b2 = {slope:.2f} [{ci_lo:.2f}, {ci_hi:.2f}]{sig}'
        ax.text(0.05, 0.95, label, transform=ax.transAxes, fontsize=9,
                color=Colors.INK, va='top', ha='left', fontstyle='italic')

    style_axis(ax, xlabel=xlabel, ylabel=ylabel)
    ax.set_facecolor('#FCFCFD')


# ============================================================
# H5c: Appraisal dissociation — omega → confidence vs anxiety
# ============================================================

def plot_h5c():
    n_samples = len(samples)
    fig, axes = plt.subplots(1, 2 * n_samples, figsize=(4.0 * 2 * n_samples, 3.5))
    if n_samples == 1:
        axes = list(axes)

    col = 0
    for name, s in samples.items():
        df = sample_dfs[name]
        label = s['label']

        # Left: omega_z → mean_confidence (expected: negative)
        ax = axes[col]
        scatter_regression(ax, df['omega_z'].values, df['mean_confidence'].values,
                          color=Colors.CERULEAN2,
                          xlabel='$\\omega$ (log z-scored)',
                          ylabel='Mean confidence')
        ax.set_title(f'{label}\nCoping appraisal', fontsize=10, color=Colors.DARK_GREY, pad=10)
        col += 1

        # Right: omega_z → mean_anxiety (expected: null/weak)
        ax = axes[col]
        scatter_regression(ax, df['omega_z'].values, df['mean_anxiety'].values,
                          color=Colors.RUBY1,
                          xlabel='$\\omega$ (log z-scored)',
                          ylabel='Mean anxiety')
        ax.set_title(f'{label}\nThreat appraisal', fontsize=10, color=Colors.DARK_GREY, pad=10)
        col += 1

    plt.tight_layout()
    fig.savefig(OUT_DIR / 'h5c_appraisal_dissociation.png', dpi=300, bbox_inches='tight', facecolor='white')
    fig.savefig(OUT_DIR / 'h5c_appraisal_dissociation.pdf', bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print('  Saved h5c_appraisal_dissociation')


# ============================================================
# H5d: Error type — confidence predicts what errors you make
# ============================================================

def plot_h5d():
    # Use confirmatory if available, else exploratory
    name = 'confirmatory' if 'confirmatory' in sample_dfs else list(sample_dfs.keys())[0]
    df = sample_dfs[name]
    label = samples[name]['label']

    fig, axes = plt.subplots(1, 2, figsize=(8.0, 3.5))

    # Ensure z-scored confidence is available
    if 'mean_confidence_z' not in df.columns:
        print('  Skipping h5d: mean_confidence_z not available')
        plt.close(fig)
        return

    # Left: confidence_z → n_overcautious (expected: negative)
    ax = axes[0]
    scatter_regression(ax, df['mean_confidence_z'].values, df['n_overcautious'].values,
                      color=Colors.CERULEAN2,
                      xlabel='Confidence (z-scored)',
                      ylabel='N overcautious choices')
    ax.set_title(f'{label}\nOvercautious errors', fontsize=10, color=Colors.DARK_GREY, pad=10)

    # Right: confidence_z → n_reckless (expected: positive)
    ax = axes[1]
    scatter_regression(ax, df['mean_confidence_z'].values, df['n_reckless'].values,
                      color=Colors.RUBY1,
                      xlabel='Confidence (z-scored)',
                      ylabel='N reckless choices')
    ax.set_title(f'{label}\nReckless errors', fontsize=10, color=Colors.DARK_GREY, pad=10)

    plt.tight_layout()
    fig.savefig(OUT_DIR / 'h5d_error_type.png', dpi=300, bbox_inches='tight', facecolor='white')
    fig.savefig(OUT_DIR / 'h5d_error_type.pdf', bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print('  Saved h5d_error_type')


# ============================================================
# H5b: Anxiety slope → choice shift
# ============================================================

def plot_h5b():
    n_samples = len(samples)
    fig, axes = plt.subplots(1, n_samples, figsize=(4.0 * n_samples, 3.5))
    if n_samples == 1:
        axes = [axes]

    for ax, (name, s) in zip(axes, samples.items()):
        df = sample_dfs[name]
        label = s['label']

        if 'anxiety_slope_z' not in df.columns or 'choice_shift_z' not in df.columns:
            ax.text(0.5, 0.5, 'Data not available', transform=ax.transAxes, ha='center',
                   fontsize=10, color=Colors.INK)
            continue

        scatter_regression(ax, df['anxiety_slope_z'].values, df['choice_shift'].values,
                          color=Colors.PERSIMMON3,
                          xlabel='Anxiety slope (z-scored)',
                          ylabel='Choice shift\nP(heavy|T=0.1) \u2212 P(heavy|T=0.9)' if ax == axes[0] else None)
        ax.set_title(label, fontsize=11, color=Colors.DARK_GREY, pad=10)

    plt.tight_layout()
    fig.savefig(OUT_DIR / 'h5b_slope_shift.png', dpi=300, bbox_inches='tight', facecolor='white')
    fig.savefig(OUT_DIR / 'h5b_slope_shift.pdf', bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print('  Saved h5b_slope_shift')


# ============================================================
# Combined: Full H5 panel (confirmatory only)
# ============================================================

def plot_h5_combined():
    """Single figure with all H5 results for confirmatory sample."""
    # Use confirmatory if available, else exploratory
    name = 'confirmatory' if 'confirmatory' in sample_dfs else list(sample_dfs.keys())[0]
    df = sample_dfs[name]
    label = samples[name]['label']

    fig = plt.figure(figsize=(9, 10.5))
    gs = GridSpec(3, 2, hspace=0.45, wspace=0.35)

    # ── Row 1: Appraisal dissociation ──
    ax_conf = fig.add_subplot(gs[0, 0])
    scatter_regression(ax_conf, df['omega_z'].values, df['mean_confidence'].values,
                      color=Colors.CERULEAN2,
                      xlabel='$\\omega$ (log z-scored)',
                      ylabel='Mean confidence')
    ax_conf.set_title('Coping appraisal', fontsize=10, color=Colors.DARK_GREY, pad=8)

    ax_anx = fig.add_subplot(gs[0, 1])
    scatter_regression(ax_anx, df['omega_z'].values, df['mean_anxiety'].values,
                      color=Colors.RUBY1,
                      xlabel='$\\omega$ (log z-scored)',
                      ylabel='Mean anxiety')
    ax_anx.set_title('Threat appraisal', fontsize=10, color=Colors.DARK_GREY, pad=8)

    # ── Row 2: Error type ──
    if 'mean_confidence_z' in df.columns:
        ax_oc = fig.add_subplot(gs[1, 0])
        scatter_regression(ax_oc, df['mean_confidence_z'].values, df['n_overcautious'].values,
                          color=Colors.CERULEAN2,
                          xlabel='Confidence (z-scored)',
                          ylabel='N overcautious choices')
        ax_oc.set_title('Overcautious errors', fontsize=10, color=Colors.DARK_GREY, pad=8)

        ax_rk = fig.add_subplot(gs[1, 1])
        scatter_regression(ax_rk, df['mean_confidence_z'].values, df['n_reckless'].values,
                          color=Colors.RUBY1,
                          xlabel='Confidence (z-scored)',
                          ylabel='N reckless choices')
        ax_rk.set_title('Reckless errors', fontsize=10, color=Colors.DARK_GREY, pad=8)

    # ── Row 3: Slope → shift ──
    if 'anxiety_slope_z' in df.columns:
        ax_ss = fig.add_subplot(gs[2, :])
        scatter_regression(ax_ss, df['anxiety_slope_z'].values, df['choice_shift'].values,
                          color=Colors.PERSIMMON3,
                          xlabel='Anxiety slope (z-scored)',
                          ylabel='Choice shift\nP(heavy|T=0.1) \u2212 P(heavy|T=0.9)')
        ax_ss.set_title('Anxiety reactivity predicts choice adaptation', fontsize=10,
                        color=Colors.DARK_GREY, pad=8)

    # Suptitle
    fig.suptitle(label, fontsize=12, color=Colors.DARK_GREY, y=1.01)

    plt.savefig(OUT_DIR / 'h5_combined.png', dpi=300, bbox_inches='tight', facecolor='white')
    plt.savefig(OUT_DIR / 'h5_combined.pdf', bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print('  Saved h5_combined')


# ============================================================
# OLS helper for forest plot
# ============================================================


def _ols_beta_with_ci(y, X, focal_idx=0, ci=0.95):
    """OLS β for the focal coefficient with normal-theory CI.

    X should NOT include an intercept column — added internally.
    Returns (beta, ci_lo, ci_hi). Returns (nan, nan, nan) if too few obs.
    """
    from scipy.stats import t as student_t

    y = np.asarray(y, dtype=float)
    X = np.asarray(X, dtype=float)
    if X.ndim == 1:
        X = X[:, None]
    mask = np.isfinite(y) & np.all(np.isfinite(X), axis=1)
    y = y[mask]
    X = X[mask]
    n = len(y)
    k = X.shape[1] + 1  # +1 for intercept
    if n <= k + 2:
        return float('nan'), float('nan'), float('nan')
    X_aug = np.column_stack([np.ones(n), X])
    beta_hat, _, _, _ = np.linalg.lstsq(X_aug, y, rcond=None)
    resid = y - X_aug @ beta_hat
    sigma2 = float((resid ** 2).sum() / (n - k))
    try:
        cov = sigma2 * np.linalg.inv(X_aug.T @ X_aug)
    except np.linalg.LinAlgError:
        return float('nan'), float('nan'), float('nan')
    se_focal = float(np.sqrt(cov[focal_idx + 1, focal_idx + 1]))
    beta_focal = float(beta_hat[focal_idx + 1])
    crit = float(student_t.ppf(0.5 + ci / 2, df=n - k))
    return beta_focal, beta_focal - crit * se_focal, beta_focal + crit * se_focal


def _zsc(x):
    x = np.asarray(x, dtype=float)
    valid = np.isfinite(x)
    if valid.sum() < 5:
        return np.full_like(x, np.nan, dtype=float)
    m = float(np.nanmean(x))
    sd = float(np.nanstd(x))
    if sd < 1e-10:
        return x - m
    return (x - m) / sd


# ============================================================
# H5 forest plot
# ============================================================


def plot_h5_forest():
    """Grouped forest plot of H5 regression results.

    4 sub-panels, one per sub-hypothesis, each with its own narrative
    title:
        H5a — Metacognitive accuracy adds variance beyond ω, κ
        H5b — Anxiety reactivity drives choice adaptation
        H5c — ω → coping appraisal, not threat appraisal
        H5d — Confidence determines error type, not error rate

    All outcomes z-scored before OLS so β values are on a common scale.
    Sub-panels share the x-axis (standardized β). H5c-anx row is
    shaded with the ROPE band [-0.10, +0.10].
    """
    if not sample_dfs:
        print('  Skipping h5_forest — no samples loaded')
        return

    # Compute β for each test in each sample
    rows = []
    for name, df in sample_dfs.items():
        # H5a: multivariate outcome_z ~ ω_z + κ_z + calibration_z, focal = calibration
        for outcome, tag in [
            ('pct_opt', 'H5a-opt'),
            ('escape_rate', 'H5a-esc'),
            ('earnings', 'H5a-earn'),
        ]:
            if outcome not in df.columns or 'calibration' not in df.columns:
                continue
            y = _zsc(df[outcome].values)
            X = np.column_stack([
                df['omega_z'].values if 'omega_z' in df.columns
                else np.full(len(df), np.nan),
                df['kappa_z'].values if 'kappa_z' in df.columns
                else np.full(len(df), np.nan),
                _zsc(df['calibration'].values),
            ])
            beta, lo, hi = _ols_beta_with_ci(y, X, focal_idx=2)
            rows.append({'sample': name, 'test': tag,
                         'beta': beta, 'lo': lo, 'hi': hi})

        # H5b: choice_shift_z ~ anxiety_slope_z
        if 'anxiety_slope' in df.columns and 'choice_shift' in df.columns:
            beta, lo, hi = _ols_beta_with_ci(
                _zsc(df['choice_shift'].values),
                _zsc(df['anxiety_slope'].values), focal_idx=0)
            rows.append({'sample': name, 'test': 'H5b',
                         'beta': beta, 'lo': lo, 'hi': hi})

        # H5c: mean_confidence_z ~ omega_z AND mean_anxiety_z ~ omega_z
        if 'mean_confidence' in df.columns and 'omega_z' in df.columns:
            beta, lo, hi = _ols_beta_with_ci(
                _zsc(df['mean_confidence'].values),
                df['omega_z'].values, focal_idx=0)
            rows.append({'sample': name, 'test': 'H5c-conf',
                         'beta': beta, 'lo': lo, 'hi': hi})
        if 'mean_anxiety' in df.columns and 'omega_z' in df.columns:
            beta, lo, hi = _ols_beta_with_ci(
                _zsc(df['mean_anxiety'].values),
                df['omega_z'].values, focal_idx=0)
            rows.append({'sample': name, 'test': 'H5c-anx',
                         'beta': beta, 'lo': lo, 'hi': hi})

        # H5d: n_overcautious_z ~ confidence_z AND n_reckless_z ~ confidence_z
        if 'mean_confidence' in df.columns and 'n_overcautious' in df.columns:
            beta, lo, hi = _ols_beta_with_ci(
                _zsc(df['n_overcautious'].values),
                _zsc(df['mean_confidence'].values), focal_idx=0)
            rows.append({'sample': name, 'test': 'H5d-oc',
                         'beta': beta, 'lo': lo, 'hi': hi})
        if 'mean_confidence' in df.columns and 'n_reckless' in df.columns:
            beta, lo, hi = _ols_beta_with_ci(
                _zsc(df['n_reckless'].values),
                _zsc(df['mean_confidence'].values), focal_idx=0)
            rows.append({'sample': name, 'test': 'H5d-rk',
                         'beta': beta, 'lo': lo, 'hi': hi})

    fr = pd.DataFrame(rows)

    # Grouped sub-panel definitions
    groups = [
        {
            'id': 'H5a',
            'title': r'Metacognitive accuracy adds variance beyond $\omega,\ \kappa$',
            'tests': [
                ('H5a-opt',  r'calibration $\rightarrow$ % optimal'),
                ('H5a-esc',  r'calibration $\rightarrow$ escape rate'),
                ('H5a-earn', r'calibration $\rightarrow$ earnings'),
            ],
            'rope': None,
        },
        {
            'id': 'H5b',
            'title': 'Anxiety reactivity drives choice adaptation',
            'tests': [
                ('H5b', r'anxiety slope $\rightarrow$ choice shift'),
            ],
            'rope': None,
        },
        {
            'id': 'H5c',
            'title': r'$\omega$ maps to coping appraisal, not threat appraisal',
            'tests': [
                ('H5c-conf', r'$\omega \rightarrow$ confidence  (coping)'),
                ('H5c-anx',  r'$\omega \rightarrow$ anxiety  (threat; ROPE null)'),
            ],
            'rope': 'H5c-anx',
        },
        {
            'id': 'H5d',
            'title': 'Confidence determines error type, not error rate',
            'tests': [
                ('H5d-oc', r'confidence $\rightarrow$ overcautious errors'),
                ('H5d-rk', r'confidence $\rightarrow$ reckless errors'),
            ],
            'rope': None,
        },
    ]

    sample_offsets = {'exploratory': -0.18, 'confirmatory': +0.18}
    sample_colors = {
        'exploratory': Colors.SLATE,
        'confirmatory': Colors.CERULEAN2,
    }
    sample_markers = {'exploratory': 'o', 'confirmatory': 's'}

    from matplotlib.patches import Rectangle

    heights = [len(g['tests']) for g in groups]
    n_groups = len(groups)
    fig, axes = plt.subplots(
        n_groups, 1,
        figsize=(9.5, 1.0 * sum(heights) + 1.6 * n_groups),
        sharex=True,
        gridspec_kw={'height_ratios': heights, 'hspace': 0.9},
    )
    if n_groups == 1:
        axes = [axes]

    # Determine a common x-range with padding
    all_lo = fr['lo'].values
    all_hi = fr['hi'].values
    x_min = float(np.nanmin(all_lo))
    x_max = float(np.nanmax(all_hi))
    x_pad = 0.12 * max(abs(x_min), abs(x_max))
    x_lo = min(x_min - x_pad, -0.20)  # ensure ROPE band visible
    x_hi = x_max + x_pad

    legend_seen = set()
    for ax, group in zip(axes, groups):
        group_tests = group['tests']
        y_positions = {t_key: i for i, (t_key, _) in enumerate(group_tests)}

        # ROPE band (for H5c-anx null prediction)
        if group['rope'] and group['rope'] in y_positions:
            y_rope = y_positions[group['rope']]
            rope_rect = Rectangle(
                (-0.10, y_rope - 0.42), 0.20, 0.84,
                facecolor=Colors.SLATE, alpha=0.22,
                edgecolor='none', zorder=0,
            )
            ax.add_patch(rope_rect)
            # Label the ROPE band subtly
            ax.text(0, y_rope - 0.48, 'ROPE', fontsize=8,
                    color=Colors.INK, alpha=0.7,
                    ha='center', va='bottom', zorder=1)

        # Reference line at β = 0
        ax.axvline(0, color=Colors.DARK_GREY, lw=1.0, alpha=0.45,
                   linestyle='--', zorder=1)

        # Plot data for this group's tests
        for _, row in fr.iterrows():
            if row['test'] not in y_positions:
                continue
            y = y_positions[row['test']] + sample_offsets[row['sample']]
            c = sample_colors[row['sample']]
            m = sample_markers[row['sample']]
            label = ('Exploratory' if row['sample'] == 'exploratory'
                     else 'Confirmatory')

            ax.plot([row['lo'], row['hi']], [y, y],
                    color=c, lw=2.0, alpha=0.85, zorder=2,
                    solid_capstyle='round')
            for x_cap in [row['lo'], row['hi']]:
                ax.plot([x_cap, x_cap], [y - 0.09, y + 0.09],
                        color=c, lw=2.0, zorder=2, solid_capstyle='round')
            ax.scatter(row['beta'], y, s=110, marker=m,
                       facecolor=c, edgecolors='white', linewidths=1.2,
                       zorder=3,
                       label=(label if label not in legend_seen else None))
            legend_seen.add(label)

        ax.set_yticks(list(y_positions.values()))
        ax.set_yticklabels([label for _, label in group_tests],
                            fontsize=10, color=Colors.INK)
        ax.invert_yaxis()
        ax.set_facecolor('#FCFCFD')
        ax.set_xlim(x_lo, x_hi)

        # Sub-panel title with the H5 ID in bold
        ax.set_title(
            f'$\\bf{{{group["id"]}}}$ — {group["title"]}',
            fontsize=11, color=Colors.DARK_GREY, loc='left', pad=6)

        style_axis(ax)
        ax.tick_params(colors=Colors.INK, labelsize=9)
        # Tighten y-limits per panel
        n = len(group_tests)
        ax.set_ylim(n - 0.45, -0.55)

    # X-label on the bottom panel only
    axes[-1].set_xlabel(r'Standardized $\beta$ (95% CI)',
                        fontsize=11, color=Colors.INK)

    # Legend on the first (top) panel
    leg = axes[0].legend(fontsize=9, loc='upper right', framealpha=0.92)
    leg.get_frame().set_facecolor('white')
    leg.get_frame().set_edgecolor('#E5E7EB')
    leg.get_frame().set_linewidth(0.8)

    fig.suptitle('H5: Metacognitive monitoring',
                 fontsize=13, color=Colors.DARK_GREY, y=1.00)

    plt.tight_layout()
    fig.savefig(OUT_DIR / 'h5_forest.png', dpi=300,
                bbox_inches='tight', facecolor='white')
    fig.savefig(OUT_DIR / 'h5_forest.pdf',
                bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print('  Saved h5_forest (grouped)')


# ============================================================
# H5 clinical extension heatmap
# ============================================================


def plot_h5_clinical():
    """Heatmap of task affect signals → clinical symptom dimensions.

    Rows: mean anxiety, mean confidence, calibration (task-level metacog)
    Cols: DASS21_Anxiety, DASS21_Depression, AMI_Total (clinical scales)
    Cells: standardized β from univariate OLS (clinical_z ~ affect_z)
           coloured with a diverging colormap; significant cells marked.

    Samples are pooled across exploratory and confirmatory to maximize
    power for the clinical regressions.
    """
    from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm

    if not samples:
        print('  Skipping h5_clinical — no samples loaded')
        return

    # Pool samples: combine the task-derived subject DFs with psych data
    rows_list = []
    for name, s in samples.items():
        df_task = sample_dfs.get(name)
        psych = s.get('psych')
        if df_task is None or psych is None:
            continue
        merged = df_task.merge(psych, on='subj', how='inner')
        merged['_sample'] = name
        rows_list.append(merged)
    if not rows_list:
        print('  Skipping h5_clinical — missing psych or derived data')
        return
    pooled = pd.concat(rows_list, ignore_index=True)

    affects = [
        ('mean_anxiety', 'Task anxiety'),
        ('mean_confidence', 'Task confidence'),
        ('calibration', 'Anxiety calibration'),
    ]
    clinical = [
        ('DASS21_Anxiety', 'DASS-21 Anxiety'),
        ('DASS21_Depression', 'DASS-21 Depression'),
        ('AMI_Total', 'AMI Apathy'),
    ]

    # Verify clinical columns exist
    missing = [c for c, _ in clinical if c not in pooled.columns]
    if missing:
        print(f'  Skipping h5_clinical — missing clinical columns: {missing}')
        return

    n_rows = len(affects)
    n_cols = len(clinical)
    beta_mat = np.full((n_rows, n_cols), np.nan)
    sig_mat = np.zeros((n_rows, n_cols), dtype=bool)

    for i, (a_col, _) in enumerate(affects):
        if a_col not in pooled.columns:
            continue
        x = _zsc(pooled[a_col].values)
        for j, (c_col, _) in enumerate(clinical):
            y = _zsc(pooled[c_col].values)
            beta, lo, hi = _ols_beta_with_ci(y, x, focal_idx=0)
            beta_mat[i, j] = beta
            sig_mat[i, j] = (np.isfinite(lo) and np.isfinite(hi)
                             and (lo > 0 or hi < 0))

    # Diverging colormap matching project palette (cerulean ↔ ruby)
    cmap = LinearSegmentedColormap.from_list(
        'cerulean_ruby',
        [
            '#003C7A',
            Colors.CERULEAN2,
            '#FFD0D8',
            Colors.RUBY1,
            '#5A0826',
        ],
        N=256,
    )

    # Symmetric norm around 0
    beta_abs_max = float(np.nanmax(np.abs(beta_mat))) if np.any(np.isfinite(beta_mat)) else 0.3
    vmax = max(beta_abs_max, 0.25)
    norm = TwoSlopeNorm(vmin=-vmax, vcenter=0.0, vmax=vmax)

    fig, ax = plt.subplots(figsize=(6.5, 4.0))
    im = ax.imshow(beta_mat, cmap=cmap, norm=norm, aspect='auto')

    # Annotate each cell with β value (* if significant)
    for i in range(n_rows):
        for j in range(n_cols):
            val = beta_mat[i, j]
            if not np.isfinite(val):
                continue
            txt = f'{val:+.2f}'
            if sig_mat[i, j]:
                txt += '*'
            # Contrast colour
            txt_color = 'white' if abs(val) > 0.15 else Colors.DARK_GREY
            ax.text(j, i, txt, ha='center', va='center',
                    fontsize=11, color=txt_color, fontweight='bold')

    ax.set_xticks(range(n_cols))
    ax.set_xticklabels([name for _, name in clinical], fontsize=10,
                        color=Colors.INK)
    ax.set_yticks(range(n_rows))
    ax.set_yticklabels([name for _, name in affects], fontsize=10,
                        color=Colors.INK)
    ax.set_title(f'Task affect → clinical dimensions  '
                 f'(pooled N={len(pooled)})',
                 fontsize=12, color=Colors.DARK_GREY, pad=12)
    ax.tick_params(colors=Colors.INK)

    cbar = fig.colorbar(im, ax=ax, shrink=0.85, pad=0.04, aspect=22)
    cbar.set_label(r'Standardized $\beta$',
                   fontsize=10, color=Colors.INK, labelpad=10)
    cbar.ax.tick_params(labelsize=8, colors=Colors.INK)

    plt.tight_layout()
    fig.savefig(OUT_DIR / 'h5_clinical.png', dpi=300,
                bbox_inches='tight', facecolor='white')
    fig.savefig(OUT_DIR / 'h5_clinical.pdf',
                bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f'  Saved h5_clinical (pooled N={len(pooled)})')


# ============================================================
# Panel E — H5 convergent-validity strip for the combined H4+H5 figure
# ============================================================


def plot_h5_panel_e(sample_name='confirmatory'):
    """Panel E: three-scatter strip summarising H5 (convergent validity).

    Designed to slot into the combined H4+H5 figure as a full-width
    bottom row. Matches the visual style of plot_h4_param_behavior.

        E1  H5a: anxiety calibration → % optimal  (partial residual on ω, κ)
        E2  H5b: anxiety reactivity  → choice shift
        E3  H5c: ω → confidence  vs  ω → anxiety (dissociation, ROPE wedge)
    """
    if sample_name not in sample_dfs:
        print(f'  Skipping h5_panel_e_{sample_name} — sample missing')
        return

    from scipy.stats import linregress
    from matplotlib.lines import Line2D

    df = sample_dfs[sample_name]
    label = samples[sample_name]['label']

    def _ols_line_with_band(x, y, x_grid, n_boot=1000, seed=42):
        """OLS line + 95% bootstrap CI band on x_grid."""
        res = linregress(x, y)
        central = res.intercept + res.slope * x_grid
        rng = np.random.default_rng(seed)
        n = len(x)
        boots = np.empty((n_boot, len(x_grid)))
        for i in range(n_boot):
            idx = rng.integers(0, n, n)
            r = linregress(x[idx], y[idx])
            boots[i] = r.intercept + r.slope * x_grid
        lo = np.percentile(boots, 2.5, axis=0)
        hi = np.percentile(boots, 97.5, axis=0)
        return central, lo, hi, res

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.6))

    # ───────────── E1: calibration → pct_opt (partial on ω, κ) ─────────────
    ax = axes[0]
    mask = df[['omega_z', 'kappa_z', 'calibration', 'pct_opt']].notna().all(axis=1)
    sub = df[mask]
    if len(sub) >= 20:
        # Residualize pct_opt on ω_z and κ_z
        n = len(sub)
        X_ctrl = np.column_stack([
            np.ones(n),
            sub['omega_z'].values,
            sub['kappa_z'].values,
        ])
        y = sub['pct_opt'].values
        beta_ctrl, _, _, _ = np.linalg.lstsq(X_ctrl, y, rcond=None)
        y_resid = y - X_ctrl @ beta_ctrl

        # Standardize calibration within the filtered subset
        cal = sub['calibration'].values
        cal_z = (cal - np.nanmean(cal)) / np.nanstd(cal)

        ax.scatter(cal_z, y_resid,
                   c=Colors.CERULEAN2, s=44, alpha=0.85,
                   edgecolors='white', linewidth=0.5, zorder=3)

        x_grid = np.linspace(np.nanpercentile(cal_z, 1),
                              np.nanpercentile(cal_z, 99), 120)
        central, lo, hi, _ = _ols_line_with_band(cal_z, y_resid, x_grid)
        ax.fill_between(x_grid, lo, hi, color=Colors.CERULEAN2,
                        alpha=0.20, zorder=2)
        ax.plot(x_grid, central, color=Colors.CERULEAN2, lw=3.0,
                alpha=0.98, zorder=4)

    style_axis(ax,
               xlabel='Anxiety calibration (z)',
               ylabel=r'% optimal  (residualised on $\omega,\ \kappa$)')
    ax.set_facecolor('#FCFCFD')
    ax.set_title(r'$\bf{H5a}$ — calibration adds variance',
                 fontsize=11, color=Colors.DARK_GREY, pad=8, loc='left')

    # ───────────── E2: anxiety slope → choice shift ─────────────
    ax = axes[1]
    mask = df[['anxiety_slope', 'choice_shift']].notna().all(axis=1)
    sub = df[mask]
    if len(sub) >= 20:
        slope_raw = sub['anxiety_slope'].values
        slope_z = (slope_raw - np.nanmean(slope_raw)) / np.nanstd(slope_raw)
        shift = sub['choice_shift'].values

        ax.scatter(slope_z, shift,
                   c=Colors.CERULEAN2, s=44, alpha=0.85,
                   edgecolors='white', linewidth=0.5, zorder=3)

        x_grid = np.linspace(np.nanpercentile(slope_z, 1),
                              np.nanpercentile(slope_z, 99), 120)
        central, lo, hi, _ = _ols_line_with_band(slope_z, shift, x_grid)
        ax.fill_between(x_grid, lo, hi, color=Colors.CERULEAN2,
                        alpha=0.20, zorder=2)
        ax.plot(x_grid, central, color=Colors.CERULEAN2, lw=3.0,
                alpha=0.98, zorder=4)

    style_axis(ax,
               xlabel='Anxiety slope (z-scored)',
               ylabel=r'Choice shift   P(heavy|T=0.1)$-$P(heavy|T=0.9)')
    ax.set_facecolor('#FCFCFD')
    ax.set_title(r'$\bf{H5b}$ — reactivity drives adaptation',
                 fontsize=11, color=Colors.DARK_GREY, pad=8, loc='left')

    # ───────────── E3: ω → confidence  vs  ω → anxiety (dissociation) ─────────────
    ax = axes[2]
    mask = df[['omega_z', 'mean_confidence', 'mean_anxiety']].notna().all(axis=1)
    sub = df[mask]

    if len(sub) >= 20:
        om_z = sub['omega_z'].values
        conf = sub['mean_confidence'].values
        conf_z = (conf - np.nanmean(conf)) / np.nanstd(conf)
        anx = sub['mean_anxiety'].values
        anx_z = (anx - np.nanmean(anx)) / np.nanstd(anx)

        color_conf = Colors.CERULEAN2
        color_anx = Colors.INK  # grey for the null/flat series

        # Shared x-grid for both lines
        x_grid = np.linspace(
            np.nanpercentile(om_z, 1),
            np.nanpercentile(om_z, 99),
            120,
        )

        # ROPE wedge: any line with slope in ±0.10 through origin
        # falls inside this shaded region (z-scored coordinates).
        y_rope_half = 0.10 * np.abs(x_grid)
        ax.fill_between(x_grid, -y_rope_half, y_rope_half,
                        color=Colors.SLATE, alpha=0.18, zorder=0)
        ax.text(x_grid[-1], 0, '  ROPE', color=Colors.INK,
                fontsize=8, alpha=0.75, ha='left', va='center', zorder=1)

        # Confidence (strong negative slope, outside ROPE)
        ax.scatter(om_z, conf_z, c=color_conf, s=44, alpha=0.85,
                   edgecolors='white', linewidth=0.5, zorder=3)
        central, lo, hi, _ = _ols_line_with_band(om_z, conf_z, x_grid)
        ax.fill_between(x_grid, lo, hi, color=color_conf,
                        alpha=0.20, zorder=2)
        ax.plot(x_grid, central, color=color_conf, lw=3.0,
                alpha=0.98, zorder=5)

        # Anxiety (flat slope, inside ROPE)
        ax.scatter(om_z, anx_z, c=color_anx, s=44, alpha=0.85,
                   edgecolors='white', linewidth=0.5, zorder=3)
        central, lo, hi, _ = _ols_line_with_band(om_z, anx_z, x_grid)
        ax.fill_between(x_grid, lo, hi, color=color_anx,
                        alpha=0.20, zorder=2)
        ax.plot(x_grid, central, color=color_anx, lw=3.0,
                alpha=0.98, zorder=5)

        # Legend
        legend_elements = [
            Line2D([0], [0], marker='o', color='w',
                   markerfacecolor=color_conf, markeredgecolor='white',
                   markersize=9, label='Confidence'),
            Line2D([0], [0], marker='o', color='w',
                   markerfacecolor=color_anx, markeredgecolor='white',
                   markersize=9, label='Anxiety'),
        ]
        leg = ax.legend(handles=legend_elements, fontsize=9,
                         loc='upper right', framealpha=0.92)
        leg.get_frame().set_facecolor('white')
        leg.get_frame().set_edgecolor('#E5E7EB')
        leg.get_frame().set_linewidth(0.8)

    style_axis(ax,
               xlabel=r'$\omega$ (z-scored)',
               ylabel='Affect rating (z-scored)')
    ax.set_facecolor('#FCFCFD')
    ax.set_title(r'$\bf{H5c}$ — $\omega$ → coping, not threat appraisal',
                 fontsize=11, color=Colors.DARK_GREY, pad=8, loc='left')

    fig.suptitle(label, fontsize=12, color=Colors.DARK_GREY, y=1.02)
    plt.tight_layout()
    fig.savefig(OUT_DIR / f'h5_panel_e_{sample_name}.png',
                dpi=300, bbox_inches='tight', facecolor='white')
    fig.savefig(OUT_DIR / f'h5_panel_e_{sample_name}.pdf',
                bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f'  Saved h5_panel_e_{sample_name}')


# ============================================================
# Run all
# ============================================================

if __name__ == '__main__':
    print('Generating H5 figures...')
    plot_h5c()
    plot_h5d()
    plot_h5b()
    plot_h5_combined()
    plot_h5_forest()
    plot_h5_clinical()
    for name in sample_dfs.keys():
        plot_h5_panel_e(name)
    print(f'Done. Saved to {OUT_DIR}/')
