"""
H4 Results Figures — Foraging Profiles & Optimality.

Generates:
  h4_omega_kappa_space.png       — omega-kappa scatter (both samples)
  h4_predictions.png             — Forest plot of H4a-d β coefficients
  h4_combined.png                — Full H4 panel (scatter + predictions)
  h4_optimality_surface_*.png    — Expected-earnings surface in (ω, κ) space
                                    with subjects overlaid + optimum marked
  h4_param_behavior_*.png        — 2-panel scatter with ω and κ overlaid

Usage:
  python scripts/plotting/plot_h4.py
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'notebooks', 'analysis'))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.lines import Line2D
from scipy.stats import zscore, linregress
from pathlib import Path

from plotter import Colors, set_plot_style, style_axis

# Apply global style
set_plot_style()
plt.rcParams.update({'savefig.dpi': 300, 'figure.dpi': 300})

OUT_DIR = Path('results/figs/h4')
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Task reward constants
R_H = 5.0   # heavy cookie reward
R_L = 1.0   # light cookie reward
C = 5.0     # capture penalty


# ============================================================
# Load data
# ============================================================

def load_sample(name):
    """Load a sample's data for H4 figures."""
    base = Path(f'data/{name}_350/processed')
    s5 = sorted(base.glob('stage5_*'))[-1]

    beh = pd.read_csv(s5 / 'behavior_rich.csv', low_memory=False)
    tv_path = s5 / 'trial_vigor.csv'
    tv = pd.read_csv(tv_path) if tv_path.exists() else None

    # Load model parameters
    params_path = Path(f'results/stats/joint_optimal/{name}/mcmc_m4_params.csv')
    params = pd.read_csv(params_path)

    # Exclusions
    exclude = [154, 197, 208] if name == 'exploratory' else []
    beh = beh[~beh['subj'].isin(exclude)]
    if tv is not None:
        tv = tv[~tv['subj'].isin(exclude)]
    params = params[~params['subj'].isin(exclude)]

    return {
        'beh': beh, 'vigor': tv, 'params': params,
        'label': f'{"Exploratory" if name == "exploratory" else "Confirmatory"} (N={beh["subj"].nunique()})',
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
# Helpers
# ============================================================

def _compute_angle(omega_z, kappa_z):
    """Compute angular position in standardised omega-kappa space."""
    return np.arctan2(kappa_z, omega_z)


def _regression_with_ci(x, y, n_boot=5000, ci=95):
    """OLS regression with bootstrapped CI for slope."""
    mask = np.isfinite(x) & np.isfinite(y)
    x, y = x[mask], y[mask]
    if len(x) < 10:
        return np.nan, (np.nan, np.nan), np.nan, x, y

    res = linregress(x, y)
    beta = res.slope

    # Bootstrap CI
    rng = np.random.default_rng(42)
    betas = np.empty(n_boot)
    n = len(x)
    for i in range(n_boot):
        idx = rng.integers(0, n, n)
        betas[i] = linregress(x[idx], y[idx]).slope

    lo = np.percentile(betas, (100 - ci) / 2)
    hi = np.percentile(betas, 100 - (100 - ci) / 2)
    r2 = res.rvalue ** 2

    return beta, (lo, hi), r2, x, y


def _plot_regression_panel(ax, x, y, xlabel, ylabel, color=Colors.CERULEAN2):
    """Scatter + regression line + 95% CI band on a single axis."""
    beta, (lo, hi), r2, x_clean, y_clean = _regression_with_ci(x, y)

    # Scatter
    ax.scatter(x_clean, y_clean, s=15, alpha=0.4, color=Colors.SLATE,
               edgecolors='none', zorder=2)

    # Regression line + CI band (trim to inner 98% of x to avoid extrapolation)
    if np.isfinite(beta):
        x_lo_trim = np.percentile(x_clean, 1)
        x_hi_trim = np.percentile(x_clean, 99)
        x_grid = np.linspace(x_lo_trim, x_hi_trim, 200)
        res = linregress(x_clean, y_clean)
        y_hat = res.intercept + res.slope * x_grid

        # Prediction CI from bootstrap resampling
        rng = np.random.default_rng(42)
        n_boot = 2000
        y_boots = np.empty((n_boot, len(x_grid)))
        n = len(x_clean)
        for i in range(n_boot):
            idx = rng.integers(0, n, n)
            b = linregress(x_clean[idx], y_clean[idx])
            y_boots[i] = b.intercept + b.slope * x_grid
        y_lo = np.percentile(y_boots, 2.5, axis=0)
        y_hi = np.percentile(y_boots, 97.5, axis=0)

        ax.plot(x_grid, y_hat, color=color, lw=2.0, zorder=3)
        ax.fill_between(x_grid, y_lo, y_hi, color=color, alpha=0.15, zorder=1)

        # Annotate
        hdi_str = f'[{lo:.3f}, {hi:.3f}]'
        ax.text(0.04, 0.96,
                f'$\\beta$ = {beta:.3f}\nHDI = {hdi_str}',
                transform=ax.transAxes, fontsize=8, color=Colors.INK,
                va='top', ha='left',
                bbox=dict(facecolor='white', edgecolor='#E5E7EB',
                          linewidth=0.8, boxstyle='round,pad=0.3', alpha=0.9))

    style_axis(ax, xlabel=xlabel, ylabel=ylabel)
    ax.set_facecolor('#FCFCFD')


def _compute_subject_metrics(s):
    """Compute per-subject behavioral metrics and z-scored parameters."""
    beh = s['beh']
    tv = s['vigor']
    params = s['params'].copy()

    # Log-transform then z-score (omega/kappa are log-normal with heavy tails)
    params['log_omega'] = np.log(params['omega'].clip(lower=1e-6))
    params['log_kappa'] = np.log(params['kappa'].clip(lower=1e-6))
    params['omega_z'] = zscore(params['log_omega'], nan_policy='omit')
    params['kappa_z'] = zscore(params['log_kappa'], nan_policy='omit')
    params['angle'] = _compute_angle(params['omega_z'], params['kappa_z'])
    params['angle_z'] = zscore(params['angle'], nan_policy='omit')

    # Free-choice trials
    free = beh[beh['type'] == 1].copy()
    free['T_round'] = free['threat'].round(1)

    # --- Escape rate on attack trials ---
    attack = free[free['isAttackTrial'] == 1]
    escape = attack.groupby('subj').apply(
        lambda g: (g['trialEndState'] == 'escaped').mean(), include_groups=False
    ).rename('escape_rate')

    # --- Overcaution: fraction of light choices (higher = more cautious) ---
    light_frac = free.groupby('subj')['choice'].apply(
        lambda g: 1.0 - g.mean()
    ).rename('light_frac')

    # --- Optimal choice and pct_optimal ---
    # EV_H(T,D) = R_H * S_H - (1-S_H)*C  where S depends on threat × distance
    # EV_L(T,D) = R_L * S_L - (1-S_L)*C
    # Simple proxy: survival ≈ exp(-T * D) for distance scaling
    free = free.copy()
    free['S_H'] = np.exp(-free['threat'] * free['distance_H'])
    free['EV_H'] = R_H * free['S_H'] - (1 - free['S_H']) * C
    # Light cookie at distance 1 (always closest)
    free['S_L'] = np.exp(-free['threat'] * 1.0)
    free['EV_L'] = R_L * free['S_L'] - (1 - free['S_L']) * C
    free['optimal'] = (free['EV_H'] > free['EV_L']).astype(int)
    free['is_optimal'] = (free['choice'] == free['optimal']).astype(int)

    pct_optimal = free.groupby('subj')['is_optimal'].mean().rename('pct_optimal')

    # --- Mean vigor ---
    vigor_means = pd.Series(dtype=float, name='mean_vigor')
    if tv is not None:
        tv_valid = tv[(tv['type'] == 1) & tv['norm_rate'].notna()]
        vigor_means = tv_valid.groupby('subj')['norm_rate'].mean().rename('mean_vigor')

    # Merge all
    subj_df = params.set_index('subj')
    subj_df = subj_df.join(escape, how='left')
    subj_df = subj_df.join(light_frac, how='left')
    subj_df = subj_df.join(pct_optimal, how='left')
    subj_df = subj_df.join(vigor_means, how='left')
    subj_df = subj_df.reset_index()

    return subj_df


def _compute_h4_metrics_proper(s, pop):
    """Per-subject H4 metrics using the model's actual survival function.

    Differences from `_compute_subject_metrics`:
      - `pct_optimal` and `oc_ratio` are computed using the model's
        survival function `S = exp(-h * T^γ * D)` (assuming u = req →
        speed = 1 at full engagement) instead of the simplified
        `S = exp(-T·D)`.
      - Adds `oc_ratio` = (#overcautious errors) / (#suboptimal choices).
        This is the metric the H4 prereg actually tests in H4b.
    """
    beh = s['beh']
    tv = s.get('vigor', None)
    params = s['params'].copy()
    gamma = pop['gamma']
    hazard = pop['hazard']

    params['log_omega'] = np.log(params['omega'].clip(lower=1e-6))
    params['log_kappa'] = np.log(params['kappa'].clip(lower=1e-6))
    params['omega_z'] = zscore(params['log_omega'], nan_policy='omit')
    params['kappa_z'] = zscore(params['log_kappa'], nan_policy='omit')
    params['angle'] = _compute_angle(params['omega_z'], params['kappa_z'])
    params['angle_z'] = zscore(params['angle'], nan_policy='omit')

    free = beh[beh['type'] == 1].copy()

    # Escape rate (attack trials only)
    attack = free[free['isAttackTrial'] == 1]
    escape = attack.groupby('subj').apply(
        lambda g: (g['trialEndState'] == 'escaped').mean(),
        include_groups=False,
    ).rename('escape_rate')

    # Proper survival under model: assume u = req (full engagement),
    # so speed(u) saturates and the surviving function is just
    # S = exp(-h * T^γ * D)
    free['S_H'] = np.exp(-hazard * free['threat'] ** gamma * free['distance_H'])
    free['S_L'] = np.exp(-hazard * free['threat'] ** gamma * 1.0)
    free['EV_H'] = R_H * free['S_H'] - (1 - free['S_H']) * C
    free['EV_L'] = R_L * free['S_L'] - (1 - free['S_L']) * C
    free['optimal_high'] = (free['EV_H'] > free['EV_L']).astype(int)
    free['is_optimal'] = (free['choice'] == free['optimal_high']).astype(int)
    free['is_suboptimal'] = (1 - free['is_optimal']).astype(int)
    # Overcautious error: chose light when heavy was optimal
    free['is_oc_error'] = ((free['choice'] == 0) &
                            (free['optimal_high'] == 1)).astype(int)

    pct_opt = free.groupby('subj')['is_optimal'].mean().rename('pct_optimal')

    # Overcaution ratio = #overcautious / #suboptimal (per subject)
    # NaN when subject has no suboptimal choices
    sub_counts = free.groupby('subj').agg(
        n_subopt=('is_suboptimal', 'sum'),
        n_oc=('is_oc_error', 'sum'),
    )
    oc_ratio = (sub_counts['n_oc'] /
                sub_counts['n_subopt'].replace(0, np.nan)).rename('oc_ratio')

    # Mean vigor
    vigor_means = pd.Series(dtype=float, name='mean_vigor')
    if tv is not None:
        tv_valid = tv[(tv['type'] == 1) & tv['norm_rate'].notna()]
        vigor_means = tv_valid.groupby('subj')['norm_rate'].mean().rename('mean_vigor')

    # Light fraction (kept for backward compat / phase scatters)
    light_frac = free.groupby('subj')['choice'].apply(
        lambda g: 1.0 - g.mean()
    ).rename('light_frac')

    subj_df = params.set_index('subj')
    subj_df = subj_df.join(escape, how='left')
    subj_df = subj_df.join(light_frac, how='left')
    subj_df = subj_df.join(oc_ratio, how='left')
    subj_df = subj_df.join(pct_opt, how='left')
    subj_df = subj_df.join(vigor_means, how='left')
    return subj_df.reset_index()


def _ols_beta_with_ci(y, X, focal_idx=0, ci=0.95):
    """OLS estimate of β for the focal coefficient with normal-theory CI.

    Parameters
    ----------
    y : (n,) array — outcome
    X : (n, k) array — covariates (DO NOT include intercept; added internally)
    focal_idx : index of the focal coefficient (0 = first column of X)
    ci : confidence level (default 0.95)

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


# ============================================================
# Figure 1: omega-kappa parameter space scatter
# ============================================================

def plot_omega_kappa_space():
    n_samples = len(samples)
    fig, axes = plt.subplots(1, n_samples, figsize=(4.5 * n_samples, 4.0), sharey=True)
    if n_samples == 1:
        axes = [axes]

    for ax, (name, s) in zip(axes, samples.items()):
        params = s['params'].copy()
        params['log_omega'] = np.log(params['omega'].clip(lower=1e-6))
        params['log_kappa'] = np.log(params['kappa'].clip(lower=1e-6))
        params['omega_z'] = zscore(params['log_omega'], nan_policy='omit')
        params['kappa_z'] = zscore(params['log_kappa'], nan_policy='omit')
        angle = _compute_angle(params['omega_z'], params['kappa_z'])

        # Diverging colormap: blue (threat-driven) to red (effort-driven)
        from matplotlib.colors import TwoSlopeNorm
        norm = TwoSlopeNorm(vmin=-np.pi, vcenter=0, vmax=np.pi)
        cmap = plt.cm.RdBu_r

        sc = ax.scatter(params['omega'], params['kappa'],
                        c=angle, cmap=cmap, norm=norm,
                        s=25, alpha=0.7, edgecolors='white', linewidth=0.3,
                        zorder=3)

        # Log scale for both axes (parameters are log-normal)
        ax.set_xscale('log')
        ax.set_yscale('log')

        # Cross-hairs at median
        med_omega = params['omega'].median()
        med_kappa = params['kappa'].median()
        ax.axvline(med_omega, color=Colors.INK, lw=0.8, ls='--', alpha=0.4, zorder=1)
        ax.axhline(med_kappa, color=Colors.INK, lw=0.8, ls='--', alpha=0.4, zorder=1)

        style_axis(ax,
                   xlabel='$\\omega$ (capture cost sensitivity)',
                   ylabel='$\\kappa$ (effort discounting)' if ax == axes[0] else None)
        ax.set_facecolor('#FCFCFD')
        ax.set_title(s['label'], fontsize=11, color=Colors.DARK_GREY, pad=10)

    # Colorbar (added after layout — bbox_inches='tight' handles spacing)
    cbar = fig.colorbar(sc, ax=axes, shrink=0.7, pad=0.03, aspect=25)
    cbar.set_label('Angle in $\\omega$-$\\kappa$ space', fontsize=9, color=Colors.INK)
    cbar.ax.tick_params(labelsize=8, colors=Colors.INK)

    fig.subplots_adjust(wspace=0.25)
    fig.savefig(OUT_DIR / 'h4_omega_kappa_space.png', dpi=300, bbox_inches='tight', facecolor='white')
    fig.savefig(OUT_DIR / 'h4_omega_kappa_space.pdf', bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print('  Saved h4_omega_kappa_space')


# ============================================================
# Figure 2: Four regression panels (confirmatory)
# ============================================================

def plot_predictions():
    """H4 forest plot — β coefficients with 95% CI for all sub-tests.

    Each row is one preregistered H4 sub-test (a-d). Both samples are
    plotted side-by-side for replication. Coefficients are computed from
    OLS (matches Bayesian within rounding when priors are weak):

      H4a:  escape_rate ~ omega_z + kappa_z   (focal: omega)
      H4b:  oc_ratio    ~ omega_z              (focal: omega)
      H4c:  mean_vigor  ~ kappa_z              (focal: kappa)
      H4d:  pct_optimal ~ angle_z              (focal: angle)

    H4b uses the proper `oc_ratio` (overcaution among errors) instead of
    raw light-choice fraction. H4d uses the proper survival function for
    the optimality labels.
    """
    if not samples:
        print('  Skipping h4_predictions — no samples loaded')
        return

    # Compute β/CI for each sub-test in each sample
    rows = []
    for name, s in samples.items():
        try:
            pop = _load_pop_params_h4(name)
        except Exception as e:
            print(f'  Skipping {name}: failed to load pop params ({e})')
            continue
        df = _compute_h4_metrics_proper(s, pop)

        # H4a — multivariate (control for kappa)
        beta, lo, hi = _ols_beta_with_ci(
            df['escape_rate'].values,
            df[['omega_z', 'kappa_z']].values,
            focal_idx=0,
        )
        rows.append({'sample': name, 'test': 'H4a',
                     'beta': beta, 'lo': lo, 'hi': hi})

        # H4b — univariate, using proper oc_ratio
        beta, lo, hi = _ols_beta_with_ci(
            df['oc_ratio'].values,
            df[['omega_z']].values,
            focal_idx=0,
        )
        rows.append({'sample': name, 'test': 'H4b',
                     'beta': beta, 'lo': lo, 'hi': hi})

        # H4c — univariate
        beta, lo, hi = _ols_beta_with_ci(
            df['mean_vigor'].values,
            df[['kappa_z']].values,
            focal_idx=0,
        )
        rows.append({'sample': name, 'test': 'H4c',
                     'beta': beta, 'lo': lo, 'hi': hi})

        # H4d — univariate, using proper pct_optimal
        beta, lo, hi = _ols_beta_with_ci(
            df['pct_optimal'].values,
            df[['angle_z']].values,
            focal_idx=0,
        )
        rows.append({'sample': name, 'test': 'H4d',
                     'beta': beta, 'lo': lo, 'hi': hi})

    fr = pd.DataFrame(rows)

    # ── Three side-by-side square forest sub-panels ──
    groups = [
        {
            'title': r'Avoidance  —  capture cost ($\omega$)',
            'tests': [
                ('H4a', 'Escape rate'),
                ('H4b', 'Overcaution'),
            ],
        },
        {
            'title': r'Activation  —  effort cost ($\kappa$)',
            'tests': [
                ('H4c', 'Mean vigor'),
            ],
        },
        {
            'title': 'Decision quality  —  balance',
            'tests': [
                ('H4d', '% optimal'),
            ],
        },
    ]

    sample_offsets = {'exploratory': -0.22, 'confirmatory': +0.22}
    sample_colors = {
        'exploratory': Colors.SLATE,
        'confirmatory': Colors.CERULEAN2,
    }
    sample_markers = {'exploratory': 'o', 'confirmatory': 's'}

    # Common x-range across all panels with padding
    all_lo = fr['lo'].values
    all_hi = fr['hi'].values
    x_min = float(np.nanmin(all_lo))
    x_max = float(np.nanmax(all_hi))
    x_pad = 0.12 * max(abs(x_min), abs(x_max))
    x_lo = x_min - x_pad
    x_hi = x_max + x_pad

    n_groups = len(groups)
    fig, axes = plt.subplots(
        1, n_groups,
        figsize=(4.5 * n_groups, 4.2),
        sharex=True,
    )
    if n_groups == 1:
        axes = [axes]

    legend_seen = set()
    for ax, group in zip(axes, groups):
        group_tests = group['tests']
        y_positions = {t_key: i for i, (t_key, _) in enumerate(group_tests)}

        ax.axvline(0, color=Colors.DARK_GREY, lw=1.0, alpha=0.45,
                   linestyle='--', zorder=1)

        for _, row in fr.iterrows():
            if row['test'] not in y_positions:
                continue
            y = y_positions[row['test']] + sample_offsets[row['sample']]
            c = sample_colors[row['sample']]
            m = sample_markers[row['sample']]
            label = ('Exploratory' if row['sample'] == 'exploratory'
                     else 'Confirmatory')

            ax.plot([row['lo'], row['hi']], [y, y],
                    color=c, lw=2.4, alpha=0.9, zorder=2,
                    solid_capstyle='round')
            for x_cap in [row['lo'], row['hi']]:
                ax.plot([x_cap, x_cap], [y - 0.10, y + 0.10],
                        color=c, lw=2.4, zorder=2, solid_capstyle='round')
            ax.scatter(row['beta'], y, s=150, marker=m,
                       facecolor=c, edgecolors='white', linewidths=1.4,
                       zorder=3,
                       label=(label if label not in legend_seen else None))
            legend_seen.add(label)

        ax.set_yticks(list(y_positions.values()))
        ax.set_yticklabels([label for _, label in group_tests],
                            fontsize=11, color=Colors.INK)
        ax.invert_yaxis()
        ax.set_facecolor('#FCFCFD')
        ax.set_xlim(x_lo, x_hi)

        ax.set_title(group['title'], fontsize=11,
                     color=Colors.DARK_GREY, pad=10)

        style_axis(ax)
        ax.tick_params(colors=Colors.INK, labelsize=9)
        n = len(group_tests)
        # Pad the y-limit so single-row panels don't look cramped
        y_pad = max(0.6, (2 - n) * 0.4)
        ax.set_ylim(n - 1 + y_pad, -y_pad)
        ax.set_xlabel(r'Standardized $\beta$ (95% CI)',
                      fontsize=10, color=Colors.INK)

    # Legend on the leftmost (first) panel
    leg = axes[0].legend(fontsize=9, loc='lower right', framealpha=0.92)
    leg.get_frame().set_facecolor('white')
    leg.get_frame().set_edgecolor('#E5E7EB')
    leg.get_frame().set_linewidth(0.8)

    plt.tight_layout()
    fig.savefig(OUT_DIR / 'h4_predictions.png', dpi=300,
                bbox_inches='tight', facecolor='white')
    fig.savefig(OUT_DIR / 'h4_predictions.pdf',
                bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print('  Saved h4_predictions (3 square panels)')


# ============================================================
# Figure 3: Combined H4 panel
# ============================================================

def plot_h4_combined():
    n_samples = len(samples)
    has_confirm = 'confirmatory' in samples

    fig = plt.figure(figsize=(4.5 * n_samples, 11.5))
    gs = GridSpec(3, n_samples, height_ratios=[1, 1, 1],
                  hspace=0.45, wspace=0.35,
                  left=0.12, right=0.95, top=0.96, bottom=0.06)

    # ── Row 0: omega-kappa scatter (both samples) ──
    from matplotlib.colors import TwoSlopeNorm
    norm = TwoSlopeNorm(vmin=-np.pi, vcenter=0, vmax=np.pi)
    cmap = plt.cm.RdBu_r

    for col, (name, s) in enumerate(samples.items()):
        ax = fig.add_subplot(gs[0, col])
        params = s['params'].copy()
        params['log_omega'] = np.log(params['omega'].clip(lower=1e-6))
        params['log_kappa'] = np.log(params['kappa'].clip(lower=1e-6))
        params['omega_z'] = zscore(params['log_omega'], nan_policy='omit')
        params['kappa_z'] = zscore(params['log_kappa'], nan_policy='omit')
        angle = _compute_angle(params['omega_z'], params['kappa_z'])

        sc = ax.scatter(params['omega'], params['kappa'],
                        c=angle, cmap=cmap, norm=norm,
                        s=20, alpha=0.7, edgecolors='white', linewidth=0.3,
                        zorder=3)

        ax.set_xscale('log')
        ax.set_yscale('log')

        med_omega = params['omega'].median()
        med_kappa = params['kappa'].median()
        ax.axvline(med_omega, color=Colors.INK, lw=0.8, ls='--', alpha=0.4, zorder=1)
        ax.axhline(med_kappa, color=Colors.INK, lw=0.8, ls='--', alpha=0.4, zorder=1)

        style_axis(ax,
                   xlabel='$\\omega$ (capture cost sensitivity)',
                   ylabel='$\\kappa$ (effort discounting)' if col == 0 else None)
        ax.set_facecolor('#FCFCFD')
        ax.set_title(s['label'], fontsize=11, color=Colors.DARK_GREY, pad=8)

    # ── Rows 1-2: Prediction panels (confirmatory only) ──
    if has_confirm:
        s = samples['confirmatory']
        subj_df = _compute_subject_metrics(s)

        panel_specs = [
            # (row, col, x_col, y_col, xlabel, ylabel, title, color)
            (1, 0, 'omega_z', 'escape_rate',
             '$\\omega$ (z)', 'Escape rate',
             '$\\omega$ $\\rightarrow$ Escape', Colors.RUBY1),
            (1, 1, 'omega_z', 'light_frac',
             '$\\omega$ (z)', 'P(choose light)',
             '$\\omega$ $\\rightarrow$ Overcaution', Colors.PERSIMMON3),
        ]
        # Second row of predictions spans columns if 2 samples, else use col 0/1
        panel_specs += [
            (2, 0, 'kappa_z', 'mean_vigor',
             '$\\kappa$ (z)', 'Mean vigor',
             '$\\kappa$ $\\rightarrow$ Vigor', Colors.CERULEAN2),
            (2, 1, 'angle_z', 'pct_optimal',
             'Angle (z)', '% optimal',
             'Angle $\\rightarrow$ Optimality', Colors.MANTIS1),
        ]

        for row, col_idx, x_col, y_col, xlabel, ylabel, title, color in panel_specs:
            ax = fig.add_subplot(gs[row, col_idx])
            _plot_regression_panel(
                ax,
                subj_df[x_col].values, subj_df[y_col].values,
                xlabel=xlabel, ylabel=ylabel, color=color,
            )
            ax.set_title(title, fontsize=10, color=Colors.DARK_GREY, pad=8)
    else:
        # Blank panels if no confirmatory
        for row in [1, 2]:
            for col_idx in range(n_samples):
                ax = fig.add_subplot(gs[row, col_idx])
                ax.text(0.5, 0.5, 'Confirmatory\nnot available',
                        transform=ax.transAxes, ha='center', va='center',
                        fontsize=10, color=Colors.SLATE)
                style_axis(ax)
                ax.set_facecolor('#FCFCFD')

    plt.savefig(OUT_DIR / 'h4_combined.png', dpi=300, bbox_inches='tight', facecolor='white')
    plt.savefig(OUT_DIR / 'h4_combined.pdf', bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print('  Saved h4_combined')


# ============================================================
# Optimality surface — expected per-trial earnings over (ω, κ)
# ============================================================

C_PEN = 5.0  # capture penalty (points lost)


def _eu_sat_h4(om, kap, T, D, R, req, gamma, hazard, sp, ug):
    """Per-trial optimal vigor u* and subjective value V* under (om, kap).
    Mirrors scripts/plotting/plot_ppc.py:_eu_sat_numpy with numerically
    stable softmax (subtract row max before exp).
    """
    from scipy.special import expit
    u = ug[None, :]
    speed = expit((u - 0.25 * req[:, None]) / sp)
    S = np.exp(-hazard * np.power(T[:, None], gamma) * D[:, None]
               / np.clip(speed, 0.01, None))
    W = (S * R[:, None]
         - (1.0 - S) * om[:, None] * (R[:, None] + C_PEN)
         - kap[:, None] * (u - req[:, None]) ** 2 * D[:, None])

    # Stable soft-argmax: subtract row max before exp
    Wm = W * 20.0
    Wm = Wm - Wm.max(axis=1, keepdims=True)
    w = np.exp(Wm)
    w_sum = w.sum(axis=1, keepdims=True)
    w = w / np.maximum(w_sum, 1e-300)
    u_star = (w * u).sum(axis=1)
    V_star = (w * W).sum(axis=1)
    return u_star, V_star


def _load_pop_params_h4(sample_name):
    diag = pd.read_csv(
        f'results/stats/joint_optimal/{sample_name}/mcmc_convergence_diagnostics.csv')
    m4 = diag[diag['model'] == 'M4'].set_index('parameter')
    return {
        'gamma': float(np.clip(np.exp(m4.loc['gr', 'mean']), 0.1, 3.0)),
        'hazard': float(np.exp(m4.loc['hr', 'mean'])),
        'tau': float(np.clip(np.exp(m4.loc['tr', 'mean']), 0.01, 50.)),
        'sp': float(np.clip(np.exp(m4.loc['spr', 'mean']), 0.01, 1.0)),
        'bc': float(m4.loc['bc', 'mean']),
        'sv': float(m4.loc['sv', 'mean']),
    }


N_FREE_CHOICE_TRIALS = 45  # per-session count for cumulative-earnings metric


def compute_optimality_surface(sample_name, *, n_grid=50,
                                omega_range=(0.05, 25.0),
                                kappa_range=(0.005, 10.0)):
    """Compute model-predicted metrics over the (ω, κ) parameter grid.

    Returns
    -------
    omega_grid : (n_grid,) log-spaced ω values
    kappa_grid : (n_grid,) log-spaced κ values
    metrics    : dict with keys
        'earnings_per_trial'  — expected earnings per trial (pts/trial)
        'cumulative_earnings' — earnings × N_FREE_CHOICE_TRIALS (pts/session)
        'survival_rate'       — average survival probability across cells
                                weighted by P(choose)
    pop        : population-level params used
    """
    from scipy.special import expit

    pop = _load_pop_params_h4(sample_name)

    threats = np.array([0.1, 0.5, 0.9])
    distances_H = np.array([1, 2, 3])
    T_arr, DH_arr = np.meshgrid(threats, distances_H, indexing='ij')
    T_arr = T_arr.ravel().astype(float)
    DH_arr = DH_arr.ravel().astype(float)
    n_cond = T_arr.size
    DL_arr = np.ones(n_cond)
    R_H_arr = np.full(n_cond, 5.0)
    R_L_arr = np.full(n_cond, 1.0)
    req_H = np.full(n_cond, 0.9)
    req_L = np.full(n_cond, 0.4)

    ug = np.linspace(0.1, 1.5, 40)

    omega_grid = np.logspace(np.log10(omega_range[0]),
                              np.log10(omega_range[1]), n_grid)
    kappa_grid = np.logspace(np.log10(kappa_range[0]),
                              np.log10(kappa_range[1]), n_grid)

    earnings = np.zeros((n_grid, n_grid))
    survival = np.zeros((n_grid, n_grid))

    for i, om in enumerate(omega_grid):
        om_arr = np.full(n_cond, om)
        for j, ka in enumerate(kappa_grid):
            ka_arr = np.full(n_cond, ka)

            # Heavy cookie value
            u_star_H, V_H_base = _eu_sat_h4(
                om_arr, ka_arr, T_arr, DH_arr, R_H_arr, req_H,
                pop['gamma'], pop['hazard'], pop['sp'], ug)
            V_H = V_H_base - ka * 0.9 * DH_arr

            # Light cookie value (D_L = 1)
            u_star_L, V_L_base = _eu_sat_h4(
                om_arr, ka_arr, T_arr, DL_arr, R_L_arr, req_L,
                pop['gamma'], pop['hazard'], pop['sp'], ug)
            V_L = V_L_base - ka * 0.4 * DL_arr

            # Choice probability
            P_H = expit(np.clip((V_H - V_L) / pop['tau'], -20, 20))

            # Survival probabilities at the chosen vigor
            speed_H = expit((u_star_H - 0.25 * req_H) / pop['sp'])
            speed_L = expit((u_star_L - 0.25 * req_L) / pop['sp'])
            S_H = np.exp(-pop['hazard'] * np.power(T_arr, pop['gamma']) * DH_arr
                          / np.clip(speed_H, 0.01, None))
            S_L = np.exp(-pop['hazard'] * np.power(T_arr, pop['gamma']) * DL_arr
                          / np.clip(speed_L, 0.01, None))

            # Expected per-trial earnings using actual reward structure
            E_R_H = S_H * R_H_arr - (1 - S_H) * C_PEN
            E_R_L = S_L * R_L_arr - (1 - S_L) * C_PEN
            E_R_trial = P_H * E_R_H + (1 - P_H) * E_R_L
            earnings[j, i] = float(E_R_trial.mean())

            # Average survival across cells, weighted by chosen option
            S_chosen = P_H * S_H + (1 - P_H) * S_L
            survival[j, i] = float(S_chosen.mean())

    metrics = {
        'earnings_per_trial': earnings,
        'cumulative_earnings': earnings * N_FREE_CHOICE_TRIALS,
        'survival_rate': survival,
    }
    return omega_grid, kappa_grid, metrics, pop


_METRIC_LABELS = {
    'earnings_per_trial': 'Expected earnings (pts / trial)',
    'cumulative_earnings': f'Cumulative earnings (pts / {N_FREE_CHOICE_TRIALS} trials)',
    'survival_rate': 'Mean survival probability',
}

_METRIC_FILENAME_SUFFIX = {
    'earnings_per_trial': '',
    'cumulative_earnings': '_cum',
    'survival_rate': '_survival',
}


def plot_optimality_surface(*, n_grid=120, flip_axes=False,
                             metric='cumulative_earnings'):
    """Joint metric landscape over (ω, κ), both samples side by side.

    Two panels (exploratory + confirmatory), each showing the model-
    predicted metric over (ω, κ) coloured with a custom CERULEAN ↔
    RUBY diverging colormap (CERULEAN = below population median,
    RUBY = above), with subjects overlaid as dots. Both panels use a
    SHARED norm (centred on the mean of the two population medians)
    so a single unified colorbar applies across both.

    Axes are LINEAR with log10 values directly (no log scaling), and
    the colormap uses smooth gouraud shading instead of discrete bands.

    Parameters
    ----------
    flip_axes : bool
        If True, plots κ on the X axis and ω on the Y axis (default
        is ω on X, κ on Y). Saves to a different filename.
    metric : {'earnings_per_trial', 'cumulative_earnings', 'survival_rate'}
        Which surface to display. Default: cumulative earnings.
    """
    from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm

    if not samples:
        print('  Skipping h4_optimality_surface — no samples loaded')
        return
    if metric not in _METRIC_LABELS:
        raise ValueError(f'unknown metric: {metric}')

    # Cerulean ↔ ruby diverging colormap (low = cerulean, high = ruby)
    cmap = LinearSegmentedColormap.from_list(
        'cerulean_ruby',
        [
            '#003C7A',           # deep navy (worst)
            Colors.CERULEAN2,    # #1A93FF
            '#FFD0D8',           # pale pink midpoint (red-shifted, no pure white)
            Colors.RUBY1,        # #D4145A
            '#5A0826',           # deep ruby (best)
        ],
        N=256,
    )

    # Compute surface per sample; collect data + sample medians
    panels_data = []
    medians = []
    z_min_all = float('inf')
    z_max_all = float('-inf')
    for name in samples.keys():
        omega_grid, kappa_grid, metrics_dict, _pop = compute_optimality_surface(
            name, n_grid=n_grid)
        z = metrics_dict[metric]
        params = samples[name]['params'].copy()
        med_om = params['omega'].median()
        med_ka = params['kappa'].median()
        i_med = int(np.argmin(np.abs(omega_grid - med_om)))
        j_med = int(np.argmin(np.abs(kappa_grid - med_ka)))
        med_z = float(z[j_med, i_med])
        medians.append(med_z)
        z_min_all = min(z_min_all, float(z.min()))
        z_max_all = max(z_max_all, float(z.max()))
        panels_data.append({
            'name': name,
            'omega_grid': omega_grid,
            'kappa_grid': kappa_grid,
            'earnings': z,
            'params': params,
        })

    # SHARED norm across both samples — centred on average median
    z_center = float(np.mean(medians))
    z_lo = min(z_min_all, z_center - 1e-3)
    z_hi = max(z_max_all, z_center + 1e-3)
    shared_norm = TwoSlopeNorm(vmin=z_lo, vcenter=z_center, vmax=z_hi)

    # Two panels side by side, each with its own colorbar
    n_panels = len(panels_data)
    fig, axes = plt.subplots(1, n_panels, figsize=(7.4 * n_panels, 5.8),
                              sharex=True, sharey=True)
    if n_panels == 1:
        axes = [axes]

    for ax, data in zip(axes, panels_data):
        omega_grid = data['omega_grid']
        kappa_grid = data['kappa_grid']
        earnings = data['earnings']
        params = data['params']

        log_om_grid = np.log10(omega_grid)
        log_ka_grid = np.log10(kappa_grid)
        log_om_subj = np.log10(
            params['omega'].clip(lower=omega_grid[0],
                                  upper=omega_grid[-1]).values)
        log_ka_subj = np.log10(
            params['kappa'].clip(lower=kappa_grid[0],
                                  upper=kappa_grid[-1]).values)

        if flip_axes:
            # κ on x, ω on y → transpose surface
            X, Y = np.meshgrid(log_ka_grid, log_om_grid)
            Z = earnings.T  # earnings is (n_kappa, n_omega) → (n_omega, n_kappa)
            sc_x, sc_y = log_ka_subj, log_om_subj
            x_label = r'$\log_{10}\,\kappa$  (effort discounting)'
            y_label = r'$\log_{10}\,\omega$  (capture cost sensitivity)'
        else:
            # ω on x, κ on y (default)
            X, Y = np.meshgrid(log_om_grid, log_ka_grid)
            Z = earnings
            sc_x, sc_y = log_om_subj, log_ka_subj
            x_label = r'$\log_{10}\,\omega$  (capture cost sensitivity)'
            y_label = r'$\log_{10}\,\kappa$  (effort discounting)'

        # Smooth continuous shading with the SHARED norm
        cs = ax.pcolormesh(X, Y, Z, cmap=cmap, norm=shared_norm,
                           alpha=0.55, zorder=1, shading='auto',
                           rasterized=True)

        # Subject dots
        ax.scatter(sc_x, sc_y, s=22, facecolor='#1f1f1f',
                   edgecolors='white', linewidths=0.4, alpha=0.85,
                   zorder=3)

        ax.set_xlim(float(X.min()), float(X.max()))
        ax.set_ylim(float(Y.min()), float(Y.max()))
        ax.set_xlabel(x_label, fontsize=11, color=Colors.INK)
        if ax is axes[0]:
            ax.set_ylabel(y_label, fontsize=11, color=Colors.INK)
        ax.set_title(samples[data['name']]['label'], fontsize=12,
                     color=Colors.DARK_GREY, pad=10)
        ax.tick_params(colors=Colors.INK, labelsize=9)
        ax.set_facecolor('#FCFCFD')

    # Single shared colorbar (applies to both panels)
    cbar = fig.colorbar(cs, ax=axes, shrink=0.80, pad=0.02, aspect=24)
    cbar.set_label(_METRIC_LABELS[metric],
                   fontsize=10, color=Colors.INK, labelpad=10)
    cbar.ax.tick_params(labelsize=8, colors=Colors.INK)

    flip_suffix = '_flipped' if flip_axes else ''
    metric_suffix = _METRIC_FILENAME_SUFFIX[metric]
    suffix = metric_suffix + flip_suffix
    fig.savefig(OUT_DIR / f'h4_optimality_surface{suffix}.png',
                dpi=300, bbox_inches='tight', facecolor='white')
    fig.savefig(OUT_DIR / f'h4_optimality_surface{suffix}.pdf',
                bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f'  Saved h4_optimality_surface{suffix}  ({metric})')


def plot_optimality_surface_joint(*, n_grid=120, flip_axes=False,
                                    metric='cumulative_earnings'):
    """Single-panel optimality surface combining both samples.

    Both samples share the same (ω, κ) grid, so the surfaces can be
    averaged voxel-wise. All subjects from both samples are overlaid as
    dots on the averaged surface.

    Parameters
    ----------
    flip_axes : bool
        If True, plots κ on X and ω on Y.
    metric : {'earnings_per_trial', 'cumulative_earnings', 'survival_rate'}
        Which surface metric to display.
    """
    from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm

    if not samples:
        print('  Skipping h4_optimality_surface_joint — no samples loaded')
        return
    if metric not in _METRIC_LABELS:
        raise ValueError(f'unknown metric: {metric}')

    cmap = LinearSegmentedColormap.from_list(
        'cerulean_ruby',
        [
            '#003C7A',
            Colors.CERULEAN2,
            '#FFD0D8',           # pale pink midpoint (red-shifted)
            Colors.RUBY1,
            '#5A0826',
        ],
        N=256,
    )

    # Compute surfaces and pool subjects across samples
    surfaces = []
    all_om = []
    all_ka = []
    medians = []
    omega_grid = None
    kappa_grid = None
    for name in samples.keys():
        omega_grid, kappa_grid, metrics_dict, _ = compute_optimality_surface(
            name, n_grid=n_grid)
        z = metrics_dict[metric]
        params = samples[name]['params'].copy()
        med_om = params['omega'].median()
        med_ka = params['kappa'].median()
        i_med = int(np.argmin(np.abs(omega_grid - med_om)))
        j_med = int(np.argmin(np.abs(kappa_grid - med_ka)))
        medians.append(float(z[j_med, i_med]))
        surfaces.append(z)
        all_om.append(params['omega'].values)
        all_ka.append(params['kappa'].values)

    z_avg = np.mean(surfaces, axis=0)
    all_om = np.concatenate(all_om)
    all_ka = np.concatenate(all_ka)
    n_total = len(all_om)

    # Norm centred on the average of the two sample medians
    z_center = float(np.mean(medians))
    z_lo = min(float(z_avg.min()), z_center - 1e-3)
    z_hi = max(float(z_avg.max()), z_center + 1e-3)
    norm = TwoSlopeNorm(vmin=z_lo, vcenter=z_center, vmax=z_hi)

    log_om_grid = np.log10(omega_grid)
    log_ka_grid = np.log10(kappa_grid)
    log_om_subj = np.log10(np.clip(all_om, omega_grid[0], omega_grid[-1]))
    log_ka_subj = np.log10(np.clip(all_ka, kappa_grid[0], kappa_grid[-1]))

    if flip_axes:
        X, Y = np.meshgrid(log_ka_grid, log_om_grid)
        Z = z_avg.T
        sc_x, sc_y = log_ka_subj, log_om_subj
        x_label = r'$\log_{10}\,\kappa$  (effort discounting)'
        y_label = r'$\log_{10}\,\omega$  (capture cost sensitivity)'
    else:
        X, Y = np.meshgrid(log_om_grid, log_ka_grid)
        Z = z_avg
        sc_x, sc_y = log_om_subj, log_ka_subj
        x_label = r'$\log_{10}\,\omega$  (capture cost sensitivity)'
        y_label = r'$\log_{10}\,\kappa$  (effort discounting)'

    fig, ax = plt.subplots(figsize=(8, 6))

    cs = ax.pcolormesh(X, Y, Z, cmap=cmap, norm=norm,
                       alpha=0.55, zorder=1, shading='auto',
                       rasterized=True)

    ax.scatter(sc_x, sc_y, s=20, facecolor='#1f1f1f',
               edgecolors='white', linewidths=0.4, alpha=0.80, zorder=3)

    ax.set_xlim(float(X.min()), float(X.max()))
    ax.set_ylim(float(Y.min()), float(Y.max()))
    ax.set_xlabel(x_label, fontsize=11, color=Colors.INK)
    ax.set_ylabel(y_label, fontsize=11, color=Colors.INK)
    ax.set_title(f'Joint sample (N={n_total})',
                 fontsize=12, color=Colors.DARK_GREY, pad=10)
    ax.tick_params(colors=Colors.INK, labelsize=9)
    ax.set_facecolor('#FCFCFD')

    cbar = fig.colorbar(cs, ax=ax, shrink=0.85, pad=0.04, aspect=22)
    cbar.set_label(_METRIC_LABELS[metric],
                   fontsize=10, color=Colors.INK, labelpad=10)
    cbar.ax.tick_params(labelsize=8, colors=Colors.INK)

    plt.tight_layout()
    flip_suffix = '_flipped' if flip_axes else ''
    metric_suffix = _METRIC_FILENAME_SUFFIX[metric]
    suffix = metric_suffix + flip_suffix
    fig.savefig(OUT_DIR / f'h4_optimality_surface_joint{suffix}.png',
                dpi=300, bbox_inches='tight', facecolor='white')
    fig.savefig(OUT_DIR / f'h4_optimality_surface_joint{suffix}.pdf',
                bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f'  Saved h4_optimality_surface_joint{suffix}'
          f'  ({metric}, N={n_total})')


def plot_optimality_1d(sample_name='confirmatory', *, n_grid=40,
                        joint_axis='sum', show_surface=True):
    """Earnings collapsed onto a 1D joint axis of (ω, κ).

    For each subject, predicted earnings under the model (interpolated
    from the optimality surface) are plotted against a 1D summary of
    their parameter pair: log10(ω) + log10(κ), interpreted as "joint
    log-avoidance." A theoretical curve overlaid shows the *best
    achievable* earnings at each joint value (the upper envelope of
    the surface), making the gap between humans and the optimum
    visible as the vertical distance between the dot cloud and the
    envelope.
    """
    if sample_name not in samples:
        print(f'  Skipping h4_optimality_1d_{sample_name} — sample missing')
        return

    from scipy.interpolate import RegularGridInterpolator
    from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm

    if joint_axis not in ('sum', 'ratio'):
        raise ValueError(f'unknown joint_axis: {joint_axis}')

    s = samples[sample_name]
    omega_grid, kappa_grid, metrics, _pop = compute_optimality_surface(
        sample_name, n_grid=n_grid)
    earnings = metrics['cumulative_earnings']
    log_om_grid = np.log10(omega_grid)
    log_ka_grid = np.log10(kappa_grid)

    interp = RegularGridInterpolator(
        (log_ka_grid, log_om_grid), earnings,
        bounds_error=False, fill_value=None,
    )

    # Per-subject predicted earnings
    params = s['params'].copy()
    log_om_subj = np.log10(np.clip(params['omega'].values, 1e-6, None))
    log_ka_subj = np.log10(np.clip(params['kappa'].values, 1e-6, None))
    log_om_clip = np.clip(log_om_subj, log_om_grid[0], log_om_grid[-1])
    log_ka_clip = np.clip(log_ka_subj, log_ka_grid[0], log_ka_grid[-1])
    points = np.column_stack([log_ka_clip, log_om_clip])
    earnings_subj = interp(points)

    if joint_axis == 'sum':
        joint_x_subj = log_om_subj + log_ka_subj
        x_label = (r'$\log_{10}\,\omega + \log_{10}\,\kappa$  '
                   r'(joint log-avoidance)')
        suffix_axis = ''
    else:  # 'ratio'
        joint_x_subj = log_om_subj - log_ka_subj
        x_label = (r'$\log_{10}\,(\omega / \kappa)$  '
                   r'(threat-vs-effort balance)')
        suffix_axis = '_ratio'

    # Cerulean ↔ ruby colormap (matching the surface plots)
    cmap = LinearSegmentedColormap.from_list(
        'cerulean_ruby',
        [
            '#003C7A',
            Colors.CERULEAN2,
            '#FFD0D8',           # pale pink midpoint (red-shifted)
            Colors.RUBY1,
            '#5A0826',
        ],
        N=256,
    )

    # Norm centred on the sample-median earnings
    valid_e = earnings_subj[np.isfinite(earnings_subj)]
    med_earnings = float(np.median(valid_e))
    e_lo = min(float(earnings.min()), med_earnings - 1e-3)
    e_hi = max(float(earnings.max()), med_earnings + 1e-3)
    norm = TwoSlopeNorm(vmin=e_lo, vcenter=med_earnings, vmax=e_hi)

    fig, ax = plt.subplots(figsize=(8, 5.5))

    # Optional heatmap-style background: project the entire (ω, κ) surface
    # onto the 1D joint axis. Each grid point is a distinct coloured dot
    # (no alpha blending / smoothing — discrete cells).
    if show_surface:
        LO, LA = np.meshgrid(log_om_grid, log_ka_grid)
        if joint_axis == 'sum':
            grid_joint = (LO + LA).flatten()
        else:
            grid_joint = (LO - LA).flatten()
        grid_earnings = earnings.flatten()

        ax.scatter(grid_joint, grid_earnings,
                   c=grid_earnings, cmap=cmap, norm=norm,
                   s=20, alpha=0.55, edgecolors='none', zorder=1)

    # Foreground: subject scatter
    sc = ax.scatter(joint_x_subj, earnings_subj,
                    c=earnings_subj, cmap=cmap, norm=norm,
                    s=44, alpha=0.92, edgecolors='white', linewidth=0.5,
                    zorder=3)

    style_axis(ax,
               xlabel=x_label,
               ylabel=f'Predicted cumulative earnings '
                      f'(pts / {N_FREE_CHOICE_TRIALS} trials)')
    ax.set_facecolor('#FCFCFD')
    ax.set_title(s['label'], fontsize=12, color=Colors.DARK_GREY, pad=10)

    cbar = fig.colorbar(sc, ax=ax, shrink=0.85, pad=0.04, aspect=22)
    cbar.set_label(f'Cumulative earnings (pts / {N_FREE_CHOICE_TRIALS} trials)',
                   fontsize=10, color=Colors.INK, labelpad=10)
    cbar.ax.tick_params(labelsize=8, colors=Colors.INK)

    plt.tight_layout()
    fig.savefig(OUT_DIR / f'h4_optimality_1d{suffix_axis}_{sample_name}.png',
                dpi=300, bbox_inches='tight', facecolor='white')
    fig.savefig(OUT_DIR / f'h4_optimality_1d{suffix_axis}_{sample_name}.pdf',
                bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f'  Saved h4_optimality_1d{suffix_axis}_{sample_name}')


# ============================================================
# H4 parameter–behavior overlay
# ============================================================


def plot_h4_param_behavior(sample_name='confirmatory'):
    """Two-panel scatter showing how each parameter relates to each behavior.

    Both ω and κ are plotted as separate series on the SAME panel,
    sharing the parameter-magnitude X axis (z-scored). This makes the
    avoid-activate dissociation visible as DIFFERENT SLOPES for the two
    parameters within each panel:

      Panel 1 — Y = P(choose High)
        ω slope = steep negative   (ω drives choice)
        κ slope = shallow           (κ doesn't drive choice much)

      Panel 2 — Y = mean vigor
        κ slope = steep negative   (κ drives vigor)
        ω slope = ~flat             (ω doesn't drive vigor)

    Inspired by the layout in examplebehaviorandparameter.png.
    """
    if sample_name not in samples:
        print(f'  Skipping h4_param_behavior_{sample_name} — sample missing')
        return

    from scipy.stats import linregress
    from matplotlib.lines import Line2D

    s = samples[sample_name]
    pop = _load_pop_params_h4(sample_name)
    subj_df = _compute_h4_metrics_proper(s, pop)

    # Z-scored log parameters (so the two series share an X scale)
    log_om = np.log(np.clip(subj_df['omega'].values, 1e-6, None))
    log_ka = np.log(np.clip(subj_df['kappa'].values, 1e-6, None))
    om_z = (log_om - np.nanmean(log_om)) / np.nanstd(log_om)
    ka_z = (log_ka - np.nanmean(log_ka)) / np.nanstd(log_ka)

    p_high = (1.0 - subj_df['light_frac'].values).astype(float)
    vigor = subj_df['mean_vigor'].values

    # Grey ω, blue κ — neutral and saturated, distinct from
    # the choice/vigor channel colours used elsewhere
    color_omega = Colors.INK        # #6B7280 — medium grey
    color_kappa = Colors.CERULEAN2  # #1A93FF — saturated blue

    panels = [
        ('P(choose High)', p_high, (-0.05, 1.05)),
        ('Mean vigor (norm. press rate)', vigor,
         (float(np.nanpercentile(vigor, 2)),
          float(np.nanpercentile(vigor, 98)))),
    ]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5.4), sharex=True)

    def _ols_line_with_band(x, y, x_grid, n_boot=1000, seed=42):
        """Return central OLS line + bootstrap 95% CI band on x_grid."""
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

    for ax, (ylabel, y_vals, ylim) in zip(axes, panels):
        # Common x-grid for both regression lines so the CI bands align
        x_grid = np.linspace(
            min(float(np.nanpercentile(om_z, 1)),
                float(np.nanpercentile(ka_z, 1))),
            max(float(np.nanpercentile(om_z, 99)),
                float(np.nanpercentile(ka_z, 99))),
            120,
        )

        # ω: scatter + regression line + CI band
        valid_om = np.isfinite(om_z) & np.isfinite(y_vals)
        x_om = om_z[valid_om]
        y_om = y_vals[valid_om]
        y_om_plot = np.clip(y_om, ylim[0], ylim[1]) if ylim else y_om
        ax.scatter(x_om, y_om_plot, c=color_omega, s=44, alpha=0.85,
                   edgecolors='white', linewidth=0.5, zorder=2)
        c_om, lo_om, hi_om, _ = _ols_line_with_band(x_om, y_om, x_grid)
        ax.fill_between(x_grid, lo_om, hi_om, color=color_omega,
                        alpha=0.20, zorder=3)
        ax.plot(x_grid, c_om, color=color_omega, lw=3.0, zorder=5,
                alpha=0.98)

        # κ: scatter + regression line + CI band
        valid_ka = np.isfinite(ka_z) & np.isfinite(y_vals)
        x_ka = ka_z[valid_ka]
        y_ka = y_vals[valid_ka]
        y_ka_plot = np.clip(y_ka, ylim[0], ylim[1]) if ylim else y_ka
        ax.scatter(x_ka, y_ka_plot, c=color_kappa, s=44, alpha=0.85,
                   edgecolors='white', linewidth=0.5, zorder=2)
        c_ka, lo_ka, hi_ka, _ = _ols_line_with_band(x_ka, y_ka, x_grid)
        ax.fill_between(x_grid, lo_ka, hi_ka, color=color_kappa,
                        alpha=0.20, zorder=3)
        ax.plot(x_grid, c_ka, color=color_kappa, lw=3.0, zorder=5,
                alpha=0.98)

        style_axis(ax, ylabel=ylabel, xlabel='Parameter magnitude (z-scored)')
        ax.set_facecolor('#FCFCFD')
        if ylim is not None:
            ax.set_ylim(*ylim)

    # Shared legend on the left panel
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor=color_omega,
               markeredgecolor='white', markersize=10,
               label=r'$\omega$  (capture cost)'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor=color_kappa,
               markeredgecolor='white', markersize=10,
               label=r'$\kappa$  (effort cost)'),
    ]
    leg = axes[0].legend(handles=legend_elements, fontsize=10,
                          loc='upper right', framealpha=0.93)
    leg.get_frame().set_facecolor('white')
    leg.get_frame().set_edgecolor('#E5E7EB')
    leg.get_frame().set_linewidth(0.8)

    fig.suptitle(s['label'], fontsize=12, color=Colors.DARK_GREY, y=1.01)
    plt.tight_layout()
    fig.savefig(OUT_DIR / f'h4_param_behavior_{sample_name}.png',
                dpi=300, bbox_inches='tight', facecolor='white')
    fig.savefig(OUT_DIR / f'h4_param_behavior_{sample_name}.pdf',
                bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f'  Saved h4_param_behavior_{sample_name}')


# ============================================================
# Run all
# ============================================================

if __name__ == '__main__':
    print('Generating H4 figures...')
    plot_omega_kappa_space()
    plot_predictions()
    plot_h4_combined()
    # Two-panel optimality surfaces — three metrics × two orientations
    for metric in ('earnings_per_trial', 'cumulative_earnings', 'survival_rate'):
        plot_optimality_surface(metric=metric, flip_axes=False)
        plot_optimality_surface(metric=metric, flip_axes=True)
    # Joint single-panel optimality (averaged across both samples)
    plot_optimality_surface_joint(metric='cumulative_earnings')
    plot_optimality_surface_joint(metric='cumulative_earnings', flip_axes=True)
    # 1D earnings projections (sum and ratio)
    for name in samples.keys():
        plot_optimality_1d(name, joint_axis='sum')
        plot_optimality_1d(name, joint_axis='ratio')
        plot_h4_param_behavior(name)
    print(f'Done. Saved to {OUT_DIR}/')
