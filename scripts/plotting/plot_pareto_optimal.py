"""
Plot Pareto-optimal foraging figure.

Three panels:
  A. Earnings-Effort Tradeoff — actual or model-predicted earnings vs effort cost,
     with Pareto frontier.
  B. Policy Space — P(choose heavy) vs mean press rate, colored by earnings.
  C. Earnings in Cost Space — contour of expected earnings over (c_effort, c_death).

Run:
    python scripts/plotting/plot_pareto_optimal.py [--cumulative]

Flags:
    --cumulative   Use cumulative (total) earnings instead of mean per-trial.
"""

import sys, argparse
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import cm
from scipy.interpolate import griddata

sys.path.insert(0, '/workspace/scripts/modeling')
from optimal_control import (
    soft_optimal_tier, choice_probability,
    REQ_RATE_HEAVY, REQ_RATE_LIGHT,
)
import jax.numpy as jnp

# ── Paths ────────────────────────────────────────────────────────────────────
DATA_DIR = '/workspace/data/exploratory_350/processed/stage5_filtered_data_20260320_191950'
PARAMS_PATH = '/workspace/results/stats/oc_3param_eps_params.csv'
FIG_DIR = '/workspace/results/figs/paper'

# ── Population-level model constants (from fit) ──────────────────────────────
LAMBDA_PARAM = 1.0
TIER_SURV_FRACS = jnp.array([0.0, 0.25, 0.5, 1.0])
T_SCALE = 1.5
TAU = 0.5
C = 5.0
R_H, R_L = 5.0, 1.0

# Task conditions: 3 threats × 3 distances
CONDITIONS = [(t, d) for t in [0.1, 0.5, 0.9] for d in [1, 2, 3]]


def _compute_policy_for_condition(ce, cd, t, d):
    """Compute model-derived policy and objective payoff for one (subject, condition)."""
    T = jnp.array([t])
    ce_a, cd_a = jnp.array([ce]), jnp.array([cd])

    eu_H, u_H, ps_H = soft_optimal_tier(
        ce_a, cd_a, T, jnp.array([float(d)]),
        jnp.array([R_H]), C, jnp.array([REQ_RATE_HEAVY]),
        LAMBDA_PARAM, TIER_SURV_FRACS, T_SCALE,
    )
    eu_L, u_L, ps_L = soft_optimal_tier(
        ce_a, cd_a, T, jnp.array([1.0]),
        jnp.array([R_L]), C, jnp.array([REQ_RATE_LIGHT]),
        LAMBDA_PARAM, TIER_SURV_FRACS, T_SCALE,
    )

    eh, el = float(eu_H[0]), float(eu_L[0])
    p_H = 1 / (1 + np.exp(np.clip(-(eh - el) / TAU, -20, 20)))

    # Objective expected payoff in game points (not subjective utility)
    obj_H = float(ps_H[0]) * R_H - (1 - float(ps_H[0])) * C
    obj_L = float(ps_L[0]) * R_L - (1 - float(ps_L[0])) * C
    obj_payoff = p_H * obj_H + (1 - p_H) * obj_L

    chosen_u = p_H * float(u_H[0]) + (1 - p_H) * float(u_L[0])
    return p_H, chosen_u, obj_payoff


def compute_subject_model_predictions(params):
    """For each subject, compute model-derived policy and expected payoff across conditions."""
    records = []
    for _, row in params.iterrows():
        ce, cd, eps = row['c_effort'], row['c_death'], row['epsilon']

        payoffs, p_heavys, u_stars = [], [], []
        for t, d in CONDITIONS:
            p_H, u, payoff = _compute_policy_for_condition(ce, cd, t, d)
            payoffs.append(payoff)
            p_heavys.append(p_H)
            u_stars.append(u)

        records.append({
            'subj': row['subj'],
            'c_effort': ce, 'c_death': cd, 'epsilon': eps,
            'mean_expected_earnings': np.mean(payoffs),
            'mean_p_heavy': np.mean(p_heavys),
            'mean_u_star': np.mean(u_stars),
            'mean_effort_cost': np.mean([u ** 2 * d
                                         for u, (_, d) in zip(u_stars, CONDITIONS)]),
        })
    return pd.DataFrame(records)


def compute_earnings_landscape(ce_range, cd_range, n_grid=80):
    """Compute expected OBJECTIVE payoff (game points) on a grid of (c_effort, c_death).

    At each grid point, derives the model-optimal policy, then evaluates the
    actual expected game-point payoff of that policy (not subjective utility).
    """
    ce_grid = np.logspace(ce_range[0], ce_range[1], n_grid)
    cd_grid = np.logspace(cd_range[0], cd_range[1], n_grid)
    CE, CD = np.meshgrid(ce_grid, cd_grid)
    PAYOFF = np.zeros_like(CE)

    for i in range(n_grid):
        for j in range(n_grid):
            ce, cd = float(CE[i, j]), float(CD[i, j])
            payoffs = []
            for t, d in CONDITIONS:
                _, _, payoff = _compute_policy_for_condition(ce, cd, t, d)
                payoffs.append(payoff)
            PAYOFF[i, j] = np.mean(payoffs)

    return np.log10(CE), np.log10(CD), PAYOFF


def _smooth_pareto_envelope(x, y, n_bins=50):
    """Build a smooth Pareto envelope by binning x and taking max y per bin."""
    mask = np.isfinite(x) & np.isfinite(y)
    x, y = x[mask], y[mask]
    bins = np.linspace(x.min(), x.max(), n_bins + 1)
    cx, cy = [], []
    for lo, hi in zip(bins[:-1], bins[1:]):
        in_bin = (x >= lo) & (x < hi)
        if in_bin.any():
            cx.append((lo + hi) / 2)
            cy.append(y[in_bin].max())
    # Enforce monotonicity (running max from left to right)
    cy_mono = np.maximum.accumulate(cy)
    return np.array(cx), np.array(cy_mono)


def pareto_frontier(x, y, maximize_x=True, maximize_y=True):
    """Return indices of Pareto-optimal points."""
    pts = np.column_stack([x if maximize_x else -x,
                           y if maximize_y else -y])
    is_pareto = np.ones(len(pts), dtype=bool)
    for i in range(len(pts)):
        if is_pareto[i]:
            # A point dominates i if it's >= on all objectives and > on at least one
            dominated = np.all(pts >= pts[i], axis=1) & np.any(pts > pts[i], axis=1)
            is_pareto[dominated] = False
    return np.where(is_pareto)[0]


def plot_pareto(use_cumulative=False):
    # ── Load data ────────────────────────────────────────────────────────────
    beh = pd.read_csv(f'{DATA_DIR}/behavior.csv')
    params = pd.read_csv(PARAMS_PATH)

    # Actual earnings per trial
    beh['reward_chosen'] = np.where(beh['choice'] == 1, R_H, R_L)
    beh['trial_earning'] = np.where(beh['outcome'] == 0, beh['reward_chosen'], -C)

    subj_stats = beh.groupby('subj').agg(
        cum_earnings=('trial_earning', 'sum'),
        mean_earnings=('trial_earning', 'mean'),
        survival_rate=('outcome', lambda x: 1 - x.mean()),
        heavy_rate=('choice', 'mean'),
    ).reset_index()

    # Model predictions
    print("Computing model predictions per subject...")
    model_df = compute_subject_model_predictions(params)
    df = subj_stats.merge(model_df, on='subj')

    # ── Compute landscape (same for both versions — objective payoff) ────────
    # Range covers where subjects are + extends to show the gradient
    print("Computing earnings landscape...")
    log_ce_grid, log_cd_grid, EU_grid = compute_earnings_landscape(
        ce_range=(-4.0, 0.5), cd_range=(-0.5, 2.5), n_grid=60,
    )

    # ── Figure ───────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle('Optimal Foraging: Where Should You Be?', fontsize=16, fontweight='bold')

    # ── Panel A: Earnings–Effort Tradeoff ────────────────────────────────────
    ax = axes[0]
    x = df['mean_effort_cost'].values

    if use_cumulative:
        y = df['cum_earnings'].values
        y_label = 'Cumulative earnings (45 trials)'
    else:
        y = df['mean_expected_earnings'].values
        y_label = 'Mean expected earnings'

    # Color by log ratio c_death/c_effort
    log_ratio = np.log10(df['c_death'].values / df['c_effort'].values)
    vmin, vmax = np.percentile(log_ratio, [2, 98])

    sc = ax.scatter(x, y, c=log_ratio, cmap='YlGnBu', s=25, alpha=0.7,
                    edgecolors='k', linewidth=0.3, vmin=vmin, vmax=vmax)

    # Smooth Pareto envelope
    front_x, front_y = _smooth_pareto_envelope(x, y, n_bins=40)
    ax.plot(front_x, front_y, 'k-', lw=2.5, label='Pareto front')

    plt.colorbar(sc, ax=ax, label='log₁₀(c_death / c_effort)', shrink=0.8)
    ax.set_xlabel('Mean effort cost (u² × D)', fontsize=11)
    ax.set_ylabel(y_label, fontsize=11)
    ax.set_title('A. Earnings–Effort Tradeoff', fontsize=13, fontweight='bold')
    ax.legend(fontsize=10)

    # No need to clip — objective payoff is on a sensible scale

    # ── Panel B: Policy Space ────────────────────────────────────────────────
    ax = axes[1]
    x_b = df['mean_p_heavy'].values  # model-predicted P(heavy)
    y_b = df['mean_u_star'].values   # model-predicted mean press rate

    if use_cumulative:
        color_b = df['cum_earnings'].values
        clabel_b = 'Cumulative earnings'
    else:
        color_b = df['mean_expected_earnings'].values
        clabel_b = 'Expected earnings'

    # Clip color for outliers
    vmin_b, vmax_b = np.percentile(color_b, [2, 98])
    sc2 = ax.scatter(x_b, y_b, c=color_b, cmap='YlGnBu', s=25, alpha=0.7,
                     edgecolors='k', linewidth=0.3, vmin=vmin_b, vmax=vmax_b)

    plt.colorbar(sc2, ax=ax, label=clabel_b, shrink=0.8)
    ax.set_xlabel('P(choose heavy)', fontsize=11)
    ax.set_ylabel('Mean press rate (u*)', fontsize=11)
    ax.set_title('B. Policy Space', fontsize=13, fontweight='bold')

    # ── Panel C: Earnings in Cost Space ──────────────────────────────────────
    ax = axes[2]

    cf = ax.contourf(log_ce_grid, log_cd_grid, EU_grid, levels=20,
                     cmap='YlGnBu', alpha=0.9)
    plt.colorbar(cf, ax=ax, label='Expected earnings', shrink=0.8)

    # EV-optimal: the payoff is monotonically best at low ce + low cd.
    # Mark the peak within the grid.
    opt_idx = np.unravel_index(np.argmax(EU_grid), EU_grid.shape)
    ax.plot(log_ce_grid[opt_idx], log_cd_grid[opt_idx], 'r*', markersize=18,
            markeredgecolor='darkred', markeredgewidth=1, label='EV-optimal', zorder=10)

    # Overlay subjects
    subj_log_ce = np.log10(df['c_effort'].values)
    subj_log_cd = np.log10(df['c_death'].values)
    # Color subjects by their expected earnings
    ee = df['mean_expected_earnings'].values
    ax.scatter(subj_log_ce, subj_log_cd, c=ee, cmap='YlGnBu',
               s=15, alpha=0.6, edgecolors='k', linewidth=0.3, zorder=5,
               vmin=EU_grid.min(), vmax=EU_grid.max())

    ax.set_xlabel('log₁₀(c_effort)', fontsize=11)
    ax.set_ylabel('log₁₀(c_death)', fontsize=11)
    ax.set_title('C. Earnings in Cost Space', fontsize=13, fontweight='bold')
    ax.legend(fontsize=10, loc='upper left')

    plt.tight_layout()

    suffix = '_cumulative' if use_cumulative else ''
    out = f'{FIG_DIR}/fig_pareto_optimal{suffix}.png'
    fig.savefig(out, dpi=150, bbox_inches='tight')
    print(f"Saved: {out}")

    # Print summary
    print(f"  EV-optimal at: log10(ce)={log_ce_grid[opt_idx]:.2f}, "
          f"log10(cd)={log_cd_grid[opt_idx]:.2f}, "
          f"payoff={EU_grid[opt_idx]:.2f}")
    ee = df['mean_expected_earnings'].values
    print(f"  Subject expected earnings: mean={np.mean(ee):.2f}, "
          f"median={np.median(ee):.2f}, range=[{ee.min():.2f}, {ee.max():.2f}]")

    plt.close()
    return out


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cumulative', action='store_true',
                        help='Use cumulative earnings instead of mean per-trial')
    args = parser.parse_args()

    # Generate both versions for comparison
    print("=" * 60)
    print("Generating mean expected earnings version...")
    print("=" * 60)
    plot_pareto(use_cumulative=False)

    print()
    print("=" * 60)
    print("Generating cumulative earnings version...")
    print("=" * 60)
    plot_pareto(use_cumulative=True)
