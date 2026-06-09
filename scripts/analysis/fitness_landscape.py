"""
Fitness landscape over (ω, κ) parameter space.

For each (ω, κ) on a fine grid, compute the expected:
  1. Per-trial earnings (objective: not weighted by ω)
  2. Per-trial survival probability
  3. Combined fitness (survival-weighted earnings)

Procedure for each (ω, κ, T, D):
  - Solve u*_heavy = argmax W(u; ω, κ, T, D_heavy, R=5, req=0.9)
  - Solve u*_light = argmax W(u; ω, κ, T, D=1, R=1, req=0.4)
  - Compute V_heavy = W(u*_heavy; ω, κ, T, D_heavy, R=5, req=0.9)
  - Compute V_light = W(u*_light; ω, κ, T, D=1, R=1, req=0.4)
  - P(heavy) = sigmoid((V_heavy − V_light) / τ)
  - Objective earnings per branch: E[earn|heavy] = S(u*_h, T, D_h) × R_H − (1−S) × C
                                   E[earn|light] = S(u*_l, T, 1) × R_L − (1−S) × C
  - Objective survival per branch: S(u*, T, D)
  - Per-condition expectation: average over branches weighted by P(heavy)

Aggregate by averaging across 9 (T, D_heavy) conditions (equally weighted).

Visualization:
  - Heatmap of E[earnings] in (ω, κ) space
  - Heatmap of E[survival] in (ω, κ) space
  - Heatmap of E[combined] = E[survival] × E[earnings] in (ω, κ) space
  - Optima marked on each
  - Observed fitted (ω, κ) distribution overlaid as scatter

Outputs:
  results/figs/joint_optimal/fitness_landscape.png
  results/stats/joint_optimal/fitness_landscape.csv
"""

from __future__ import annotations

import os
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
os.chdir(REPO_ROOT)
NB_DIR = REPO_ROOT / "notebooks" / "analysis"
sys.path.insert(0, str(NB_DIR))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

from load_data import load_both  # type: ignore


# ── Constants ─────────────────────────────────────────────────────────────
GAMMA = 0.86
HAZARD = 0.832
TAU = 2.01
C_PENALTY = 5.0
R_HEAVY = 5.0
R_LIGHT = 1.0
REQ_HEAVY = 0.9
REQ_LIGHT = 0.4
D_LIGHT = 1.0
D_HEAVY_LEVELS = [1.0, 2.0, 3.0]
T_LEVELS = [0.1, 0.5, 0.9]

U_GRID = np.linspace(0.1, 2.0, 200)
OMEGA_GRID = np.logspace(-1, 1, 30)   # 0.1 to 10
KAPPA_GRID = np.logspace(-1.3, 1.3, 30)  # 0.05 to ~20


def survival(u, T, D):
    return np.exp(-HAZARD * (T ** GAMMA) * D / np.clip(u, 0.1, None))


def W_value(u, T, D, R, req, omega, kappa):
    S = survival(u, T, D)
    return S * R - (1 - S) * omega * (R + C_PENALTY) - kappa * (u - req) ** 2 * D


def solve_optimum_branch(T, D, R, req, omega, kappa):
    W = W_value(U_GRID, T, D, R, req, omega, kappa)
    idx = int(np.argmax(W))
    return U_GRID[idx], W[idx]


def expected_outcomes(omega, kappa):
    """For (ω, κ), compute expected earnings and expected survival across all conditions."""
    cond_earn = []
    cond_surv = []
    for T in T_LEVELS:
        u_L, V_L = solve_optimum_branch(T, D_LIGHT, R_LIGHT, REQ_LIGHT, omega, kappa)
        S_L = float(survival(u_L, T, D_LIGHT))
        earn_L = S_L * R_LIGHT - (1 - S_L) * C_PENALTY
        for D_H in D_HEAVY_LEVELS:
            u_H, V_H = solve_optimum_branch(T, D_H, R_HEAVY, REQ_HEAVY, omega, kappa)
            S_H = float(survival(u_H, T, D_H))
            earn_H = S_H * R_HEAVY - (1 - S_H) * C_PENALTY
            p_heavy = 1.0 / (1.0 + np.exp(-(V_H - V_L) / TAU))
            cond_earn.append(p_heavy * earn_H + (1 - p_heavy) * earn_L)
            cond_surv.append(p_heavy * S_H + (1 - p_heavy) * S_L)
    return float(np.mean(cond_earn)), float(np.mean(cond_surv))


def build_landscape():
    rows = []
    for i, om in enumerate(OMEGA_GRID):
        for j, kp in enumerate(KAPPA_GRID):
            earn, surv = expected_outcomes(om, kp)
            rows.append({"omega": om, "kappa": kp, "earnings": earn, "survival": surv})
        if i % 5 == 0:
            print(f"  ω = {om:.3f} done ({i+1}/{len(OMEGA_GRID)})")
    return pd.DataFrame(rows)


def get_observed_params():
    exp, conf = load_both()
    rows = []
    for sample, d in [("exploratory", exp), ("confirmatory", conf)]:
        m = d["master"].reset_index().rename(columns={"index": "subj"}).copy()
        m["sample"] = sample
        rows.append(m[["subj", "sample", "omega", "kappa"]])
    return pd.concat(rows, ignore_index=True)


def plot_landscapes(land, observed):
    earn_grid = land.pivot(index="omega", columns="kappa", values="earnings").values
    surv_grid = land.pivot(index="omega", columns="kappa", values="survival").values
    combined_grid = earn_grid * surv_grid  # combined fitness

    # Optima
    i_e, j_e = np.unravel_index(np.argmax(earn_grid), earn_grid.shape)
    i_s, j_s = np.unravel_index(np.argmax(surv_grid), surv_grid.shape)
    i_c, j_c = np.unravel_index(np.argmax(combined_grid), combined_grid.shape)

    om_e_opt, kp_e_opt = OMEGA_GRID[i_e], KAPPA_GRID[j_e]
    om_s_opt, kp_s_opt = OMEGA_GRID[i_s], KAPPA_GRID[j_s]
    om_c_opt, kp_c_opt = OMEGA_GRID[i_c], KAPPA_GRID[j_c]

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    titles = [
        f"Expected EARNINGS\nOptimum at ω={om_e_opt:.2f}, κ={kp_e_opt:.2f}",
        f"Expected SURVIVAL\nOptimum at ω={om_s_opt:.2f}, κ={kp_s_opt:.2f}",
        f"Combined FITNESS (earnings × survival)\nOptimum at ω={om_c_opt:.2f}, κ={kp_c_opt:.2f}",
    ]
    grids = [earn_grid, surv_grid, combined_grid]
    opt_points = [(om_e_opt, kp_e_opt), (om_s_opt, kp_s_opt), (om_c_opt, kp_c_opt)]
    cmaps = ["viridis", "plasma", "magma"]

    log_om_min, log_om_max = np.log10(OMEGA_GRID.min()), np.log10(OMEGA_GRID.max())
    log_kp_min, log_kp_max = np.log10(KAPPA_GRID.min()), np.log10(KAPPA_GRID.max())

    for ax, grid, title, (om_opt, kp_opt), cmap in zip(axes, grids, titles, opt_points, cmaps):
        # Display: x=κ, y=ω, both log
        im = ax.imshow(grid, origin="lower", cmap=cmap,
                       extent=[log_kp_min, log_kp_max, log_om_min, log_om_max],
                       aspect="auto")
        plt.colorbar(im, ax=ax)
        ax.set_xlabel("log10(κ)")
        ax.set_ylabel("log10(ω)")
        ax.set_title(title)
        # Mark optimum
        ax.scatter([np.log10(kp_opt)], [np.log10(om_opt)], marker="*",
                   c="red", s=300, edgecolors="white", linewidths=2,
                   label=f"Optimum", zorder=5)
        # Overlay observed subjects
        obs_log_om = np.log10(observed["omega"].clip(0.01, 100))
        obs_log_kp = np.log10(observed["kappa"].clip(0.01, 100))
        ax.scatter(obs_log_kp, obs_log_om, s=3, c="white", alpha=0.3, label="Subjects", zorder=4)
        ax.legend(loc="upper right", fontsize=9)

    plt.tight_layout()
    out_dir = REPO_ROOT / "results" / "figs" / "joint_optimal"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "fitness_landscape.png"
    plt.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"\nSaved figure: {out_path}")

    # Print summary stats
    print("\n" + "=" * 78)
    print("FITNESS LANDSCAPE SUMMARY")
    print("=" * 78)
    print(f"\n  ω grid: [{OMEGA_GRID.min():.3f}, {OMEGA_GRID.max():.3f}] log-spaced, n={len(OMEGA_GRID)}")
    print(f"  κ grid: [{KAPPA_GRID.min():.3f}, {KAPPA_GRID.max():.3f}] log-spaced, n={len(KAPPA_GRID)}")
    print(f"\n  Earnings optimum:  ω* = {om_e_opt:.3f}, κ* = {kp_e_opt:.3f}, max = {earn_grid.max():.3f}")
    print(f"  Survival optimum: ω* = {om_s_opt:.3f}, κ* = {kp_s_opt:.3f}, max = {surv_grid.max():.3f}")
    print(f"  Combined optimum: ω* = {om_c_opt:.3f}, κ* = {kp_c_opt:.3f}, max = {combined_grid.max():.3f}")
    print(f"\n  Observed subject (ω, κ) summary:")
    print(f"    ω: median = {observed['omega'].median():.3f}, "
          f"5th-95th = [{observed['omega'].quantile(0.05):.3f}, {observed['omega'].quantile(0.95):.3f}]")
    print(f"    κ: median = {observed['kappa'].median():.3f}, "
          f"5th-95th = [{observed['kappa'].quantile(0.05):.3f}, {observed['kappa'].quantile(0.95):.3f}]")

    return out_path, (om_e_opt, kp_e_opt), (om_s_opt, kp_s_opt), (om_c_opt, kp_c_opt)


def main():
    print("=" * 78)
    print("FITNESS LANDSCAPE: expected outcomes across (ω, κ) parameter space")
    print("=" * 78)

    print(f"\nGrid: ω in [{OMEGA_GRID.min():.3f}, {OMEGA_GRID.max():.3f}] × "
          f"κ in [{KAPPA_GRID.min():.3f}, {KAPPA_GRID.max():.3f}]")
    print(f"Total grid points: {len(OMEGA_GRID) * len(KAPPA_GRID)}")
    print("\nComputing landscape...")

    land = build_landscape()

    out = REPO_ROOT / "results" / "stats" / "joint_optimal" / "fitness_landscape.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    land.to_csv(out, index=False)
    print(f"\nSaved landscape data: {out}")

    print("\nLoading observed (ω, κ) subjects...")
    observed = get_observed_params()
    print(f"  N = {len(observed)}")

    plot_landscapes(land, observed)


if __name__ == "__main__":
    main()
