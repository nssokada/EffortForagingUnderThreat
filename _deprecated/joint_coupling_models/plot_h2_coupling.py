#!/usr/bin/env python3
"""
H2 coupling figures:
  v1: Scatter + quadrant labels (connects to dissociation story)
  v2: Scatter + marginal density plots

Uses plotter.py styling.
Run: python scripts/plotting/plot_h2_coupling.py
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from scipy import stats
from scipy.stats import gaussian_kde
from scripts.plotting.plotter import Colors, set_plot_style, style_axis

# ── Paths ─────────────────────────────────────────────────────────────────────
DATA_DIR = "data/exploratory_350/processed/stage5_filtered_data_20260320_191950"
OUT_DIR = "results/figs/paper"
os.makedirs(OUT_DIR, exist_ok=True)

# ── Load & compute shift scores ──────────────────────────────────────────────
print("Loading data...")
beh = pd.read_csv(os.path.join(DATA_DIR, "behavior.csv"))
br = pd.read_csv(os.path.join(DATA_DIR, "behavior_rich.csv"))

br["effort_chosen"] = np.where(br["choice"] == 1, br["effort_H"], br["effort_L"])
br["excess_effort"] = br["mean_trial_effort"] - br["effort_chosen"]
br = br.dropna(subset=["excess_effort", "threat", "subj"])

choice_by_threat = beh.groupby(["subj", "threat"])["choice"].mean().unstack("threat")
delta_choice = (choice_by_threat[0.9] - choice_by_threat[0.1]).values

vigor_by_threat = br.groupby(["subj", "threat"])["excess_effort"].mean().unstack("threat")
delta_vigor = (vigor_by_threat[0.9] - vigor_by_threat[0.1]).values

r, p = stats.pearsonr(delta_choice, delta_vigor)
print(f"r = {r:.3f}, p = {p:.2e}, N = {len(delta_choice)}")

# ── Shared constants ──────────────────────────────────────────────────────────
set_plot_style()

# Quadrant colors
C_SHIFT_BOTH = Colors.CERULEAN2   # upper-left: the main story
C_VIGOR_ONLY = Colors.PERSIMMON3  # upper-right
C_CHOICE_ONLY = Colors.SLATE      # lower-left
C_NEITHER = "#CCCCCC"             # lower-right

DOT_SIZE = 36
DOT_EDGE = "white"
DOT_EDGE_W = 0.4
LINE_COLOR = Colors.RUBY1

# Regression
slope, intercept = np.polyfit(delta_choice, delta_vigor, 1)
x_fit = np.linspace(delta_choice.min() - 0.05, delta_choice.max() + 0.05, 200)
y_fit = slope * x_fit + intercept

# Confidence band (95%)
n = len(delta_choice)
x_mean = delta_choice.mean()
ss_x = np.sum((delta_choice - x_mean) ** 2)
residuals = delta_vigor - (slope * delta_choice + intercept)
se_resid = np.sqrt(np.sum(residuals ** 2) / (n - 2))
se_fit = se_resid * np.sqrt(1 / n + (x_fit - x_mean) ** 2 / ss_x)
t_crit = stats.t.ppf(0.975, n - 2)
ci_upper = y_fit + t_crit * se_fit
ci_lower = y_fit - t_crit * se_fit

# Assign quadrant per point
quad = np.where(
    (delta_choice < 0) & (delta_vigor > 0), 0,   # shift both (UL)
    np.where(
        (delta_choice >= 0) & (delta_vigor > 0), 1,  # vigor only (UR)
        np.where(
            (delta_choice < 0) & (delta_vigor <= 0), 2,  # choice only (LL)
            3  # neither (LR)
        )
    )
)
quad_colors = np.array([C_SHIFT_BOTH, C_VIGOR_ONLY, C_CHOICE_ONLY, C_NEITHER])
point_colors = quad_colors[quad]

n_ul = np.sum(quad == 0)
n_ur = np.sum(quad == 1)
n_ll = np.sum(quad == 2)
n_lr = np.sum(quad == 3)

# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 1: Scatter + quadrant labels
# ══════════════════════════════════════════════════════════════════════════════
print("Generating quadrant figure...")
fig1, ax1 = plt.subplots(figsize=(7, 6))

xlim = (delta_choice.min() - 0.1, delta_choice.max() + 0.1)
ylim = (delta_vigor.min() - 0.06, delta_vigor.max() + 0.06)

# Quadrant background fills
ax1.axvspan(xlim[0], 0, ymin=0.5, ymax=1.0, alpha=0.06, color=C_SHIFT_BOTH, zorder=0,
            transform=ax1.get_xaxis_transform())
ax1.axvspan(0, xlim[1], ymin=0.5, ymax=1.0, alpha=0.06, color=C_VIGOR_ONLY, zorder=0,
            transform=ax1.get_xaxis_transform())
ax1.axvspan(xlim[0], 0, ymin=0, ymax=0.5, alpha=0.04, color=C_CHOICE_ONLY, zorder=0,
            transform=ax1.get_xaxis_transform())
ax1.axvspan(0, xlim[1], ymin=0, ymax=0.5, alpha=0.04, color=C_NEITHER, zorder=0,
            transform=ax1.get_xaxis_transform())

# Reference lines
ax1.axvline(0, color=Colors.DARK_GREY, ls="-", lw=0.9, alpha=0.25, zorder=1)
ax1.axhline(0, color=Colors.DARK_GREY, ls="-", lw=0.9, alpha=0.25, zorder=1)

# Scatter — color-coded by quadrant
ax1.scatter(delta_choice, delta_vigor, s=DOT_SIZE, c=point_colors,
            alpha=0.65, edgecolors=DOT_EDGE, linewidths=DOT_EDGE_W, zorder=3)

# Regression line + CI band
ax1.fill_between(x_fit, ci_lower, ci_upper, color=LINE_COLOR, alpha=0.12, zorder=2)
ax1.plot(x_fit, y_fit, color=LINE_COLOR, lw=2.5, zorder=4)

# Quadrant labels — bold count, descriptive subtitle
label_kw = dict(fontsize=9, ha="center", va="center", zorder=5)
count_kw = dict(fontsize=14, ha="center", va="center", fontweight="bold", zorder=5)

# Positions in data coords
cx_l = xlim[0] + (0 - xlim[0]) * 0.5  # center of left half
cx_r = 0 + (xlim[1] - 0) * 0.5        # center of right half
cy_u = 0 + (ylim[1] - 0) * 0.82       # upper region
cy_d = ylim[0] + (0 - ylim[0]) * 0.35  # lower region

ax1.text(cx_l, cy_u, f"{n_ul}", color=C_SHIFT_BOTH, **count_kw)
ax1.text(cx_l, cy_u - 0.055, "Shift both", color=C_SHIFT_BOTH, alpha=0.8, **label_kw)

ax1.text(cx_r, cy_u, f"{n_ur}", color=C_VIGOR_ONLY, **count_kw)
ax1.text(cx_r, cy_u - 0.055, "Vigor only", color=C_VIGOR_ONLY, alpha=0.8, **label_kw)

ax1.text(cx_l, cy_d, f"{n_ll}", color=C_CHOICE_ONLY, **count_kw)
ax1.text(cx_l, cy_d - 0.045, "Choice only", color=C_CHOICE_ONLY, alpha=0.8, **label_kw)

ax1.text(cx_r, cy_d, f"{n_lr}", color=C_NEITHER, **count_kw)
ax1.text(cx_r, cy_d - 0.045, "Neither", color="#999999", alpha=0.8, **label_kw)

# Stats annotation
ax1.text(0.97, 0.03, f"r = {r:.2f}\np < 0.001\nN = {n}",
         transform=ax1.transAxes, fontsize=10, color=Colors.DARK_GREY,
         ha="right", va="bottom", linespacing=1.4,
         bbox=dict(boxstyle="round,pad=0.4", fc="white", ec=Colors.SLATE,
                   lw=0.8, alpha=0.9))

style_axis(ax1,
           xlabel="$\\Delta$Choice  [P(high | T=0.9) $-$ P(high | T=0.1)]",
           ylabel="$\\Delta$Vigor  [excess effort(T=0.9) $-$ excess effort(T=0.1)]")
ax1.set_xlim(xlim)
ax1.set_ylim(ylim)
ax1.set_title("Choice-vigor coupling under threat", fontsize=13,
              fontweight="bold", color=Colors.DARK_GREY, loc="left", pad=10)

fig1.tight_layout()
out1 = os.path.join(OUT_DIR, "fig_h2_coupling_quadrants.pdf")
fig1.savefig(out1, bbox_inches="tight")
fig1.savefig(out1.replace(".pdf", ".png"), bbox_inches="tight")
print(f"Saved: {out1}")
plt.close(fig1)

# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 2: Scatter + marginal densities
# ══════════════════════════════════════════════════════════════════════════════
print("Generating marginal density figure...")

fig2 = plt.figure(figsize=(7, 6))
gs = GridSpec(2, 2, width_ratios=[5, 1.2], height_ratios=[1.2, 5],
              hspace=0.02, wspace=0.02)

ax_main = fig2.add_subplot(gs[1, 0])
ax_top = fig2.add_subplot(gs[0, 0], sharex=ax_main)
ax_right = fig2.add_subplot(gs[1, 1], sharey=ax_main)

# Main scatter — color-coded by quadrant
ax_main.scatter(delta_choice, delta_vigor, s=DOT_SIZE, c=point_colors,
                alpha=0.65, edgecolors=DOT_EDGE, linewidths=DOT_EDGE_W, zorder=3)

# Regression + CI
ax_main.fill_between(x_fit, ci_lower, ci_upper, color=LINE_COLOR, alpha=0.12, zorder=2)
ax_main.plot(x_fit, y_fit, color=LINE_COLOR, lw=2.5, zorder=4)

# Reference lines
ax_main.axvline(0, color=Colors.DARK_GREY, ls="-", lw=0.9, alpha=0.25, zorder=1)
ax_main.axhline(0, color=Colors.DARK_GREY, ls="-", lw=0.9, alpha=0.25, zorder=1)

# Stats
ax_main.text(0.97, 0.03, f"r = {r:.2f}\np < 0.001\nN = {n}",
             transform=ax_main.transAxes, fontsize=10, color=Colors.DARK_GREY,
             ha="right", va="bottom", linespacing=1.4,
             bbox=dict(boxstyle="round,pad=0.4", fc="white", ec=Colors.SLATE,
                       lw=0.8, alpha=0.9))

style_axis(ax_main,
           xlabel="$\\Delta$Choice  [P(high | T=0.9) $-$ P(high | T=0.1)]",
           ylabel="$\\Delta$Vigor  [excess effort(T=0.9) $-$ excess effort(T=0.1)]")

# ── Top marginal: Δchoice ────────────────────────────────────────────────────
kde_x = gaussian_kde(delta_choice, bw_method=0.25)
x_grid = np.linspace(delta_choice.min() - 0.15, delta_choice.max() + 0.15, 300)
density_x = kde_x(x_grid)

# Fill left of zero (most people shift choice negative) vs right
mask_neg = x_grid <= 0
mask_pos = x_grid > 0
ax_top.fill_between(x_grid[mask_neg], density_x[mask_neg], alpha=0.25, color=C_SHIFT_BOTH)
ax_top.fill_between(x_grid[mask_pos], density_x[mask_pos], alpha=0.15, color=C_VIGOR_ONLY)
ax_top.plot(x_grid, density_x, color=Colors.DARK_GREY, lw=1.2, alpha=0.6)
ax_top.axvline(0, color=Colors.DARK_GREY, ls="-", lw=0.9, alpha=0.25)
ax_top.axvline(np.mean(delta_choice), color=Colors.DARK_GREY, ls="--", lw=1.2, alpha=0.5)
ax_top.text(np.mean(delta_choice) - 0.03, ax_top.get_ylim()[1] * 0.7,
            f"M = {np.mean(delta_choice):.2f}", fontsize=8, color=Colors.DARK_GREY,
            ha="right", va="top")

for spine in ax_top.spines.values():
    spine.set_visible(False)
ax_top.set_yticks([])
ax_top.tick_params(labelbottom=False, bottom=False)
ax_top.grid(False)

# ── Right marginal: Δvigor ───────────────────────────────────────────────────
kde_y = gaussian_kde(delta_vigor, bw_method=0.25)
y_grid = np.linspace(delta_vigor.min() - 0.06, delta_vigor.max() + 0.06, 300)
density_y = kde_y(y_grid)

# Fill above zero (most people boost vigor) vs below
mask_above = y_grid >= 0
mask_below = y_grid < 0
ax_right.fill_betweenx(y_grid[mask_above], density_y[mask_above], alpha=0.25, color=C_SHIFT_BOTH)
ax_right.fill_betweenx(y_grid[mask_below], density_y[mask_below], alpha=0.15, color=C_CHOICE_ONLY)
ax_right.plot(density_y, y_grid, color=Colors.DARK_GREY, lw=1.2, alpha=0.6)
ax_right.axhline(0, color=Colors.DARK_GREY, ls="-", lw=0.9, alpha=0.25)
ax_right.axhline(np.mean(delta_vigor), color=Colors.DARK_GREY, ls="--", lw=1.2, alpha=0.5)
ax_right.text(ax_right.get_xlim()[1] * 0.6, np.mean(delta_vigor) + 0.02,
              f"M = {np.mean(delta_vigor):.2f}", fontsize=8, color=Colors.DARK_GREY,
              ha="center", va="bottom")

for spine in ax_right.spines.values():
    spine.set_visible(False)
ax_right.set_xticks([])
ax_right.tick_params(labelleft=False, left=False)
ax_right.grid(False)

# Title
ax_top.set_title("Choice-vigor coupling under threat", fontsize=13,
                 fontweight="bold", color=Colors.DARK_GREY, loc="left", pad=8)

out2 = os.path.join(OUT_DIR, "fig_h2_coupling_marginals.pdf")
fig2.savefig(out2, bbox_inches="tight")
fig2.savefig(out2.replace(".pdf", ".png"), bbox_inches="tight")
print(f"Saved: {out2}")
plt.close(fig2)

print(f"\nQuadrant counts: UL={n_ul}, UR={n_ur}, LL={n_ll}, LR={n_lr}")
print(f"Pct shift-both (UL): {n_ul/len(delta_choice)*100:.1f}%")
print("Done.")
