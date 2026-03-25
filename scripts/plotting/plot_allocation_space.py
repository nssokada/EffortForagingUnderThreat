#!/usr/bin/env python3
"""
β-δ allocation space: same reactivity magnitude, different outcomes based on direction.
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm, LinearSegmentedColormap
from scipy import stats
from scipy.stats import gaussian_kde
from scripts.plotting.plotter import Colors, set_plot_style, style_axis

OUT_DIR = "results/figs/paper"
os.makedirs(OUT_DIR, exist_ok=True)

# ── Load data ─────────────────────────────────────────────────────────────────
mcmc_choice = pd.read_csv("results/stats/mcmc_choice_params.csv")
mcmc_vigor = pd.read_csv("results/stats/mcmc_vigor_params.csv")
br = pd.read_csv("data/exploratory_350/processed/stage5_filtered_data_20260320_191950/behavior_rich.csv")

merged = mcmc_choice.merge(mcmc_vigor, on='subj')
logb = merged['logb_mcmc'].values
delta = merged['delta_mcmc'].values

logb_z = (logb - logb.mean()) / logb.std()
delta_z = (delta - delta.mean()) / delta.std()

br['effort_chosen'] = np.where(br['choice']==1, br['effort_H'], br['effort_L'])
earnings = br.groupby('subj')['trialReward'].sum().loc[merged['subj']].values

set_plot_style()

# Diverging colormap: red (loss) → white (zero) → blue (gain)
cmap_earn = LinearSegmentedColormap.from_list("earnings", [
    "#D4145A", "#E8788A", "#F5F5F5", "#7DB8E0", "#1A93FF"
])
norm_earn = TwoSlopeNorm(vmin=earnings.min(), vcenter=0, vmax=earnings.max())

# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 1: Scatter in β-δ space, colored by earnings
# ══════════════════════════════════════════════════════════════════════════════
print("Generating allocation scatter...")
fig, ax = plt.subplots(figsize=(7, 6))

sc = ax.scatter(logb_z, delta_z, c=earnings, cmap=cmap_earn, norm=norm_earn,
                s=30, alpha=0.7, edgecolors="white", linewidths=0.3, zorder=3)

# Reference lines
ax.axhline(0, color=Colors.DARK_GREY, ls=":", lw=0.7, alpha=0.3, zorder=1)
ax.axvline(0, color=Colors.DARK_GREY, ls=":", lw=0.7, alpha=0.3, zorder=1)

# Colorbar
cb = fig.colorbar(sc, ax=ax, shrink=0.75, pad=0.02, aspect=25)
cb.set_label("Total earnings (pts)", fontsize=10, color=Colors.INK)
cb.ax.tick_params(labelsize=8, colors=Colors.INK)

style_axis(ax,
           xlabel="log($\\beta$) threat bias in choice (z-scored)",
           ylabel="$\\delta$ vigor mobilization (z-scored)")

fig.tight_layout()
out = os.path.join(OUT_DIR, "fig_allocation_scatter.pdf")
fig.savefig(out, bbox_inches="tight")
fig.savefig(out.replace(".pdf", ".png"), bbox_inches="tight")
print(f"Saved: {out}")
plt.close(fig)

# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 2: Polar — magnitude vs angle, colored by earnings
# ══════════════════════════════════════════════════════════════════════════════
print("Generating polar allocation...")
angle = np.arctan2(delta_z, logb_z)
magnitude = np.sqrt(logb_z**2 + delta_z**2)

fig, ax = plt.subplots(figsize=(7, 6))

sc = ax.scatter(np.degrees(angle), magnitude, c=earnings, cmap=cmap_earn, norm=norm_earn,
                s=30, alpha=0.7, edgecolors="white", linewidths=0.3, zorder=3)

ax.axvline(0, color=Colors.DARK_GREY, ls=":", lw=0.7, alpha=0.3)
ax.axvline(45, color=Colors.SLATE, ls="--", lw=1, alpha=0.4)  # equal split line

cb = fig.colorbar(sc, ax=ax, shrink=0.75, pad=0.02, aspect=25)
cb.set_label("Total earnings (pts)", fontsize=10, color=Colors.INK)
cb.ax.tick_params(labelsize=8, colors=Colors.INK)

style_axis(ax,
           xlabel="Allocation angle (°): 0° = pure $\\beta$, 90° = pure $\\delta$",
           ylabel="Reactivity magnitude")

fig.tight_layout()
out = os.path.join(OUT_DIR, "fig_allocation_polar.pdf")
fig.savefig(out, bbox_inches="tight")
fig.savefig(out.replace(".pdf", ".png"), bbox_inches="tight")
print(f"Saved: {out}")
plt.close(fig)

# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 3: Earnings surface with contours + scatter overlay
# ══════════════════════════════════════════════════════════════════════════════
print("Generating earnings surface...")
import statsmodels.api as sm

X = pd.DataFrame({
    'logb_z': logb_z, 'delta_z': delta_z,
    'logb_sq': logb_z**2, 'delta_sq': delta_z**2,
    'interaction': logb_z * delta_z
})
X_const = sm.add_constant(X)
model = sm.OLS(earnings, X_const).fit()

# Prediction grid
grid_b = np.linspace(logb_z.min() - 0.3, logb_z.max() + 0.3, 100)
grid_d = np.linspace(delta_z.min() - 0.3, delta_z.max() + 0.3, 100)
Gb, Gd = np.meshgrid(grid_b, grid_d)
Xgrid = pd.DataFrame({
    'logb_z': Gb.ravel(), 'delta_z': Gd.ravel(),
    'logb_sq': Gb.ravel()**2, 'delta_sq': Gd.ravel()**2,
    'interaction': Gb.ravel() * Gd.ravel()
})
Xgrid_const = sm.add_constant(Xgrid)
pred_earn = model.predict(Xgrid_const).values.reshape(Gb.shape)

fig, ax = plt.subplots(figsize=(7, 6))

# Filled contours of predicted earnings
cmap_surface = LinearSegmentedColormap.from_list("surface", [
    "#D4145A", "#E8788A", "#F5F5F5", "#C5DDFF", "#7DB8E0", "#1A93FF"
])
norm_surface = TwoSlopeNorm(vmin=pred_earn.min(), vcenter=0, vmax=pred_earn.max())
cf = ax.contourf(Gb, Gd, pred_earn, levels=15, cmap=cmap_surface, norm=norm_surface, zorder=1)
ax.contour(Gb, Gd, pred_earn, levels=[0], colors=[Colors.DARK_GREY], linewidths=1.5,
           linestyles="-", zorder=2)  # zero-earnings contour

# Scatter on top
ax.scatter(logb_z, delta_z, c=earnings, cmap=cmap_earn, norm=norm_earn,
           s=22, alpha=0.6, edgecolors="white", linewidths=0.3, zorder=3)

# Reference
ax.axhline(0, color=Colors.DARK_GREY, ls=":", lw=0.7, alpha=0.3, zorder=1)
ax.axvline(0, color=Colors.DARK_GREY, ls=":", lw=0.7, alpha=0.3, zorder=1)

cb = fig.colorbar(cf, ax=ax, shrink=0.75, pad=0.02, aspect=25)
cb.set_label("Predicted earnings (pts)", fontsize=10, color=Colors.INK)
cb.ax.tick_params(labelsize=8, colors=Colors.INK)

style_axis(ax,
           xlabel="log($\\beta$) threat bias in choice (z-scored)",
           ylabel="$\\delta$ vigor mobilization (z-scored)")

fig.tight_layout()
out = os.path.join(OUT_DIR, "fig_allocation_surface.pdf")
fig.savefig(out, bbox_inches="tight")
fig.savefig(out.replace(".pdf", ".png"), bbox_inches="tight")
print(f"Saved: {out}")
plt.close(fig)

# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 4: The punchline — same magnitude, different outcomes
# ══════════════════════════════════════════════════════════════════════════════
print("Generating magnitude-vs-angle punchline...")

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Panel A: magnitude → earnings (no relationship)
ax = axes[0]
ax.scatter(magnitude, earnings, s=22, c=Colors.SLATE, alpha=0.4, edgecolors="none")
slope, intercept = np.polyfit(magnitude, earnings, 1)
x_fit = np.linspace(magnitude.min(), magnitude.max(), 100)
ax.plot(x_fit, slope * x_fit + intercept, color=Colors.SLATE, lw=2)
r_mag, p_mag = stats.pearsonr(magnitude, earnings)
style_axis(ax, xlabel="Reactivity magnitude ($\\sqrt{\\beta_z^2 + \\delta_z^2}$)",
           ylabel="Total earnings (pts)")

# Panel B: angle → earnings (strong relationship)
ax = axes[1]
ax.scatter(np.degrees(angle), earnings, s=22, c=Colors.CERULEAN2, alpha=0.4, edgecolors="none")
slope2, intercept2 = np.polyfit(np.degrees(angle), earnings, 1)
x_fit2 = np.linspace(np.degrees(angle).min(), np.degrees(angle).max(), 100)
ax.plot(x_fit2, slope2 * x_fit2 + intercept2, color=Colors.CERULEAN2, lw=2)
r_ang, p_ang = stats.pearsonr(angle, earnings)
style_axis(ax, xlabel="Allocation angle (°): $\\beta$-heavy ← → $\\delta$-heavy",
           ylabel="Total earnings (pts)")

fig.tight_layout(w_pad=3)
out = os.path.join(OUT_DIR, "fig_allocation_punchline.pdf")
fig.savefig(out, bbox_inches="tight")
fig.savefig(out.replace(".pdf", ".png"), bbox_inches="tight")
print(f"Saved: {out}")
plt.close(fig)

print(f"\nMagnitude → earnings: r={r_mag:.3f}, p={p_mag:.3f}")
print(f"Angle → earnings: r={r_ang:.3f}, p={p_ang:.2e}")
print("Done.")
