#!/usr/bin/env python3
"""
H2 coupling — clean scatter matching vigor timecourse Panel C style.
Ruby dots, regression line + CI band, minimal labeling.
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from scripts.plotting.plotter import Colors, set_plot_style, style_axis

DATA_DIR = "data/exploratory_350/processed/stage5_filtered_data_20260320_191950"
OUT_DIR = "results/figs/paper"
os.makedirs(OUT_DIR, exist_ok=True)

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
n = len(delta_choice)

set_plot_style()

# Regression + CI
slope, intercept = np.polyfit(delta_choice, delta_vigor, 1)
x_fit = np.linspace(delta_choice.min() - 0.05, delta_choice.max() + 0.05, 200)
y_fit = slope * x_fit + intercept

x_mean = delta_choice.mean()
ss_x = np.sum((delta_choice - x_mean) ** 2)
residuals = delta_vigor - (slope * delta_choice + intercept)
se_resid = np.sqrt(np.sum(residuals ** 2) / (n - 2))
se_fit = se_resid * np.sqrt(1 / n + (x_fit - x_mean) ** 2 / ss_x)
t_crit = stats.t.ppf(0.975, n - 2)

fig, ax = plt.subplots(figsize=(5.5, 5))

# Scatter
ax.scatter(delta_choice, delta_vigor, s=24, c=Colors.CERULEAN2, alpha=0.4,
           edgecolors="none", zorder=3)

# CI band
ax.fill_between(x_fit, y_fit - t_crit * se_fit, y_fit + t_crit * se_fit,
                color=Colors.CERULEAN2, alpha=0.15, zorder=2)

# Regression line
ax.plot(x_fit, y_fit, color=Colors.CERULEAN2, lw=2, zorder=4)

style_axis(ax,
           xlabel="$\\Delta$Choice  [P(high | T=0.9) $-$ P(high | T=0.1)]",
           ylabel="$\\Delta$Excess effort  [T=0.9 $-$ T=0.1]")

fig.tight_layout()
out = os.path.join(OUT_DIR, "fig_h2_coupling_scatter.pdf")
fig.savefig(out, bbox_inches="tight")
fig.savefig(out.replace(".pdf", ".png"), bbox_inches="tight")
print(f"Saved: {out}")
print(f"r = {r:.2f}, p = {p:.2e}, N = {n}")
plt.close(fig)
