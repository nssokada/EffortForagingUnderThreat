#!/usr/bin/env python3
"""
Figure for H1: Threat shifts choice, vigor, and subjective experience.

Three-panel figure:
  A) P(choose high-effort) by distance (x) × threat level (lines)
  B) Excess effort by distance (x) × threat level (lines)
  C) Anxiety & confidence by distance (x) × threat level (lines)

Uses plotter.py styling (Colors, set_plot_style, style_axis).
Run: python scripts/plotting/plot_h1_figure.py
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from scripts.plotting.plotter import Colors, set_plot_style, style_axis

# ── Paths ──────────────────────────────────────────────────────────────
DATA_DIR = "data/exploratory_350/processed/stage5_filtered_data_20260320_191950"
VIGOR_DIR = "data/exploratory_350/processed/vigor_processed"
OUT_DIR = "results/figs/paper"
os.makedirs(OUT_DIR, exist_ok=True)

# ── Threat styling ─────────────────────────────────────────────────────
THREAT_LEVELS = [0.1, 0.5, 0.9]
THREAT_COLORS = {0.1: Colors.CERULEAN2, 0.5: Colors.SLATE, 0.9: Colors.RUBY1}
THREAT_LABELS = {0.1: "Low (T=0.1)", 0.5: "Mid (T=0.5)", 0.9: "High (T=0.9)"}
DISTANCE_LABELS = {1: "1 (near)", 2: "2 (mid)", 3: "3 (far)"}
DISTANCES = [1, 2, 3]
MARKER = 'o'
MS = 7
LW = 2.2

# ── Load data ──────────────────────────────────────────────────────────
print("Loading data...")
behavior = pd.read_csv(os.path.join(DATA_DIR, "behavior.csv"))
behavior_rich = pd.read_csv(os.path.join(DATA_DIR, "behavior_rich.csv"))
feelings = pd.read_csv(os.path.join(DATA_DIR, "feelings.csv"))

# ── Panel A: P(choose high) by distance × threat ──────────────────────
# choice == 1 means chose high-effort option
choice_agg = (
    behavior
    .groupby(["threat", "distance_H"])["choice"]
    .agg(["mean", "sem", "count"])
    .reset_index()
    .rename(columns={"distance_H": "distance"})
)

# ── Panel B: Excess effort by distance × threat ───────────────────────
# Excess effort = mean_trial_effort / calibrationMax - effort demand of chosen option
br = behavior_rich.copy()
br["vigor_norm"] = br["mean_trial_effort"] / br["calibrationMax"]
br["effort_chosen"] = np.where(br["choice"] == 1, br["effort_H"], br["effort_L"])
br["distance_chosen"] = np.where(br["choice"] == 1, br["distance_H"], br["distance_L"])
br["excess_effort"] = br["vigor_norm"] - br["effort_chosen"]

excess_agg = (
    br
    .groupby(["threat", "distance_chosen"])["excess_effort"]
    .agg(["mean", "sem"])
    .reset_index()
    .rename(columns={"distance_chosen": "distance"})
)

# ── Panel C: Affect by distance × threat ──────────────────────────────
# feelings has: threat, distance (0-indexed), questionLabel (anxiety/confidence), response
feelings["distance_1idx"] = feelings["distance"] + 1  # convert to 1-indexed

anxiety = feelings[feelings["questionLabel"] == "anxiety"]
confidence = feelings[feelings["questionLabel"] == "confidence"]

anx_agg = (
    anxiety
    .groupby(["threat", "distance_1idx"])["response"]
    .agg(["mean", "sem"])
    .reset_index()
    .rename(columns={"distance_1idx": "distance"})
)

conf_agg = (
    confidence
    .groupby(["threat", "distance_1idx"])["response"]
    .agg(["mean", "sem"])
    .reset_index()
    .rename(columns={"distance_1idx": "distance"})
)

# ── Build figure ───────────────────────────────────────────────────────
set_plot_style()
fig, axes = plt.subplots(1, 3, figsize=(14, 4.2))
fig.subplots_adjust(wspace=0.35, left=0.06, right=0.97, top=0.88, bottom=0.15)

# ── Panel A: Choice ───────────────────────────────────────────────────
ax = axes[0]
for t in THREAT_LEVELS:
    sub = choice_agg[choice_agg["threat"] == t]
    ax.errorbar(
        sub["distance"], sub["mean"], yerr=sub["sem"],
        color=THREAT_COLORS[t], marker=MARKER, ms=MS, lw=LW,
        capsize=3, capthick=1.5, label=THREAT_LABELS[t],
    )
style_axis(ax, ylabel="P(choose high-effort)", xlabel="Distance")
ax.set_xticks(DISTANCES)
ax.set_xticklabels([DISTANCE_LABELS[d] for d in DISTANCES])
ax.set_ylim(0, 1.0)
ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
ax.set_title("A   Choice", fontsize=12, fontweight="bold", color=Colors.DARK_GREY, loc="left")

# ── Panel B: Excess effort ────────────────────────────────────────────
ax = axes[1]
for t in THREAT_LEVELS:
    sub = excess_agg[excess_agg["threat"] == t]
    ax.errorbar(
        sub["distance"], sub["mean"], yerr=sub["sem"],
        color=THREAT_COLORS[t], marker=MARKER, ms=MS, lw=LW,
        capsize=3, capthick=1.5, label=THREAT_LABELS[t],
    )
style_axis(ax, ylabel="Excess effort (vigor − demand)", xlabel="Distance")
ax.set_xticks(DISTANCES)
ax.set_xticklabels([DISTANCE_LABELS[d] for d in DISTANCES])
ax.set_title("B   Excess effort", fontsize=12, fontweight="bold", color=Colors.DARK_GREY, loc="left")
# Add zero reference line
ax.axhline(0, color=Colors.SLATE, ls="--", lw=0.8, alpha=0.5)

# ── Panel C: Affect ───────────────────────────────────────────────────
ax = axes[2]
for t in THREAT_LEVELS:
    sub_a = anx_agg[anx_agg["threat"] == t]
    sub_c = conf_agg[conf_agg["threat"] == t]
    # Anxiety: solid lines
    ax.errorbar(
        sub_a["distance"], sub_a["mean"], yerr=sub_a["sem"],
        color=THREAT_COLORS[t], marker=MARKER, ms=MS, lw=LW,
        capsize=3, capthick=1.5, ls="-",
    )
    # Confidence: dashed lines
    ax.errorbar(
        sub_c["distance"], sub_c["mean"], yerr=sub_c["sem"],
        color=THREAT_COLORS[t], marker="s", ms=MS - 1, lw=LW,
        capsize=3, capthick=1.5, ls="--",
    )
style_axis(ax, ylabel="Rating (0–7)", xlabel="Distance")
ax.set_xticks(DISTANCES)
ax.set_xticklabels([DISTANCE_LABELS[d] for d in DISTANCES])
ax.set_ylim(0, 7)
ax.set_yticks([0, 1, 2, 3, 4, 5, 6, 7])
ax.set_title("C   Subjective experience", fontsize=12, fontweight="bold", color=Colors.DARK_GREY, loc="left")

# ── Legends ────────────────────────────────────────────────────────────
# Threat legend on Panel A
axes[0].legend(fontsize=8, frameon=False, loc="upper right")

# Affect type legend on Panel C (solid = anxiety, dashed = confidence)
affect_handles = [
    Line2D([0], [0], color=Colors.INK, ls="-", lw=2, marker="o", ms=5, label="Anxiety"),
    Line2D([0], [0], color=Colors.INK, ls="--", lw=2, marker="s", ms=4, label="Confidence"),
]
ax_leg = axes[2].legend(handles=affect_handles, fontsize=8, frameon=False, loc="center right")
axes[2].add_artist(ax_leg)

# ── Save ───────────────────────────────────────────────────────────────
out_pdf = os.path.join(OUT_DIR, "fig_h1_threat_shifts.pdf")
out_png = os.path.join(OUT_DIR, "fig_h1_threat_shifts.png")
fig.savefig(out_pdf, bbox_inches="tight")
fig.savefig(out_png, bbox_inches="tight")
print(f"Saved: {out_pdf}")
print(f"Saved: {out_png}")
plt.close()

# ── Print summary stats for reference ──────────────────────────────────
print("\n=== H1 Summary Stats ===")
print("\nPanel A — P(choose high) by threat × distance:")
print(choice_agg.pivot(index="distance", columns="threat", values="mean").round(3))

print("\nPanel B — Excess effort by threat × distance:")
print(excess_agg.pivot(index="distance", columns="threat", values="mean").round(3))

print("\nPanel C — Anxiety by threat × distance:")
print(anx_agg.pivot(index="distance", columns="threat", values="mean").round(2))

print("\nPanel C — Confidence by threat × distance:")
print(conf_agg.pivot(index="distance", columns="threat", values="mean").round(2))
