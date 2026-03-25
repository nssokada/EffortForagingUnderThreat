#!/usr/bin/env python3
"""
Figure for H1: Threat shifts choice, vigor, and subjective experience.

2×2 layout:
  A) P(choose high-effort) by distance × threat
  B) Excess effort DIFFERENCE from low-threat baseline, by distance × threat
  C) Anxiety by distance × threat
  D) Confidence by distance × threat

Uses Cousineau-Morey within-subject SEM and plotter.py styling.
Run: python scripts/plotting/plot_h1_figure_v2.py
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scripts.plotting.plotter import Colors, set_plot_style, style_axis

# ── Paths ──────────────────────────────────────────────────────────────
DATA_DIR = "data/exploratory_350/processed/stage5_filtered_data_20260320_191950"
OUT_DIR = "results/figs/paper"
os.makedirs(OUT_DIR, exist_ok=True)

# ── Styling constants ─────────────────────────────────────────────────
THREAT_LEVELS = [0.1, 0.5, 0.9]
THREAT_COLORS = {0.1: Colors.CERULEAN2, 0.5: Colors.SLATE, 0.9: Colors.RUBY1}
THREAT_LABELS = {0.1: "Low (T=0.1)", 0.5: "Mid (T=0.5)", 0.9: "High (T=0.9)"}
DISTANCE_LABELS = {1: "Near", 2: "Mid", 3: "Far"}
DISTANCES = [1, 2, 3]
MARKER = 'o'
MS = 7
LW = 2.2
JITTER = {0.1: -0.06, 0.5: 0.0, 0.9: 0.06}  # slight x-jitter to reduce overlap

# ── Cousineau-Morey within-subject SEM ────────────────────────────────
def within_subject_sem(df, subject_col, value_col, group_cols):
    """
    Cousineau-Morey correction: remove between-subject variance so
    error bars reflect within-subject effects (what the LMM tests).
    """
    # Subject mean across all conditions
    subj_means = df.groupby(subject_col)[value_col].transform('mean')
    # Grand mean
    grand_mean = df[value_col].mean()
    # Normalized values: remove subject intercept, add back grand mean
    df = df.copy()
    df['_normed'] = df[value_col] - subj_means + grand_mean
    # Number of conditions for Morey correction
    n_conds = df.groupby(group_cols).ngroups
    morey = np.sqrt(n_conds / (n_conds - 1))
    # SEM on normalized values, then apply Morey correction
    agg = (
        df.groupby(group_cols)['_normed']
        .agg(['mean', 'sem', 'count'])
        .reset_index()
    )
    # Use original means (not normed) but normed SEM
    orig_means = df.groupby(group_cols)[value_col].mean().reset_index()
    agg['mean'] = orig_means[value_col].values
    agg['sem'] = agg['sem'] * morey
    return agg


# ── Load data ──────────────────────────────────────────────────────────
print("Loading data...")
behavior = pd.read_csv(os.path.join(DATA_DIR, "behavior.csv"))
behavior_rich = pd.read_csv(os.path.join(DATA_DIR, "behavior_rich.csv"))
feelings = pd.read_csv(os.path.join(DATA_DIR, "feelings.csv"))

# ── Panel A: P(choose high) ───────────────────────────────────────────
# Compute per-subject cell means first
choice_subj = (
    behavior
    .groupby(["subj", "threat", "distance_H"])["choice"]
    .mean()
    .reset_index()
    .rename(columns={"distance_H": "distance"})
)
choice_agg = within_subject_sem(choice_subj, 'subj', 'choice', ['threat', 'distance'])

# ── Panel B: Excess effort (difference from low-threat baseline) ──────
br = behavior_rich.copy()
br["vigor_norm"] = br["mean_trial_effort"] / br["calibrationMax"]
br["effort_chosen"] = np.where(br["choice"] == 1, br["effort_H"], br["effort_L"])
br["distance_chosen"] = np.where(br["choice"] == 1, br["distance_H"], br["distance_L"])
br["excess_effort"] = br["vigor_norm"] - br["effort_chosen"]

# Per-subject cell means
excess_subj = (
    br
    .groupby(["subj", "threat", "distance_chosen"])["excess_effort"]
    .mean()
    .reset_index()
    .rename(columns={"distance_chosen": "distance"})
)

# Compute difference from each subject's low-threat baseline at each distance
baseline = (
    excess_subj[excess_subj["threat"] == 0.1]
    .set_index(["subj", "distance"])["excess_effort"]
    .rename("baseline")
)
excess_subj = excess_subj.join(baseline, on=["subj", "distance"])
excess_subj["excess_diff"] = excess_subj["excess_effort"] - excess_subj["baseline"]

excess_diff_agg = within_subject_sem(
    excess_subj, 'subj', 'excess_diff', ['threat', 'distance']
)

# ── Panel C & D: Anxiety & Confidence ─────────────────────────────────
feelings["distance_1idx"] = feelings["distance"] + 1

anxiety = feelings[feelings["questionLabel"] == "anxiety"].copy()
confidence = feelings[feelings["questionLabel"] == "confidence"].copy()

# Per-subject cell means
anx_subj = (
    anxiety.groupby(["subj", "threat", "distance_1idx"])["response"]
    .mean().reset_index().rename(columns={"distance_1idx": "distance"})
)
conf_subj = (
    confidence.groupby(["subj", "threat", "distance_1idx"])["response"]
    .mean().reset_index().rename(columns={"distance_1idx": "distance"})
)

anx_agg = within_subject_sem(anx_subj, 'subj', 'response', ['threat', 'distance'])
conf_agg = within_subject_sem(conf_subj, 'subj', 'response', ['threat', 'distance'])

# ── Build 2×2 figure ──────────────────────────────────────────────────
set_plot_style()
fig, axes = plt.subplots(2, 2, figsize=(10, 8))
fig.subplots_adjust(hspace=0.38, wspace=0.32, left=0.09, right=0.96, top=0.93, bottom=0.08)

def plot_panel(ax, agg_df, ylabel, title, ylim=None, yticks=None,
               ref_line=None, legend=False):
    for t in THREAT_LEVELS:
        sub = agg_df[agg_df["threat"] == t]
        ax.errorbar(
            sub["distance"] + JITTER[t], sub["mean"], yerr=sub["sem"],
            color=THREAT_COLORS[t], marker=MARKER, ms=MS, lw=LW,
            capsize=4, capthick=1.5, label=THREAT_LABELS[t],
        )
    style_axis(ax, ylabel=ylabel, xlabel="Distance")
    ax.set_xticks(DISTANCES)
    ax.set_xticklabels([DISTANCE_LABELS[d] for d in DISTANCES])
    ax.set_title(title, fontsize=12, fontweight="bold", color=Colors.DARK_GREY, loc="left")
    if ylim is not None:
        ax.set_ylim(ylim)
    if yticks is not None:
        ax.set_yticks(yticks)
    if ref_line is not None:
        ax.axhline(ref_line, color=Colors.SLATE, ls="--", lw=0.8, alpha=0.5)
    if legend:
        ax.legend(fontsize=8.5, frameon=False, loc="best")

# Panel A — Choice
plot_panel(axes[0, 0], choice_agg,
           ylabel="P(choose high-effort)", title="A   Choice",
           ylim=(0, 1.0), yticks=[0, 0.25, 0.5, 0.75, 1.0], legend=True)

# Panel B — Excess effort difference from low-threat
plot_panel(axes[0, 1], excess_diff_agg,
           ylabel="\u0394 Excess effort\n(vs low-threat baseline)",
           title="B   Threat-driven effort boost",
           ref_line=0)

# Panel C — Anxiety
plot_panel(axes[1, 0], anx_agg,
           ylabel="Anxiety rating (0\u20137)", title="C   Anxiety",
           ylim=(0, 7), yticks=[0, 1, 2, 3, 4, 5, 6, 7])

# Panel D — Confidence
plot_panel(axes[1, 1], conf_agg,
           ylabel="Confidence rating (0\u20137)", title="D   Confidence",
           ylim=(0, 7), yticks=[0, 1, 2, 3, 4, 5, 6, 7])

# ── Save ───────────────────────────────────────────────────────────────
out_pdf = os.path.join(OUT_DIR, "fig_h1_threat_shifts_v2.pdf")
out_png = os.path.join(OUT_DIR, "fig_h1_threat_shifts_v2.png")
fig.savefig(out_pdf, bbox_inches="tight")
fig.savefig(out_png, bbox_inches="tight", dpi=200)
print(f"Saved: {out_pdf}")
print(f"Saved: {out_png}")
plt.close()

# ── Summary stats ──────────────────────────────────────────────────────
print("\n=== H1 Summary Stats ===")
print("\nPanel A — P(choose high) by threat × distance:")
print(choice_agg.pivot(index="distance", columns="threat", values="mean").round(3))

print("\nPanel B — Δ Excess effort (vs low-threat) by threat × distance:")
print(excess_diff_agg.pivot(index="distance", columns="threat", values="mean").round(4))

print("\nPanel C — Anxiety by threat × distance:")
print(anx_agg.pivot(index="distance", columns="threat", values="mean").round(2))

print("\nPanel D — Confidence by threat × distance:")
print(conf_agg.pivot(index="distance", columns="threat", values="mean").round(2))
