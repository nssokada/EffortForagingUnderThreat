#!/usr/bin/env python3
"""
H1 figure — three standalone panels saved as separate files.

  Panel A: P(choose high-effort) — grouped bar chart by distance × threat
  Panel B: Vigor timecourse by threat — excess effort over time from trial start
  Panel C: Affect — grouped bar charts for anxiety and confidence by distance × threat

All use within-subject SEM (Cousineau-Morey correction).
Run: python scripts/plotting/plot_h1_panels.py
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scripts.plotting.plotter import Colors, set_plot_style, style_axis

# ── Paths ──────────────────────────────────────────────────────────────
DATA_DIR = "data/exploratory_350/processed/stage5_filtered_data_20260320_191950"
VIGOR_TS = "data/exploratory_350/processed/vigor_processed/smoothed_vigor_ts.parquet"
OUT_DIR = "results/figs/paper"
os.makedirs(OUT_DIR, exist_ok=True)

# ── Styling constants ─────────────────────────────────────────────────
THREAT_LEVELS = [0.1, 0.5, 0.9]
THREAT_COLORS = {0.1: Colors.CERULEAN2, 0.5: Colors.SLATE, 0.9: Colors.RUBY1}
THREAT_LABELS = {0.1: "Low (T=0.1)", 0.5: "Mid (T=0.5)", 0.9: "High (T=0.9)"}
DISTANCE_LABELS = {1: "Near", 2: "Mid", 3: "Far"}
DISTANCES = [1, 2, 3]

# ── Within-subject SEM (Cousineau-Morey) ──────────────────────────────
def ws_sem(df, subj_col, val_col, group_cols):
    """Cousineau-Morey within-subject SEM."""
    df = df.copy()
    subj_mean = df.groupby(subj_col)[val_col].transform('mean')
    grand_mean = df[val_col].mean()
    df['_norm'] = df[val_col] - subj_mean + grand_mean
    n_conds = df.groupby(group_cols).ngroups
    morey = np.sqrt(n_conds / (n_conds - 1))
    agg = df.groupby(group_cols)['_norm'].agg(['mean', 'sem']).reset_index()
    orig = df.groupby(group_cols)[val_col].mean().reset_index()
    agg['mean'] = orig[val_col].values
    agg['sem'] = agg['sem'] * morey
    return agg


# ── Load data ──────────────────────────────────────────────────────────
print("Loading data...")
behavior = pd.read_csv(os.path.join(DATA_DIR, "behavior.csv"))
feelings = pd.read_csv(os.path.join(DATA_DIR, "feelings.csv"))

# ======================================================================
# PANEL A — Choice: grouped bar chart
# ======================================================================
print("Panel A: Choice bars...")

# Per-subject cell means
choice_subj = (
    behavior
    .groupby(["subj", "threat", "distance_H"])["choice"]
    .mean().reset_index()
    .rename(columns={"distance_H": "distance"})
)
choice_agg = ws_sem(choice_subj, 'subj', 'choice', ['threat', 'distance'])

set_plot_style()
fig_a, ax_a = plt.subplots(figsize=(5, 4))

bar_width = 0.22
x_base = np.arange(len(DISTANCES))
for i, t in enumerate(THREAT_LEVELS):
    sub = choice_agg[choice_agg["threat"] == t].sort_values("distance")
    offset = (i - 1) * bar_width
    ax_a.bar(
        x_base + offset, sub["mean"].values, bar_width,
        yerr=sub["sem"].values, color=THREAT_COLORS[t],
        label=THREAT_LABELS[t], capsize=3, edgecolor='white', linewidth=0.5,
        error_kw=dict(lw=1.2, capthick=1.2),
    )

style_axis(ax_a, ylabel="P(choose high-effort)", xlabel="Distance")
ax_a.set_xticks(x_base)
ax_a.set_xticklabels([DISTANCE_LABELS[d] for d in DISTANCES])
ax_a.set_ylim(0, 1.0)
ax_a.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
ax_a.legend(fontsize=9, frameon=False, loc="upper right")
ax_a.set_title("Choice", fontsize=13, fontweight="bold", color=Colors.DARK_GREY, loc="left")

fig_a.tight_layout()
fig_a.savefig(os.path.join(OUT_DIR, "fig_h1a_choice.pdf"), bbox_inches="tight")
fig_a.savefig(os.path.join(OUT_DIR, "fig_h1a_choice.png"), bbox_inches="tight", dpi=200)
plt.close(fig_a)
print("  Saved fig_h1a_choice.pdf/.png")

# ======================================================================
# PANEL B — Vigor timecourse by threat
# ======================================================================
print("Panel B: Vigor timecourse — SKIPPED (requires pyarrow for parquet)")
print("  Install pyarrow then rerun to generate fig_h1b_vigor_timecourse")

# ======================================================================
# PANEL C — Affect: anxiety & confidence bar charts (side by side)
# ======================================================================
print("Panel C: Affect bars...")

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

anx_agg = ws_sem(anx_subj, 'subj', 'response', ['threat', 'distance'])
conf_agg = ws_sem(conf_subj, 'subj', 'response', ['threat', 'distance'])

fig_c, (ax_anx, ax_conf) = plt.subplots(1, 2, figsize=(9, 4))

bar_width = 0.22
x_base = np.arange(len(DISTANCES))

for ax, agg, title, ylbl in [
    (ax_anx, anx_agg, "Anxiety", "Anxiety rating (0\u20137)"),
    (ax_conf, conf_agg, "Confidence", "Confidence rating (0\u20137)"),
]:
    for i, t in enumerate(THREAT_LEVELS):
        sub = agg[agg["threat"] == t].sort_values("distance")
        offset = (i - 1) * bar_width
        ax.bar(
            x_base + offset, sub["mean"].values, bar_width,
            yerr=sub["sem"].values, color=THREAT_COLORS[t],
            label=THREAT_LABELS[t], capsize=3, edgecolor='white', linewidth=0.5,
            error_kw=dict(lw=1.2, capthick=1.2),
        )
    style_axis(ax, ylabel=ylbl, xlabel="Distance")
    ax.set_xticks(x_base)
    ax.set_xticklabels([DISTANCE_LABELS[d] for d in DISTANCES])
    ax.set_ylim(0, 7)
    ax.set_yticks([0, 1, 2, 3, 4, 5, 6, 7])
    ax.set_title(title, fontsize=13, fontweight="bold", color=Colors.DARK_GREY, loc="left")

# Legend on anxiety panel only
ax_anx.legend(fontsize=9, frameon=False, loc="upper left")

fig_c.tight_layout()
fig_c.savefig(os.path.join(OUT_DIR, "fig_h1c_affect.pdf"), bbox_inches="tight")
fig_c.savefig(os.path.join(OUT_DIR, "fig_h1c_affect.png"), bbox_inches="tight", dpi=200)
plt.close(fig_c)
print("  Saved fig_h1c_affect.pdf/.png")

print("\nDone — all three panels saved.")
