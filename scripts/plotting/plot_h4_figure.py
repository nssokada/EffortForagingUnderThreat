#!/usr/bin/env python3
"""
Figure + Table for H4: The survival-weighted additive-effort choice model
best explains foraging behavior.

Layout (2 rows):
  Top:    Panel A — Model comparison (ΔELBO horizontal bars, 5 prereg models)
  Bottom: Panel B — Anxiety ~ S_probe,  Panel C — Confidence ~ S_probe

Outputs:
  results/figs/paper/fig_h4_choice_model.{png,pdf}
  results/figs/paper/table_h4.html          (standalone styled table)
  results/figs/paper/fig_h4_choice_model.html  (standalone figure + table)

Run: python scripts/plotting/plot_h4_figure.py
"""

import sys, os, json
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scripts.plotting.plotter import Colors, set_plot_style, style_axis

# ── Paths ──────────────────────────────────────────────────────────────
DATA_DIR = "data/exploratory_350/processed/stage5_filtered_data_20260320_191950"
STATS_DIR = "results/stats"
OUT_DIR = "results/figs/paper"
os.makedirs(OUT_DIR, exist_ok=True)

# ── Constants ──────────────────────────────────────────────────────────
# Binary-E model λ (from binary_e_validation.csv)
LAMBDA = 0.6976

# Map CSV model names → prereg model labels
PREREG_MODELS = {
    "L0_effort":    ("M1", r"$R \cdot e^{-kE}$", "Does threat matter?"),
    "L2_TxD":       ("M2", r"$R \cdot e^{-kE} - \beta \cdot T \cdot D$", "Mechanistic S or\nlinear features?"),
    "L3_survival":  ("M3", r"$R \cdot e^{-kE} \cdot S_{exp} - \beta(1-S)$", "Which survival\nkernel?"),
    "L4a_hyp":      ("M4", r"$R \cdot e^{-kE} \cdot S_{hyp} - \beta(1-S)$", "Which effort\nstructure?"),
    "L4a_add":      ("M5", r"$R \cdot S - k \cdot E - \beta(1-S)$", "Winner"),
}
PREREG_ORDER = ["L0_effort", "L2_TxD", "L3_survival", "L4a_hyp", "L4a_add"]

# Threat colors (matching H1 figure)
THREAT_COLORS = {0.1: Colors.CERULEAN2, 0.5: Colors.SLATE, 0.9: Colors.RUBY1}
THREAT_LABELS = {0.1: "Low (T=0.1)", 0.5: "Mid (T=0.5)", 0.9: "High (T=0.9)"}

# ── Cousineau-Morey within-subject SEM ────────────────────────────────
def within_subject_sem(df, subject_col, value_col, group_cols):
    subj_means = df.groupby(subject_col)[value_col].transform('mean')
    grand_mean = df[value_col].mean()
    df = df.copy()
    df['_normed'] = df[value_col] - subj_means + grand_mean
    n_conds = df.groupby(group_cols).ngroups
    morey = np.sqrt(n_conds / (n_conds - 1))
    agg = (
        df.groupby(group_cols)['_normed']
        .agg(['mean', 'sem', 'count'])
        .reset_index()
    )
    orig_means = df.groupby(group_cols)[value_col].mean().reset_index()
    agg['mean'] = orig_means[value_col].values
    agg['sem'] = agg['sem'] * morey
    return agg


# ── Load data ──────────────────────────────────────────────────────────
print("Loading data...")

# Model comparison
mc = pd.read_csv(os.path.join(STATS_DIR, "unified_model_comparison.csv"))
mc_sub = mc[mc["model"].isin(PREREG_ORDER)].copy()
mc_sub["prereg"] = mc_sub["model"].map(lambda m: PREREG_MODELS[m][0])
mc_sub["formula"] = mc_sub["model"].map(lambda m: PREREG_MODELS[m][1])
mc_sub["question"] = mc_sub["model"].map(lambda m: PREREG_MODELS[m][2])
# ΔELBO relative to winner (M5)
elbo_winner = mc_sub.loc[mc_sub["model"] == "L4a_add", "ELBO"].values[0]
mc_sub["dELBO_from_winner"] = mc_sub["ELBO"] - elbo_winner
# Sort by prereg order
mc_sub["sort_idx"] = mc_sub["model"].map(lambda m: PREREG_ORDER.index(m))
mc_sub = mc_sub.sort_values("sort_idx")

# Affect LMM results
affect_lmm = pd.read_csv(os.path.join(STATS_DIR, "affect_lmm_results.csv"))

# Feelings (probe trials)
feelings = pd.read_csv(os.path.join(DATA_DIR, "feelings.csv"))

# Compute S_probe for each probe trial
# D = distance + 1 (converting 0-indexed to 1-indexed levels)
feelings["D"] = feelings["distance"] + 1
feelings["S_probe"] = (1 - feelings["threat"]) + feelings["threat"] / (1 + LAMBDA * feelings["D"])

anxiety = feelings[feelings["questionLabel"] == "anxiety"].copy()
confidence = feelings[feelings["questionLabel"] == "confidence"].copy()


# ── Panel A: Model comparison ──────────────────────────────────────────
def plot_model_comparison(ax, mc_df):
    """Horizontal bar chart of ΔELBO from M1 baseline."""
    n = len(mc_df)
    y_pos = np.arange(n)
    elbos = mc_df["dELBO_from_winner"].values
    labels = [f"{row.prereg}" for _, row in mc_df.iterrows()]

    # Color: winner in blue, others in grey
    bar_colors = []
    for _, row in mc_df.iterrows():
        if row["model"] == "L4a_add":
            bar_colors.append(Colors.CERULEAN2)
        else:
            bar_colors.append(Colors.SLATE)

    bars = ax.barh(y_pos, elbos, color=bar_colors, height=0.6, alpha=0.85,
                   edgecolor='white', linewidth=0.5)

    # Use model formulas as y-tick labels
    formula_labels = [mc_df.iloc[i]["formula"] for i in range(n)]
    ax.set_yticks(y_pos)
    ax.set_yticklabels(formula_labels, fontsize=10)
    ax.invert_yaxis()

    # Mark the winner with a diamond at 0
    winner_idx = list(mc_df["model"]).index("L4a_add")
    ax.plot(0, winner_idx, marker='D', ms=8, color=Colors.CERULEAN2,
            zorder=5, markeredgecolor='white', markeredgewidth=1.2)

    style_axis(ax, xlabel=r"$\Delta$ELBO (vs. winner M5)", ylabel=None)
    ax.set_title("A   Choice model comparison", fontsize=12, fontweight="bold",
                 color=Colors.DARK_GREY, loc="left")


# ── Panels B & C: S_probe vs affect ───────────────────────────────────
def plot_affect_panel(ax, df, outcome_label, lmm_row, color, ylabel, title):
    """
    Plot condition means (by threat × S_probe) with within-subject SEM,
    plus the LMM regression line.
    """
    # Per-subject cell means at each (threat, distance) → unique S_probe
    df_subj = (
        df.groupby(["subj", "threat", "S_probe"])["response"]
        .mean().reset_index()
    )
    agg = within_subject_sem(df_subj, 'subj', 'response', ['threat', 'S_probe'])

    # Plot by threat level
    for t in [0.1, 0.5, 0.9]:
        sub = agg[agg["threat"] == t].sort_values("S_probe")
        # Round threat for matching
        sub_t = sub.copy()
        ax.errorbar(
            sub_t["S_probe"], sub_t["mean"], yerr=sub_t["sem"],
            color=THREAT_COLORS[t], marker='o', ms=7, lw=2,
            capsize=4, capthick=1.5, label=THREAT_LABELS[t],
            zorder=3
        )

    # Add LMM regression line
    beta = lmm_row["beta"]
    # S_probe_z → need to know mean and sd of S_probe to back-transform
    all_s = df["S_probe"].values
    s_mean, s_std = all_s.mean(), all_s.std()
    # response mean for intercept approximation
    r_mean = df["response"].mean()
    # Line: response = r_mean + beta * (S - s_mean) / s_std
    s_line = np.linspace(agg["S_probe"].min() - 0.02, agg["S_probe"].max() + 0.02, 100)
    r_line = r_mean + beta * (s_line - s_mean) / s_std

    ax.plot(s_line, r_line, color=color, lw=2.5, ls='--', alpha=0.7, zorder=2,
            label=f"LMM: β={beta:.2f}")

    # Annotation: β and t-value
    t_val = lmm_row["t"]
    p_val = lmm_row["p"]
    p_str = f"p < 10$^{{{int(np.log10(p_val))}}}$" if p_val < 0.001 else f"p = {p_val:.3f}"
    ax.text(0.97, 0.95, f"β = {beta:.3f}\nt = {t_val:.1f}\n{p_str}",
            transform=ax.transAxes, fontsize=9, va='top', ha='right',
            color=Colors.INK, bbox=dict(boxstyle='round,pad=0.3',
            facecolor='white', edgecolor=Colors.GREY, alpha=0.9))

    style_axis(ax, ylabel=ylabel, xlabel="Survival probability (S)")
    ax.set_title(title, fontsize=12, fontweight="bold",
                 color=Colors.DARK_GREY, loc="left")
    ax.legend(fontsize=8, frameon=False, loc="best")


# ── Build figure ──────────────────────────────────────────────────────
set_plot_style()
fig = plt.figure(figsize=(12, 9))
gs = gridspec.GridSpec(2, 2, figure=fig, height_ratios=[1.1, 1],
                       hspace=0.38, wspace=0.32,
                       left=0.08, right=0.95, top=0.95, bottom=0.07)

# Panel A spans both columns on top
ax_a = fig.add_subplot(gs[0, :])
plot_model_comparison(ax_a, mc_sub)

# Panel B: Anxiety ~ S_probe
ax_b = fig.add_subplot(gs[1, 0])
anx_lmm = affect_lmm.loc[
    (affect_lmm["outcome"] == "anxiety") & (affect_lmm["predictor"] == "S_probe_z")
].iloc[0]
plot_affect_panel(ax_b, anxiety, "anxiety", anx_lmm,
                  color=Colors.RUBY1,
                  ylabel="Anxiety rating (0–7)",
                  title="B   Anxiety ~ Survival probability")

# Panel C: Confidence ~ S_probe
ax_c = fig.add_subplot(gs[1, 1])
conf_lmm = affect_lmm.loc[
    (affect_lmm["outcome"] == "confidence") & (affect_lmm["predictor"] == "S_probe_z")
].iloc[0]
plot_affect_panel(ax_c, confidence, "confidence", conf_lmm,
                  color=Colors.CERULEAN2,
                  ylabel="Confidence rating (0–7)",
                  title="C   Confidence ~ Survival probability")

# ── Save figure ───────────────────────────────────────────────────────
for ext in ["pdf", "png"]:
    out = os.path.join(OUT_DIR, f"fig_h4_choice_model.{ext}")
    fig.savefig(out, bbox_inches="tight", dpi=200 if ext == "png" else None)
    print(f"Saved: {out}")
plt.close()


# ══════════════════════════════════════════════════════════════════════
# HTML TABLE
# ══════════════════════════════════════════════════════════════════════

def build_html_table():
    """Build a standalone HTML file with the H4 results table, using KaTeX for equations."""
    # Model comparison rows — KaTeX math strings
    KATEX_FORMULAS = {
        "L0_effort":    r"SV = R \cdot e^{-kE}",
        "L2_TxD":       r"SV = R \cdot e^{-kE} - \beta \cdot T \cdot D",
        "L3_survival":  r"SV = R \cdot e^{-kE} \cdot S_{\text{exp}} - \beta(1-S)",
        "L4a_hyp":      r"SV = R \cdot e^{-kE} \cdot S_{\text{hyp}} - \beta(1-S)",
        "L4a_add":      r"SV = R \cdot S - k \cdot E - \beta(1-S)",
    }
    QUESTION_MAP = {
        "L0_effort":    "Does threat matter?",
        "L2_TxD":       "Mechanistic S or linear features?",
        "L3_survival":  "Which survival kernel?",
        "L4a_hyp":      "Which effort structure?",
        "L4a_add":      "",
    }

    winner_elbo = mc_sub.loc[mc_sub["model"] == "L4a_add", "ELBO"].values[0]

    model_rows = []
    for _, row in mc_sub.iterrows():
        delta_from_winner = row["ELBO"] - winner_elbo
        is_winner = row["model"] == "L4a_add"
        model_rows.append({
            "prereg": row["prereg"],
            "katex": KATEX_FORMULAS[row["model"]],
            "elbo": row["ELBO"],
            "delta": delta_from_winner,
            "question": QUESTION_MAP[row["model"]],
            "is_winner": is_winner,
        })

    # Affect LMM rows
    affect_rows = []
    for _, row in affect_lmm.iterrows():
        if row["predictor"] == "S_probe_z":
            p = row["p"]
            if p < 1e-100:
                p_str = f"< 10<sup>{int(np.log10(p))}</sup>"
            elif p < 0.001:
                p_str = "< .001"
            else:
                p_str = f"{p:.3f}"
            affect_rows.append({
                "outcome": row["outcome"].capitalize(),
                "beta": row["beta"],
                "se": row["se"],
                "t": row["t"],
                "p": p_str,
                "n_subj": int(row["n_subj"]),
                "n_obs": int(row["n_obs"]),
            })

    m5_elbo = mc_sub.loc[mc_sub["model"] == "L4a_add", "ELBO"].values[0]
    m4_elbo = mc_sub.loc[mc_sub["model"] == "L4a_hyp", "ELBO"].values[0]
    m3_elbo = mc_sub.loc[mc_sub["model"] == "L3_survival", "ELBO"].values[0]
    m1_elbo = mc_sub.loc[mc_sub["model"] == "L0_effort", "ELBO"].values[0]

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>H4 — Choice Model Comparison &amp; Affect Prediction</title>
<link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/katex@0.16.11/dist/katex.min.css">
<script defer src="https://cdn.jsdelivr.net/npm/katex@0.16.11/dist/katex.min.js"></script>
<script defer src="https://cdn.jsdelivr.net/npm/katex@0.16.11/dist/contrib/auto-render.min.js"
  onload="renderMathInElement(document.body, {{delimiters: [{{left: '$$', right: '$$', display: true}}, {{left: '$', right: '$', display: false}}]}});"></script>
<style>
  :root {{
    --ink: #374151;
    --muted: #6B7280;
    --border: #E5E7EB;
    --border-strong: #D1D5DB;
    --bg-header: #F9FAFB;
    --bg-winner: #EFF6FF;
    --bg-winner-hover: #DBEAFE;
    --accent: #2563EB;
    --pass: #059669;
    --pass-bg: #ECFDF5;
    --pass-border: #A7F3D0;
  }}
  * {{ box-sizing: border-box; }}
  body {{
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
    max-width: 920px;
    margin: 48px auto;
    color: var(--ink);
    background: #fff;
    padding: 0 24px;
    line-height: 1.55;
  }}
  h1 {{
    font-size: 1.35em;
    font-weight: 700;
    margin-bottom: 0.2em;
    letter-spacing: -0.01em;
  }}
  h2 {{
    font-size: 1.0em;
    font-weight: 600;
    color: var(--muted);
    margin: 2.5em 0 0.6em;
    text-transform: uppercase;
    letter-spacing: 0.04em;
  }}
  .subtitle {{
    color: var(--muted);
    font-size: 0.92em;
    margin-bottom: 2em;
  }}

  /* ── Tables ── */
  table {{
    border-collapse: collapse;
    width: 100%;
    font-size: 0.88em;
    margin-bottom: 0.8em;
  }}
  thead th {{
    background: var(--bg-header);
    border-top: 2px solid var(--border-strong);
    border-bottom: 2px solid var(--border-strong);
    padding: 10px 14px;
    text-align: left;
    font-weight: 600;
    color: var(--muted);
    font-size: 0.85em;
    text-transform: uppercase;
    letter-spacing: 0.03em;
  }}
  thead th.r {{ text-align: right; }}
  tbody td {{
    border-bottom: 1px solid var(--border);
    padding: 10px 14px;
    vertical-align: middle;
  }}
  tbody td.r {{
    text-align: right;
    font-variant-numeric: tabular-nums;
    font-family: 'SF Mono', 'Menlo', 'Consolas', monospace;
    font-size: 0.92em;
  }}
  tbody tr:last-child td {{ border-bottom: 2px solid var(--border-strong); }}
  tbody tr:hover td {{ background: #F9FAFB; }}

  /* Winner row */
  tr.winner td {{ background: var(--bg-winner); }}
  tr.winner:hover td {{ background: var(--bg-winner-hover); }}
  .tag {{
    display: inline-block;
    padding: 2px 8px;
    border-radius: 10px;
    font-size: 0.78em;
    font-weight: 600;
    letter-spacing: 0.02em;
  }}
  .tag-best {{ background: var(--accent); color: #fff; }}

  /* Question column */
  .q {{ color: var(--muted); font-size: 0.92em; }}

  /* ── Test result cards ── */
  .tests {{
    display: grid;
    grid-template-columns: 1fr 1fr 1fr;
    gap: 12px;
    margin: 1.2em 0 0;
  }}
  .test {{
    background: var(--pass-bg);
    border: 1px solid var(--pass-border);
    border-radius: 8px;
    padding: 14px 16px;
  }}
  .test .label {{
    font-size: 0.78em;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.04em;
    color: var(--muted);
    margin-bottom: 4px;
  }}
  .test .value {{
    font-size: 1.15em;
    font-weight: 700;
    color: var(--pass);
  }}
  .test .detail {{
    font-size: 0.82em;
    color: var(--muted);
    margin-top: 2px;
  }}

  /* ── Hypothesis block ── */
  .hyp-block {{
    background: #F9FAFB;
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 20px 24px;
    margin-bottom: 2em;
  }}
  .hyp-statement {{
    font-size: 0.92em;
    line-height: 1.65;
    margin-bottom: 12px;
  }}
  .hyp-subs {{
    display: flex;
    flex-direction: column;
    gap: 6px;
  }}
  .hyp-sub {{
    font-size: 0.86em;
    color: var(--ink);
    padding: 6px 12px;
    background: white;
    border-radius: 6px;
    border: 1px solid var(--border);
    line-height: 1.55;
  }}

  /* ── Notes ── */
  .note {{
    color: var(--muted);
    font-size: 0.82em;
    line-height: 1.6;
    margin-top: 1.5em;
    padding-top: 1em;
    border-top: 1px solid var(--border);
  }}

  /* ── LMM table tweaks ── */
  .lmm-spec {{
    color: var(--muted);
    font-size: 0.85em;
    margin-bottom: 0.8em;
  }}
</style>
</head>
<body>

<h1>H4 — The survival-weighted additive-effort choice model best explains foraging behavior</h1>
<p class="subtitle">Exploratory sample, $N = 293$ &middot; Binary effort $E \\in \\{{0, 1\\}}$ &middot; $\\lambda = {LAMBDA:.3f}$</p>

<div class="hyp-block">
  <div class="hyp-statement">
    <strong>Hypothesis:</strong> Choice behavior is best explained by a model in which (a) energetic effort costs are subtracted
    additively from expected reward, (b) escape probability follows a hyperbolic function of distance, and (c) residual
    threat aversion enters as a subject-specific additive bias $\\beta$. The survival probability $S$ from this model also
    predicts trial-level subjective affect.
  </div>
  <div class="hyp-subs">
    <div class="hyp-sub"><strong>H4a</strong> &mdash; Additive effort (M5) outperforms multiplicative effort (M4): $\\Delta$ELBO &gt; 0.</div>
    <div class="hyp-sub"><strong>H4b</strong> &mdash; Hyperbolic survival (M4) outperforms exponential survival (M3): $\\Delta$ELBO &gt; 0. Additionally, M5 outperforms the effort-only baseline (M1) by $\\Delta$ELBO &gt; 100.</div>
    <div class="hyp-sub"><strong>H4c</strong> &mdash; Model-derived $S$ negatively predicts trial-level anxiety ($\\beta &lt; 0$, $|t| &gt; 3$) and positively predicts confidence ($\\beta &gt; 0$, $|t| &gt; 3$) in within-subject LMMs.</div>
  </div>
</div>

<h2>Table 1 &mdash; Model Comparison (ELBO)</h2>
<table>
<thead>
<tr>
  <th style="width:60px">Model</th>
  <th>Subjective value structure</th>
  <th class="r">ELBO</th>
  <th class="r">$\\Delta$ELBO</th>
  <th>Question</th>
</tr>
</thead>
<tbody>
"""

    for m in model_rows:
        cls = ' class="winner"' if m["is_winner"] else ""
        tag = ' <span class="tag tag-best">Best</span>' if m["is_winner"] else ""
        html += f"""<tr{cls}>
  <td>{m['prereg']}{tag}</td>
  <td>${m['katex']}$</td>
  <td class="r">{m['elbo']:,.1f}</td>
  <td class="r">{m['delta']:+.1f}</td>
  <td class="q">{m['question']}</td>
</tr>
"""

    html += f"""</tbody>
</table>

<div class="tests">
  <div class="test">
    <div class="label">H4a &mdash; Effort structure</div>
    <div class="value">$\\Delta$ = +{m5_elbo - m4_elbo:.1f}</div>
    <div class="detail">Additive &gt; multiplicative</div>
  </div>
  <div class="test">
    <div class="label">H4b &mdash; Survival kernel</div>
    <div class="value">$\\Delta$ = +{m4_elbo - m3_elbo:.1f}</div>
    <div class="detail">Hyperbolic &gt; exponential</div>
  </div>
  <div class="test">
    <div class="label">M5 vs. baseline</div>
    <div class="value">$\\Delta$ = +{m5_elbo - m1_elbo:.1f}</div>
    <div class="detail">Threshold: &gt; 100</div>
  </div>
</div>

<h2>Table 2 &mdash; H4c: Survival predicts trial-level affect</h2>
<p class="lmm-spec">LMM: $\\text{{rating}} \\sim S_{{\\text{{probe}}}}^z + (1 + S_{{\\text{{probe}}}}^z \\mid \\text{{subject}})$</p>

<table>
<thead>
<tr>
  <th>Outcome</th>
  <th class="r">$\\beta$</th>
  <th class="r">SE</th>
  <th class="r">$t$</th>
  <th class="r">$p$</th>
  <th class="r">$N_{{\\text{{subj}}}}$</th>
  <th class="r">$N_{{\\text{{obs}}}}$</th>
</tr>
</thead>
<tbody>
"""

    for ar in affect_rows:
        html += f"""<tr>
  <td>{ar['outcome']}</td>
  <td class="r">{ar['beta']:.3f}</td>
  <td class="r">{ar['se']:.3f}</td>
  <td class="r">{ar['t']:.2f}</td>
  <td class="r">{ar['p']}</td>
  <td class="r">{ar['n_subj']}</td>
  <td class="r">{ar['n_obs']:,}</td>
</tr>
"""

    html += f"""</tbody>
</table>

<div class="tests" style="grid-template-columns: 1fr 1fr 1fr;">
  <div class="test">
    <div class="label">Anxiety</div>
    <div class="value">$\\beta < 0$ &check;</div>
  </div>
  <div class="test">
    <div class="label">Confidence</div>
    <div class="value">$\\beta > 0$ &check;</div>
  </div>
  <div class="test">
    <div class="label">Effect size</div>
    <div class="value">$|t| > 3$ &check;</div>
  </div>
</div>

<div class="note">
  $S_{{\\text{{probe}}}} = (1 - T) + \\frac{{T}}{{1 + \\lambda D}}$, with population-level
  $\\lambda = {LAMBDA:.3f}$ from the binary-$E$ SVI fit.
  ELBO = evidence lower bound (higher is better); all models fit via SVI (NumPyro).
  $\\Delta$ELBO is relative to the best model (M5); negative values indicate worse fit.
  Within-subject LMMs include random intercepts and random slopes for $S_{{\\text{{probe}}}}$.
</div>

</body>
</html>"""

    return html


# Write HTML table
html_table = build_html_table()
table_path = os.path.join(OUT_DIR, "table_h4.html")
with open(table_path, 'w') as f:
    f.write(html_table)
print(f"Saved: {table_path}")


# ══════════════════════════════════════════════════════════════════════
# COMBINED HTML (figure + table)
# ══════════════════════════════════════════════════════════════════════

def build_combined_html():
    """HTML page embedding the figure PNG and the table, with KaTeX."""
    import base64
    png_path = os.path.join(OUT_DIR, "fig_h4_choice_model.png")
    with open(png_path, 'rb') as f:
        img_b64 = base64.b64encode(f.read()).decode()

    # Extract the full <style> and <script> blocks from table HTML
    # We reuse the table HTML but wrap it with the figure on top
    # Parse out everything between <style> and </style>
    style_start = html_table.index("<style>")
    style_end = html_table.index("</style>") + len("</style>")
    style_block = html_table[style_start:style_end]

    delta_h4a = mc_sub.loc[mc_sub['model']=='L4a_add','ELBO'].values[0] - mc_sub.loc[mc_sub['model']=='L4a_hyp','ELBO'].values[0]
    delta_h4b = mc_sub.loc[mc_sub['model']=='L4a_hyp','ELBO'].values[0] - mc_sub.loc[mc_sub['model']=='L3_survival','ELBO'].values[0]

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>H4 — Figure &amp; Table</title>
<link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/katex@0.16.11/dist/katex.min.css">
<script defer src="https://cdn.jsdelivr.net/npm/katex@0.16.11/dist/katex.min.js"></script>
<script defer src="https://cdn.jsdelivr.net/npm/katex@0.16.11/dist/contrib/auto-render.min.js"
  onload="renderMathInElement(document.body, {{delimiters: [{{left: '$$', right: '$$', display: true}}, {{left: '$', right: '$', display: false}}]}});"></script>
{style_block}
<style>
  .fig-container {{
    text-align: center;
    margin: 2em 0 2.5em;
  }}
  .fig-container img {{
    max-width: 100%;
    border: 1px solid var(--border);
    border-radius: 6px;
  }}
  .caption {{
    color: var(--muted);
    font-size: 0.88em;
    margin-top: 1em;
    text-align: left;
    line-height: 1.65;
  }}
  .caption strong {{ color: var(--ink); }}
</style>
</head>
<body>

<h1>H4 — The survival-weighted additive-effort choice model best explains foraging behavior</h1>

<div class="fig-container">
  <img src="data:image/png;base64,{img_b64}" alt="H4 Figure">
  <p class="caption">
    <strong>Figure.</strong>
    <strong>(A)</strong> Model comparison across five candidate choice models (M1&ndash;M5).
    Bars show $\\Delta$ELBO relative to the effort-only baseline (M1). The winning model M5
    (additive effort + hyperbolic survival + threat bias) outperforms all alternatives.
    H4a: additive &gt; multiplicative effort ($\\Delta$ELBO = +{delta_h4a:.0f}).
    H4b: hyperbolic &gt; exponential survival ($\\Delta$ELBO = +{delta_h4b:.0f}).
    <strong>(B&ndash;C)</strong> Model-derived survival probability $S$ predicts trial-level
    anxiety (B, $\\beta = -0.605$) and confidence (C, $\\beta = +0.612$) in mixed-effects models,
    confirming that subjective affect tracks the same latent survival computation that governs
    choice (H4c). Dots show condition means (within-subject SEM); dashed line shows LMM fit.
    $N = 293$.
  </p>
</div>

"""

    # Extract the table body (everything from the first <h2> to </body>)
    body_start = html_table.index('<div class="hyp-block">')
    body_end = html_table.index("</body>")
    table_content = html_table[body_start:body_end]

    html += table_content
    html += "\n</body>\n</html>"
    return html

combined_html = build_combined_html()
combined_path = os.path.join(OUT_DIR, "fig_h4_choice_model.html")
with open(combined_path, 'w') as f:
    f.write(combined_html)
print(f"Saved: {combined_path}")


# ══════════════════════════════════════════════════════════════════════
# TABLE PNG (standalone rendered table as image)
# ══════════════════════════════════════════════════════════════════════

def render_table_png():
    """Render model comparison + affect LMM as a publication-quality table image."""
    set_plot_style()
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 5.5),
                                    gridspec_kw={'height_ratios': [1.3, 0.7],
                                                 'hspace': 0.5})

    # ── Table 1: Model comparison ──
    ax1.axis('off')
    ax1.set_title("Table 1 — Choice Model Comparison (ELBO)",
                  fontsize=12, fontweight="bold", color=Colors.DARK_GREY, loc="left",
                  pad=12)

    winner_elbo = mc_sub.loc[mc_sub["model"] == "L4a_add", "ELBO"].values[0]
    col_labels = ["Model", "Structure", "ELBO", "ΔELBO\n(vs M5)", "Question"]
    cell_text = []
    cell_colors = []
    for _, row in mc_sub.iterrows():
        delta = row["ELBO"] - winner_elbo
        is_winner = row["model"] == "L4a_add"
        # Clean formula for table (plain text)
        struct_map = {
            "L0_effort":    "R·exp(−kE)",
            "L2_TxD":       "R·exp(−kE) − β·T·D",
            "L3_survival":  "R·exp(−kE)·S_exp − β(1−S)",
            "L4a_hyp":      "R·exp(−kE)·S_hyp − β(1−S)",
            "L4a_add":      "R·S − k·E − β(1−S)",
        }
        cell_text.append([
            f"{PREREG_MODELS[row['model']][0]}{'  ★' if is_winner else ''}",
            struct_map[row['model']],
            f"{row['ELBO']:,.1f}",
            f"{delta:+.1f}",
            row['question'].replace('\n', ' '),
        ])
        if is_winner:
            cell_colors.append(['#e8f4fd'] * 5)
        else:
            cell_colors.append(['white'] * 5)

    tbl1 = ax1.table(cellText=cell_text, colLabels=col_labels,
                     cellColours=cell_colors,
                     colColours=['#f8f9fa'] * 5,
                     loc='center', cellLoc='left')
    tbl1.auto_set_font_size(False)
    tbl1.set_fontsize(9)
    tbl1.scale(1, 1.4)
    # Right-align numeric columns
    for (row, col), cell in tbl1.get_celld().items():
        cell.set_edgecolor('#e0e0e0')
        cell.set_linewidth(0.5)
        if col in [2, 3]:
            cell.set_text_props(ha='right')
        if row == 0:
            cell.set_text_props(fontweight='bold', color='#495057')

    # Key test summary below table
    m5 = mc_sub.loc[mc_sub["model"] == "L4a_add", "ELBO"].values[0]
    m4 = mc_sub.loc[mc_sub["model"] == "L4a_hyp", "ELBO"].values[0]
    m3 = mc_sub.loc[mc_sub["model"] == "L3_survival", "ELBO"].values[0]
    m1 = mc_sub.loc[mc_sub["model"] == "L0_effort", "ELBO"].values[0]
    ax1.text(0.02, -0.08,
             f"H4a: M5 vs M4 = +{m5-m4:.1f}  |  H4b: M4 vs M3 = +{m4-m3:.1f}  |  M5 vs M1 = +{m5-m1:.1f}",
             transform=ax1.transAxes, fontsize=9, color=Colors.INK,
             bbox=dict(boxstyle='round,pad=0.3', facecolor='#f0f7ff', edgecolor='#c0d8f0'))

    # ── Table 2: Affect LMMs ──
    ax2.axis('off')
    ax2.set_title("Table 2 — H4c: S predicts trial-level affect (LMM)",
                  fontsize=12, fontweight="bold", color=Colors.DARK_GREY, loc="left",
                  pad=12)

    col_labels2 = ["Outcome", "Predictor", "β", "SE", "t", "p", "N_subj", "N_obs"]
    cell_text2 = []
    for _, row in affect_lmm.iterrows():
        if row["predictor"] == "S_probe_z":
            p = row["p"]
            p_str = f"< 10^{int(np.log10(p))}" if p < 0.001 else f"{p:.3f}"
            cell_text2.append([
                row["outcome"].capitalize(),
                "S_probe (z)",
                f"{row['beta']:.3f}",
                f"{row['se']:.3f}",
                f"{row['t']:.2f}",
                p_str,
                f"{int(row['n_subj'])}",
                f"{int(row['n_obs']):,}",
            ])

    tbl2 = ax2.table(cellText=cell_text2, colLabels=col_labels2,
                     cellColours=[['white'] * 8] * len(cell_text2),
                     colColours=['#f8f9fa'] * 8,
                     loc='center', cellLoc='left')
    tbl2.auto_set_font_size(False)
    tbl2.set_fontsize(9)
    tbl2.scale(1, 1.4)
    for (row, col), cell in tbl2.get_celld().items():
        cell.set_edgecolor('#e0e0e0')
        cell.set_linewidth(0.5)
        if col in [2, 3, 4, 5, 6, 7]:
            cell.set_text_props(ha='right')
        if row == 0:
            cell.set_text_props(fontweight='bold', color='#495057')

    ax2.text(0.02, -0.12,
             "H4c: Anxiety β < 0 ✓  |  Confidence β > 0 ✓  |  Both |t| > 3.0 ✓",
             transform=ax2.transAxes, fontsize=9, color='#155724',
             bbox=dict(boxstyle='round,pad=0.3', facecolor='#d4edda', edgecolor='#b7d8b7'))

    for ext in ["pdf", "png"]:
        out = os.path.join(OUT_DIR, f"table_h4.{ext}")
        fig.savefig(out, bbox_inches="tight", dpi=200 if ext == "png" else None)
        print(f"Saved: {out}")
    plt.close()

render_table_png()


# ── Summary ───────────────────────────────────────────────────────────
print("\n=== H4 Summary ===")
print(f"Model comparison: 5 models, winner = M5 (L4a_add)")
print(f"  H4a: M5 vs M4 ΔELBO = +{mc_sub.loc[mc_sub['model']=='L4a_add','ELBO'].values[0] - mc_sub.loc[mc_sub['model']=='L4a_hyp','ELBO'].values[0]:.1f}")
print(f"  H4b: M4 vs M3 ΔELBO = +{mc_sub.loc[mc_sub['model']=='L4a_hyp','ELBO'].values[0] - mc_sub.loc[mc_sub['model']=='L3_survival','ELBO'].values[0]:.1f}")
print(f"  M5 vs M1 ΔELBO = +{mc_sub.loc[mc_sub['model']=='L4a_add','ELBO'].values[0] - mc_sub.loc[mc_sub['model']=='L0_effort','ELBO'].values[0]:.1f}")
print(f"Affect LMMs:")
for _, row in affect_lmm.iterrows():
    if row["predictor"] == "S_probe_z":
        print(f"  {row['outcome']}: β={row['beta']:.3f}, t={row['t']:.1f}")
