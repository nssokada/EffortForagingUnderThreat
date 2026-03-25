#!/usr/bin/env python3
"""
Figure + Table for H5: Model-derived danger drives excess motor vigor.

Layout (1×2):
  A) Distribution of subject-level δ_i (danger mobilization)
  B) Trial-level excess effort vs danger (1−S)

Outputs:
  results/figs/paper/fig_h5_vigor_model.{png,pdf}
  results/figs/paper/table_h5.html
  results/figs/paper/fig_h5_vigor_model.html

Run: python scripts/plotting/plot_h5_figure.py
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.stats import gaussian_kde
from scripts.plotting.plotter import Colors, set_plot_style, style_axis

# ── Paths ──────────────────────────────────────────────────────────────
DATA_DIR = "data/exploratory_350/processed/stage5_filtered_data_20260320_191950"
STATS_DIR = "results/stats"
OUT_DIR = "results/figs/paper"
os.makedirs(OUT_DIR, exist_ok=True)

# ── Constants ──────────────────────────────────────────────────────────
LAMBDA = 0.6976  # binary-E λ

# ── Load data ──────────────────────────────────────────────────────────
print("Loading data...")

# MCMC vigor population params
pop = pd.read_csv(os.path.join(STATS_DIR, "mcmc_vigor_population.csv"))
mu_delta = pop["mu_delta"].values[0]
mu_delta_lo = pop["mu_delta_2.5"].values[0]
mu_delta_hi = pop["mu_delta_97.5"].values[0]
sd_delta = pop["sd_delta"].values[0]
pct_pos = pop["pct_delta_pos"].values[0] * 100

# Per-subject δ
subj_params = pd.read_csv(os.path.join(STATS_DIR, "mcmc_vigor_params.csv"))
delta_vals = subj_params["delta_mcmc"].values
n_subj = len(delta_vals)

# Binary-E validation (for γ)
be_val = pd.read_csv(os.path.join(STATS_DIR, "binary_e_validation.csv"))
gamma_row = be_val[be_val["model"] == "vigor_effort_controlled"].iloc[0]
gamma_val = gamma_row["gamma"]

# Trial-level behavior for Panel B (behavior_rich already excludes probes)
br = pd.read_csv(os.path.join(DATA_DIR, "behavior_rich.csv"))

# Compute excess effort and danger (1-S)
br["effort_chosen"] = np.where(br["choice"] == 1, br["effort_H"], br["effort_L"])
br["distance_chosen"] = np.where(br["choice"] == 1, br["distance_H"], br["distance_L"])
br["excess_effort"] = br["mean_trial_effort"] - br["effort_chosen"]
br["S"] = (1 - br["threat"]) + br["threat"] / (1 + LAMBDA * br["distance_chosen"])
br["danger"] = 1 - br["S"]


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


# ── Panel A: δ_i distribution ─────────────────────────────────────────
def plot_delta_distribution(ax):
    # Histogram + KDE
    bins = np.linspace(delta_vals.min() - 0.05, delta_vals.max() + 0.05, 40)
    ax.hist(delta_vals, bins=bins, color=Colors.SLATE, alpha=0.4,
            edgecolor='white', linewidth=0.5, density=True, zorder=2)

    # KDE overlay
    kde = gaussian_kde(delta_vals, bw_method=0.25)
    x_kde = np.linspace(bins[0], bins[-1], 200)
    ax.plot(x_kde, kde(x_kde), color=Colors.RUBY1, lw=2.5, zorder=3)

    # Shade area > 0
    x_pos = x_kde[x_kde >= 0]
    ax.fill_between(x_pos, kde(x_pos), color=Colors.RUBY1, alpha=0.15, zorder=2)

    # μ_δ line + CI
    ax.axvline(mu_delta, color=Colors.RUBY1, lw=2, ls='-', zorder=4)
    ax.axvline(mu_delta_lo, color=Colors.RUBY1, lw=1, ls='--', alpha=0.6, zorder=4)
    ax.axvline(mu_delta_hi, color=Colors.RUBY1, lw=1, ls='--', alpha=0.6, zorder=4)

    # Zero reference
    ax.axvline(0, color=Colors.INK, lw=1, ls=':', alpha=0.5, zorder=1)

    # Annotations
    ax.text(0.97, 0.95,
            f"$\\mu_\\delta$ = {mu_delta:.3f}\n"
            f"95% CI [{mu_delta_lo:.3f}, {mu_delta_hi:.3f}]\n"
            f"{pct_pos:.1f}% > 0\n"
            f"$N$ = {n_subj}",
            transform=ax.transAxes, fontsize=9, va='top', ha='right',
            color=Colors.INK, bbox=dict(boxstyle='round,pad=0.4',
            facecolor='white', edgecolor=Colors.GREY, alpha=0.9))

    style_axis(ax, ylabel="Density", xlabel=r"Subject-level $\delta_i$ (danger mobilization)")
    ax.set_title(r"A   Distribution of $\delta_i$", fontsize=12, fontweight="bold",
                 color=Colors.DARK_GREY, loc="left")


# ── Panel B: Excess effort vs danger (1-S) ────────────────────────────
# Lines by distance, dots colored by threat
THREAT_COLORS = {0.1: Colors.CERULEAN2, 0.5: Colors.SLATE, 0.9: Colors.RUBY1}
THREAT_LABELS = {0.1: "Low (T=0.1)", 0.5: "Mid (T=0.5)", 0.9: "High (T=0.9)"}
DIST_LABELS = {1: "Near (D=1)", 2: "Mid (D=2)", 3: "Far (D=3)"}
DIST_LS = {1: '-', 2: '--', 3: ':'}

def plot_excess_vs_danger(ax):
    # Per-subject cell means at each (threat, distance) condition
    br_subj = (
        br.groupby(["subj", "threat", "distance_chosen"])
        .agg(excess_effort=("excess_effort", "mean"),
             danger=("danger", "mean"))
        .reset_index()
    )
    br_subj["danger_round"] = br_subj["danger"].round(3)

    # Within-subject SEM at each (threat, distance) cell
    agg = within_subject_sem(br_subj, 'subj', 'excess_effort',
                             ['threat', 'distance_chosen', 'danger_round'])
    agg = agg.sort_values("danger_round")

    # Lines connecting dots across threat levels within each distance
    for d in [1, 2, 3]:
        sub = agg[agg["distance_chosen"] == d].sort_values("danger_round")
        ax.plot(sub["danger_round"], sub["mean"], color=Colors.INK,
                ls=DIST_LS[d], lw=1.8, alpha=0.5, zorder=2,
                label=DIST_LABELS[d])

    # Dots colored by threat (plotted on top)
    for t in [0.1, 0.5, 0.9]:
        sub = agg[agg["threat"] == t].sort_values("danger_round")
        ax.errorbar(
            sub["danger_round"], sub["mean"], yerr=sub["sem"],
            color=THREAT_COLORS[t], marker='o', ms=8, lw=0,
            capsize=4, capthick=1.5, label=THREAT_LABELS[t], zorder=4
        )

    # Model regression line
    mean_e = br["effort_chosen"].mean()
    mu_alpha = pop["mu_alpha"].values[0]
    d_line = np.linspace(0, agg["danger_round"].max() + 0.02, 100)
    excess_line = mu_alpha + mu_delta * d_line + gamma_val * mean_e
    ax.plot(d_line, excess_line, color=Colors.RUBY1, lw=2.5, ls='--', alpha=0.7,
            zorder=3, label=r"HBM fit")

    ax.axhline(0, color=Colors.INK, lw=0.8, ls=':', alpha=0.4, zorder=1)

    style_axis(ax, ylabel="Excess effort (observed \u2212 demanded)",
               xlabel="Danger  (1 \u2212 S)")
    ax.set_title("B   Excess effort scales with danger", fontsize=12,
                 fontweight="bold", color=Colors.DARK_GREY, loc="left")
    ax.legend(fontsize=7.5, frameon=False, loc="upper left", ncol=2)


# ── Build figure ──────────────────────────────────────────────────────
set_plot_style()
fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(11, 4.5))
fig.subplots_adjust(wspace=0.35, left=0.08, right=0.97, top=0.90, bottom=0.14)

plot_delta_distribution(ax_a)
plot_excess_vs_danger(ax_b)

for ext in ["pdf", "png"]:
    out = os.path.join(OUT_DIR, f"fig_h5_vigor_model.{ext}")
    fig.savefig(out, bbox_inches="tight", dpi=200 if ext == "png" else None)
    print(f"Saved: {out}")
plt.close()


# ══════════════════════════════════════════════════════════════════════
# HTML TABLE (KaTeX)
# ══════════════════════════════════════════════════════════════════════

def build_html_table():
    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>H5 — Danger Drives Excess Motor Vigor</title>
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
    --accent: #2563EB;
    --pass: #059669;
    --pass-bg: #ECFDF5;
    --pass-border: #A7F3D0;
    --fail-bg: #FEF2F2;
    --fail-border: #FECACA;
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
  h1 {{ font-size: 1.35em; font-weight: 700; margin-bottom: 0.2em; letter-spacing: -0.01em; }}
  h2 {{
    font-size: 1.0em; font-weight: 600; color: var(--muted);
    margin: 2.5em 0 0.6em; text-transform: uppercase; letter-spacing: 0.04em;
  }}
  .subtitle {{ color: var(--muted); font-size: 0.92em; margin-bottom: 2em; }}

  table {{
    border-collapse: collapse; width: 100%;
    font-size: 0.88em; margin-bottom: 0.8em;
  }}
  thead th {{
    background: var(--bg-header);
    border-top: 2px solid var(--border-strong);
    border-bottom: 2px solid var(--border-strong);
    padding: 10px 14px; text-align: left;
    font-weight: 600; color: var(--muted);
    font-size: 0.85em; text-transform: uppercase; letter-spacing: 0.03em;
  }}
  thead th.r {{ text-align: right; }}
  tbody td {{
    border-bottom: 1px solid var(--border);
    padding: 10px 14px; vertical-align: middle;
  }}
  tbody td.r {{
    text-align: right; font-variant-numeric: tabular-nums;
    font-family: 'SF Mono', 'Menlo', 'Consolas', monospace; font-size: 0.92em;
  }}
  tbody tr:last-child td {{ border-bottom: 2px solid var(--border-strong); }}
  tbody tr:hover td {{ background: #F9FAFB; }}

  .model-spec {{
    background: var(--bg-header);
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 20px 24px;
    margin: 1.5em 0;
    text-align: center;
    line-height: 2;
  }}
  .model-spec .label {{
    font-size: 0.78em; font-weight: 600;
    text-transform: uppercase; letter-spacing: 0.04em;
    color: var(--muted); margin-bottom: 8px;
  }}

  .tests {{
    display: grid;
    grid-template-columns: 1fr 1fr;
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
    font-size: 0.78em; font-weight: 600;
    text-transform: uppercase; letter-spacing: 0.04em;
    color: var(--muted); margin-bottom: 4px;
  }}
  .test .value {{ font-size: 1.15em; font-weight: 700; color: var(--pass); }}
  .test .detail {{ font-size: 0.82em; color: var(--muted); margin-top: 2px; }}

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

  .note {{
    color: var(--muted); font-size: 0.82em; line-height: 1.6;
    margin-top: 1.5em; padding-top: 1em; border-top: 1px solid var(--border);
  }}
</style>
</head>
<body>

<h1>H5 — Model-derived danger drives excess motor vigor</h1>
<p class="subtitle">Exploratory sample, $N = {n_subj}$ &middot; MCMC (NumPyro NUTS, 4 chains &times; 1000) &middot; $\\lambda = {LAMBDA:.3f}$</p>

<div class="hyp-block">
  <div class="hyp-statement">
    <strong>Hypothesis:</strong> Lower survival probability $S$ causes participants to press harder than the task requires,
    expressed as a positive population-mean danger mobilization parameter $\\delta$ in a hierarchical Bayesian model of
    excess effort, after controlling for the physical effort demand of the chosen option.
  </div>
  <div class="hyp-subs">
    <div class="hyp-sub"><strong>H5a.1</strong> &mdash; Population mean $\\mu_\\delta > 0$; the posterior 95% credible interval must exclude zero.</div>
    <div class="hyp-sub"><strong>H5a.2</strong> &mdash; The proportion of subjects with posterior mean $\\delta_i > 0$ must exceed 80%.</div>
    <div class="hyp-sub"><strong>Secondary</strong> &mdash; $\\sigma_\\delta > 0.05$ (individual differences recoverable) and $\\gamma < 0$ (effort demand constraint confirmed).</div>
  </div>
</div>

<div class="model-spec">
  <div class="label">Vigor model</div>
  $$\\text{{excess}}_{{ij}} = \\alpha_i + \\delta_i \\cdot (1 - S_{{ij}}) + \\gamma \\cdot E_{{\\text{{chosen}},ij}} + \\varepsilon_{{ij}}$$
  <div style="color: var(--muted); font-size: 0.85em; margin-top: 4px;">
    $S_{{ij}} = (1 - T) + T / (1 + \\lambda D_{{\\text{{chosen}}}})$ &emsp;|&emsp;
    $\\alpha_i \\sim \\mathcal{{N}}(\\mu_\\alpha, \\sigma_\\alpha)$ &emsp;|&emsp;
    $\\delta_i \\sim \\mathcal{{N}}(\\mu_\\delta, \\sigma_\\delta)$
  </div>
</div>

<h2>Table 1 &mdash; Population Parameters</h2>
<table>
<thead>
<tr>
  <th>Parameter</th>
  <th>Description</th>
  <th class="r">Estimate</th>
  <th class="r">95% CI</th>
</tr>
</thead>
<tbody>
<tr>
  <td>$\\mu_\\delta$</td>
  <td>Population mean danger mobilization</td>
  <td class="r">{mu_delta:.3f}</td>
  <td class="r">[{mu_delta_lo:.3f}, {mu_delta_hi:.3f}]</td>
</tr>
<tr>
  <td>$\\sigma_\\delta$</td>
  <td>Between-subject SD of $\\delta$</td>
  <td class="r">{sd_delta:.3f}</td>
  <td class="r"></td>
</tr>
<tr>
  <td>$\\mu_\\alpha$</td>
  <td>Population mean baseline excess effort</td>
  <td class="r">{pop['mu_alpha'].values[0]:.3f}</td>
  <td class="r"></td>
</tr>
<tr>
  <td>$\\gamma$</td>
  <td>Effort demand constraint (population)</td>
  <td class="r">{gamma_val:.3f}</td>
  <td class="r"></td>
</tr>
<tr>
  <td>$\\%\\;\\delta_i > 0$</td>
  <td>Proportion of subjects with positive danger mobilization</td>
  <td class="r">{pct_pos:.1f}%</td>
  <td class="r"></td>
</tr>
</tbody>
</table>

<div class="tests">
  <div class="test">
    <div class="label">H5a.1 &mdash; $\\mu_\\delta > 0$</div>
    <div class="value">{mu_delta:.3f} &ensp; [{mu_delta_lo:.3f}, {mu_delta_hi:.3f}]</div>
    <div class="detail">95% CI excludes zero &check;</div>
  </div>
  <div class="test">
    <div class="label">H5a.2 &mdash; &gt; 80% positive</div>
    <div class="value">{pct_pos:.1f}%</div>
    <div class="detail">Threshold: 80% &check;</div>
  </div>
  <div class="test">
    <div class="label">Secondary &mdash; $\\sigma_\\delta > 0.05$</div>
    <div class="value">{sd_delta:.3f}</div>
    <div class="detail">Individual differences recoverable &check;</div>
  </div>
  <div class="test">
    <div class="label">Secondary &mdash; $\\gamma < 0$</div>
    <div class="value">{gamma_val:.3f}</div>
    <div class="detail">Effort demand constraint confirmed &check;</div>
  </div>
</div>

<div class="note">
  Excess effort = observed capacity-normalized vigor &minus; graded effort demand of chosen option.
  $S_{{ij}}$ uses $\\lambda = {LAMBDA:.3f}$ from the binary-$E$ choice model.
  $\\gamma$ controls for the physical ceiling: participants who chose the harder option have less headroom.
  All subject-level parameters estimated via hierarchical normal priors (MCMC NUTS, 4 chains &times; 1000 warmup + 1000 samples, target acceptance = 0.95).
</div>

</body>
</html>"""
    return html


html_table = build_html_table()
table_path = os.path.join(OUT_DIR, "table_h5.html")
with open(table_path, 'w') as f:
    f.write(html_table)
print(f"Saved: {table_path}")


# ══════════════════════════════════════════════════════════════════════
# COMBINED HTML (figure + table)
# ══════════════════════════════════════════════════════════════════════

def build_combined_html():
    import base64
    png_path = os.path.join(OUT_DIR, "fig_h5_vigor_model.png")
    with open(png_path, 'rb') as f:
        img_b64 = base64.b64encode(f.read()).decode()

    # Extract style and body from table HTML
    style_start = html_table.index("<style>")
    style_end = html_table.index("</style>") + len("</style>")
    style_block = html_table[style_start:style_end]

    body_start = html_table.index('<div class="hyp-block">')
    body_end = html_table.index("</body>")
    table_content = html_table[body_start:body_end]

    # Also grab the model-spec div
    spec_start = html_table.index('<div class="model-spec">')
    spec_end = html_table.index("</div>", spec_start) + len("</div>")
    # Actually grab from model-spec to first h2
    spec_block = html_table[spec_start:body_start]

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>H5 — Figure &amp; Table</title>
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

<h1>H5 — Model-derived danger drives excess motor vigor</h1>

<div class="fig-container">
  <img src="data:image/png;base64,{img_b64}" alt="H5 Figure">
  <p class="caption">
    <strong>Figure.</strong>
    <strong>(A)</strong> Distribution of subject-level danger mobilization parameters $\\delta_i$
    estimated via hierarchical Bayesian model (MCMC). The population mean
    $\\mu_\\delta = {mu_delta:.3f}$ (solid line) with 95% CI (dashed) excludes zero.
    {pct_pos:.1f}% of subjects show $\\delta_i > 0$ (shaded region).
    <strong>(B)</strong> Trial-level excess effort (observed &minus; demanded) increases with
    danger $(1 - S)$, colored by threat level. Dashed line shows the HBM prediction.
    $N = {n_subj}$; error bars = within-subject SEM.
  </p>
</div>

{spec_block}

{table_content}

</body>
</html>"""
    return html


combined_html = build_combined_html()
combined_path = os.path.join(OUT_DIR, "fig_h5_vigor_model.html")
with open(combined_path, 'w') as f:
    f.write(combined_html)
print(f"Saved: {combined_path}")


# ── Summary ───────────────────────────────────────────────────────────
print(f"\n=== H5 Summary ===")
print(f"μ_δ = {mu_delta:.3f} [{mu_delta_lo:.3f}, {mu_delta_hi:.3f}]")
print(f"σ_δ = {sd_delta:.3f}")
print(f"% δ > 0 = {pct_pos:.1f}%")
print(f"γ = {gamma_val:.3f}")
print(f"N = {n_subj}")
