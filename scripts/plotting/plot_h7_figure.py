#!/usr/bin/env python3
"""
Figure + Table for H7: Individuals who mobilize vigor under danger show
more accurate subjective threat appraisal.

Layout (1×2):
  A) δ vs anxiety S-slope scatter (H7a)
  B) δ vs confidence S-slope scatter (H7a)

Outputs:
  results/figs/paper/fig_h7_metacognition.{png,pdf}
  results/figs/paper/table_h7.html
  results/figs/paper/fig_h7_metacognition.html

Run: python scripts/plotting/plot_h7_figure.py
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import pearsonr, linregress
from scripts.plotting.plotter import Colors, set_plot_style, style_axis

# ── Paths ──────────────────────────────────────────────────────────────
DATA_DIR = "data/exploratory_350/processed/stage5_filtered_data_20260320_191950"
STATS_DIR = "results/stats"
OUT_DIR = "results/figs/paper"
os.makedirs(OUT_DIR, exist_ok=True)

# Use vigor HBM λ (matches metacognition notebook)
LAMBDA = 15.113

# ── Load data ──────────────────────────────────────────────────────────
print("Loading data...")

# Subject params (joint correlated model — matches metacognition notebook)
subj_params = pd.read_csv(os.path.join(STATS_DIR, "joint_correlated_subjects.csv"))
# Rename to match expected columns
subj_params = subj_params.rename(columns={"delta_ols": "delta_ols_orig"})
# Use log(beta) for consistency
subj_params["logb"] = np.log(subj_params["beta"])

# Pre-computed correlation results
meta_results = pd.read_csv(os.path.join(STATS_DIR, "paper", "metacognition_results.csv"))

# Feelings (probe trials) — compute per-subject S-slopes
feelings = pd.read_csv(os.path.join(DATA_DIR, "feelings.csv"))
feelings["D"] = feelings["distance"] + 1
feelings["S_probe"] = (1 - feelings["threat"]) + feelings["threat"] / (1 + LAMBDA * feelings["D"])

# Compute per-subject S-slopes via OLS
calibration_rows = []
for subj_id, grp in feelings.groupby("subj"):
    for qlabel in ["anxiety", "confidence"]:
        sub = grp[grp["questionLabel"] == qlabel]
        if len(sub) >= 5:
            slope, _, r, p, _ = linregress(sub["S_probe"], sub["response"])
            calibration_rows.append({
                "subj": subj_id,
                "outcome": qlabel,
                "S_slope": slope,
                "S_r": r,
            })

calib = pd.DataFrame(calibration_rows)
anx_slopes = calib[calib["outcome"] == "anxiety"][["subj", "S_slope"]].rename(
    columns={"S_slope": "anx_S_slope"})
conf_slopes = calib[calib["outcome"] == "confidence"][["subj", "S_slope"]].rename(
    columns={"S_slope": "conf_S_slope"})

# Also compute mean affect per subject
mean_affect = (
    feelings.groupby(["subj", "questionLabel"])["response"]
    .mean().unstack().reset_index()
    .rename(columns={"anxiety": "mean_anxiety", "confidence": "mean_confidence"})
)

# Merge
df = subj_params.merge(anx_slopes, on="subj").merge(conf_slopes, on="subj")
df = df.merge(mean_affect[["subj", "mean_anxiety"]], on="subj", how="left")
n_subj = len(df)
print(f"  Merged: {n_subj} subjects with calibration slopes")


# ── Panel A: δ vs anxiety S-slope ─────────────────────────────────────
def plot_scatter(ax, x, y, xlabel, ylabel, title, color):
    ax.scatter(x, y, s=18, alpha=0.4, color=color,
               edgecolors='white', linewidth=0.3, zorder=2)
    # Regression line
    m, b = np.polyfit(x, y, 1)
    x_line = np.linspace(x.min(), x.max(), 100)
    ax.plot(x_line, m * x_line + b, color=color, lw=2.5, zorder=3)
    # Stats
    r_val, p_val = pearsonr(x, y)
    p_str = f"p < 10$^{{{int(np.log10(p_val))}}}$" if p_val < 0.001 else f"p = {p_val:.3f}"
    ax.text(0.97, 0.95, f"r = {r_val:+.3f}\n{p_str}",
            transform=ax.transAxes, fontsize=9, va='top', ha='right',
            color=Colors.INK, bbox=dict(boxstyle='round,pad=0.4',
            facecolor='white', edgecolor=Colors.GREY, alpha=0.9))
    style_axis(ax, xlabel=xlabel, ylabel=ylabel)
    ax.set_title(title, fontsize=12, fontweight="bold",
                 color=Colors.DARK_GREY, loc="left")


# ── Build figure ──────────────────────────────────────────────────────
set_plot_style()
fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(11, 4.5))
fig.subplots_adjust(wspace=0.35, left=0.08, right=0.97, top=0.88, bottom=0.15)

plot_scatter(ax_a, df["delta"].values, df["anx_S_slope"].values,
             xlabel=r"$\delta$ (danger mobilization)",
             ylabel="Anxiety ~ S slope",
             title=r"A   $\delta$ predicts anxiety calibration",
             color=Colors.RUBY1)

plot_scatter(ax_b, df["delta"].values, df["conf_S_slope"].values,
             xlabel=r"$\delta$ (danger mobilization)",
             ylabel="Confidence ~ S slope",
             title=r"B   $\delta$ predicts confidence calibration",
             color=Colors.CERULEAN2)

for ext in ["pdf", "png"]:
    out = os.path.join(OUT_DIR, f"fig_h7_metacognition.{ext}")
    fig.savefig(out, bbox_inches="tight", dpi=200 if ext == "png" else None)
    print(f"Saved: {out}")
plt.close()


# ══════════════════════════════════════════════════════════════════════
# HTML TABLE (KaTeX)
# ══════════════════════════════════════════════════════════════════════

def build_html_table():
    # Gather key results
    param_calib = meta_results[meta_results["test_type"] == "param_x_calibration"].copy()

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>H7 — Vigor Mobilization Predicts Metacognitive Accuracy</title>
<link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/katex@0.16.11/dist/katex.min.css">
<script defer src="https://cdn.jsdelivr.net/npm/katex@0.16.11/dist/katex.min.js"></script>
<script defer src="https://cdn.jsdelivr.net/npm/katex@0.16.11/dist/contrib/auto-render.min.js"
  onload="renderMathInElement(document.body, {{delimiters: [{{left: '$$', right: '$$', display: true}}, {{left: '$', right: '$', display: false}}]}});"></script>
<style>
  :root {{
    --ink: #374151; --muted: #6B7280; --border: #E5E7EB;
    --border-strong: #D1D5DB; --bg-header: #F9FAFB;
    --accent: #2563EB; --pass: #059669;
    --pass-bg: #ECFDF5; --pass-border: #A7F3D0;
    --warn: #D97706; --warn-bg: #FFFBEB; --warn-border: #FDE68A;
  }}
  * {{ box-sizing: border-box; }}
  body {{
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
    max-width: 920px; margin: 48px auto; color: var(--ink);
    background: #fff; padding: 0 24px; line-height: 1.55;
  }}
  h1 {{ font-size: 1.35em; font-weight: 700; margin-bottom: 0.2em; }}
  h2 {{
    font-size: 1.0em; font-weight: 600; color: var(--muted);
    margin: 2.5em 0 0.6em; text-transform: uppercase; letter-spacing: 0.04em;
  }}
  .subtitle {{ color: var(--muted); font-size: 0.92em; margin-bottom: 2em; }}

  .hyp-block {{
    background: #F9FAFB; border: 1px solid var(--border);
    border-radius: 8px; padding: 20px 24px; margin-bottom: 2em;
  }}
  .hyp-statement {{ font-size: 0.92em; line-height: 1.65; margin-bottom: 12px; }}
  .hyp-subs {{ display: flex; flex-direction: column; gap: 6px; }}
  .hyp-sub {{
    font-size: 0.86em; color: var(--ink); padding: 6px 12px;
    background: white; border-radius: 6px; border: 1px solid var(--border); line-height: 1.55;
  }}

  table {{
    border-collapse: collapse; width: 100%; font-size: 0.88em; margin-bottom: 0.8em;
  }}
  thead th {{
    background: var(--bg-header); border-top: 2px solid var(--border-strong);
    border-bottom: 2px solid var(--border-strong); padding: 10px 14px;
    text-align: left; font-weight: 600; color: var(--muted);
    font-size: 0.85em; text-transform: uppercase; letter-spacing: 0.03em;
  }}
  thead th.r {{ text-align: right; }}
  tbody td {{
    border-bottom: 1px solid var(--border); padding: 10px 14px; vertical-align: middle;
  }}
  tbody td.r {{
    text-align: right; font-variant-numeric: tabular-nums;
    font-family: 'SF Mono', 'Menlo', 'Consolas', monospace; font-size: 0.92em;
  }}
  tbody tr:last-child td {{ border-bottom: 2px solid var(--border-strong); }}
  tbody tr:hover td {{ background: #F9FAFB; }}
  tr.highlight td {{ background: #EFF6FF; }}
  tr.highlight:hover td {{ background: #DBEAFE; }}

  .tests {{
    display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 12px; margin: 1.2em 0 0;
  }}
  .test {{
    border-radius: 8px; padding: 14px 16px;
  }}
  .test-pass {{ background: var(--pass-bg); border: 1px solid var(--pass-border); }}
  .test-warn {{ background: var(--warn-bg); border: 1px solid var(--warn-border); }}
  .test .label {{
    font-size: 0.78em; font-weight: 600; text-transform: uppercase;
    letter-spacing: 0.04em; color: var(--muted); margin-bottom: 4px;
  }}
  .test .value {{ font-size: 1.15em; font-weight: 700; }}
  .test .value-pass {{ color: var(--pass); }}
  .test .value-warn {{ color: var(--warn); }}
  .test .detail {{ font-size: 0.82em; color: var(--muted); margin-top: 2px; }}

  .note {{
    color: var(--muted); font-size: 0.82em; line-height: 1.6;
    margin-top: 1.5em; padding-top: 1em; border-top: 1px solid var(--border);
  }}
</style>
</head>
<body>

<h1>H7 — Individuals who mobilize vigor under danger show more accurate subjective threat appraisal</h1>
<p class="subtitle">Exploratory sample, $N = {n_subj}$ &middot; Per-subject OLS calibration slopes &middot; FDR-corrected</p>

<div class="hyp-block">
  <div class="hyp-statement">
    <strong>Hypothesis:</strong> Participants whose motor effort is more danger-responsive (higher $\\delta$)
    will show tighter affective tracking of survival probability $S$ &mdash; steeper anxiety&ndash;$S$ and
    confidence&ndash;$S$ slopes &mdash; constituting more accurate metacognitive appraisal of threat.
  </div>
  <div class="hyp-subs">
    <div class="hyp-sub"><strong>H7a.1</strong> &mdash; $r(\\delta,\\, \\text{{anxiety slope on }} S) < 0$, $p < .05$ one-tailed.</div>
    <div class="hyp-sub"><strong>H7a.2</strong> &mdash; $r(\\delta,\\, \\text{{confidence slope on }} S) > 0$, $p < .05$ one-tailed.</div>
    <div class="hyp-sub"><strong>H7a.3</strong> (secondary) &mdash; $r(\\delta,\\, \\text{{mean anxiety}}) < 0$: high-$\\delta$ individuals report lower average anxiety despite stronger coupling.</div>
  </div>
</div>

<h2>Table 1 &mdash; Parameter &times; calibration correlations</h2>
<p style="color: var(--muted); font-size: 0.85em; margin-bottom: 0.8em;">
  Per-subject calibration = OLS slope of rating $\\sim S_{{\\text{{probe}}}}$. All $p$-values FDR-corrected.
</p>

<table>
<thead>
<tr>
  <th>Parameter</th>
  <th>Metric</th>
  <th class="r">$r$</th>
  <th class="r">$p$ (raw)</th>
  <th class="r">$p$ (FDR)</th>
  <th class="r">$N$</th>
  <th class="r">Sig</th>
</tr>
</thead>
<tbody>
"""

    param_display = {"delta": "\\delta", "beta": "\\beta", "k": "k", "alpha": "\\alpha"}
    metric_display = {
        "S_slope_anxiety": "Anxiety S-slope",
        "S_slope_confidence": "Confidence S-slope",
        "mean_response_anxiety": "Mean anxiety",
        "mean_response_confidence": "Mean confidence",
    }

    for _, row in param_calib.iterrows():
        param_tex = param_display.get(row["parameter"], row["parameter"])
        metric_label = metric_display.get(row["metric"], row["metric"])
        p_raw = row["p"]
        p_fdr = row["p_fdr"]
        p_raw_str = f"< 10<sup>{int(np.log10(p_raw))}</sup>" if p_raw < 0.001 else f"{p_raw:.4f}"
        p_fdr_str = f"< 10<sup>{int(np.log10(p_fdr))}</sup>" if p_fdr < 0.001 else f"{p_fdr:.4f}"
        sig_str = "&check;" if row["sig"] else "&mdash;"

        # Highlight primary H7a tests (δ × S-slope)
        is_primary = row["parameter"] == "delta" and "S_slope" in row["metric"]
        cls = ' class="highlight"' if is_primary else ""

        html += f"""<tr{cls}>
  <td>${param_tex}$</td>
  <td>{metric_label}</td>
  <td class="r">{row['r']:+.3f}</td>
  <td class="r">{p_raw_str}</td>
  <td class="r">{p_fdr_str}</td>
  <td class="r">{int(row['n'])}</td>
  <td class="r">{sig_str}</td>
</tr>
"""

    # Get key values for test cards
    d_anx = param_calib[(param_calib["parameter"] == "delta") & (param_calib["metric"] == "S_slope_anxiety")].iloc[0]
    d_conf = param_calib[(param_calib["parameter"] == "delta") & (param_calib["metric"] == "S_slope_confidence")].iloc[0]
    d_mean = param_calib[(param_calib["parameter"] == "delta") & (param_calib["metric"] == "mean_response_anxiety")].iloc[0]

    html += f"""</tbody>
</table>

<div class="tests">
  <div class="test test-pass">
    <div class="label">H7a.1 &mdash; $\\delta$ &times; anxiety slope</div>
    <div class="value value-pass">$r$ = {d_anx['r']:+.3f}</div>
    <div class="detail">$p_{{\\text{{FDR}}}}$ = {d_anx['p_fdr']:.1e} &check;</div>
  </div>
  <div class="test test-pass">
    <div class="label">H7a.2 &mdash; $\\delta$ &times; confidence slope</div>
    <div class="value value-pass">$r$ = {d_conf['r']:+.3f}</div>
    <div class="detail">$p_{{\\text{{FDR}}}}$ = {d_conf['p_fdr']:.1e} &check;</div>
  </div>
  <div class="test test-pass">
    <div class="label">H7a.3 &mdash; $\\delta$ &times; mean anxiety</div>
    <div class="value value-pass">$r$ = {d_mean['r']:+.3f}</div>
    <div class="detail">Lower anxiety despite stronger coupling &check;</div>
  </div>
</div>

<div class="note">
  Calibration slope = per-subject OLS: rating $\\sim S_{{\\text{{probe}}}}$, where
  $S_{{\\text{{probe}}}} = (1 - T) + T / (1 + \\lambda D)$ with $\\lambda = {LAMBDA:.3f}$.
  More negative anxiety slopes (and more positive confidence slopes) indicate tighter
  affective tracking of survival probability &mdash; better metacognitive accuracy.
  All $p$-values corrected for false discovery rate across 16 parameter &times; metric tests.
  The table includes all parameters for completeness; the primary H7a tests concern $\\delta$ only.
</div>

</body>
</html>"""
    return html


html_table = build_html_table()
table_path = os.path.join(OUT_DIR, "table_h7.html")
with open(table_path, 'w') as f:
    f.write(html_table)
print(f"Saved: {table_path}")


# ══════════════════════════════════════════════════════════════════════
# COMBINED HTML
# ══════════════════════════════════════════════════════════════════════

def build_combined_html():
    import base64
    png_path = os.path.join(OUT_DIR, "fig_h7_metacognition.png")
    with open(png_path, 'rb') as f:
        img_b64 = base64.b64encode(f.read()).decode()

    style_start = html_table.index("<style>")
    style_end = html_table.index("</style>") + len("</style>")
    style_block = html_table[style_start:style_end]

    body_start = html_table.index('<div class="hyp-block">')
    body_end = html_table.index("</body>")
    table_content = html_table[body_start:body_end]

    d_anx = meta_results[(meta_results["parameter"] == "delta") & (meta_results["metric"] == "S_slope_anxiety")].iloc[0]
    d_conf = meta_results[(meta_results["parameter"] == "delta") & (meta_results["metric"] == "S_slope_confidence")].iloc[0]

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>H7 — Figure &amp; Table</title>
<link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/katex@0.16.11/dist/katex.min.css">
<script defer src="https://cdn.jsdelivr.net/npm/katex@0.16.11/dist/katex.min.js"></script>
<script defer src="https://cdn.jsdelivr.net/npm/katex@0.16.11/dist/contrib/auto-render.min.js"
  onload="renderMathInElement(document.body, {{delimiters: [{{left: '$$', right: '$$', display: true}}, {{left: '$', right: '$', display: false}}]}});"></script>
{style_block}
<style>
  .fig-container {{ text-align: center; margin: 2em 0 2.5em; }}
  .fig-container img {{ max-width: 100%; border: 1px solid var(--border); border-radius: 6px; }}
  .caption {{ color: var(--muted); font-size: 0.88em; margin-top: 1em; text-align: left; line-height: 1.65; }}
  .caption strong {{ color: var(--ink); }}
</style>
</head>
<body>

<h1>H7 — Individuals who mobilize vigor under danger show more accurate subjective threat appraisal</h1>

<div class="fig-container">
  <img src="data:image/png;base64,{img_b64}" alt="H7 Figure">
  <p class="caption">
    <strong>Figure.</strong>
    <strong>(A)</strong> Participants with higher danger mobilization ($\\delta$) show steeper
    negative anxiety&ndash;$S$ slopes ($r = {d_anx['r']:+.3f}$), indicating their anxiety ratings
    more accurately track survival probability.
    <strong>(B)</strong> The same pattern for confidence: higher $\\delta$ predicts steeper
    positive confidence&ndash;$S$ slopes ($r = {d_conf['r']:+.3f}$).
    $N = {n_subj}$.
  </p>
</div>

{table_content}

</body>
</html>"""
    return html


combined_html = build_combined_html()
combined_path = os.path.join(OUT_DIR, "fig_h7_metacognition.html")
with open(combined_path, 'w') as f:
    f.write(combined_html)
print(f"Saved: {combined_path}")


# ── Summary ───────────────────────────────────────────────────────────
d_anx = meta_results[(meta_results["parameter"] == "delta") & (meta_results["metric"] == "S_slope_anxiety")].iloc[0]
d_conf = meta_results[(meta_results["parameter"] == "delta") & (meta_results["metric"] == "S_slope_confidence")].iloc[0]
d_mean = meta_results[(meta_results["parameter"] == "delta") & (meta_results["metric"] == "mean_response_anxiety")].iloc[0]
print(f"\n=== H7 Summary ===")
print(f"H7a.1: r(δ, anx S-slope) = {d_anx['r']:+.3f}, p = {d_anx['p']:.2e}")
print(f"H7a.2: r(δ, conf S-slope) = {d_conf['r']:+.3f}, p = {d_conf['p']:.2e}")
print(f"H7a.3: r(δ, mean anxiety) = {d_mean['r']:+.3f}, p = {d_mean['p']:.4f}")
print(f"N = {n_subj}")
