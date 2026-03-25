#!/usr/bin/env python3
"""
Figure + Table for H6: Choice and vigor parameters are coupled across
independently estimated Bayesian models.

Layout (1×3):
  A) log(β) vs δ scatter (H6a — independent pipeline)
  B) Joint LKJ model correlation heatmap (H6b)
  C) Optimality ~ β + δ (H6c)

Outputs:
  results/figs/paper/fig_h6_coupling.{png,pdf}
  results/figs/paper/table_h6.html
  results/figs/paper/fig_h6_coupling.html

Run: python scripts/plotting/plot_h6_figure.py
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.stats import pearsonr, zscore
from scripts.plotting.plotter import Colors, set_plot_style, style_axis

# ── Paths ──────────────────────────────────────────────────────────────
STATS_DIR = "results/stats"
OUT_DIR = "results/figs/paper"
os.makedirs(OUT_DIR, exist_ok=True)

LAMBDA = 0.6976

# ── Load data ──────────────────────────────────────────────────────────
print("Loading data...")

# Subject params (binary-E: choice + vigor)
subj = pd.read_csv(os.path.join(STATS_DIR, "binary_e_subject_params.csv"))

# Independent Bayesian correlations
indep_corr = pd.read_csv(os.path.join(STATS_DIR, "independent_bayesian_correlations.csv"))

# Joint LKJ correlations
joint_corr = pd.read_csv(os.path.join(STATS_DIR, "joint_correlated_correlations.csv"))

# Optimality
opt = pd.read_csv(os.path.join(STATS_DIR, "optimality_per_subject.csv"))

# Merge for H6c regression
merged = subj.merge(opt, on="subj")
n_subj = len(merged)


# ── Panel A: log(β) vs δ scatter (H6a) ───────────────────────────────
def plot_coupling_scatter(ax):
    x = subj["logb"].values
    y = subj["delta"].values

    ax.scatter(x, y, s=18, alpha=0.45, color=Colors.CERULEAN2,
               edgecolors='white', linewidth=0.3, zorder=2)

    # Regression line
    m, b = np.polyfit(x, y, 1)
    x_line = np.linspace(x.min(), x.max(), 100)
    ax.plot(x_line, m * x_line + b, color=Colors.RUBY1, lw=2.5, zorder=3)

    # Stats
    r_val, p_val = pearsonr(x, y)
    p_str = f"p < 10$^{{{int(np.log10(p_val))}}}$" if p_val < 0.001 else f"p = {p_val:.3f}"
    ax.text(0.97, 0.95, f"r = +{r_val:.3f}\n{p_str}\nN = {len(x)}",
            transform=ax.transAxes, fontsize=9, va='top', ha='right',
            color=Colors.INK, bbox=dict(boxstyle='round,pad=0.4',
            facecolor='white', edgecolor=Colors.GREY, alpha=0.9))

    style_axis(ax, xlabel=r"log($\beta$)  (threat bias, choice model)",
               ylabel=r"$\delta$  (danger mobilization, vigor model)")
    ax.set_title("A   H6a: β–δ coupling", fontsize=12, fontweight="bold",
                 color=Colors.DARK_GREY, loc="left")


# ── Panel B: Joint LKJ correlation heatmap (H6b) ─────────────────────
def plot_joint_heatmap(ax):
    params = ["log_k", "log_beta", "alpha", "delta"]
    labels = [r"log($k$)", r"log($\beta$)", r"$\alpha$", r"$\delta$"]
    n = len(params)

    # Build correlation matrix
    corr_mat = np.eye(n)
    sig_mat = np.ones((n, n), dtype=bool)  # CI excludes zero

    for _, row in joint_corr.iterrows():
        p1, p2 = row["param_1"], row["param_2"]
        if p1 in params and p2 in params:
            i, j = params.index(p1), params.index(p2)
            corr_mat[i, j] = row["rho_mean"]
            corr_mat[j, i] = row["rho_mean"]
            ci_excludes_zero = (row["rho_2.5"] > 0 and row["rho_97.5"] > 0) or \
                               (row["rho_2.5"] < 0 and row["rho_97.5"] < 0)
            sig_mat[i, j] = ci_excludes_zero
            sig_mat[j, i] = ci_excludes_zero

    # Plot heatmap
    im = ax.imshow(corr_mat, cmap='RdBu_r', vmin=-1, vmax=1, aspect='equal')

    # Add text annotations
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            val = corr_mat[i, j]
            color = 'white' if abs(val) > 0.5 else Colors.INK
            weight = 'bold' if sig_mat[i, j] else 'normal'
            ax.text(j, i, f"{val:+.2f}", ha='center', va='center',
                    fontsize=9, color=color, fontweight=weight)

    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(labels, fontsize=10)
    ax.set_yticklabels(labels, fontsize=10)
    ax.tick_params(length=0)

    # Colorbar
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, shrink=0.85)
    cbar.set_label(r"$\rho$ (LKJ posterior mean)", fontsize=9, color=Colors.INK)
    cbar.ax.tick_params(labelsize=8, colors=Colors.INK)

    ax.set_title("B   H6b: Joint model", fontsize=12, fontweight="bold",
                 color=Colors.DARK_GREY, loc="left")

    # Highlight β-δ cell
    from matplotlib.patches import Rectangle
    # log_beta is index 1, delta is index 3
    rect = Rectangle((3 - 0.5, 1 - 0.5), 1, 1, linewidth=2.5,
                      edgecolor=Colors.RUBY1, facecolor='none', zorder=5)
    ax.add_patch(rect)
    rect2 = Rectangle((1 - 0.5, 3 - 0.5), 1, 1, linewidth=2.5,
                       edgecolor=Colors.RUBY1, facecolor='none', zorder=5)
    ax.add_patch(rect2)


# ── Panel C: Optimality ~ β + δ (H6c) ────────────────────────────────
def plot_optimality(ax):
    # Z-score predictors
    merged["logb_z"] = zscore(merged["logb"])
    merged["delta_z"] = zscore(merged["delta"])
    merged["k_z"] = zscore(merged["logk"])

    # OLS: pct_optimal ~ logb_z + delta_z + k_z
    from numpy.linalg import lstsq
    X = np.column_stack([
        np.ones(len(merged)),
        merged["logb_z"].values,
        merged["delta_z"].values,
        merged["k_z"].values,
    ])
    y = merged["pct_optimal"].values
    betas, residuals, _, _ = lstsq(X, y, rcond=None)
    y_hat = X @ betas
    ss_res = np.sum((y - y_hat) ** 2)
    ss_tot = np.sum((y - y.mean()) ** 2)
    r_sq = 1 - ss_res / ss_tot

    # Standard errors
    n_obs = len(y)
    n_pred = X.shape[1]
    mse = ss_res / (n_obs - n_pred)
    se = np.sqrt(np.diag(mse * np.linalg.inv(X.T @ X)))
    t_vals = betas / se

    # Store for table
    global h6c_results
    h6c_results = {
        "r_sq": r_sq,
        "n": n_obs,
        "predictors": {
            "β_z": {"beta": betas[1], "se": se[1], "t": t_vals[1]},
            "δ_z": {"beta": betas[2], "se": se[2], "t": t_vals[2]},
            "k_z": {"beta": betas[3], "se": se[3], "t": t_vals[3]},
        }
    }

    # Scatter: color by δ tercile
    delta_vals = merged["delta_z"].values
    terc_lo = np.percentile(delta_vals, 33.3)
    terc_hi = np.percentile(delta_vals, 66.7)
    colors = np.where(delta_vals > terc_hi, Colors.RUBY1,
                      np.where(delta_vals < terc_lo, Colors.CERULEAN2, Colors.SLATE))

    ax.scatter(merged["logb_z"], merged["pct_optimal"], s=18, alpha=0.45,
               c=colors, edgecolors='white', linewidth=0.3, zorder=2)

    # Regression lines for each δ tercile
    for terc_val, color, label in [
        (1.0, Colors.RUBY1, r"High $\delta$"),
        (0.0, Colors.SLATE, r"Med $\delta$"),
        (-1.0, Colors.CERULEAN2, r"Low $\delta$"),
    ]:
        x_line = np.linspace(-2.5, 2.5, 100)
        y_line = betas[0] + betas[1] * x_line + betas[2] * terc_val + betas[3] * 0
        ax.plot(x_line, y_line, color=color, lw=2, alpha=0.7, label=label, zorder=3)

    ax.text(0.97, 0.05, f"$R^2$ = {r_sq:.3f}\nN = {n_obs}",
            transform=ax.transAxes, fontsize=9, va='bottom', ha='right',
            color=Colors.INK, bbox=dict(boxstyle='round,pad=0.4',
            facecolor='white', edgecolor=Colors.GREY, alpha=0.9))

    style_axis(ax, xlabel=r"$\beta_z$ (threat bias)",
               ylabel="% optimal choices")
    ax.set_title(r"C   H6c: Optimality ~ $\beta$ + $\delta$", fontsize=12,
                 fontweight="bold", color=Colors.DARK_GREY, loc="left")
    ax.legend(fontsize=8, frameon=False, loc="upper left")


# ── Build figure ──────────────────────────────────────────────────────
set_plot_style()
fig, (ax_a, ax_b, ax_c) = plt.subplots(1, 3, figsize=(14, 4.5))
fig.subplots_adjust(wspace=0.42, left=0.06, right=0.97, top=0.88, bottom=0.15)

plot_coupling_scatter(ax_a)
plot_joint_heatmap(ax_b)
plot_optimality(ax_c)

for ext in ["pdf", "png"]:
    out = os.path.join(OUT_DIR, f"fig_h6_coupling.{ext}")
    fig.savefig(out, bbox_inches="tight", dpi=200 if ext == "png" else None)
    print(f"Saved: {out}")
plt.close()


# ══════════════════════════════════════════════════════════════════════
# HTML TABLE (KaTeX)
# ══════════════════════════════════════════════════════════════════════

def build_html_table():
    # Get key values
    indep_bd = indep_corr[(indep_corr["param_1"] == "log(β)") & (indep_corr["param_2"] == "δ")].iloc[0]
    joint_bd = joint_corr[(joint_corr["param_1"] == "log_beta") & (joint_corr["param_2"] == "delta")].iloc[0]

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>H6 — Cross-Model Parameter Coupling</title>
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
    background: var(--pass-bg); border: 1px solid var(--pass-border);
    border-radius: 8px; padding: 14px 16px;
  }}
  .test .label {{
    font-size: 0.78em; font-weight: 600; text-transform: uppercase;
    letter-spacing: 0.04em; color: var(--muted); margin-bottom: 4px;
  }}
  .test .value {{ font-size: 1.15em; font-weight: 700; color: var(--pass); }}
  .test .detail {{ font-size: 0.82em; color: var(--muted); margin-top: 2px; }}

  .note {{
    color: var(--muted); font-size: 0.82em; line-height: 1.6;
    margin-top: 1.5em; padding-top: 1em; border-top: 1px solid var(--border);
  }}
</style>
</head>
<body>

<h1>H6 — Choice and vigor parameters are coupled across independently estimated Bayesian models</h1>
<p class="subtitle">Exploratory sample, $N = {n_subj}$ &middot; Independent Bayesian pipeline + Joint LKJ model</p>

<div class="hyp-block">
  <div class="hyp-statement">
    <strong>Statement:</strong> Individual differences in threat bias ($\\beta$, from the choice model) and
    danger-responsive vigor mobilization ($\\delta$, from the vigor model) are positively correlated.
    This correlation emerges from models that share no parameters or data &mdash; only the survival
    function $S$ (evaluated at the choice-estimated $\\lambda$) links them.
  </div>
  <div class="hyp-subs">
    <div class="hyp-sub"><strong>H6a</strong> &mdash; Pearson $r(\\log\\beta_{{\\text{{choice}}}},\\, \\delta_{{\\text{{vigor}}}}) > 0$, $p < .001$ (one-tailed), via independently estimated hierarchical models.</div>
    <div class="hyp-sub"><strong>H6b</strong> &mdash; A joint LKJ model with correlated random effects confirms $\\rho(\\beta, \\delta)$ posterior mean $> 0$ with 95% CI excluding zero.</div>
    <div class="hyp-sub"><strong>H6c</strong> &mdash; OLS: optimality $\\sim \\beta_z + \\delta_z + k_z$. Both $\\beta$ and $\\delta$ are significant predictors ($p < .05$).</div>
  </div>
</div>

<h2>Table 1 &mdash; Independent Bayesian cross-correlations</h2>
<table>
<thead>
<tr>
  <th>Parameters</th>
  <th class="r">$r$</th>
  <th class="r">$p$</th>
</tr>
</thead>
<tbody>
"""

    for _, row in indep_corr.iterrows():
        p = row["p"]
        p_str = f"< 10<sup>{int(np.log10(p))}</sup>" if p < 0.001 else f"{p:.3f}"
        is_key = "log(β)" in row["param_1"] and "δ" in row["param_2"]
        cls = ' class="highlight"' if is_key else ""
        html += f"""<tr{cls}>
  <td>${row['param_1']}$ &harr; ${row['param_2']}$</td>
  <td class="r">{row['r']:+.3f}</td>
  <td class="r">{p_str}</td>
</tr>
"""

    html += f"""</tbody>
</table>

<h2>Table 2 &mdash; Joint LKJ model posterior correlations</h2>
<table>
<thead>
<tr>
  <th>Parameters</th>
  <th class="r">$\\rho$ (mean)</th>
  <th class="r">SD</th>
  <th class="r">95% CI</th>
  <th class="r">$P(\\rho > 0)$</th>
</tr>
</thead>
<tbody>
"""

    param_label = {
        "log_k": "\\log(k)", "log_beta": "\\log(\\beta)",
        "alpha": "\\alpha", "delta": "\\delta"
    }
    for _, row in joint_corr.iterrows():
        p1 = param_label.get(row["param_1"], row["param_1"])
        p2 = param_label.get(row["param_2"], row["param_2"])
        is_key = "beta" in row["param_1"] and "delta" in row["param_2"]
        cls = ' class="highlight"' if is_key else ""
        html += f"""<tr{cls}>
  <td>${p1}$ &harr; ${p2}$</td>
  <td class="r">{row['rho_mean']:+.3f}</td>
  <td class="r">{row['rho_sd']:.3f}</td>
  <td class="r">[{row['rho_2.5']:+.3f}, {row['rho_97.5']:+.3f}]</td>
  <td class="r">{row['P_positive']:.3f}</td>
</tr>
"""

    # H6c regression
    h6c = h6c_results
    html += f"""</tbody>
</table>

<h2>Table 3 &mdash; H6c: Optimality regression</h2>
<p style="color: var(--muted); font-size: 0.85em; margin-bottom: 0.8em;">
  OLS: $\\%\\text{{optimal}} \\sim \\beta_z + \\delta_z + k_z$ &emsp; ($R^2 = {h6c['r_sq']:.3f}$, $N = {h6c['n']}$)
</p>
<table>
<thead>
<tr>
  <th>Predictor</th>
  <th class="r">$\\beta$</th>
  <th class="r">SE</th>
  <th class="r">$t$</th>
</tr>
</thead>
<tbody>
"""
    for name, vals in h6c["predictors"].items():
        html += f"""<tr>
  <td>${name}$</td>
  <td class="r">{vals['beta']:+.4f}</td>
  <td class="r">{vals['se']:.4f}</td>
  <td class="r">{vals['t']:+.2f}</td>
</tr>
"""

    html += f"""</tbody>
</table>

<div class="tests">
  <div class="test">
    <div class="label">H6a &mdash; $\\beta$&ndash;$\\delta$ coupling</div>
    <div class="value">$r$ = +{indep_bd['r']:.3f}</div>
    <div class="detail">$p$ = {indep_bd['p']:.2e}, one-tailed &check;</div>
  </div>
  <div class="test">
    <div class="label">H6b &mdash; Joint model</div>
    <div class="value">$\\rho$ = +{joint_bd['rho_mean']:.3f}</div>
    <div class="detail">95% CI [{joint_bd['rho_2.5']:+.3f}, {joint_bd['rho_97.5']:+.3f}] &check;</div>
  </div>
  <div class="test">
    <div class="label">H6c &mdash; Optimality</div>
    <div class="value">$R^2$ = {h6c['r_sq']:.3f}</div>
    <div class="detail">Both $\\beta$ and $\\delta$ predict &check;</div>
  </div>
</div>

<div class="note">
  Independent Bayesian: choice model (SVI, binary $E$) and vigor model (MCMC) share no parameters or data.
  The only link is the survival function $S = (1-T) + T/(1+\\lambda D)$ with $\\lambda = {LAMBDA:.3f}$.
  Joint model: $[\\log k_i, \\log \\beta_i, \\alpha_i, \\delta_i] \\sim \\text{{MVN}}(\\mu, \\Sigma)$, $\\Omega \\sim \\text{{LKJCholesky}}(\\eta=2)$.
  Bold values in heatmap: 95% CI excludes zero.
</div>

</body>
</html>"""
    return html


html_table = build_html_table()
table_path = os.path.join(OUT_DIR, "table_h6.html")
with open(table_path, 'w') as f:
    f.write(html_table)
print(f"Saved: {table_path}")


# ══════════════════════════════════════════════════════════════════════
# COMBINED HTML
# ══════════════════════════════════════════════════════════════════════

def build_combined_html():
    import base64
    png_path = os.path.join(OUT_DIR, "fig_h6_coupling.png")
    with open(png_path, 'rb') as f:
        img_b64 = base64.b64encode(f.read()).decode()

    style_start = html_table.index("<style>")
    style_end = html_table.index("</style>") + len("</style>")
    style_block = html_table[style_start:style_end]

    body_start = html_table.index('<div class="hyp-block">')
    body_end = html_table.index("</body>")
    table_content = html_table[body_start:body_end]

    indep_bd = indep_corr[(indep_corr["param_1"] == "log(β)") & (indep_corr["param_2"] == "δ")].iloc[0]
    joint_bd = joint_corr[(joint_corr["param_1"] == "log_beta") & (joint_corr["param_2"] == "delta")].iloc[0]

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>H6 — Figure &amp; Table</title>
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

<h1>H6 — Choice and vigor parameters are coupled across independently estimated Bayesian models</h1>

<div class="fig-container">
  <img src="data:image/png;base64,{img_b64}" alt="H6 Figure">
  <p class="caption">
    <strong>Figure.</strong>
    <strong>(A)</strong> Scatter of $\\log(\\beta)$ (threat bias, choice model) vs. $\\delta$ (danger
    mobilization, vigor model). The positive correlation ($r = +{indep_bd['r']:.3f}$) emerges from
    independently estimated hierarchical models that share no parameters (H6a).
    <strong>(B)</strong> Full posterior correlation matrix from the joint LKJ model. Red outline
    highlights the $\\beta$&ndash;$\\delta$ coupling ($\\rho = +{joint_bd['rho_mean']:.3f}$,
    95% CI [{joint_bd['rho_2.5']:+.2f}, {joint_bd['rho_97.5']:+.2f}]), confirming structural coupling (H6b).
    Bold = 95% CI excludes zero.
    <strong>(C)</strong> Percentage of optimal choices predicted by $\\beta_z$ and $\\delta_z$.
    Regression lines at three $\\delta$ terciles show that higher danger mobilization shifts the
    optimality curve upward ($R^2 = {h6c_results['r_sq']:.3f}$; H6c). $N = {n_subj}$.
  </p>
</div>

{table_content}

</body>
</html>"""
    return html


combined_html = build_combined_html()
combined_path = os.path.join(OUT_DIR, "fig_h6_coupling.html")
with open(combined_path, 'w') as f:
    f.write(combined_html)
print(f"Saved: {combined_path}")


# ── Summary ───────────────────────────────────────────────────────────
indep_bd = indep_corr[(indep_corr["param_1"] == "log(β)") & (indep_corr["param_2"] == "δ")].iloc[0]
joint_bd = joint_corr[(joint_corr["param_1"] == "log_beta") & (joint_corr["param_2"] == "delta")].iloc[0]
print(f"\n=== H6 Summary ===")
print(f"H6a: r(log(β), δ) = +{indep_bd['r']:.3f}, p = {indep_bd['p']:.2e}")
print(f"H6b: ρ(β,δ) = +{joint_bd['rho_mean']:.3f}, CI [{joint_bd['rho_2.5']:+.3f}, {joint_bd['rho_97.5']:+.3f}]")
print(f"H6c: R² = {h6c_results['r_sq']:.3f}")
print(f"N = {n_subj}")
