#!/usr/bin/env python3
"""
Generate combined HTML (figure + hypothesis statement + tables) for H1, H2, H3.

Reads JSON results, embeds existing PNG figures as base64, outputs styled HTML
matching the H4-H7 pattern (KaTeX, :root variables, .hyp-block, .tests grid).

Outputs:
  results/prereg_html/h1_threat_shifts.html
  results/prereg_html/h2_coupling.html
  results/prereg_html/h3_optimality.html

Run: python scripts/plotting/plot_h1_h2_h3_html.py
"""

import os, sys, json, base64, math

# ── Paths ──────────────────────────────────────────────────────────────
ROOT = os.path.join(os.path.dirname(__file__), '..', '..')
STATS_DIR = os.path.join(ROOT, "results", "stats")
FIG_DIR = os.path.join(ROOT, "results", "figs", "paper")
OUT_DIR = os.path.join(ROOT, "results", "prereg_html")
os.makedirs(OUT_DIR, exist_ok=True)


# ── Utilities ──────────────────────────────────────────────────────────
def load_json(name):
    path = os.path.join(STATS_DIR, name)
    with open(path) as f:
        return json.load(f)


def embed_png(filename):
    """Return base64-encoded PNG string, or None if file not found."""
    path = os.path.join(FIG_DIR, filename)
    if not os.path.exists(path):
        # Try fallback names
        fallbacks = {
            "fig_h2_coupling_contour.png": "fig_h2_coupling_scatter.png",
            "fig_pareto_frontier.png": "fig_allocation_surface.png",
        }
        fallback = fallbacks.get(filename)
        if fallback:
            path = os.path.join(FIG_DIR, fallback)
        if not os.path.exists(path):
            print(f"  WARNING: PNG not found: {filename} (and no fallback)")
            return None
        print(f"  Using fallback: {fallback} for {filename}")
    with open(path, 'rb') as f:
        return base64.b64encode(f.read()).decode()


def fmt_p(p):
    """Format p-value per spec: approx 0 when p < 1e-16 or p == 0,
    < 10^N with <sup> for other small values, else = .NNN."""
    if p is None:
        return "N/A"
    if p == 0 or p < 1e-16:
        return "&asymp; 0"
    if p < 0.001:
        exp = int(math.floor(math.log10(p)))
        return f"< 10<sup>{exp}</sup>"
    return f"= {p:.3f}"


def fmt_p_katex(p):
    """Format p-value for KaTeX inline math."""
    if p is None:
        return "\\text{N/A}"
    if p == 0 or p < 1e-16:
        return "\\approx 0"
    if p < 0.001:
        exp = int(math.floor(math.log10(p)))
        return f"< 10^{{{exp}}}"
    return f"= {p:.3f}"


# ── Shared CSS (matches H4-H7 pattern) ────────────────────────────────
SHARED_CSS = """
  :root {
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
    --fail: #DC2626;
    --fail-bg: #FEF2F2;
    --fail-border: #FECACA;
  }
  * { box-sizing: border-box; }
  body {
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
    max-width: 920px;
    margin: 48px auto;
    color: var(--ink);
    background: #fff;
    padding: 0 24px;
    line-height: 1.55;
  }
  h1 {
    font-size: 1.35em;
    font-weight: 700;
    margin-bottom: 0.2em;
    letter-spacing: -0.01em;
  }
  h2 {
    font-size: 1.0em;
    font-weight: 600;
    color: var(--muted);
    margin: 2.5em 0 0.6em;
    text-transform: uppercase;
    letter-spacing: 0.04em;
  }
  .subtitle {
    color: var(--muted);
    font-size: 0.92em;
    margin-bottom: 2em;
  }

  /* ── Tables ── */
  table {
    border-collapse: collapse;
    width: 100%;
    font-size: 0.88em;
    margin-bottom: 0.8em;
  }
  thead th {
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
  }
  thead th.r { text-align: right; }
  tbody td {
    border-bottom: 1px solid var(--border);
    padding: 10px 14px;
    vertical-align: middle;
  }
  tbody td.r {
    text-align: right;
    font-variant-numeric: tabular-nums;
    font-family: 'SF Mono', 'Menlo', 'Consolas', monospace;
    font-size: 0.92em;
  }
  tbody tr:last-child td { border-bottom: 2px solid var(--border-strong); }
  tbody tr:hover td { background: #F9FAFB; }

  /* Winner row */
  tr.winner td { background: var(--bg-winner); }
  tr.winner:hover td { background: var(--bg-winner-hover); }
  .tag {
    display: inline-block;
    padding: 2px 8px;
    border-radius: 10px;
    font-size: 0.78em;
    font-weight: 600;
    letter-spacing: 0.02em;
  }
  .tag-best { background: var(--accent); color: #fff; }

  .q { color: var(--muted); font-size: 0.92em; }

  /* ── Test result cards ── */
  .tests {
    display: grid;
    grid-template-columns: 1fr 1fr 1fr;
    gap: 12px;
    margin: 1.2em 0 0;
  }
  .tests-2col {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 12px;
    margin: 1.2em 0 0;
  }
  .test {
    background: var(--pass-bg);
    border: 1px solid var(--pass-border);
    border-radius: 8px;
    padding: 14px 16px;
  }
  .test.fail {
    background: var(--fail-bg);
    border: 1px solid var(--fail-border);
  }
  .test .label {
    font-size: 0.78em;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.04em;
    color: var(--muted);
    margin-bottom: 4px;
  }
  .test .value {
    font-size: 1.15em;
    font-weight: 700;
    color: var(--pass);
  }
  .test.fail .value {
    color: var(--fail);
  }
  .test .detail {
    font-size: 0.82em;
    color: var(--muted);
    margin-top: 2px;
  }

  /* ── Hypothesis block ── */
  .hyp-block {
    background: #F9FAFB;
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 20px 24px;
    margin-bottom: 2em;
  }
  .hyp-statement {
    font-size: 0.92em;
    line-height: 1.65;
    margin-bottom: 12px;
  }
  .hyp-subs {
    display: flex;
    flex-direction: column;
    gap: 6px;
  }
  .hyp-sub {
    font-size: 0.86em;
    color: var(--ink);
    padding: 6px 12px;
    background: white;
    border-radius: 6px;
    border: 1px solid var(--border);
    line-height: 1.55;
  }

  /* ── Notes ── */
  .note {
    color: var(--muted);
    font-size: 0.82em;
    line-height: 1.6;
    margin-top: 1.5em;
    padding-top: 1em;
    border-top: 1px solid var(--border);
  }

  /* ── Figure container ── */
  .fig-container {
    text-align: center;
    margin: 2em 0 2.5em;
  }
  .fig-container img {
    max-width: 100%;
    border: 1px solid var(--border);
    border-radius: 6px;
  }
  .caption {
    color: var(--muted);
    font-size: 0.88em;
    margin-top: 1em;
    text-align: left;
    line-height: 1.65;
  }
  .caption strong { color: var(--ink); }

  .lmm-spec {
    color: var(--muted);
    font-size: 0.85em;
    margin-bottom: 0.8em;
  }
"""

KATEX_HEAD = """<link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/katex@0.16.11/dist/katex.min.css">
<script defer src="https://cdn.jsdelivr.net/npm/katex@0.16.11/dist/katex.min.js"></script>
<script defer src="https://cdn.jsdelivr.net/npm/katex@0.16.11/dist/contrib/auto-render.min.js"
  onload="renderMathInElement(document.body, {delimiters: [{left: '$$', right: '$$', display: true}, {left: '$', right: '$', display: false}]});"></script>"""


def html_shell(title, body):
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>{title}</title>
{KATEX_HEAD}
<style>
{SHARED_CSS}
</style>
</head>
<body>
{body}
</body>
</html>"""


# ======================================================================
# H1 — Threat shifts
# ======================================================================
def build_h1():
    print("Building H1...")
    d = load_json("h1_lmm_results.json")
    img = embed_png("fig_h1_threat_shifts_v2.png")
    N = d["dataset"]["N_subjects"]
    N_trials = d["dataset"]["N_choice_trials"]

    # H1a fixed effects
    h1a = d["H1a"]
    mono = h1a["monotonicity"]

    # H1b
    h1b = d["H1b"]

    # H1c
    h1c_anx = d["H1c"]["anxiety"]
    h1c_conf = d["H1c"]["confidence"]

    body = f"""
<h1>H1 — Threat reduces high-effort choice, amplifies avoidance, and shifts affect</h1>
<p class="subtitle">Exploratory sample, $N = {N}$ &middot; {N_trials:,} choice trials</p>

<div class="hyp-block">
  <div class="hyp-statement">
    <strong>Hypothesis:</strong> Threat will reduce high-effort choice, amplify distance-driven avoidance,
    and increase excess motor effort &mdash; while shifting anxiety upward and confidence downward.
  </div>
  <div class="hyp-subs">
    <div class="hyp-sub"><strong>H1a</strong> &mdash; Logistic LMM: <code>choice ~ threat_z + dist_z + threat_z:dist_z + (1|subj)</code>.
    All $\\beta < 0$, $p < .01$, plus monotonicity of $P(\\text{{high}})$ across adjacent threat levels within each distance.</div>
    <div class="hyp-sub"><strong>H1b</strong> &mdash; Linear LMM: <code>excess_effort ~ threat_z &times; dist_z + effort_chosen_z + (1|subj)</code>.
    $\\beta(\\text{{threat}}) > 0$; $\\beta(\\text{{threat}} \\times \\text{{dist}}) < 0$.</div>
    <div class="hyp-sub"><strong>H1c</strong> &mdash; LMMs: <code>anxiety ~ threat_z + dist_z + (1+threat_z|subj)</code>: $\\beta > 0$.
    Confidence: $\\beta < 0$. Both $p < .001$.</div>
  </div>
</div>
"""

    # Figure
    if img:
        body += f"""
<div class="fig-container">
  <img src="data:image/png;base64,{img}" alt="H1 Figure">
  <p class="caption">
    <strong>Figure.</strong> Threat shifts choice, effort, and affect.
    <strong>(A)</strong> P(choose high-effort) decreases with threat and distance.
    <strong>(B)</strong> Excess motor effort increases with threat.
    <strong>(C&ndash;D)</strong> Anxiety increases and confidence decreases with threat probability.
    $N = {N}$.
  </p>
</div>
"""

    # H1a table
    body += """
<h2>Table 1 &mdash; H1a: Logistic LMM &mdash; Choice</h2>
<p class="lmm-spec">Model: $\\text{choice} \\sim \\text{threat}_z + \\text{dist}_z + \\text{threat}_z \\!\\times\\! \\text{dist}_z + (1 \\mid \\text{subj})$</p>
<table>
<thead>
<tr>
  <th>Predictor</th>
  <th class="r">$\\beta$</th>
  <th class="r">SE</th>
  <th class="r">$z$</th>
  <th class="r">$p$</th>
</tr>
</thead>
<tbody>
"""

    for pred_key, pred_label in [("Intercept", "Intercept"),
                                  ("threat_z", "Threat (z)"),
                                  ("dist_z", "Distance (z)"),
                                  ("threat_x_dist", "Threat &times; Distance")]:
        row = h1a[pred_key]
        body += f"""<tr>
  <td>{pred_label}</td>
  <td class="r">{row['beta']:.4f}</td>
  <td class="r">{row['SE']:.4f}</td>
  <td class="r">{row['z']:.2f}</td>
  <td class="r">{fmt_p(row['p'])}</td>
</tr>
"""

    body += f"""</tbody>
</table>
<p class="lmm-spec">Random intercept SD = {h1a['random_intercept_sd']:.3f}</p>
"""

    # Monotonicity table
    body += """
<h2>H1a Monotonicity Tests</h2>
<table>
<thead>
<tr>
  <th>Comparison</th>
  <th class="r">$t$</th>
  <th class="r">$p$ (one-tailed)</th>
  <th class="r">Cohen's $d$</th>
</tr>
</thead>
<tbody>
"""
    for m in mono:
        body += f"""<tr>
  <td>{m['comparison']}</td>
  <td class="r">{m['t']:.2f}</td>
  <td class="r">{fmt_p(m['p_one_tailed'])}</td>
  <td class="r">{m['d']:.3f}</td>
</tr>
"""

    body += f"""</tbody>
</table>
"""

    # Test cards for H1a
    all_sig = all(h1a[k]['p'] < 0.01 for k in ['threat_z', 'dist_z', 'threat_x_dist'])
    all_neg = all(h1a[k]['beta'] < 0 for k in ['threat_z', 'dist_z', 'threat_x_dist'])
    all_mono = h1a['all_monotonicity_pass']

    body += f"""
<div class="tests">
  <div class="test">
    <div class="label">All &beta; &lt; 0</div>
    <div class="value">{"&check;" if all_neg else "&cross;"}</div>
    <div class="detail">Threat, distance, interaction</div>
  </div>
  <div class="test">
    <div class="label">All $p$ &lt; .01</div>
    <div class="value">{"&check;" if all_sig else "&cross;"}</div>
  </div>
  <div class="test">
    <div class="label">Monotonicity</div>
    <div class="value">{"6/6 pass" if all_mono else "Fail"} {"&check;" if all_mono else "&cross;"}</div>
    <div class="detail">P(high) decreases with threat at each D</div>
  </div>
</div>
"""

    # H1b table
    body += f"""
<h2>Table 2 &mdash; H1b: Linear LMM &mdash; Excess Effort</h2>
<p class="lmm-spec">Model: $\\text{{excess}} \\sim \\text{{threat}}_z \\times \\text{{dist}}_z + \\text{{effort\\_chosen}}_z + (1 \\mid \\text{{subj}})$
&middot; {h1b['N_trials']:,} trials, {h1b['N_subjects']} subjects</p>
<table>
<thead>
<tr>
  <th>Predictor</th>
  <th class="r">$\\beta$</th>
  <th class="r">SE</th>
  <th class="r">$z$</th>
  <th class="r">$p$</th>
  <th class="r">95% CI</th>
</tr>
</thead>
<tbody>
"""
    for pred_key, pred_label in [("Intercept", "Intercept"),
                                  ("threat_z", "Threat (z)"),
                                  ("dist_z", "Distance (z)"),
                                  ("effort_chosen_z", "Effort chosen (z)"),
                                  ("threat_x_dist", "Threat &times; Distance")]:
        row = h1b[pred_key]
        body += f"""<tr>
  <td>{pred_label}</td>
  <td class="r">{row['beta']:.4f}</td>
  <td class="r">{row['SE']:.4f}</td>
  <td class="r">{row['z']:.2f}</td>
  <td class="r">{fmt_p(row['p'])}</td>
  <td class="r">[{row['CI_lo']:.4f}, {row['CI_hi']:.4f}]</td>
</tr>
"""

    body += """</tbody>
</table>
"""

    # H1b test cards
    threat_pos = h1b['threat_z']['beta'] > 0 and h1b['threat_z']['p'] < 0.05
    interaction_neg = h1b['threat_x_dist']['beta'] < 0 and h1b['threat_x_dist']['p'] < 0.05

    body += f"""
<div class="tests-2col">
  <div class="test">
    <div class="label">H1b &mdash; $\\beta$(threat) &gt; 0</div>
    <div class="value">&beta; = {h1b['threat_z']['beta']:.4f} {"&check;" if threat_pos else "&cross;"}</div>
    <div class="detail">$p$ {fmt_p_katex(h1b['threat_z']['p'])}</div>
  </div>
  <div class="test">
    <div class="label">H1b &mdash; $\\beta$(threat &times; dist) &lt; 0</div>
    <div class="value">&beta; = {h1b['threat_x_dist']['beta']:.4f} {"&check;" if interaction_neg else "&cross;"}</div>
    <div class="detail">$p$ {fmt_p_katex(h1b['threat_x_dist']['p'])}</div>
  </div>
</div>
"""

    # H1c table
    body += """
<h2>Table 3 &mdash; H1c: Affect LMMs</h2>
"""
    for label, data, expected_sign in [("Anxiety", h1c_anx, "positive"), ("Confidence", h1c_conf, "negative")]:
        body += f"""
<p class="lmm-spec">{label}: $\\text{{{label.lower()}}} \\sim \\text{{threat}}_z + \\text{{dist}}_z + (1 + \\text{{threat}}_z \\mid \\text{{subj}})$
&middot; {data['N_trials']:,} trials, {data['N_subjects']} subjects</p>
<table>
<thead>
<tr>
  <th>Predictor</th>
  <th class="r">$\\beta$</th>
  <th class="r">SE</th>
  <th class="r">$z$</th>
  <th class="r">$p$</th>
  <th class="r">95% CI</th>
</tr>
</thead>
<tbody>
"""
        for pred_key, pred_label in [("Intercept", "Intercept"),
                                      ("threat_z", "Threat (z)"),
                                      ("dist_z", "Distance (z)")]:
            row = data[pred_key]
            body += f"""<tr>
  <td>{pred_label}</td>
  <td class="r">{row['beta']:.4f}</td>
  <td class="r">{row['SE']:.4f}</td>
  <td class="r">{row['z']:.2f}</td>
  <td class="r">{fmt_p(row['p'])}</td>
  <td class="r">[{row['CI_lo']:.4f}, {row['CI_hi']:.4f}]</td>
</tr>
"""
        body += """</tbody>
</table>
"""

    # H1c test cards
    anx_pass = h1c_anx['threat_z']['beta'] > 0 and h1c_anx['threat_z']['p'] < 0.001
    conf_pass = h1c_conf['threat_z']['beta'] < 0 and h1c_conf['threat_z']['p'] < 0.001

    body += f"""
<div class="tests-2col">
  <div class="test">
    <div class="label">H1c &mdash; Anxiety $\\beta$(threat) &gt; 0</div>
    <div class="value">&beta; = {h1c_anx['threat_z']['beta']:.4f} {"&check;" if anx_pass else "&cross;"}</div>
    <div class="detail">$z$ = {h1c_anx['threat_z']['z']:.2f}, $p$ {fmt_p_katex(h1c_anx['threat_z']['p'])}</div>
  </div>
  <div class="test">
    <div class="label">H1c &mdash; Confidence $\\beta$(threat) &lt; 0</div>
    <div class="value">&beta; = {h1c_conf['threat_z']['beta']:.4f} {"&check;" if conf_pass else "&cross;"}</div>
    <div class="detail">$z$ = {h1c_conf['threat_z']['z']:.2f}, $p$ {fmt_p_katex(h1c_conf['threat_z']['p'])}</div>
  </div>
</div>
"""

    # Overall note
    overall = d["H1_overall"]
    body += f"""
<div class="note">
  <strong>H1 overall:</strong> All sub-hypotheses {"supported" if overall['all_supported'] else "NOT fully supported"}.
  H1a (choice): {"&check;" if overall['H1a_supported'] else "&cross;"}
  &middot; H1b (effort): {"&check;" if overall['H1b_supported'] else "&cross;"}
  &middot; H1c anxiety: {"&check;" if overall['H1c_anxiety_supported'] else "&cross;"}
  &middot; H1c confidence: {"&check;" if overall['H1c_confidence_supported'] else "&cross;"}.
  LMMs estimated via Laplace approximation (H1a) and REML (H1b, H1c).
</div>
"""

    html = html_shell("H1 — Threat Shifts", body)
    out_path = os.path.join(OUT_DIR, "h1_threat_shifts.html")
    with open(out_path, 'w') as f:
        f.write(html)
    print(f"  Saved: {out_path}")


# ======================================================================
# H2 — Coupling
# ======================================================================
def build_h2():
    print("Building H2...")
    d = load_json("h2_coupling_results.json")
    img = embed_png("fig_h2_coupling_contour.png")
    N = d["dataset"]["N_subjects"]

    h2a = d["H2a"]
    h2b = d["H2b"]

    body = f"""
<h1>H2 — Choice shift and vigor shift are inversely correlated</h1>
<p class="subtitle">Exploratory sample, $N = {N}$</p>

<div class="hyp-block">
  <div class="hyp-statement">
    <strong>Hypothesis:</strong> Choice shift and vigor shift under threat will be inversely correlated across
    individuals &mdash; participants who shift choices most toward safety will show the largest increase in excess effort.
  </div>
  <div class="hyp-subs">
    <div class="hyp-sub"><strong>H2a</strong> &mdash; Pearson $r(\\Delta\\text{{choice}}, \\Delta\\text{{vigor}}) < 0$, $p < .01$ one-tailed.</div>
    <div class="hyp-sub"><strong>H2b</strong> &mdash; Split-half robustness: $r(\\Delta\\text{{choice}}_{{\\text{{odd}}}}, \\Delta\\text{{vigor}}_{{\\text{{even}}}}) < 0$, $p < .05$. Same for reversed split.</div>
  </div>
</div>
"""

    if img:
        body += f"""
<div class="fig-container">
  <img src="data:image/png;base64,{img}" alt="H2 Figure">
  <p class="caption">
    <strong>Figure.</strong> Across-subject coupling between threat-driven choice shift ($\\Delta$choice)
    and vigor shift ($\\Delta$vigor). Participants who reduce high-effort choices more under threat also
    increase their pressing rate more. $N = {N}$.
  </p>
</div>
"""

    # H2a results
    body += f"""
<h2>H2a &mdash; Full-sample correlation</h2>
<table>
<thead>
<tr>
  <th>Statistic</th>
  <th class="r">Value</th>
</tr>
</thead>
<tbody>
<tr><td>Pearson $r$</td><td class="r">{h2a['r']:.4f}</td></tr>
<tr><td>$p$ (two-tailed)</td><td class="r">{fmt_p(h2a['p_two_tailed'])}</td></tr>
<tr><td>$p$ (one-tailed)</td><td class="r">{fmt_p(h2a['p_one_tailed'])}</td></tr>
<tr><td>$N$</td><td class="r">{h2a['N']}</td></tr>
<tr><td>$\\Delta$choice mean (SD)</td><td class="r">{h2a['delta_choice_mean']:.4f} ({h2a['delta_choice_sd']:.4f})</td></tr>
<tr><td>$\\Delta$vigor mean (SD)</td><td class="r">{h2a['delta_vigor_mean']:.4f} ({h2a['delta_vigor_sd']:.4f})</td></tr>
</tbody>
</table>
"""

    # H2b split-half
    s1 = h2b["split1_choice_odd_vigor_even"]
    s2 = h2b["split2_choice_even_vigor_odd"]
    body += f"""
<h2>H2b &mdash; Split-half robustness</h2>
<table>
<thead>
<tr>
  <th>Split</th>
  <th class="r">$r$</th>
  <th class="r">$p$ (one-tailed)</th>
  <th class="r">Pass</th>
</tr>
</thead>
<tbody>
<tr>
  <td>$\\Delta$choice (odd) vs $\\Delta$vigor (even)</td>
  <td class="r">{s1['r']:.4f}</td>
  <td class="r">{fmt_p(s1['p_one_tailed'])}</td>
  <td class="r">{"&check;" if s1['pass'] else "&cross;"}</td>
</tr>
<tr>
  <td>$\\Delta$choice (even) vs $\\Delta$vigor (odd)</td>
  <td class="r">{s2['r']:.4f}</td>
  <td class="r">{fmt_p(s2['p_one_tailed'])}</td>
  <td class="r">{"&check;" if s2['pass'] else "&cross;"}</td>
</tr>
</tbody>
</table>
"""

    # Test cards
    h2a_pass = h2a['r'] < 0 and h2a['p_one_tailed'] < 0.01
    h2b_pass = h2b['supported']

    body += f"""
<div class="tests-2col">
  <div class="test">
    <div class="label">H2a &mdash; $r < 0$, $p < .01$</div>
    <div class="value">$r = {h2a['r']:.3f}$ {"&check;" if h2a_pass else "&cross;"}</div>
    <div class="detail">$p$ {fmt_p_katex(h2a['p_one_tailed'])} (one-tailed)</div>
  </div>
  <div class="test">
    <div class="label">H2b &mdash; Split-half robust</div>
    <div class="value">Both splits $r < 0$ {"&check;" if h2b_pass else "&cross;"}</div>
    <div class="detail">$r_1 = {s1['r']:.3f}$, $r_2 = {s2['r']:.3f}$</div>
  </div>
</div>
"""

    overall = d["H2_overall"]
    body += f"""
<div class="note">
  <strong>H2 overall:</strong> {"Supported" if overall['overall'] else "NOT supported"}.
  $\\Delta$choice = $P(\\text{{high}} \\mid T{{=}}0.9) - P(\\text{{high}} \\mid T{{=}}0.1)$.
  $\\Delta$vigor = mean excess effort at $T{{=}}0.9$ minus $T{{=}}0.1$.
  Split-half uses odd/even trial indices.
</div>
"""

    html = html_shell("H2 — Coupling", body)
    out_path = os.path.join(OUT_DIR, "h2_coupling.html")
    with open(out_path, 'w') as f:
        f.write(html)
    print(f"  Saved: {out_path}")


# ======================================================================
# H3 — Optimality
# ======================================================================
def build_h3():
    print("Building H3...")
    d = load_json("h3_optimality_results.json")
    img = embed_png("fig_pareto_frontier.png")
    N = d["dataset"]["N_subjects"]

    h3a = d["H3a"]
    h3b = d["H3b"]
    ev_table = d.get("ev_by_condition", [])

    body = f"""
<h1>H3 — Reallocation approximates the optimal policy</h1>
<p class="subtitle">Exploratory sample, $N = {N}$</p>

<div class="hyp-block">
  <div class="hyp-statement">
    <strong>Hypothesis:</strong> The reallocation strategy will approximate the expected-value-maximizing policy.
    Participants who reallocate more (shifting both choice and vigor in response to threat) will earn more, and the
    dominant deviation from optimality will be excessive caution rather than excessive risk-taking.
  </div>
  <div class="hyp-subs">
    <div class="hyp-sub"><strong>H3a</strong> &mdash; Pearson $r(\\text{{reallocation index}}, \\text{{total earnings}}) > 0$, $p < .01$ one-tailed.</div>
    <div class="hyp-sub"><strong>H3b</strong> &mdash; Among suboptimal trials, $> 50\\%$ of errors are &ldquo;too cautious.&rdquo; One-sample $t$-test vs 50%.</div>
  </div>
</div>
"""

    if img:
        body += f"""
<div class="fig-container">
  <img src="data:image/png;base64,{img}" alt="H3 Figure">
  <p class="caption">
    <strong>Figure.</strong> Expected value surface as a function of threat bias ($\\log \\beta$, x-axis) and
    vigor mobilization ($\\delta$, y-axis). Contours show predicted foraging earnings; black curve
    marks the break-even boundary. Participants who combine high $\\beta$ with high $\\delta$
    (upper right) achieve the best outcomes. $N = {N}$.
  </p>
</div>
"""

    # H3a
    body += f"""
<h2>H3a &mdash; Reallocation predicts earnings</h2>
<table>
<thead>
<tr>
  <th>Statistic</th>
  <th class="r">Value</th>
</tr>
</thead>
<tbody>
<tr><td>Pearson $r$</td><td class="r">{h3a['r']:.4f}</td></tr>
<tr><td>$p$ (one-tailed)</td><td class="r">{fmt_p(h3a['p_one_tailed'])}</td></tr>
<tr><td>$N$</td><td class="r">{h3a['N']}</td></tr>
</tbody>
</table>
"""

    # H3b
    body += f"""
<h2>H3b &mdash; Error asymmetry: too cautious vs too risky</h2>
<table>
<thead>
<tr>
  <th>Statistic</th>
  <th class="r">Value</th>
</tr>
</thead>
<tbody>
<tr><td>Optimal trials</td><td class="r">{h3b['n_optimal']:,}</td></tr>
<tr><td>Suboptimal trials</td><td class="r">{h3b['n_suboptimal']:,}</td></tr>
<tr><td>&ldquo;Too cautious&rdquo; errors</td><td class="r">{h3b['n_too_cautious']:,}</td></tr>
<tr><td>&ldquo;Too risky&rdquo; errors</td><td class="r">{h3b['n_too_risky']:,}</td></tr>
<tr><td>% cautious (of errors)</td><td class="r">{h3b['pct_cautious_of_errors']*100:.1f}%</td></tr>
<tr><td>Per-subject mean (SD)</td><td class="r">{h3b['per_subject_mean_pct_cautious']*100:.1f}% ({h3b['per_subject_sd_pct_cautious']*100:.1f}%)</td></tr>
<tr><td>$t$ vs 50%</td><td class="r">{h3b['t']:.2f}</td></tr>
<tr><td>$p$ (one-tailed)</td><td class="r">{fmt_p(h3b['p_one_tailed'])}</td></tr>
</tbody>
</table>
"""

    # EV by condition table
    if ev_table:
        body += """
<h2>Expected value by condition</h2>
<table>
<thead>
<tr>
  <th>Threat</th>
  <th class="r">Distance</th>
  <th class="r">EV(high)</th>
  <th class="r">EV(low)</th>
  <th class="r">Optimal</th>
  <th class="r">Actual P(high)</th>
</tr>
</thead>
<tbody>
"""
        for ev in ev_table:
            opt_label = "High" if ev["optimal_is_high"] == 1.0 else "Low"
            body += f"""<tr>
  <td>{ev['threat']}</td>
  <td class="r">{ev['distance_H']}</td>
  <td class="r">{ev['EV_H']:.3f}</td>
  <td class="r">{ev['EV_L']:.3f}</td>
  <td class="r">{opt_label}</td>
  <td class="r">{ev['actual_chose_high']:.3f}</td>
</tr>
"""
        body += """</tbody>
</table>
"""

    # Test cards
    h3a_pass = h3a['r'] > 0 and h3a['p_one_tailed'] < 0.01
    h3b_pass = h3b['pct_cautious_of_errors'] > 0.5 and h3b['p_one_tailed'] < 0.05

    body += f"""
<div class="tests-2col">
  <div class="test">
    <div class="label">H3a &mdash; $r > 0$, $p < .01$</div>
    <div class="value">$r = {h3a['r']:.3f}$ {"&check;" if h3a_pass else "&cross;"}</div>
    <div class="detail">$p$ {fmt_p_katex(h3a['p_one_tailed'])} (one-tailed)</div>
  </div>
  <div class="test">
    <div class="label">H3b &mdash; &gt; 50% cautious</div>
    <div class="value">{h3b['pct_cautious_of_errors']*100:.1f}% {"&check;" if h3b_pass else "&cross;"}</div>
    <div class="detail">$t = {h3b['t']:.1f}$, $p$ {fmt_p_katex(h3b['p_one_tailed'])}</div>
  </div>
</div>
"""

    overall = d["H3_overall"]
    body += f"""
<div class="note">
  <strong>H3 overall:</strong> {"Supported" if overall['overall'] else "NOT supported"}.
  Reallocation index = $|\\Delta\\text{{choice}}| + |\\Delta\\text{{vigor}}|$ (z-scored).
  Optimal choice determined by empirical escape rates (method: {h3b.get('method', 'empirical')}).
  &ldquo;Too cautious&rdquo; = chose low-effort when high-effort had higher expected value.
</div>
"""

    html = html_shell("H3 — Optimality", body)
    out_path = os.path.join(OUT_DIR, "h3_optimality.html")
    with open(out_path, 'w') as f:
        f.write(html)
    print(f"  Saved: {out_path}")


# ======================================================================
# Main
# ======================================================================
if __name__ == "__main__":
    build_h1()
    build_h2()
    build_h3()
    print("\nAll H1-H3 HTML files generated.")
