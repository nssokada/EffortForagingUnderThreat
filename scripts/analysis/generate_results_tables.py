#!/usr/bin/env python3
"""
Generate HTML tables for each preregistered hypothesis from results JSONs.
Outputs to results/tables/html/
"""

import json
import numpy as np
from pathlib import Path

OUT_DIR = Path("/workspace/results/tables/html")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ── Shared CSS ────────────────────────────────────────────────────────────────
CSS = """
<style>
  body { font-family: 'Helvetica Neue', Arial, sans-serif; color: #191919; margin: 24px; }
  h2 { color: #1A93FF; margin-bottom: 4px; }
  h3 { color: #6B7280; margin-top: 18px; margin-bottom: 6px; }
  .verdict { font-size: 1.1em; font-weight: bold; margin: 12px 0; padding: 8px 14px;
             border-radius: 6px; display: inline-block; }
  .pass { background: #D1FAE5; color: #065F46; }
  .fail { background: #FEE2E2; color: #991B1B; }
  table { border-collapse: collapse; margin: 10px 0 18px 0; min-width: 500px; }
  th { background: #F3F4F6; color: #374151; padding: 8px 14px; text-align: left;
       border-bottom: 2px solid #D1D5DB; font-size: 0.9em; }
  td { padding: 7px 14px; border-bottom: 1px solid #E5E7EB; font-size: 0.9em; }
  tr:hover { background: #F9FAFB; }
  .sig { color: #065F46; font-weight: bold; }
  .ns  { color: #991B1B; }
  .note { color: #6B7280; font-size: 0.85em; margin-top: 4px; }
  .stat-box { background: #F9FAFB; border: 1px solid #E5E7EB; border-radius: 6px;
              padding: 10px 16px; margin: 8px 0; display: inline-block; }
</style>
"""

def fmt_p(p):
    if p < 0.001: return "< 0.001"
    if p < 0.01:  return f"{p:.3f}"
    return f"{p:.3f}"

def fmt_p_sci(p):
    if p < 0.001: return f"{p:.2e}"
    return f"{p:.3f}"

def sig_class(p, threshold=0.05):
    return "sig" if p < threshold else "ns"

def verdict_html(supported):
    cls = "pass" if supported else "fail"
    text = "SUPPORTED" if supported else "NOT SUPPORTED"
    return f'<span class="verdict {cls}">{text}</span>'

# ══════════════════════════════════════════════════════════════════════════════
# H1 TABLES
# ══════════════════════════════════════════════════════════════════════════════
print("Generating H1 tables...")
with open("/workspace/results/stats/h1_lmm_results.json") as f:
    h1 = json.load(f)

# ── H1a: Choice GLMM ─────────────────────────────────────────────────────────
h1a = h1["H1a"]
h1a_terms = ["Intercept", "threat_z", "dist_z", "threat_x_dist"]
h1a_labels = {"Intercept": "Intercept", "threat_z": "Threat (z)",
              "dist_z": "Distance (z)", "threat_x_dist": "Threat &times; Distance"}

rows_h1a = ""
for term in h1a_terms:
    d = h1a[term]
    rows_h1a += f"""<tr>
  <td>{h1a_labels[term]}</td>
  <td>{d['beta']:.3f}</td><td>{d['SE']:.3f}</td>
  <td>{d['z']:.2f}</td><td class="{sig_class(d['p'], 0.01)}">{fmt_p(d['p'])}</td>
</tr>\n"""

# Monotonicity sub-table
mono_rows = ""
for m in h1a["monotonicity"]:
    mono_rows += f"""<tr>
  <td>{m['comparison']}</td><td>{m['t']:.2f}</td>
  <td class="{sig_class(m['p_one_tailed'], 0.01)}">{fmt_p_sci(m['p_one_tailed'])}</td>
  <td>{m['d']:.2f}</td>
</tr>\n"""

h1a_supported = (h1a["threat_z"]["p"] < 0.01 and h1a["dist_z"]["p"] < 0.01
                 and h1a["threat_x_dist"]["p"] < 0.01 and h1a["all_monotonicity_pass"])

h1a_html = f"""<!DOCTYPE html><html><head><meta charset="utf-8">{CSS}</head><body>
<h2>H1a &mdash; High-effort choice decreases with threat and distance</h2>
<p>Logistic GLMM (VB/Laplace): <code>choice ~ threat_z + dist_z + threat_z:dist_z + (1 | subject)</code></p>
<p>N = {h1["dataset"]["N_subjects"]} subjects, {h1["dataset"]["N_choice_trials"]:,} trials</p>

<h3>Fixed Effects</h3>
<table>
<tr><th>Predictor</th><th>&beta;</th><th>SE</th><th>z</th><th>p</th></tr>
{rows_h1a}</table>

<h3>Monotonicity Tests (paired t, one-tailed)</h3>
<table>
<tr><th>Comparison</th><th>t</th><th>p (one-tailed)</th><th>Cohen's d</th></tr>
{mono_rows}</table>

<p>All monotonicity tests p &lt; 0.01: <strong>{'Yes' if h1a['all_monotonicity_pass'] else 'No'}</strong></p>
{verdict_html(h1a_supported)}
</body></html>"""

(OUT_DIR / "h1a_choice_glmm.html").write_text(h1a_html)

# ── H1b: Excess effort LMM ───────────────────────────────────────────────────
h1b = h1["H1b"]
h1b_terms = ["Intercept", "threat_z", "dist_z", "effort_chosen_z", "threat_x_dist"]
h1b_labels = {"Intercept": "Intercept", "threat_z": "Threat (z)",
              "dist_z": "Distance (z)", "effort_chosen_z": "Effort chosen (z)",
              "threat_x_dist": "Threat &times; Distance"}

rows_h1b = ""
for term in h1b_terms:
    d = h1b[term]
    rows_h1b += f"""<tr>
  <td>{h1b_labels[term]}</td>
  <td>{d['beta']:.4f}</td><td>{d['SE']:.4f}</td>
  <td>{d['z']:.2f}</td><td class="{sig_class(d['p'])}">{fmt_p(d['p'])}</td>
  <td>[{d['CI_lo']:.4f}, {d['CI_hi']:.4f}]</td>
</tr>\n"""

h1b_html = f"""<!DOCTYPE html><html><head><meta charset="utf-8">{CSS}</head><body>
<h2>H1b &mdash; Excess effort increases with threat</h2>
<p>Linear LMM (REML): <code>excess_effort ~ threat_z * dist_z + effort_chosen_z + (1 | subject)</code></p>
<p>N = {h1b['N_subjects']} subjects, {h1b['N_trials']:,} trials</p>

<h3>Fixed Effects</h3>
<table>
<tr><th>Predictor</th><th>&beta;</th><th>SE</th><th>z</th><th>p</th><th>95% CI</th></tr>
{rows_h1b}</table>

<p class="note">Criterion: &beta;(threat) &gt; 0, p &lt; 0.05; &beta;(threat &times; dist) &lt; 0, p &lt; 0.05</p>
{verdict_html(h1b['supported'])}
</body></html>"""

(OUT_DIR / "h1b_excess_effort_lmm.html").write_text(h1b_html)

# ── H1c: Affect LMMs ─────────────────────────────────────────────────────────
h1c = h1["H1c"]

for measure in ["anxiety", "confidence"]:
    res = h1c[measure]
    terms = ["Intercept", "threat_z", "dist_z"]
    labels = {"Intercept": "Intercept", "threat_z": "Threat (z)", "dist_z": "Distance (z)"}

    rows = ""
    for term in terms:
        d = res[term]
        rows += f"""<tr>
  <td>{labels[term]}</td>
  <td>{d['beta']:.3f}</td><td>{d['SE']:.3f}</td>
  <td>{d['z']:.2f}</td><td class="{sig_class(d['p'], 0.001)}">{fmt_p(d['p'])}</td>
  <td>[{d['CI_lo']:.3f}, {d['CI_hi']:.3f}]</td>
</tr>\n"""

    re = res["random_effects"]
    expected = "positive" if measure == "anxiety" else "negative"
    criterion = "&beta;(threat) &gt; 0" if measure == "anxiety" else "&beta;(threat) &lt; 0"

    html = f"""<!DOCTYPE html><html><head><meta charset="utf-8">{CSS}</head><body>
<h2>H1c &mdash; {measure.title()} shifts with threat</h2>
<p>Linear LMM (REML): <code>{measure} ~ threat_z + dist_z + (1 + threat_z | subject)</code></p>
<p>N = {res['N_subjects']} subjects, {res['N_trials']:,} probe ratings</p>

<h3>Fixed Effects</h3>
<table>
<tr><th>Predictor</th><th>&beta;</th><th>SE</th><th>z</th><th>p</th><th>95% CI</th></tr>
{rows}</table>

<h3>Random Effects</h3>
<div class="stat-box">
  Intercept variance: {re['intercept_var']:.3f} &nbsp;|&nbsp;
  Threat slope variance: {re['threat_z_var']:.3f} &nbsp;|&nbsp;
  Covariance: {re['covariance']:.3f}
</div>

<p class="note">Criterion: {criterion}, p &lt; 0.001. Expected direction: {expected}.</p>
{verdict_html(res['supported'])}
</body></html>"""

    (OUT_DIR / f"h1c_{measure}_lmm.html").write_text(html)

# ── H1 Summary ───────────────────────────────────────────────────────────────
h1_overall = h1["H1_overall"]
summary_rows = ""
checks = [
    ("H1a: Threat &rarr; choice", f"&beta; = {h1a['threat_z']['beta']:.3f}", fmt_p(h1a['threat_z']['p']), "p &lt; 0.01", h1a['threat_z']['p'] < 0.01),
    ("H1a: Distance &rarr; choice", f"&beta; = {h1a['dist_z']['beta']:.3f}", fmt_p(h1a['dist_z']['p']), "p &lt; 0.01", h1a['dist_z']['p'] < 0.01),
    ("H1a: Threat &times; Distance", f"&beta; = {h1a['threat_x_dist']['beta']:.3f}", fmt_p(h1a['threat_x_dist']['p']), "p &lt; 0.01", h1a['threat_x_dist']['p'] < 0.01),
    ("H1a: Monotonicity (6 tests)", "d = 0.50&ndash;1.07", "all &lt; 10<sup>-15</sup>", "all p &lt; 0.01", h1a['all_monotonicity_pass']),
    ("H1b: Threat &rarr; excess effort", f"&beta; = {h1b['threat_z']['beta']:.4f}", fmt_p(h1b['threat_z']['p']), "&beta; &gt; 0, p &lt; 0.05", h1b['threat_z']['beta'] > 0 and h1b['threat_z']['p'] < 0.05),
    ("H1b: Threat &times; Distance", f"&beta; = {h1b['threat_x_dist']['beta']:.4f}", fmt_p(h1b['threat_x_dist']['p']), "&beta; &lt; 0, p &lt; 0.05", h1b['threat_x_dist']['beta'] < 0 and h1b['threat_x_dist']['p'] < 0.05),
    ("H1c: Threat &rarr; anxiety", f"&beta; = {h1c['anxiety']['threat_z']['beta']:.3f}", fmt_p(h1c['anxiety']['threat_z']['p']), "&beta; &gt; 0, p &lt; 0.001", h1c['anxiety']['supported']),
    ("H1c: Threat &rarr; confidence", f"&beta; = {h1c['confidence']['threat_z']['beta']:.3f}", fmt_p(h1c['confidence']['threat_z']['p']), "&beta; &lt; 0, p &lt; 0.001", h1c['confidence']['supported']),
]

for label, stat, pval, criterion, passed in checks:
    cls = "sig" if passed else "ns"
    result = "PASS" if passed else "FAIL"
    summary_rows += f"""<tr>
  <td>{label}</td><td>{stat}</td><td>{pval}</td>
  <td>{criterion}</td><td class="{cls}"><strong>{result}</strong></td>
</tr>\n"""

h1_summary_html = f"""<!DOCTYPE html><html><head><meta charset="utf-8">{CSS}</head><body>
<h2>H1 Summary &mdash; Threat shifts choice, vigor, and affect</h2>
<p>Exploratory sample: N = {h1['dataset']['N_subjects']}</p>

<table>
<tr><th>Test</th><th>Statistic</th><th>p</th><th>Criterion</th><th>Result</th></tr>
{summary_rows}</table>

{verdict_html(h1_overall['all_supported'])}
</body></html>"""

(OUT_DIR / "h1_summary.html").write_text(h1_summary_html)

# ══════════════════════════════════════════════════════════════════════════════
# H2 TABLES
# ══════════════════════════════════════════════════════════════════════════════
print("Generating H2 tables...")
with open("/workspace/results/stats/h2_coupling_results.json") as f:
    h2 = json.load(f)

h2a = h2["H2a"]
h2b = h2["H2b"]

h2_html = f"""<!DOCTYPE html><html><head><meta charset="utf-8">{CSS}</head><body>
<h2>H2 &mdash; Choice-vigor coupling under threat</h2>
<p>N = {h2a['N']} subjects</p>

<h3>H2a: Choice-vigor coupling</h3>
<div class="stat-box">
  &Delta;choice: M = {h2a['delta_choice_mean']:.3f}, SD = {h2a['delta_choice_sd']:.3f}<br>
  &Delta;vigor: M = {h2a['delta_vigor_mean']:.4f}, SD = {h2a['delta_vigor_sd']:.4f}
</div>

<table>
<tr><th>Test</th><th>r</th><th>p (two-tailed)</th><th>p (one-tailed)</th><th>Criterion</th><th>Result</th></tr>
<tr>
  <td>Pearson r(&Delta;choice, &Delta;vigor)</td>
  <td><strong>{h2a['r']:.3f}</strong></td>
  <td>{fmt_p_sci(h2a['p_two_tailed'])}</td>
  <td class="{sig_class(h2a['p_one_tailed'], 0.01)}">{fmt_p_sci(h2a['p_one_tailed'])}</td>
  <td>r &lt; 0, p &lt; 0.01</td>
  <td class="{sig_class(h2a['p_one_tailed'], 0.01)}"><strong>{'PASS' if h2a['supported'] else 'FAIL'}</strong></td>
</tr>
</table>

<h3>H2b: Split-half robustness</h3>
<table>
<tr><th>Split</th><th>r</th><th>p (one-tailed)</th><th>N</th><th>Criterion</th><th>Result</th></tr>
<tr>
  <td>&Delta;choice(odd) vs &Delta;vigor(even)</td>
  <td>{h2b['split1_choice_odd_vigor_even']['r']:.3f}</td>
  <td class="{sig_class(h2b['split1_choice_odd_vigor_even']['p_one_tailed'])}">{fmt_p_sci(h2b['split1_choice_odd_vigor_even']['p_one_tailed'])}</td>
  <td>{h2b['split1_choice_odd_vigor_even']['N']}</td>
  <td>r &lt; 0, p &lt; 0.05</td>
  <td class="{'sig' if h2b['split1_choice_odd_vigor_even']['pass'] else 'ns'}"><strong>{'PASS' if h2b['split1_choice_odd_vigor_even']['pass'] else 'FAIL'}</strong></td>
</tr>
<tr>
  <td>&Delta;choice(even) vs &Delta;vigor(odd)</td>
  <td>{h2b['split2_choice_even_vigor_odd']['r']:.3f}</td>
  <td class="{sig_class(h2b['split2_choice_even_vigor_odd']['p_one_tailed'])}">{fmt_p_sci(h2b['split2_choice_even_vigor_odd']['p_one_tailed'])}</td>
  <td>{h2b['split2_choice_even_vigor_odd']['N']}</td>
  <td>r &lt; 0, p &lt; 0.05</td>
  <td class="{'sig' if h2b['split2_choice_even_vigor_odd']['pass'] else 'ns'}"><strong>{'PASS' if h2b['split2_choice_even_vigor_odd']['pass'] else 'FAIL'}</strong></td>
</tr>
</table>

{verdict_html(h2['H2_overall']['overall'])}
</body></html>"""

(OUT_DIR / "h2_coupling.html").write_text(h2_html)

# ══════════════════════════════════════════════════════════════════════════════
print(f"\nAll tables saved to {OUT_DIR}/")
for f in sorted(OUT_DIR.glob("*.html")):
    print(f"  {f.name}")
