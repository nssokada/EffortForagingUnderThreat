#!/usr/bin/env python3
"""
Extract sections 0, 1, 2 from drafts/discovery_results_with_figs.html,
fix p-value formatting, add back-link, save to results/prereg_html/discovery_behavioral.html.

Uses only stdlib — no pandas/numpy/matplotlib needed.
"""

import os, re

ROOT = os.path.join(os.path.dirname(__file__), '..', '..')
SRC = os.path.join(ROOT, "drafts", "discovery_results_with_figs.html")
OUT = os.path.join(ROOT, "results", "prereg_html", "discovery_behavioral.html")
os.makedirs(os.path.dirname(OUT), exist_ok=True)

with open(SRC, 'r') as f:
    content = f.read()

# ── Find section boundaries ──
s0_start = content.index('<h2 id="0-task-behavior')
s3_start = content.index('<h2 id="3-coordinated')
sections_html = content[s0_start:s3_start]

# ── Fix p-value formatting ──
# 1) Bare "10^-NNN" where exponent has 3+ digits → use "≈ 0"
sections_html = re.sub(r'10\^(-\d{3,})', '&asymp; 0', sections_html)
# 2) "X x 10^-NNN" (3+ digit exponent) → "≈ 0"
sections_html = re.sub(r'\d+\.?\d*\s*x\s*10\^(-\d{3,})', '&asymp; 0', sections_html)
# 3) "X x 10^-NN" (1-2 digit exponent) → proper <sup>
sections_html = re.sub(
    r'(\d+\.?\d*)\s*x\s*10\^(-\d{1,2})(?!\d)',
    r'\1 &times; 10<sup>\2</sup>',
    sections_html
)
# 4) Bare "10^-NN" (1-2 digit) → <sup>
sections_html = re.sub(r'10\^(-\d{1,2})(?!\d)', r'10<sup>\1</sup>', sections_html)
# 5) "< 10^-NN" already inside text
sections_html = re.sub(r'< 10\^(-\d{1,2})(?!\d)', r'< 10<sup>\1</sup>', sections_html)

# ── Build output ──
out_html = '''<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Discovery Behavioral Results — Effort Reallocation Under Threat</title>
<style>

body {
    font-family: 'Georgia', 'Times New Roman', serif;
    max-width: 1000px;
    margin: 0 auto;
    padding: 40px 20px;
    background: #fff;
    color: #222;
    line-height: 1.7;
}
h1 { font-size: 1.8em; border-bottom: 2px solid #333; padding-bottom: 10px; margin-top: 40px; }
h2 { font-size: 1.4em; border-bottom: 1px solid #ccc; padding-bottom: 6px; margin-top: 35px; color: #333; }
h3 { font-size: 1.15em; margin-top: 25px; color: #444; }
h4 { font-size: 1.05em; margin-top: 20px; color: #555; }
p { margin: 10px 0; }
table { border-collapse: collapse; width: 100%; margin: 15px 0; font-size: 0.92em; }
th, td { border: 1px solid #ccc; padding: 8px 12px; text-align: left; }
th { background: #f5f5f5; font-weight: bold; }
tr:nth-child(even) { background: #fafafa; }
pre, code { font-family: 'Consolas', 'Monaco', monospace; }
pre { background: #f4f4f4; padding: 15px; border-radius: 4px; overflow-x: auto; font-size: 0.9em; border: 1px solid #ddd; }
code { background: #f0f0f0; padding: 1px 4px; border-radius: 2px; font-size: 0.9em; }
pre code { background: none; padding: 0; }
.figure { margin: 25px 0; text-align: center; }
.figure img { border: 1px solid #ddd; border-radius: 4px; box-shadow: 0 2px 8px rgba(0,0,0,0.08); max-width: 100%; }
.caption { font-style: italic; color: #666; font-size: 0.9em; margin-top: 8px; }
hr { border: none; border-top: 1px solid #ddd; margin: 30px 0; }
strong { color: #111; }
blockquote { border-left: 3px solid #ccc; margin: 15px 0; padding: 10px 20px; color: #555; background: #fafafa; }
a { color: #2266aa; }
.back-link {
    display: inline-block;
    padding: 6px 14px;
    border-radius: 6px;
    font-size: 0.9em;
    font-weight: 600;
    text-decoration: none;
    border: 1px solid #D1D5DB;
    color: #2563EB;
    background: white;
    margin-bottom: 1.5em;
}
.back-link:hover { background: #EFF6FF; }

</style>
</head>
<body>

<a class="back-link" href="prereg_aspredicted.html">&larr; Back to preregistration</a>

<h1>Discovery Behavioral Results</h1>
<p><strong>Date:</strong> 2026-03-23</p>
<p><strong>Sample:</strong> Exploratory N=293, 13,185 choice trials, 10,546 affect ratings</p>
<p><strong>All numbers MCMC-verified</strong> (4 chains &times; 4,000 samples, all Rhat = 1.00, zero divergences)</p>
<hr>
'''

out_html += sections_html
out_html += '''
</body>
</html>'''

with open(OUT, 'w') as f:
    f.write(out_html)

print(f"Saved: {OUT}")
print(f"Size: {len(out_html):,} bytes")
