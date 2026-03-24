#!/usr/bin/env python3
"""Generate HTML documents with embedded base64 images from markdown sources."""

import base64
import os
import re
import html

IMG_DIR = "/workspace/results/figs/paper"
DRAFTS_DIR = "/workspace/drafts"


def load_images():
    """Load all PNGs as base64 strings."""
    images = {}
    for fname in os.listdir(IMG_DIR):
        if fname.endswith(".png"):
            with open(os.path.join(IMG_DIR, fname), "rb") as f:
                images[fname] = base64.b64encode(f.read()).decode()
    return images


def img_tag(images, fname, caption="", max_width="100%"):
    """Create an HTML img tag with base64 data."""
    b64 = images.get(fname, "")
    if not b64:
        return f"<!-- Image {fname} not found -->"
    return f"""<div class="figure">
<img src="data:image/png;base64,{b64}" alt="{html.escape(caption)}" style="max-width:{max_width}; height:auto;">
<p class="caption">{html.escape(caption)}</p>
</div>"""


CSS = """
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
.figure img { border: 1px solid #ddd; border-radius: 4px; box-shadow: 0 2px 8px rgba(0,0,0,0.08); }
.caption { font-style: italic; color: #666; font-size: 0.9em; margin-top: 8px; }
hr { border: none; border-top: 1px solid #ddd; margin: 30px 0; }
strong { color: #111; }
blockquote { border-left: 3px solid #ccc; margin: 15px 0; padding: 10px 20px; color: #555; background: #fafafa; }
.toc { background: #f8f8f8; padding: 20px 30px; border-radius: 6px; border: 1px solid #e0e0e0; margin-bottom: 30px; }
.toc h2 { border: none; margin-top: 10px; }
.toc ul { list-style: none; padding-left: 0; }
.toc li { margin: 5px 0; }
.toc a { text-decoration: none; color: #2266aa; }
.toc a:hover { text-decoration: underline; }
.toc ul ul { padding-left: 20px; }
a { color: #2266aa; }
"""


def md_to_html_block(text):
    """Convert markdown text to HTML (simple converter for our needs)."""
    lines = text.split("\n")
    result = []
    in_code = False
    in_table = False
    in_list = False
    table_rows = []

    def flush_table():
        nonlocal table_rows, in_table
        if not table_rows:
            return ""
        html_out = "<table>\n"
        for i, row in enumerate(table_rows):
            cells = [c.strip() for c in row.split("|")]
            cells = [c for c in cells if c != ""]
            # Skip separator rows
            if all(re.match(r"^[-:]+$", c) for c in cells):
                continue
            tag = "th" if i == 0 else "td"
            html_out += "<tr>" + "".join(f"<{tag}>{format_inline(c)}</{tag}>" for c in cells) + "</tr>\n"
        html_out += "</table>\n"
        table_rows = []
        in_table = False
        return html_out

    def format_inline(text):
        """Format inline markdown: bold, italic, code, math."""
        # Code spans first
        text = re.sub(r"`([^`]+)`", r"<code>\1</code>", text)
        # Bold
        text = re.sub(r"\*\*([^*]+)\*\*", r"<strong>\1</strong>", text)
        # Italic
        text = re.sub(r"\*([^*]+)\*", r"<em>\1</em>", text)
        # LaTeX-style math ($...$)
        text = re.sub(r"\$([^$]+)\$", r"<em>\1</em>", text)
        return text

    i = 0
    while i < len(lines):
        line = lines[i]

        # Code blocks
        if line.strip().startswith("```"):
            if in_table:
                result.append(flush_table())
            if in_code:
                result.append("</code></pre>")
                in_code = False
            else:
                lang = line.strip()[3:]
                result.append(f"<pre><code>")
                in_code = True
            i += 1
            continue

        if in_code:
            result.append(html.escape(line))
            result.append("\n")
            i += 1
            continue

        # Blank line
        if line.strip() == "":
            if in_table:
                result.append(flush_table())
            if in_list:
                in_list = False
            i += 1
            continue

        # Horizontal rule
        if line.strip() == "---":
            if in_table:
                result.append(flush_table())
            result.append("<hr>")
            i += 1
            continue

        # Headers
        m = re.match(r"^(#{1,4})\s+(.*)", line)
        if m:
            if in_table:
                result.append(flush_table())
            level = len(m.group(1))
            text = format_inline(m.group(2))
            anchor = re.sub(r"[^a-z0-9]+", "-", m.group(2).lower()).strip("-")
            result.append(f"<h{level} id=\"{anchor}\">{text}</h{level}>")
            i += 1
            continue

        # Table row
        if "|" in line and line.strip().startswith("|"):
            in_table = True
            table_rows.append(line)
            i += 1
            continue

        # Unordered list
        m = re.match(r"^(\s*)[-*]\s+(.*)", line)
        if m:
            if in_table:
                result.append(flush_table())
            if not in_list:
                result.append("<ul>")
                in_list = True
            result.append(f"<li>{format_inline(m.group(2))}</li>")
            i += 1
            continue

        # Numbered list
        m = re.match(r"^(\s*)\d+\.\s+(.*)", line)
        if m:
            if in_table:
                result.append(flush_table())
            if not in_list:
                result.append("<ol>")
                in_list = True
            result.append(f"<li>{format_inline(m.group(2))}</li>")
            i += 1
            # Check if next lines continue as numbered list
            if i < len(lines) and not re.match(r"^\s*\d+\.\s+", lines[i]) and in_list:
                result.append("</ol>")
                in_list = False
            continue

        # Regular paragraph
        if in_table:
            result.append(flush_table())
        result.append(f"<p>{format_inline(line)}</p>")
        i += 1

    if in_table:
        result.append(flush_table())
    if in_code:
        result.append("</code></pre>")

    return "\n".join(result)


def generate_prereg_html(images):
    """Generate the preregistration HTML with embedded figures."""
    with open(os.path.join(DRAFTS_DIR, "preregistration.md"), "r") as f:
        md_text = f.read()

    # Split by sections to insert figures
    sections = md_text.split("\n---\n")

    # Convert full text to HTML first, then insert figures at strategic points
    full_html = md_to_html_block(md_text)

    # Insert figures after each hypothesis section using actual anchor IDs
    # Process in reverse order so insertions don't shift positions of earlier anchors
    figure_insertions = [
        ("h1-the-survival-weighted-additive-effort-choice-model-best-explains-foraging-behavior",
            img_tag(images, "fig_s_ppc.png", "Figure S1: Posterior predictive checks for the winning choice model (M5). All 9 experimental conditions fall within 95% HDI.", "90%")),
        ("h2-model-derived-survival-predicts-trial-level-anxiety-and-confidence",
            img_tag(images, "fig_s_metacognition.png", "Figure S2: Metacognition (Panels A-B) -- Survival probability S predicts trial-level anxiety (negative) and confidence (positive).", "90%")),
        ("h3-danger-drives-excess-motor-vigor",
            img_tag(images, "fig_vigor_timecourse.png", "Figure 3: Vigor timecourse -- danger mobilization across threat levels, showing excess motor effort under high threat.", "90%")),
        ("h4-choice-shift-and-vigor-shift-under-threat-are-coherently-coupled-and-predict-outcomes",
            img_tag(images, "fig4_coherent_shift.png", "Figure 4: Coherent behavioral shift -- choice conservatism and vigor mobilization are coupled across threat levels.", "90%")),
        ("h5-choice-and-vigor-parameters-are-coupled-across-independently-estimated-bayesian-models",
            "\n".join([
                img_tag(images, "fig5_joint_model.png", "Figure 5: Joint model -- cross-domain parameter correlations from independently estimated Bayesian models.", "90%"),
                img_tag(images, "fig_s_recovery.png", "Figure S3: Parameter recovery -- simulated data recovery validates model identifiability.", "90%"),
            ])),
        ("h6-vigor-mobilization-predicts-metacognitive-accuracy",
            img_tag(images, "fig_s_metacognition.png", "Figure S2: Metacognition (Panels C-F) -- Vigor mobilization (delta) predicts metacognitive calibration accuracy.", "90%")),
    ]

    # Insert in reverse so position shifts don't affect earlier insertions
    for anchor, fig_html in reversed(figure_insertions):
        pattern = f'id="{anchor}"'
        pos = full_html.find(pattern)
        if pos == -1:
            print(f"  WARNING: anchor not found: {anchor}")
            continue

        # Find the next <hr> after this heading (section boundary)
        search_start = pos + 100
        next_hr = full_html.find("<hr>", search_start)
        next_h2 = full_html.find("<h2", search_start)

        boundaries = []
        if next_hr > 0: boundaries.append(next_hr)
        if next_h2 > 0: boundaries.append(next_h2)

        if boundaries:
            insert_pos = min(boundaries)
            full_html = full_html[:insert_pos] + "\n" + fig_html + "\n" + full_html[insert_pos:]

    # Mental health section (Section 7)
    psych_pos = full_html.lower().find("psychiatric battery associations")
    if psych_pos > 0:
        next_break = full_html.find("<h", psych_pos + 100)
        if next_break < 0:
            next_break = len(full_html)
        fig = img_tag(images, "fig_s_mental_health.png", "Figure S4: Mental health -- psychiatric factor associations with task parameters.", "90%")
        full_html = full_html[:next_break] + "\n" + fig + "\n" + full_html[next_break:]

    html_doc = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Preregistration: Confirmatory Replication of Effort-Threat Integration in Human Foraging</title>
<style>
{CSS}
</style>
</head>
<body>
{full_html}
</body>
</html>"""

    with open(os.path.join(DRAFTS_DIR, "prereg_with_figs.html"), "w") as f:
        f.write(html_doc)
    print(f"Written prereg_with_figs.html ({len(html_doc)} chars)")


def generate_discovery_html(images):
    """Generate the discovery results HTML with embedded figures."""
    with open(os.path.join(DRAFTS_DIR, "discovery_results.md"), "r") as f:
        md_text = f.read()

    full_html = md_to_html_block(md_text)

    # Table of contents
    toc = """<div class="toc">
<h2>Table of Contents</h2>
<ul>
<li><a href="#1-choice-survival-weighted-value-governs-foraging-decisions">1. Choice: Survival-weighted value governs foraging decisions</a></li>
<li><a href="#2-vigor-affect-s-governs-motor-effort-and-subjective-experience">2. Vigor + Affect: S governs motor effort and subjective experience</a>
<ul>
<li><a href="#vigor">Vigor</a></li>
<li><a href="#affect">Affect</a></li>
</ul>
</li>
<li><a href="#3-coordinated-effort-reallocation">3. Coordinated effort reallocation</a>
<ul>
<li><a href="#behavioral-coupling">Behavioral coupling</a></li>
<li><a href="#independent-bayesian-models-mcmc-validated">Independent Bayesian models</a></li>
<li><a href="#svi-joint-model-robustness-supp-note-2">SVI joint model</a></li>
<li><a href="#outcome-prediction">Outcome prediction</a></li>
</ul>
</li>
<li><a href="#4-metacognitive-motor-bridge">4. Metacognitive-motor bridge</a></li>
<li><a href="#5-mental-health-performance-phenotype-not-clinical">5. Mental health: Performance phenotype, not clinical</a></li>
<li><a href="#6-model-validation">6. Model validation</a></li>
<li><a href="#not-in-the-main-story-supplementary">Supplementary</a></li>
</ul>
</div>"""

    # Insert figures into sections (using correct anchor IDs, processed in reverse)
    figure_insertions = [
        ("1-choice-survival-weighted-value-governs-foraging-decisions",
         img_tag(images, "fig_s_ppc.png", "Figure S1: Posterior predictive checks -- all 9 conditions within 95% HDI. Accuracy 76.1%, AUC 0.863.", "90%")),
        ("vigor",
         img_tag(images, "fig_vigor_timecourse.png", "Figure 3: Vigor timecourse -- excess motor effort scales with danger (1-S). Population mean delta = +0.211.", "90%")),
        ("affect",
         img_tag(images, "fig_s_metacognition.png", "Figure S2: Metacognition -- S predicts anxiety (beta = -0.281) and confidence (beta = +0.280) at the trial level.", "90%")),
        ("behavioral-coupling",
         img_tag(images, "fig4_coherent_shift.png", "Figure 4: Coherent shift -- choice shift x vigor shift r = -0.78. Cross-validated r = -0.55.", "90%")),
        ("independent-bayesian-models-mcmc-validated",
         img_tag(images, "fig5_joint_model.png", "Figure 5: Joint model -- log(beta) x delta: r = +0.53; log(k) x delta: r = -0.33. Independent Bayesian models.", "90%")),
        ("4-metacognitive-motor-bridge",
         img_tag(images, "fig_s_metacognition.png", "Figure S2: Metacognitive calibration -- delta predicts anxiety-slope (r = -0.311) and confidence-slope (r = +0.325) on S.", "90%")),
        ("5-mental-health-performance-phenotype-not-clinical",
         img_tag(images, "fig_s_mental_health.png", "Figure S4: Mental health -- alpha predicts apathy (R2 = 0.12). All other associations null (ROPE-confirmed).", "90%")),
        ("mcmc-convergence",
         "\n".join([
             img_tag(images, "fig_s_ppc.png", "Figure S1: Posterior predictive checks confirming model adequacy.", "85%"),
             img_tag(images, "fig_s_recovery.png", "Figure S3: Parameter recovery -- k: r=0.86, beta: r=0.40, delta: r=0.67, alpha: r=0.94.", "85%"),
         ])),
    ]

    for anchor, fig_html in reversed(figure_insertions):
        pattern = f'id="{anchor}"'
        pos = full_html.find(pattern)
        if pos == -1:
            print(f"  WARNING: discovery anchor not found: {anchor}")
            continue

        # Find next heading after this section's content
        search_start = pos + 50
        next_h2 = full_html.find("<h2", search_start)
        next_h3 = full_html.find("<h3", search_start)

        boundaries = []
        if next_h2 > 0: boundaries.append(next_h2)
        if next_h3 > 0: boundaries.append(next_h3)

        if boundaries:
            insert_pos = min(boundaries)
        else:
            insert_pos = len(full_html)

        full_html = full_html[:insert_pos] + "\n" + fig_html + "\n" + full_html[insert_pos:]

    html_doc = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Discovery Results — Effort Reallocation Under Threat</title>
<style>
{CSS}
</style>
</head>
<body>
{toc}
{full_html}
</body>
</html>"""

    with open(os.path.join(DRAFTS_DIR, "discovery_results_with_figs.html"), "w") as f:
        f.write(html_doc)
    print(f"Written discovery_results_with_figs.html ({len(html_doc)} chars)")


if __name__ == "__main__":
    images = load_images()
    print(f"Loaded {len(images)} images")
    generate_prereg_html(images)
    generate_discovery_html(images)
    print("Done!")
