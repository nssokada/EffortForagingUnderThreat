#!/usr/bin/env python3
"""Convert markdown files to styled HTML for PDF printing."""

import sys
from pathlib import Path
from markdown_it import MarkdownIt

CSS = """
<style>
@page { margin: 2.5cm; size: A4; }
body {
    font-family: 'Times New Roman', Times, Georgia, serif;
    font-size: 11pt;
    line-height: 1.6;
    max-width: 700px;
    margin: 0 auto;
    padding: 40px 20px;
    color: #1a1a1a;
}
h1 { font-size: 18pt; margin-top: 0; margin-bottom: 8px; line-height: 1.3; }
h2 { font-size: 14pt; margin-top: 28px; margin-bottom: 10px; border-bottom: 1px solid #ccc; padding-bottom: 4px; }
h3 { font-size: 12pt; margin-top: 20px; margin-bottom: 8px; }
h4 { font-size: 11pt; margin-top: 16px; margin-bottom: 6px; font-style: italic; }
p { margin: 0 0 10px 0; text-align: justify; }
table { border-collapse: collapse; margin: 16px 0; font-size: 10pt; width: 100%; }
th, td { border: 1px solid #999; padding: 5px 8px; text-align: left; }
th { background: #f0f0f0; font-weight: bold; }
tr:nth-child(even) { background: #fafafa; }
sup { font-size: 8pt; }
hr { border: none; border-top: 1px solid #ccc; margin: 24px 0; }
blockquote { border-left: 3px solid #ccc; margin: 12px 0; padding: 4px 16px; color: #555; }
code { font-family: 'Courier New', monospace; font-size: 10pt; background: #f5f5f5; padding: 1px 4px; }
pre { background: #f5f5f5; padding: 12px; overflow-x: auto; font-size: 10pt; }
strong { font-weight: bold; }
em { font-style: italic; }
ul, ol { margin: 8px 0 8px 24px; }
li { margin-bottom: 4px; }
.author { font-size: 11pt; color: #444; margin-bottom: 4px; }
.abstract { font-size: 10.5pt; }
@media print {
    body { font-size: 10.5pt; padding: 0; }
    h2 { page-break-after: avoid; }
    table { page-break-inside: avoid; }
}
</style>
"""

def convert(md_path: str, html_path: str):
    md = MarkdownIt('commonmark', {'typographer': True})
    md.enable('table')
    md.enable('strikethrough')

    text = Path(md_path).read_text()
    html_body = md.render(text)

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{Path(md_path).stem}</title>
{CSS}
</head>
<body>
{html_body}
</body>
</html>
"""
    Path(html_path).write_text(html)
    print(f"  {md_path} -> {html_path}")


if __name__ == '__main__':
    base = Path(__file__).parent

    # Convert paper
    paper_md = base / 'paper.md'
    paper_html = base / 'paper.html'
    if paper_md.exists():
        convert(str(paper_md), str(paper_html))

    # Convert preregistration
    prereg_md = Path('/workspace/drafts/prereg_evc_aspredicted.md')
    prereg_html = base / 'prereg.html'
    if prereg_md.exists():
        convert(str(prereg_md), str(prereg_html))

    print("\nDone. Open HTML files in a browser and print to PDF (Ctrl+P).")
    print("Use 'Save as PDF' in the print dialog for best results.")
