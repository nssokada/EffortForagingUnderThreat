#!/usr/bin/env python3
"""Convert markdown to styled HTML for paper and prereg documents.
Uses only stdlib (no pandoc, no markdown library). Handles the subset
of markdown used in this project's drafts.
"""

import re
import sys
from pathlib import Path

CSS = """
@page { margin: 2.5cm; size: A4; }
body {
    font-family: 'Times New Roman', Times, Georgia, serif;
    font-size: 11pt;
    line-height: 1.6;
    max-width: 720px;
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
code { font-family: 'Courier New', monospace; font-size: 10pt; background: #f5f5f5; padding: 1px 4px; border-radius: 2px; }
pre { background: #f5f5f5; padding: 12px; overflow-x: auto; font-size: 10pt; border-radius: 4px; }
pre code { background: none; padding: 0; }
strong { font-weight: bold; }
em { font-style: italic; }
ul, ol { margin: 8px 0 8px 24px; }
li { margin-bottom: 4px; }
.author { font-size: 11pt; color: #444; margin-bottom: 4px; }
.abstract { font-size: 10.5pt; }
.math { font-family: 'Cambria Math', 'Times New Roman', serif; font-style: italic; }
@media print {
    body { font-size: 10.5pt; padding: 0; }
    h2 { page-break-after: avoid; }
    table { page-break-inside: avoid; }
}
"""

def md_to_html(md_text, title="Document"):
    """Convert markdown text to a complete HTML document."""
    lines = md_text.split('\n')
    html_parts = []
    in_code_block = False
    in_table = False
    in_list = False
    list_type = None
    table_rows = []

    def flush_table():
        nonlocal table_rows, in_table
        if not table_rows:
            return ""
        # First row is header
        result = '<table>\n<thead><tr>'
        for cell in table_rows[0]:
            result += f'<th>{inline(cell.strip())}</th>'
        result += '</tr></thead>\n<tbody>\n'
        for row in table_rows[1:]:
            result += '<tr>'
            for cell in row:
                result += f'<td>{inline(cell.strip())}</td>'
            result += '</tr>\n'
        result += '</tbody></table>\n'
        table_rows = []
        in_table = False
        return result

    def flush_list():
        nonlocal in_list, list_type
        if in_list:
            in_list = False
            return f'</{list_type}>\n'
        return ''

    def inline(text):
        """Process inline markdown."""
        # LaTeX math: $...$ → <span class="math">...</span>
        text = re.sub(r'\$([^$]+)\$', r'<span class="math">\1</span>', text)
        # Bold: **text** or __text__
        text = re.sub(r'\*\*(.+?)\*\*', r'<strong>\1</strong>', text)
        text = re.sub(r'__(.+?)__', r'<strong>\1</strong>', text)
        # Italic: *text* or _text_
        text = re.sub(r'\*(.+?)\*', r'<em>\1</em>', text)
        text = re.sub(r'(?<!\w)_(.+?)_(?!\w)', r'<em>\1</em>', text)
        # Code: `text`
        text = re.sub(r'`([^`]+)`', r'<code>\1</code>', text)
        # Superscript: ^text^ (markdown style)
        text = re.sub(r'\^([^^]+?)\^', r'<sup>\1</sup>', text)
        # Subscript: ~text~
        text = re.sub(r'~([^~]+?)~', r'<sub>\1</sub>', text)
        # Links: [text](url)
        text = re.sub(r'\[([^\]]+)\]\(([^)]+)\)', r'<a href="\2">\1</a>', text)
        return text

    i = 0
    while i < len(lines):
        line = lines[i]

        # Code blocks
        if line.strip().startswith('```'):
            if in_code_block:
                html_parts.append('</code></pre>\n')
                in_code_block = False
            else:
                lang = line.strip()[3:].strip()
                html_parts.append(flush_list())
                html_parts.append(f'<pre><code>')
                in_code_block = True
            i += 1
            continue

        if in_code_block:
            html_parts.append(line.replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;') + '\n')
            i += 1
            continue

        # Horizontal rule
        if re.match(r'^---+\s*$', line):
            html_parts.append(flush_list())
            html_parts.append(flush_table())
            html_parts.append('<hr>\n')
            i += 1
            continue

        # Headers
        m = re.match(r'^(#{1,4})\s+(.+)$', line)
        if m:
            html_parts.append(flush_list())
            html_parts.append(flush_table())
            level = len(m.group(1))
            html_parts.append(f'<h{level}>{inline(m.group(2))}</h{level}>\n')
            i += 1
            continue

        # Table rows
        if '|' in line and line.strip().startswith('|'):
            cells = [c for c in line.strip().split('|')[1:-1]]
            # Check if separator row
            if all(re.match(r'^[\s\-:]+$', c) for c in cells):
                i += 1
                continue
            if not in_table:
                html_parts.append(flush_list())
                in_table = True
            table_rows.append(cells)
            i += 1
            continue
        elif in_table:
            html_parts.append(flush_table())

        # Lists
        m_ul = re.match(r'^(\s*)[-*]\s+(.+)$', line)
        m_ol = re.match(r'^(\s*)\d+\.\s+(.+)$', line)
        if m_ul or m_ol:
            new_type = 'ul' if m_ul else 'ol'
            content = m_ul.group(2) if m_ul else m_ol.group(2)
            indent = len(m_ul.group(1) if m_ul else m_ol.group(1))
            if not in_list:
                html_parts.append(f'<{new_type}>\n')
                in_list = True
                list_type = new_type
            # Check for continuation on next lines (indented)
            full_content = content
            while i + 1 < len(lines) and lines[i + 1].startswith('    ') and not re.match(r'^\s*[-*]\s+', lines[i + 1]) and not re.match(r'^\s*\d+\.\s+', lines[i + 1]):
                i += 1
                full_content += ' ' + lines[i].strip()
            html_parts.append(f'<li>{inline(full_content)}</li>\n')
            i += 1
            continue
        elif in_list and line.strip() == '':
            html_parts.append(flush_list())
            i += 1
            continue
        elif in_list and not line.startswith(' '):
            html_parts.append(flush_list())

        # Empty line
        if line.strip() == '':
            i += 1
            continue

        # Paragraph — collect until blank line
        para_lines = [line]
        while i + 1 < len(lines) and lines[i + 1].strip() != '' and not lines[i + 1].startswith('#') and not lines[i + 1].startswith('```') and not re.match(r'^---+\s*$', lines[i + 1]) and '|' not in lines[i + 1].split()[0:1] and not re.match(r'^\s*[-*]\s+', lines[i + 1]) and not re.match(r'^\s*\d+\.\s+', lines[i + 1]):
            i += 1
            para_lines.append(lines[i])
        text = ' '.join(l.strip() for l in para_lines)
        html_parts.append(f'<p>{inline(text)}</p>\n')
        i += 1

    html_parts.append(flush_list())
    html_parts.append(flush_table())

    body = ''.join(html_parts)
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{title}</title>
<style>
{CSS}
</style>
</head>
<body>
{body}
</body>
</html>"""


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python md_to_html.py input.md [output.html] [title]")
        sys.exit(1)

    input_path = Path(sys.argv[1])
    output_path = Path(sys.argv[2]) if len(sys.argv) > 2 else input_path.with_suffix('.html')
    title = sys.argv[3] if len(sys.argv) > 3 else input_path.stem

    md_text = input_path.read_text()
    html = md_to_html(md_text, title)
    output_path.write_text(html)
    print(f"Wrote {output_path} ({len(html)} bytes)")
