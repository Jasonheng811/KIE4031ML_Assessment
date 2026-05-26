"""Render report.md to report.html so it can be opened in a browser and
printed to PDF (Ctrl+P -> Save as PDF) without needing pandoc/LaTeX."""

import markdown

MD_PATH   = "report.md"
HTML_PATH = "report.html"

with open(MD_PATH, "r", encoding="utf-8") as f:
    md_text = f.read()

html_body = markdown.markdown(
    md_text,
    extensions=["tables", "fenced_code", "toc", "sane_lists", "smarty"],
)

# Minimal print-friendly stylesheet — A4 page, readable line length,
# table formatting, header anchors.
CSS = """
:root { font-family: -apple-system, "Segoe UI", Helvetica, Arial, sans-serif; }
body {
    max-width: 820px;
    margin: 2em auto;
    padding: 0 1.5em;
    line-height: 1.55;
    color: #222;
    font-size: 13.5pt;
}
h1, h2, h3, h4 { color: #143b6e; line-height: 1.25; margin-top: 1.4em; }
h1 { font-size: 1.85em; border-bottom: 2px solid #143b6e; padding-bottom: .25em; }
h2 { font-size: 1.40em; border-bottom: 1px solid #ccc; padding-bottom: .15em; }
h3 { font-size: 1.15em; }
p  { margin: 0.5em 0 0.9em; }
hr { border: 0; border-top: 1px solid #ccc; margin: 2em 0; }
code {
    background: #f1f3f6; padding: 1px 4px;
    border-radius: 3px; font-size: 0.92em;
    font-family: "SF Mono", "Consolas", monospace;
}
pre {
    background: #f6f8fa; padding: 0.9em 1em;
    border-radius: 4px; overflow-x: auto;
    font-size: 0.88em; line-height: 1.4;
}
table {
    border-collapse: collapse; margin: 1em 0;
    font-size: 0.94em; width: 100%;
}
th, td { border: 1px solid #cfd6dc; padding: 6px 10px; text-align: left; }
th { background: #eef2f7; }
td:has(+ td:last-child:empty), td:last-child { text-align: right; }
strong { color: #143b6e; }
blockquote {
    border-left: 4px solid #c5cdd6;
    padding: 0.2em 0.9em;
    color: #555; margin: 1em 0;
}
@media print {
    body { font-size: 11pt; max-width: none; margin: 0; padding: 0 12mm; }
    h1, h2, h3 { page-break-after: avoid; }
    table, pre, img { page-break-inside: avoid; }
    a { color: inherit; text-decoration: none; }
}
"""

html_doc = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>KIE4031 - Stock Price Prediction with RNN</title>
<style>{CSS}</style>
</head>
<body>
{html_body}
</body>
</html>
"""

with open(HTML_PATH, "w", encoding="utf-8") as f:
    f.write(html_doc)

print(f"Rendered {MD_PATH} -> {HTML_PATH} ({len(html_doc):,} bytes)")
print("Open report.html in any browser and use Ctrl+P -> 'Save as PDF' to produce report.pdf.")
