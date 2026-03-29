"""
extract_docx.py
Extracts MSIN0221_Group_Assignment_v2.docx to a clean Markdown file (report_draft.md).
Run from the final_report_latex directory.
"""

import os
import re
import sys
import docx

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DOCX_PATH  = os.path.join(SCRIPT_DIR, "MSIN0221_Group_Assignment_v2.docx")
OUT_MD     = os.path.join(SCRIPT_DIR, "report_draft.md")

doc = docx.Document(DOCX_PATH)

# ── Figure filename hints embedded in paragraph text ──────────────────────────
# Maps a fragment that appears in the caption/text to the actual PNG filename
FIGURE_HINTS = {
    "1_2_joint_distribution":      "figures/fig01_joint_distribution.png",
    "1_4_scenario_urgency":        "figures/fig08_scenario_affinity.png",
    "1_5_style_emotion":           "figures/fig10_style_emotion_independence.png",
    "3_1_text_length":             "figures/fig11_text_length.png",
    "3_3_token_truncation":        "figures/fig12_token_truncation.png",
    "deberta_error_distribution":  "figures/fig07_error_distribution.png",
    "fig1_model_comparison":       "figures/fig02_model_comparison.png",
    "fig2_per_class_f1":           "figures/fig03_per_class_f1.png",
    "fig3_training_dynamics":      "figures/fig05_training_dynamics.png",
    "fig5_confusion_matrices":     "figures/fig06_confusion_matrices.png",
    "training_runs_f1":            "figures/fig04_training_runs.png",
}

def detect_figure(text: str) -> str | None:
    """Return PNG filename if this paragraph looks like a figure caption."""
    low = text.lower()
    for hint, fname in FIGURE_HINTS.items():
        if hint.lower() in low:
            return fname
    return None

def render_runs(para) -> str:
    """Render paragraph runs preserving bold/italic as Markdown."""
    parts = []
    for run in para.runs:
        t = run.text
        if not t:
            continue
        if run.bold and run.italic:
            t = f"***{t}***"
        elif run.bold:
            t = f"**{t}**"
        elif run.italic:
            t = f"*{t}*"
        parts.append(t)
    return "".join(parts)

def render_table(table) -> str:
    """Render a docx table as a Markdown pipe table."""
    rows = []
    for row in table.rows:
        cells = [cell.text.replace("\n", " ").strip() for cell in row.cells]
        rows.append(cells)

    if not rows:
        return ""

    col_count = max(len(r) for r in rows)
    # Pad rows to equal width
    rows = [r + [""] * (col_count - len(r)) for r in rows]

    lines = []
    lines.append("| " + " | ".join(rows[0]) + " |")
    lines.append("| " + " | ".join(["---"] * col_count) + " |")
    for row in rows[1:]:
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines)

# ── Walk document body elements in order (paragraphs + tables interleaved) ───
# python-docx exposes body.xml children; we iterate them to preserve order.
from docx.oxml.ns import qn
from docx.table import Table as DocxTable
from docx.text.paragraph import Paragraph as DocxParagraph

body = doc.element.body

output_lines: list[str] = []

# Track state
in_references = False
in_appendix   = False
table_index   = 0
all_tables    = doc.tables

# Authors / title preamble
output_lines.append("---")
output_lines.append("title: \"Decoupling Urgency and Emotion in Automated Telecom Complaint Triage\"")
output_lines.append("authors: \"Liqiang Cheng, Olena Brylinska, Yuansheng Tao, Yuyi Song, Yuzhi Zhao\"")
output_lines.append("course: \"MSIN0221 Group Assignment — Group 16\"")
output_lines.append("---\n")

for child in body:
    tag = child.tag.split("}")[-1]  # 'p' or 'tbl'

    # ── TABLE ────────────────────────────────────────────────────────────────
    if tag == "tbl":
        if in_appendix:
            continue
        tbl = DocxTable(child, doc)
        md_table = render_table(tbl)
        output_lines.append("\n" + md_table + "\n")
        table_index += 1
        continue

    # ── PARAGRAPH ────────────────────────────────────────────────────────────
    if tag != "p":
        continue

    para = DocxParagraph(child, doc)
    style = para.style.name
    raw   = para.text.strip()

    if not raw:
        continue

    # Detect section transitions
    if raw == "References":
        in_references = True
        output_lines.append("\n## References\n")
        continue
    if raw.startswith("Appendix"):
        in_appendix = True
        continue
    if in_appendix:
        continue

    # Skip title-page noise (course code, blank lines, author lines already in frontmatter)
    if raw in ("MSIN0221 Group Assignment", "Group 16:"):
        continue
    student_re = re.compile(r"^(Liqiang Cheng|Olena Brylinska|Yuansheng Tao|Yuyi Song|Yuzhi Zhao)\s+\d{8}$")
    if student_re.match(raw):
        continue

    # ── Headings ─────────────────────────────────────────────────────────────
    if style == "Heading 1":
        # Strip leading number+dot, e.g. "1. Introduction" → "Introduction"
        clean = re.sub(r"^\d+\.\s*", "", raw)
        output_lines.append(f"\n# {clean}\n")
        continue

    if style == "Heading 2":
        clean = re.sub(r"^\d+\.\d+\s*", "", raw).strip()
        output_lines.append(f"\n## {clean}\n")
        continue

    if style == "Heading 3":
        clean = re.sub(r"^\d+\.\d+\.\d+\s*", "", raw).strip()
        output_lines.append(f"\n### {clean}\n")
        continue

    # ── References section: plain text list ──────────────────────────────────
    if in_references:
        output_lines.append(raw)
        continue

    # ── Figure caption heuristic ──────────────────────────────────────────────
    fig_file = detect_figure(raw)
    if fig_file:
        # Extract caption text (strip the filename hint in brackets)
        caption = re.sub(r"\[.*?\]", "", raw).strip()
        caption = re.sub(r"^\s*Figure\s*\d+[:\.]?\s*", "", caption).strip()
        output_lines.append(f"\n![{caption}]({fig_file})\n*{caption}*\n")
        continue

    # ── Table caption heuristic ───────────────────────────────────────────────
    if re.match(r"^\s*Table\s+\d+", raw, re.IGNORECASE):
        caption = re.sub(r"^\s*Table\s+\d+[:\.]?\s*", "", raw).strip()
        output_lines.append(f"\n**Table: {caption}**\n")
        continue

    # ── Ordinary paragraph ────────────────────────────────────────────────────
    rendered = render_runs(para)
    if rendered.strip():
        output_lines.append(rendered + "\n")

# ── Write output ──────────────────────────────────────────────────────────────
with open(OUT_MD, "w", encoding="utf-8") as f:
    f.write("\n".join(output_lines))

print(f"Written: {OUT_MD}")
print(f"Lines:   {len(output_lines)}")
