import os
import json
from fpdf import FPDF

# Paths
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
IN_JSONL = os.path.join(SCRIPT_DIR, "results", "emotion_contrastive_insights.jsonl")
OUT_PDF = os.path.join(SCRIPT_DIR, "results", "final_emotion_contrastive_report.pdf")

def sanitize_text(text):
    """Replaces common unicode characters not supported by fpdf default font"""
    if not isinstance(text, str):
        return str(text)
    replacements = {
        '‘': "'", '’': "'", '“': '"', '”': '"', 
        '£': 'GBP ', '—': '-', '–': '-', '…': '...',
        'é': 'e', 'è': 'e', 'á': 'a', '\xa0': ' '
    }
    for k, v in replacements.items():
        text = text.replace(k, v)
    return text.encode('latin-1', 'replace').decode('latin-1')

class PDFReport(FPDF):
    def header(self):
        self.set_font('Arial', 'B', 14)
        self.cell(0, 10, 'Final Emotion Contrastive Error Analysis Report', 0, 1, 'C')
        self.set_font('Arial', '', 10)
        self.cell(0, 10, 'DeBERTa Telecommunications Emotion Model', 0, 1, 'C')
        self.ln(5)

    def footer(self):
        self.set_y(-15)
        self.set_font('Arial', 'I', 8)
        self.cell(0, 10, f'Page {self.page_no()}', 0, 0, 'C')

def generate_pdf():
    print(f"Loading insights from {IN_JSONL}")
    if not os.path.exists(IN_JSONL):
        print("Insights not found. Run run_emotion_icp_analysis.py first.")
        return

    pairs = []
    with open(IN_JSONL, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                try:
                    pairs.append(json.loads(line))
                except json.JSONDecodeError:
                    pass

    pdf = PDFReport()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()
    
    # Intro
    pdf.set_font('Arial', '', 11)
    intro = (
        f"This report presents the Qualitative Contrastive Error Analysis for {len(pairs)} failure cases. "
        "Each case pairs an incorrect prediction with a correct prediction from the exact same Ground Truth Emotion, "
        "Scenario, and Style. The Gemini model was used to strictly extract the linguistic delta, "
        "the algorithmic blindspot, and propose actionable dataset fixes."
    )
    pdf.multi_cell(0, 6, sanitize_text(intro))
    pdf.ln(10)
    
    # Loop pairs
    for i, p in enumerate(pairs, 1):
        pdf.set_font('Arial', 'B', 12)
        true_emo = p.get("true_label", "Unknown").upper()
        pred_emo = p.get("error_prediction", "Unknown")
        pdf.set_fill_color(240, 240, 240)
        pdf.cell(0, 8, f"Pair #{i} | True: {true_emo} -> Predicted: {pred_emo}", 0, 1, 'L', fill=True)
        
        # Original Texts (Truncated for space if necessary, or full)
        pdf.set_font('Arial', 'I', 9)
        pdf.set_text_color(200, 0, 0) # Red
        pdf.multi_cell(0, 5, sanitize_text(f"ERROR TEXT: {p.get('error_text', '')}"))
        
        pdf.set_text_color(0, 150, 0) # Green
        pdf.multi_cell(0, 5, sanitize_text(f"CORRECT TEXT: {p.get('correct_text', '')}"))
        pdf.set_text_color(0, 0, 0)
        pdf.ln(2)
        
        # LLM Insights
        insights = p.get("contrastive_analysis", {})
        
        pdf.set_font('Arial', 'B', 10)
        pdf.cell(0, 6, "1. Linguistic Delta:", 0, 1)
        pdf.set_font('Arial', '', 10)
        pdf.multi_cell(0, 5, sanitize_text(insights.get("linguistic_delta", "N/A")))
        pdf.ln(1)
        
        pdf.set_font('Arial', 'B', 10)
        pdf.cell(0, 6, "2. Algorithmic Blindspot:", 0, 1)
        pdf.set_font('Arial', '', 10)
        pdf.multi_cell(0, 5, sanitize_text(insights.get("algorithmic_blindspot", "N/A")))
        pdf.ln(1)
        
        pdf.set_font('Arial', 'B', 10)
        pdf.cell(0, 6, "3. Actionable Fix:", 0, 1)
        pdf.set_font('Arial', '', 10)
        pdf.multi_cell(0, 5, sanitize_text(insights.get("actionable_fix", "N/A")))
        pdf.ln(8)

    os.makedirs(os.path.dirname(OUT_PDF), exist_ok=True)
    pdf.output(OUT_PDF)
    print(f"PDF successfully generated at: {OUT_PDF}")

if __name__ == "__main__":
    generate_pdf()
