import os
import json
from google import genai
from dotenv import load_dotenv

# ── Config ──────────────────────────────────────────────────────────────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ENV_PATH = os.path.join(SCRIPT_DIR, "..", ".env")
IN_JSONL = os.path.join(SCRIPT_DIR, "results", "emotion_contrastive_insights.jsonl")
OUT_MD = os.path.join(SCRIPT_DIR, "results", "emotion_contrastive_summary.md")

# Load environment variables
load_dotenv(ENV_PATH)
API_KEY = os.getenv("GEMINI_API_KEY")
client = genai.Client(api_key=API_KEY)

def generate_summary():
    print("Loading insights...")
    if not os.path.exists(IN_JSONL):
        print("Error: insights file not found.")
        return

    insights = []
    with open(IN_JSONL, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                try:
                    data = json.loads(line)
                    ca = data.get("contrastive_analysis", {})
                    blindspot = ca.get("algorithmic_blindspot", "")
                    fix = ca.get("actionable_fix", "")
                    if blindspot and fix:
                        insights.append(f"- Blindspot: {blindspot}\n  Fix: {fix}")
                except json.JSONDecodeError:
                    pass

    if not insights:
        print("No valid insights found.")
        return

    print(f"Loaded {len(insights)} individual insights. Sending to Gemini for synthesis...")
    
    prompt = f"""
    You are an expert Machine Learning engineer. Below is a raw list of {len(insights)} individual 'algorithmic blindspots' and 'actionable fixes' generated from a Contrastive Error Analysis of a DeBERTa model meant to classify telecommunication complaint emotion.

    Your task is to SYNTHESIZE these {len(insights)} specific points into a single, concise SUMMARY TABLE. 
    Group similar fixes together to identify the Top 5 to 7 most critical, overarching systemic patterns.
    
    Format the output strictly as a clean Markdown table with the following three columns:
    | Core Algorithmic Blindspot | Description | Actionable Dataset Strategy |
    
    Keep the table extremely concise and high signal-to-noise. Do not output pages of text.

    RAW INSIGHTS:
    {chr(10).join(insights)}
    """

    response = client.models.generate_content(
        model='gemini-2.5-flash',
        contents=prompt,
        config=genai.types.GenerateContentConfig(
            temperature=0.2,
        )
    )

    summary_md = response.text
    
    with open(OUT_MD, "w", encoding="utf-8") as f:
        f.write("# Emotion Contrastive Error Analysis: Executive Summary Table\n\n")
        f.write(summary_md)

    print(f"Summary successfully generated at: {OUT_MD}")

if __name__ == "__main__":
    generate_summary()
