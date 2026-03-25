"""
run_stage2.py — Stage 2 of the ErrorMap Pipeline
Clusters Stage 1 error labels into a hierarchical taxonomy via a single LLM call.
Output: error_analysis/results/final_taxonomy.json
"""

import json
import os
import sys
from pathlib import Path

from dotenv import load_dotenv
from google import genai
from google.genai import types

# ── Paste your Stage 2 taxonomy prompt here before running ─────────────────
STAGE_2_SYSTEM_PROMPT = """
You are an expert AI data analyst. Your task is to review a list of specific
error titles and summaries generated from a model's incorrect predictions,
and group them into a clean taxonomy of failure modes.

You will be provided with a list of raw errors from a customer complaint
urgency and emotion classification model (DeBERTa-v3-base). The errors
represent cases where the model made wrong predictions on the test set.

Your instructions:
1. Group the provided errors into 3 to 5 high-level categories
   (e.g., "Severity Underestimation", "Context Misinterpretation").
2. Map every single original error title to exactly one category.
   Do not leave any original titles out.
3. Do not create sub-categories — keep the taxonomy completely flat.
4. Base your categories purely on the patterns you observe in the data.
   Do not invent categories that are not supported by the errors provided.

Output the result strictly as a JSON object with this exact structure,
no preamble, no markdown backticks, valid JSON only:
{
  "taxonomy": [
    {
      "category_name": "...",
      "description": "A short description of this failure mode category",
      "original_error_titles": [
        "exact error title from input",
        "exact error title from input"
      ]
    }
  ]
}
"""

# ── Config ──────────────────────────────────────────────────────────────────
SCRIPT_DIR  = Path(__file__).parent
INPUT_JSONL = SCRIPT_DIR / "stage1_errors.jsonl"
OUTPUT_JSON = SCRIPT_DIR / "results" / "final_taxonomy.json"
MODEL       = "gemini-2.5-flash"

# ── Load API key ────────────────────────────────────────────────────────────
load_dotenv(SCRIPT_DIR.parent / ".env")
api_key = os.environ.get("Gemini_API_Key") or os.environ.get("GOOGLE_API_KEY")
if not api_key:
    print("ERROR: No Gemini API key found. Set Gemini_API_Key in .env")
    sys.exit(1)

client = genai.Client(api_key=api_key)

# ── Load Stage 1 data ───────────────────────────────────────────────────────
errors = []
with open(INPUT_JSONL, "r", encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if line:
            record = json.loads(line)
            errors.append({
                "error_title":   record["error_title"],
                "error_summary": record["error_summary"],
            })

print(f"Loaded {len(errors)} error entries from Stage 1.")

# ── Build prompt payload ────────────────────────────────────────────────────
errors_json = json.dumps(errors, ensure_ascii=False)
user_message = f"{STAGE_2_SYSTEM_PROMPT}\n\nHere are the Stage 1 error labels:\n{errors_json}"

# ── Single API call ─────────────────────────────────────────────────────────
print(f"Calling {MODEL} ...")
response = client.models.generate_content(
    model=MODEL,
    contents=user_message,
    config=types.GenerateContentConfig(
        response_mime_type="application/json",
    ),
)

# ── Parse & validate ────────────────────────────────────────────────────────
try:
    taxonomy = json.loads(response.text)
except json.JSONDecodeError as e:
    print("ERROR: Could not parse model response as JSON.")
    print(f"JSONDecodeError: {e}")
    print("\n--- Raw response ---")
    print(response.text)
    sys.exit(1)

# ── Save output ─────────────────────────────────────────────────────────────
os.makedirs(OUTPUT_JSON.parent, exist_ok=True)
OUTPUT_JSON.write_text(json.dumps(taxonomy, indent=2, ensure_ascii=False), encoding="utf-8")

n = len(taxonomy)
print(f"Taxonomy saved. {n} top-level categories generated.")
print(f"Output: {OUTPUT_JSON}")
