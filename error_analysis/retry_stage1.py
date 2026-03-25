"""
retry_stage1.py — Retry failed rows from Stage 1 of the ErrorMap Pipeline
Identifies missing rows by comparing CSV vs JSONL, re-runs them through Gemini.
Appends results to the existing stage1_errors.jsonl.
"""

import json
import time
import csv
import os
import sys

from dotenv import load_dotenv
from google import genai

# ── System prompt (identical to run_stage1.py) ──────────────────────────────
STAGE_1_SYSTEM_PROMPT = """You are an expert analyst. Your job is to evaluate evidence step by step, consider alternatives, and reach a justified conclusion. Reasoning: high.

You are given the following:
- A context
- A model response that was labeled incorrect
- A reference

Your task:
1. Structured Correct Solution: Analyze the correct responses and extract from them the main required criteria or reasoning steps for the context.
2. Step-by-step Evaluation: Evaluate the incorrect response against each of the required criteria. For each criterion, provide the following fields:
   - present_in_wrong: Whether it is present in the incorrect response
   - quality: The quality of its execution (correct, partially correct, incorrect, or null if missing)
   - evidence: Supporting evidence from the incorrect response (quote)
   - comment: Any relevant comments
3. Error Diagnosis: Identify the first major error in the incorrect response that led to the incorrect answer, and provide the following fields in final_answer:
   - error_summary: If such an error exists, summarize the model's reasoning weakness in error_summary. This should focus on model thinking (e.g., 'the model failed to recognize fact X') rather than technical execution (e.g., 'the model selected the wrong answer').
   - error_title: Provide a short, free-form title that describes the specific type of error.

* If you didn't find any error in the incorrect response leave all the fields of final_answer with an empty string. If the whole solution is incorrect, write 'whole solution incorrect' in final_answer fields.
* Avoid ambiguous titles or ones that cannot be mapped to a specific skill.

Output the final result in the following JSON format:
{
  "required_criteria": [
    {
      "criterion": "...",
      "present_in_wrong": true,
      "quality": "...",
      "evidence": "...",
      "comment": "..."
    }
  ],
  "final_answer": {
    "error_summary": "...",
    "error_title": "..."
  }
}"""

# ── Config ──────────────────────────────────────────────────────────────────
SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR  = os.path.dirname(SCRIPT_DIR)
INPUT_CSV    = os.path.join(SCRIPT_DIR, "data", "incorrect_predictions.csv")
OUTPUT_JSONL = os.path.join(SCRIPT_DIR, "results", "stage1_errors.jsonl")
MODEL        = "gemini-2.5-flash"
SLEEP_SECS   = 1.5

# ── Load API key ────────────────────────────────────────────────────────────
load_dotenv(os.path.join(PROJECT_DIR, ".env"))
api_key = os.environ.get("Gemini_API_Key") or os.environ.get("GOOGLE_API_KEY")
if not api_key:
    print("ERROR: No Gemini API key found. Set Gemini_API_Key in .env")
    sys.exit(1)
client = genai.Client(api_key=api_key)

# ── Load CSV rows ──────────────────────────────────────────────────────────
rows = []
with open(INPUT_CSV, "r", encoding="utf-8") as f:
    for row in csv.DictReader(f):
        rows.append(row)
total_csv = len(rows)

# ── Load already-processed row indices from JSONL ───────────────────────────
processed_indices = set()
if os.path.exists(OUTPUT_JSONL):
    with open(OUTPUT_JSONL, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    record = json.loads(line)
                    processed_indices.add(record["row_index"])
                except (json.JSONDecodeError, KeyError):
                    pass

# ── Identify missing rows ──────────────────────────────────────────────────
missing = [(idx, row) for idx, row in enumerate(rows) if idx not in processed_indices]

print(f"CSV rows: {total_csv}")
print(f"Already processed: {len(processed_indices)}")
print(f"Missing rows to retry: {len(missing)}")
print(f"Model: {MODEL}")
print(f"Output: {OUTPUT_JSONL}\n")

if not missing:
    print("Nothing to retry — all rows are already processed!")
    sys.exit(0)

# ── Retry loop ──────────────────────────────────────────────────────────────
success_count = 0
error_count = 0

for i, (idx, row) in enumerate(missing):
    text = row["text"]
    ground_truth = row["ground_truth_urgency"]
    predicted = row["predicted_urgency"]

    user_message = (
        f"Context: {text}\n"
        f"Reference: {ground_truth}\n"
        f"Incorrect Prediction: {predicted}"
    )

    try:
        response = None
        max_retries = 5
        for attempt in range(max_retries):
            try:
                response = client.models.generate_content(
                    model=MODEL,
                    contents=user_message,
                    config=genai.types.GenerateContentConfig(
                        system_instruction=STAGE_1_SYSTEM_PROMPT,
                        response_mime_type="application/json",
                        max_output_tokens=4096,
                    ),
                )
                break
            except Exception as retry_err:
                err_str = str(retry_err).lower()
                is_transient = any(kw in err_str for kw in ["429", "rate", "quota", "overloaded", "unavailable", "503", "500"])
                if is_transient and attempt < max_retries - 1:
                    wait = 8 * (attempt + 1)
                    print(f"  [retry {i+1}/{len(missing)}, row {idx}] ... Retry {attempt+1}/{max_retries} in {wait}s ({type(retry_err).__name__})")
                    time.sleep(wait)
                elif is_transient:
                    raise Exception(f"All {max_retries} retries failed: {retry_err}")
                else:
                    raise

        if response is None:
            raise Exception(f"All {max_retries} retries failed")

        raw_text = response.text.strip()

        if raw_text.startswith("```"):
            raw_text = raw_text.split("```")[1]
            if raw_text.startswith("json"):
                raw_text = raw_text[4:]
            raw_text = raw_text.strip()

        parsed = json.loads(raw_text)
        final_answer = parsed.get("final_answer", parsed)
        error_title = final_answer.get("error_title", "UNKNOWN")
        error_summary = final_answer.get("error_summary", "UNKNOWN")

        record = {
            "row_index": idx,
            "text": text,
            "ground_truth_urgency": ground_truth,
            "predicted_urgency": predicted,
            "error_title": error_title,
            "error_summary": error_summary,
        }

        with open(OUTPUT_JSONL, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

        success_count += 1
        print(f"  [{i+1}/{len(missing)}] row {idx} -> {error_title}")

    except json.JSONDecodeError as e:
        error_count += 1
        print(f"  [{i+1}/{len(missing)}] row {idx} -> JSON parse error: {e}")

    except Exception as e:
        error_count += 1
        print(f"  [{i+1}/{len(missing)}] row {idx} -> ERROR: {e}")

    time.sleep(SLEEP_SECS)

# ── Summary ─────────────────────────────────────────────────────────────────
print(f"\n{'='*50}")
print(f"RETRY COMPLETE")
print(f"  Successful: {success_count}")
print(f"  Errors:     {error_count}")
print(f"  Total in JSONL: {len(processed_indices) + success_count}")
print(f"{'='*50}")
