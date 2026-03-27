import os
import json
import time
from google import genai
from dotenv import load_dotenv

# ── Config ──────────────────────────────────────────────────────────────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ENV_PATH = os.path.join(SCRIPT_DIR, "..", ".env")
IN_JSONL = os.path.join(SCRIPT_DIR, "data", "emotion_icp_pairs.jsonl")
OUT_JSONL = os.path.join(SCRIPT_DIR, "results", "emotion_contrastive_insights.jsonl")

# Load environment variables
load_dotenv(ENV_PATH)
API_KEY = os.getenv("GEMINI_API_KEY")
if not API_KEY:
    raise ValueError("GEMINI_API_KEY not found in .env file.")

client = genai.Client(api_key=API_KEY)

# ── Prompting ───────────────────────────────────────────────────────────────

ICP_SYSTEM_PROMPT = """
You are an expert NLP model debugger. Your task is to perform Contrastive Error Analysis to identify the specific textual triggers that cause a model to fail.

You will be provided with:
1. The True Label (intended emotion).
2. The Error Text (which the model incorrectly predicted).
3. The Correct Text (which the model correctly predicted).

Both texts share the same True Label and underlying context. The model succeeded on one and failed on the other. 

Your Instructions:
1. Do NOT guess why the model failed in isolation.
2. Compare the two texts. Identify the specific linguistic, structural, or contextual feature that is present in the Correct Text but missing, obscured, or distorted in the Error Text.
3. Determine the "Decision Boundary"—the exact mechanism that tipped the model off in the correct text but failed to trigger it in the error text (e.g., "The correct text used explicit angry phrasing ('this is unacceptable'); the error text relied on a passive-aggressive tone").

Output your analysis strictly in this JSON format:
{
  "linguistic_delta": "A concise description of the specific textual difference between the two texts regarding the emotional cue or tone.",
  "algorithmic_blindspot": "Based on this delta, what specific feature is the model's attention mechanism failing to capture in the error text?",
  "actionable_fix": "A 1-sentence recommendation on how to augment the training data to fix this (e.g., 'Add more examples where the frustration is implicitly stated rather than using explicit emotion words.')"
}
"""

def generate_insights():
    print("Loading ICP pairs...")
    if not os.path.exists(IN_JSONL):
        raise FileNotFoundError(f"Missing {IN_JSONL}")
        
    # Read existing insights to resume if interrupted
    processed_texts = set()
    if os.path.exists(OUT_JSONL):
        with open(OUT_JSONL, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    try:
                        obj = json.loads(line)
                        processed_texts.add(obj.get("error_text", ""))
                    except json.JSONDecodeError:
                        pass


    pairs = []
    with open(IN_JSONL, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                pairs.append(json.loads(line))

    to_process = [p for p in pairs if p["error_text"] not in processed_texts]
    print(f"Total pairs: {len(pairs)}. Already processed: {len(processed_texts)}. Remaining: {len(to_process)}")

    if not to_process:
        print("All pairs processed.")
        return

    os.makedirs(os.path.dirname(OUT_JSONL), exist_ok=True)

    success_count = 0
    fail_count = 0

    with open(OUT_JSONL, "a", encoding="utf-8") as f_out:
        for i, pair in enumerate(to_process, 1):
            true_label = pair["true_label"]
            error_text = pair["error_text"]
            error_pred = pair["error_prediction"]
            correct_text = pair["correct_text"]

            user_msg = (
                f"True Emotion Label: {true_label}\n\n"
                f"FAILED PREDICTION (Predicted as '{error_pred}'):\n"
                f"{error_text}\n\n"
                f"CORRECT PREDICTION (Predicted as '{true_label}'):\n"
                f"{correct_text}\n"
            )

            try:
                # Add retry backoff for rate limiting
                max_retries = 3
                for attempt in range(max_retries):
                    try:
                        response = client.models.generate_content(
                            model='gemini-2.5-flash',
                            contents=user_msg,
                            config=genai.types.GenerateContentConfig(
                                system_instruction=ICP_SYSTEM_PROMPT,
                                temperature=0.1,
                                max_output_tokens=4096,
                                response_mime_type="application/json",
                            )
                        )
                        break
                    except Exception as retry_err:
                        err_str = str(retry_err).lower()
                        if any(kw in err_str for kw in ["429", "rate", "quota", "overloaded", "unavailable", "503", "500"]) and attempt < max_retries - 1:
                            wait = 8 * (attempt + 1)
                            print(f"  [{i}/{len(to_process)}] Retry {attempt+1}/{max_retries} in {wait}s ({type(retry_err).__name__})")
                            time.sleep(wait)
                        else:
                            raise retry_err

                # Try to parse the LLM output as JSON to ensure validity before saving
                try:
                    insights_json = json.loads(response.text)
                except json.JSONDecodeError:
                    print(f"[{i}/{len(to_process)}] Error: Invalid JSON returned by model. Skipping.")
                    fail_count += 1
                    time.sleep(2)
                    continue

                # Merge original metadata with new insights
                combined_result = {**pair, "contrastive_analysis": insights_json}

                f_out.write(json.dumps(combined_result, ensure_ascii=False) + "\n")
                f_out.flush()
                
                success_count += 1
                print(f"[{i}/{len(to_process)}] Success. Saved insights for True Label: {true_label}")
                
            except Exception as e:
                print(f"[{i}/{len(to_process)}] API Error: {e}")
                fail_count += 1
            
            # Rate limiting
            time.sleep(2)

    print(f"\nFinished processing. Success: {success_count}, Failed: {fail_count}.")
    print(f"Insights appended to {OUT_JSONL}")

if __name__ == "__main__":
    generate_insights()
