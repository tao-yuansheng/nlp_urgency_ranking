"""Few-shot Claude Haiku 4.5 baseline for urgency and emotion classification.

Sends batches of 10 complaints per API call with 9 few-shot examples
(one per urgency x emotion cell, sampled from training set).  Uses the
exact same test split as train.py and the TF-IDF baseline so all three
models are evaluated on identical data.

Speed optimisation: batching 10 complaints per API call reduces total
calls from 750 to 75, cutting runtime from ~28 min to ~4 min at the
50 req/min rate limit.
"""

import json
import os
import subprocess
import sys
import time
from datetime import datetime

import pandas as pd
from dotenv import load_dotenv
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.model_selection import train_test_split

load_dotenv(os.path.join(os.path.dirname(__file__), "..", "..", ".env"))

# ── Config ──────────────────────────────────────────────────────────────────
CSV_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "..", "data", "telecoms_complaints.csv"
)
LABEL_MAP = {"Low": 0, "Medium": 1, "High": 2}
LABEL_NAMES = ["Low", "Medium", "High"]
OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL = "claude-haiku-4-5-20251001"
COMPLAINTS_PER_CALL = 10     # Complaints batched into each API request
API_CALLS_PER_BATCH = 45     
BATCH_PAUSE = 62.0           # Seconds between batches
MAX_RETRIES = 3
RETRY_DELAY = 10.0
FEW_SHOT_SEED = 42           # Fixed seed for reproducible example selection

# ── Data — identical split to train.py ──────────────────────────────────────
df = pd.read_csv(CSV_PATH)
df["urgency_label"] = df["intended_urgency"].map(LABEL_MAP)
df["emotion_label"] = df["intended_emotion"].map(LABEL_MAP)

df["strat_key"] = df["urgency_label"].astype(str) + "_" + df["emotion_label"].astype(str)
train_df, temp_df = train_test_split(
    df, test_size=0.30, stratify=df["strat_key"], random_state=42
)
val_df, test_df = train_test_split(
    temp_df, test_size=0.50, stratify=temp_df["strat_key"], random_state=42
)

print(f"Train: {len(train_df)} | Test: {len(test_df)}")

# ── Build few-shot examples from training set ───────────────────────────────
def build_few_shot_examples(training_data: pd.DataFrame, seed: int) -> str:
    """Sample 1 complaint per urgency x emotion cell (9 total) from training data."""
    examples = []
    for urg in LABEL_NAMES:
        for emo in LABEL_NAMES:
            cell = training_data[
                (training_data["intended_urgency"] == urg)
                & (training_data["intended_emotion"] == emo)
            ]
            sample = cell.sample(n=1, random_state=seed).iloc[0]
            text = sample["complaint_text"]
            # Truncate long complaints to keep prompt manageable
            if len(text) > 300:
                text = text[:297] + "..."
            examples.append(
                f'Complaint: "{text}"\n'
                f'Classification: {{"urgency": "{urg}", "emotion": "{emo}"}}'
            )
    return "\n\n".join(examples)


few_shot_block = build_few_shot_examples(train_df, FEW_SHOT_SEED)
print(f"Built 9 few-shot examples from training set (seed={FEW_SHOT_SEED})")

SYSTEM_PROMPT = (
    "You are a complaint classifier for a UK telecoms company. "
    "For each customer complaint, classify it on two independent dimensions:\n\n"
    "1. Urgency (how urgently the issue needs resolving):\n"
    "   - Low: Minor inconvenience with no immediate impact on essential services or finances.\n"
    "   - Medium: Noticeable disruption that requires attention within days.\n"
    "   - High: Severe, time-critical impact — complete loss of service, significant financial harm, or safety concern.\n\n"
    "2. Emotion (how emotionally the customer is writing):\n"
    "   - Low: Calm and factual, composed and matter-of-fact.\n"
    "   - Medium: Frustrated tone, visible dissatisfaction but not extreme.\n"
    "   - High: Strong dissatisfaction — cold controlled anger or explicit distress.\n\n"
    "IMPORTANT: Urgency and emotion are independent. A customer can describe a catastrophic "
    "outage in a calm tone (High urgency, Low emotion) or be furious about a trivial issue "
    "(Low urgency, High emotion). Judge each dimension separately.\n\n"
    "Here are 9 labelled examples, one for each urgency x emotion combination:\n\n"
    f"{few_shot_block}"
)

# ── Anthropic client setup ──────────────────────────────────────────────────
try:
    import anthropic
except ImportError:
    print("anthropic package not found — installing...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "anthropic", "-q"])
    import anthropic


def classify_batch(
    client: anthropic.Anthropic,
    texts: list[str],
    batch_label: str,
) -> list[dict]:
    """Classify a batch of complaints in a single API call, with retries."""
    n = len(texts)

    # Build numbered list of complaints
    numbered = "\n\n".join(
        f"[{i + 1}] {text}" for i, text in enumerate(texts)
    )
    user_prompt = (
        f"Classify each of the following {n} complaints. "
        f"Return a JSON object with a single key \"results\" containing "
        f"a list of exactly {n} objects, each with \"urgency\" and \"emotion\" keys.\n\n"
        f"{numbered}"
    )

    for attempt in range(1, MAX_RETRIES + 2):
        try:
            response = client.messages.create(
                model=MODEL,
                max_tokens=1024,
                system=SYSTEM_PROMPT,
                messages=[{"role": "user", "content": user_prompt}],
                temperature=0.0,
            )
            raw = response.content[0].text.strip()
            # Strip markdown code fences if present
            if raw.startswith("```"):
                raw = raw.split("\n", 1)[-1].rsplit("```", 1)[0].strip()
            parsed = json.loads(raw)

            results = parsed.get("results", parsed.get("classifications", []))
            if isinstance(parsed, list):
                results = parsed

            # Validate all results have valid labels
            validated = []
            for r in results[:n]:
                urg = r.get("urgency", "").strip()
                emo = r.get("emotion", "").strip()
                if urg in LABEL_NAMES and emo in LABEL_NAMES:
                    validated.append({"urgency": urg, "emotion": emo})
                else:
                    validated.append({"urgency": "Medium", "emotion": "Medium"})

            if len(validated) == n:
                return validated

            print(f"  {batch_label}: got {len(validated)}/{n} valid, attempt {attempt}")

        except json.JSONDecodeError:
            print(f"  {batch_label}: JSON parse error, attempt {attempt}")
        except Exception as e:
            print(f"  {batch_label}: {type(e).__name__}, attempt {attempt}")
            if "rate_limit" in str(e).lower() or "429" in str(e):
                time.sleep(RETRY_DELAY * attempt)

    # All retries exhausted — default to Medium/Medium for all
    print(f"  {batch_label}: retries exhausted, defaulting {n} to Medium/Medium")
    return [{"urgency": "Medium", "emotion": "Medium"}] * n


# ── Run classification ──────────────────────────────────────────────────────
test_texts = test_df["complaint_text"].tolist()
total = len(test_texts)

# Split into API-call-sized chunks (10 complaints each)
api_chunks: list[list[str]] = [
    test_texts[i:i + COMPLAINTS_PER_CALL]
    for i in range(0, total, COMPLAINTS_PER_CALL)
]
total_api_calls = len(api_chunks)
num_batches = (total_api_calls + API_CALLS_PER_BATCH - 1) // API_CALLS_PER_BATCH

print(f"\nClassifying {total} complaints with {MODEL} (few-shot)")
print(f"  {total_api_calls} API calls ({COMPLAINTS_PER_CALL} complaints each)")
print(f"  {num_batches} rate-limit batches, ~{BATCH_PAUSE:.0f}s pause between")
est_minutes = (num_batches * BATCH_PAUSE + total_api_calls * 3) / 60
print(f"  Estimated time: ~{est_minutes:.0f} minutes\n")

api_key = os.getenv("ANTHROPIC_API_KEY")
if not api_key:
    print("Error: ANTHROPIC_API_KEY not set. Add it to .env or environment.")
    sys.exit(1)

client = anthropic.Anthropic(api_key=api_key)
start = time.time()

results: list[dict] = []
for batch_idx in range(num_batches):
    chunk_start = batch_idx * API_CALLS_PER_BATCH
    chunk_end = min(chunk_start + API_CALLS_PER_BATCH, total_api_calls)
    batch_chunks = api_chunks[chunk_start:chunk_end]

    complaint_start = chunk_start * COMPLAINTS_PER_CALL + 1
    complaint_end = min(chunk_end * COMPLAINTS_PER_CALL, total)
    print(f"  Batch {batch_idx + 1}/{num_batches}: "
          f"{len(batch_chunks)} API calls "
          f"[complaints {complaint_start}–{complaint_end}]")

    for i, chunk in enumerate(batch_chunks):
        call_idx = chunk_start + i
        label = f"call {call_idx + 1}/{total_api_calls}"
        chunk_results = classify_batch(client, chunk, label)
        results.extend(chunk_results)

    elapsed_so_far = time.time() - start
    classified_so_far = len(results)
    print(f"    {classified_so_far}/{total} classified "
          f"({elapsed_so_far:.0f}s elapsed)")

    # Pause between batches to respect rate limit (skip after last)
    if batch_idx < num_batches - 1:
        print(f"    Waiting {BATCH_PAUSE:.0f}s for rate limit window...")
        time.sleep(BATCH_PAUSE)

elapsed = time.time() - start
print(f"\nDone in {elapsed:.1f}s")

# Count fallbacks
fallback_count = sum(
    1 for r, (_, row) in zip(results, test_df.iterrows())
    if r["urgency"] == "Medium" and r["emotion"] == "Medium"
    and row["intended_urgency"] != "Medium" and row["intended_emotion"] != "Medium"
)

# ── Map predictions to numeric labels ───────────────────────────────────────
test_urg_preds = [LABEL_MAP[r["urgency"]] for r in results]
test_emo_preds = [LABEL_MAP[r["emotion"]] for r in results]
test_urg_labels = test_df["urgency_label"].tolist()
test_emo_labels = test_df["emotion_label"].tolist()

# ── Evaluation ──────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("TEST RESULTS")
print("=" * 60)

urg_f1_per_class = f1_score(test_urg_labels, test_urg_preds, average=None, zero_division=0)
emo_f1_per_class = f1_score(test_emo_labels, test_emo_preds, average=None, zero_division=0)

print("\n--- Urgency ---")
for i, name in enumerate(LABEL_NAMES):
    print(f"  F1 [{name}]: {urg_f1_per_class[i]:.4f}")
urg_macro = f1_score(test_urg_labels, test_urg_preds, average="macro", zero_division=0)
print(f"  Macro F1: {urg_macro:.4f}")

print("\nConfusion Matrix (Urgency) — rows=true, cols=pred:")
print(
    pd.DataFrame(
        confusion_matrix(test_urg_labels, test_urg_preds, labels=[0, 1, 2]),
        index=LABEL_NAMES,
        columns=LABEL_NAMES,
    ).to_string()
)

print("\n--- Emotion ---")
for i, name in enumerate(LABEL_NAMES):
    print(f"  F1 [{name}]: {emo_f1_per_class[i]:.4f}")
emo_macro = f1_score(test_emo_labels, test_emo_preds, average="macro", zero_division=0)
print(f"  Macro F1: {emo_macro:.4f}")

print("\nConfusion Matrix (Emotion) — rows=true, cols=pred:")
print(
    pd.DataFrame(
        confusion_matrix(test_emo_labels, test_emo_preds, labels=[0, 1, 2]),
        index=LABEL_NAMES,
        columns=LABEL_NAMES,
    ).to_string()
)

print("\n--- Classification Reports ---")
print("\nUrgency:")
print(classification_report(
    test_urg_labels, test_urg_preds,
    target_names=LABEL_NAMES, zero_division=0,
))
print("Emotion:")
print(classification_report(
    test_emo_labels, test_emo_preds,
    target_names=LABEL_NAMES, zero_division=0,
))

# ── Save results ────────────────────────────────────────────────────────────
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

output = {
    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    "run_id": timestamp,
    "model": MODEL,
    "method": "few-shot (9 examples, 1 per urgency x emotion cell)",
    "few_shot_seed": FEW_SHOT_SEED,
    "complaints_per_api_call": COMPLAINTS_PER_CALL,
    "temperature": 0.0,
    "test_samples": total,
    "elapsed_seconds": round(elapsed, 1),
    "fallback_count": fallback_count,
    # Test — urgency
    "test_urgency_macro_f1": round(float(urg_macro), 4),
    "test_urgency_f1_low": round(float(urg_f1_per_class[0]), 4),
    "test_urgency_f1_medium": round(float(urg_f1_per_class[1]), 4),
    "test_urgency_f1_high": round(float(urg_f1_per_class[2]), 4),
    # Test — emotion
    "test_emotion_macro_f1": round(float(emo_macro), 4),
    "test_emotion_f1_low": round(float(emo_f1_per_class[0]), 4),
    "test_emotion_f1_medium": round(float(emo_f1_per_class[1]), 4),
    "test_emotion_f1_high": round(float(emo_f1_per_class[2]), 4),
}

results_path = os.path.join(OUTPUT_DIR, f"llm_baseline_results_{timestamp}.json")
with open(results_path, "w") as f:
    json.dump(output, f, indent=2)

# Also save per-sample predictions for analysis
predictions_df = test_df[["id", "complaint_text", "intended_urgency", "intended_emotion"]].copy()
predictions_df["predicted_urgency"] = [r["urgency"] for r in results]
predictions_df["predicted_emotion"] = [r["emotion"] for r in results]
predictions_df["urgency_correct"] = predictions_df["intended_urgency"] == predictions_df["predicted_urgency"]
predictions_df["emotion_correct"] = predictions_df["intended_emotion"] == predictions_df["predicted_emotion"]

preds_path = os.path.join(OUTPUT_DIR, f"llm_predictions_{timestamp}.csv")
predictions_df.to_csv(preds_path, index=False)

print(f"\nResults saved to '{results_path}'")
print(f"Per-sample predictions saved to '{preds_path}'")
