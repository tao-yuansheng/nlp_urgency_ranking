# Error Analysis — Logbook

## 2026-03-25

### Step 1 — Clone Repository (12:35)
- **Task:** Clone the `nlp_urgency_ranking` repo from GitHub.
- **Command:** `git clone https://github.com/tao-yuansheng/nlp_urgency_ranking.git`
- **Location:** `C:\Users\elenb\Downloads\NLP_Group_assignment\`
- **Result:** ✅ Successfully cloned (102 objects, 3.82 MB).

### Step 2 — Create Folder Structure (12:39)
- **Task:** Create three subdirectories inside `error_analysis/`.
- **Folders created:**
  - `error_analysis/data/`
  - `error_analysis/prompts/`
  - `error_analysis/results/`
- **Result:** ✅ All three folders created successfully.

### Step 3 — Create Logbook (12:58)
- **Task:** Create this logbook to track all steps going forward.
- **Result:** ✅ Logbook created at `error_analysis/logbook.md`.

---

> **Process note:** From this point on, every new task will follow this workflow:
> 1. Receive task from user.
> 2. Propose a plan and wait for confirmation.
> 3. Execute the plan step by step.
> 4. Log each step in this logbook.

### Step 4 — Investigate Evaluation Logic (12:59–13:02)
- **Task:** Find where F1 score and evaluation logic live in the repository.
- **Findings:**
  - **`model_training/train.py`** (lines 217–260): Test evaluation loop. Key variables: `all_urg_preds`, `all_urg_labels`, `all_emo_preds`, `all_emo_labels`. Original text in `test_df["complaint_text"]`.
  - **`model_training/adversarial_test.py`** (lines 124–176): 10 hand-crafted adversarial tests only — not suitable for systematic extraction.
  - Label mapping: `LABEL_NAMES = ["Low", "Medium", "High"]` (line 37).
  - `test_loader` uses `shuffle=False` (line 87), so prediction order aligns with `test_df`.
- **Result:** ✅ Identified correct location. Proposed plan approved by user.

### Step 5 — Modify train.py to Extract Incorrect Predictions (13:28)
- **Task:** Add code to `train.py` to filter and save incorrect urgency predictions.
- **Location:** After confusion matrix printout (line 260), added 18 lines.
- **What the code does:**
  1. Aligns `test_df` index with prediction lists.
  2. Creates a boolean mask where `predicted ≠ ground_truth` for urgency.
  3. Builds a DataFrame with columns: `text`, `ground_truth_urgency`, `predicted_urgency`.
  4. Saves to `error_analysis/data/incorrect_predictions.csv`.
- **Result:** ✅ Code added. CSV will be generated on next training run.

### Step 6 — Create Standalone Extraction Script (14:16–14:27)
- **Task:** Avoid 8+ hours of retraining by using a pre-trained model instead.
- **Actions:**
  - Scanned repo for `.pt`, `.pth`, `.safetensors`, `.bin` files — none found.
  - Discovered `download_model.py` which downloads fine-tuned weights from HuggingFace Hub (`yuansheng-tao/emotion_urgency_classifier`).
  - Downloaded model (738MB) to `model_output/`.
  - Created `extract_errors_only.py` — a standalone inference-only script using `DebertaV2Model` directly.
- **Result:** ✅ Script created and model downloaded.

### Step 7 — Run Extraction & Generate CSV (14:27)
- **Task:** Execute `extract_errors_only.py` to generate the incorrect predictions CSV.
- **Output:**
  - **154 incorrect urgency predictions** out of 750 test samples (20.5% error rate).
  - Saved to `error_analysis/data/incorrect_predictions.csv`.
  - Columns: `text`, `ground_truth_urgency`, `predicted_urgency`.
- **Result:** ✅ CSV generated successfully.

### Step 8 — Write Stage 1 ErrorMap Script (14:45)
- **Task:** Create `error_analysis/run_stage1.py` for Stage 1 of ErrorMap pipeline.
- **Details:**
  - Uses Anthropic Claude API (`claude-3-haiku-20240307`).
  - Loads `.env` for API key via `python-dotenv`.
  - Placeholder `STAGE_1_SYSTEM_PROMPT` for user's custom prompt.
  - Rate limited: `time.sleep(1.5)` per request (~40 RPM, within Tier 1 limit of 50).
  - Output: `error_analysis/results/stage1_errors.jsonl`.
  - Includes resume support (skips already-processed rows).
- **Result:** ✅ Script created. Switched from Anthropic → Gemini API (Claude models deprecated).

### Step 9 — Run Stage 1 ErrorMap Script (15:41–16:31)
- **Task:** Execute `run_stage1.py` to classify all 154 incorrect predictions.
- **Fixes applied during debugging:**
  - `.env` key mapping (`Claude_API_Key` → `ANTHROPIC_API_KEY`, then `Gemini_API_Key`)
  - Model selection: `claude-3-haiku` → `claude-3-5-haiku` → `claude-sonnet-4` (all deprecated/overloaded) → `gemini-2.5-flash` ✓
  - Increased `max_output_tokens` from 512 → 4096 (fixed JSON truncation)
  - Added 5-retry backoff for transient `ServerError` / `UNAVAILABLE`
- **Output:**
  - **133 records saved** to `error_analysis/results/stage1_errors.jsonl`
  - **21 rows skipped** (JSON parse errors / exhausted retries)
  - **127 unique error titles** identified
  - Top error: "Underestimation of Cumulative Urgency Indicators" (3x)
  - Runtime: ~50 minutes
- **Result:** ✅ Stage 1 complete.
