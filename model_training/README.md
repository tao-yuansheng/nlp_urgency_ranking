# Model Training

Fine-tunes **DeBERTa-v3-base** (`microsoft/deberta-v3-base`) with two linear classification heads — one for urgency, one for emotion — on the synthetic telecoms complaints dataset.

---

## Architecture

```
DeBERTa-v3-base backbone
        │
      [CLS] token representation
      ┌────┴────┐
 urgency_head   emotion_head
 Linear(768→3)  Linear(768→3)
```

Both heads are trained jointly with a combined cross-entropy loss. The urgency head uses class weights `[1.0, 1.5, 1.2]` (Low / Medium / High) to compensate for the harder Medium class.

---

## Training Configuration

| Parameter | Value |
|---|---|
| Base model | `microsoft/deberta-v3-base` |
| Max sequence length | 192 tokens |
| Batch size | 16 |
| Learning rate | 2e-5 (AdamW) |
| LR schedule | Linear warmup (10%) + decay |
| Epochs (max) | 10 |
| Early stopping patience | 3 |
| Urgency class weights | [1.0, 1.5, 1.2] |
| Train / Val / Test split | 70 / 15 / 15 (stratified on urgency×emotion cell) |

---

## Usage

### Download the pre-trained model (recommended)

No training required — downloads directly from HuggingFace Hub:

```bash
python model_training/download_model.py
```

Model page: [yuansheng-tao/emotion_urgency_classifier](https://huggingface.co/yuansheng-tao/emotion_urgency_classifier)

### Train from scratch

Requires the dataset at `data/telecoms_complaints.csv`. See [data_generation/README.md](../data_generation/README.md) to generate it.

```bash
python model_training/train_deberta.py
```

The best model weights are saved to `model_training/model_output/` at the end of each run.

### Run the adversarial test

10 hand-crafted telecom edge cases designed to probe model weaknesses (sarcasm, cold formal language, urgency/emotion decoupling):

```bash
python model_training/adversarial_test.py
```

Results are saved as a timestamped `.txt` file in `model_training/`.

### Compare all models

Runs all three models against the same held-out test set in one command and saves a merged prediction CSV plus a metrics summary:

```bash
python model_training/compare_models.py
```

Output is saved to `model_training/results/`:
- `test_predictions_<timestamp>.csv` — every test complaint with true labels and each model's predictions side by side
- `metrics_summary_<timestamp>.json` — per-class and macro F1 for all three models

### Run a baseline model

```bash
python model_training/baseline_tfidf_lr.py    # TF-IDF + Logistic Regression
python model_training/baseline_sbert_lr.py    # Sentence-BERT (frozen) + Logistic Regression
```

---

## Results

### Across dataset versions

| Dataset | Model | Urgency Macro F1 | Emotion Macro F1 |
|---|---|---|---|
| v1 (GPT-4o-mini, batch=15) | DeBERTa-v3-base | 0.649 | 0.989 |
| v2 (GPT-4o-mini, batch=15) | DeBERTa-v3-base | 0.622 | 0.868 |
| **v3 (GPT-5-mini, batch=5)** | **DeBERTa-v3-base** | **0.801** | **0.852** |

> v1 emotion score (0.989) reflects an easier dataset with simpler emotional expression. v3 is harder and more realistic.

### Best run — per-class breakdown (v3 dataset)

| Head | Low F1 | Medium F1 | High F1 | Macro F1 |
|---|---|---|---|---|
| Urgency | 0.823 | 0.736 | 0.844 | **0.801** |
| Emotion | 0.891 | 0.825 | 0.841 | **0.852** |

### Baseline comparison (v3 dataset, test set)

| Model | Urgency Macro F1 | Emotion Macro F1 |
|---|---|---|
| TF-IDF + Logistic Regression | 0.819 | 0.764 |
| Sentence-BERT (frozen) + LR | 0.691 | 0.631 |
| **Fine-tuned DeBERTa-v3-base** | **0.801** | **0.852** |

Fine-tuning dominates on emotion (contextual understanding required). TF-IDF is competitive on urgency due to strong lexical signals from the scenario affinity map. Frozen Sentence-BERT underperforms both, confirming that general-purpose embeddings without task adaptation are insufficient.

### Adversarial test (v3 model)

**6 / 10 passed.** Remaining failures:

| # | Case | Failure reason |
|---|---|---|
| 1 | Sarcastic Praise | Sarcasm read as Low emotion — absent from training data |
| 4 | Legal Threat | Cold formal language predicted as Low emotion (1.00 confidence) |
| 9 | Ambiguous Broken | Brief vague complaint predicted Low/Low instead of Medium/Medium |
| 10 | Overreaction to Fix | Sarcasm not detected; resolved issue predicted as Medium urgency |

---

## Logs

Every training run appends to:

- `model_training/logs/run_<timestamp>.json` — full per-epoch history and test metrics
- `model_training/logs/summary.csv` — one row per run for easy comparison

Log files are prefixed `run_v1_`, `run_v2_`, or `run_v3_` to indicate which dataset version was used.
