"""Sentence-BERT (frozen) + Logistic Regression baseline.

Uses all-MiniLM-L6-v2 as a frozen sentence encoder — no fine-tuning.
The 384-dim embeddings feed into two independent Logistic Regression
classifiers (urgency and emotion), identical in structure to train_baseline.py.

Same 70/15/15 stratified split and evaluation metrics as the DeBERTa model.
First run downloads all-MiniLM-L6-v2 (~90 MB) from HuggingFace automatically.
"""

import json
import os
import sys
from datetime import datetime

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from sklearn.model_selection import train_test_split

try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "sentence-transformers", "-q"])
    from sentence_transformers import SentenceTransformer

# ── Config ───────────────────────────────────────────────────────────────────
CSV_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "data", "telecoms_complaints.csv"
)
SBERT_MODEL  = "all-MiniLM-L6-v2"   # 384-dim, ~90 MB download
LABEL_MAP    = {"Low": 0, "Medium": 1, "High": 2}
LABEL_NAMES  = ["Low", "Medium", "High"]
OUTPUT_DIR   = os.path.dirname(os.path.abspath(__file__))

# ── Data ─────────────────────────────────────────────────────────────────────
df = pd.read_csv(CSV_PATH)
df["urgency_label"] = df["intended_urgency"].map(LABEL_MAP)
df["emotion_label"] = df["intended_emotion"].map(LABEL_MAP)

# Identical split to train.py
df["strat_key"] = df["urgency_label"].astype(str) + "_" + df["emotion_label"].astype(str)
train_df, temp_df = train_test_split(
    df, test_size=0.30, stratify=df["strat_key"], random_state=42
)
val_df, test_df = train_test_split(
    temp_df, test_size=0.50, stratify=temp_df["strat_key"], random_state=42
)

print(f"Train: {len(train_df)} | Val: {len(val_df)} | Test: {len(test_df)}")

# ── Sentence-BERT embeddings (frozen — no fine-tuning) ───────────────────────
print(f"\nLoading '{SBERT_MODEL}' (downloads ~90 MB on first run)...")
sbert = SentenceTransformer(SBERT_MODEL)

print("Encoding train set...")
X_train = sbert.encode(train_df["complaint_text"].tolist(), show_progress_bar=True, batch_size=64)
print("Encoding val set...")
X_val   = sbert.encode(val_df["complaint_text"].tolist(),   show_progress_bar=True, batch_size=64)
print("Encoding test set...")
X_test  = sbert.encode(test_df["complaint_text"].tolist(),  show_progress_bar=True, batch_size=64)

print(f"Embedding shape: {X_train.shape}  (n_samples × {X_train.shape[1]}-dim)")

# ── Train two independent Logistic Regression classifiers ────────────────────
lr_params = {
    "max_iter": 1000,
    "solver": "lbfgs",
    "C": 1.0,
    "random_state": 42,
}

print("\nTraining urgency classifier...")
urgency_clf = LogisticRegression(**lr_params)
urgency_clf.fit(X_train, train_df["urgency_label"])

print("Training emotion classifier...")
emotion_clf = LogisticRegression(**lr_params)
emotion_clf.fit(X_train, train_df["emotion_label"])

# ── Evaluate on validation set ───────────────────────────────────────────────
val_urg_preds = urgency_clf.predict(X_val)
val_emo_preds = emotion_clf.predict(X_val)

val_urg_f1 = f1_score(val_df["urgency_label"], val_urg_preds, average="macro", zero_division=0)
val_emo_f1 = f1_score(val_df["emotion_label"], val_emo_preds, average="macro", zero_division=0)

print(f"\nValidation — Urgency Macro F1: {val_urg_f1:.4f}")
print(f"Validation — Emotion Macro F1: {val_emo_f1:.4f}")

# ── Evaluate on test set ─────────────────────────────────────────────────────
test_urg_preds = urgency_clf.predict(X_test)
test_emo_preds = emotion_clf.predict(X_test)

print("\n" + "=" * 60)
print("TEST RESULTS")
print("=" * 60)

urg_f1_per_class = f1_score(test_df["urgency_label"], test_urg_preds, average=None, zero_division=0)
emo_f1_per_class = f1_score(test_df["emotion_label"], test_emo_preds, average=None, zero_division=0)

print("\n--- Urgency Head ---")
for i, name in enumerate(LABEL_NAMES):
    print(f"  F1 [{name}]: {urg_f1_per_class[i]:.4f}")
urg_macro = f1_score(test_df["urgency_label"], test_urg_preds, average="macro", zero_division=0)
print(f"  Macro F1: {urg_macro:.4f}")

print("\nConfusion Matrix (Urgency) — rows=true, cols=pred:")
print(
    pd.DataFrame(
        confusion_matrix(test_df["urgency_label"], test_urg_preds, labels=[0, 1, 2]),
        index=LABEL_NAMES,
        columns=LABEL_NAMES,
    ).to_string()
)

print("\n--- Emotion Head ---")
for i, name in enumerate(LABEL_NAMES):
    print(f"  F1 [{name}]: {emo_f1_per_class[i]:.4f}")
emo_macro = f1_score(test_df["emotion_label"], test_emo_preds, average="macro", zero_division=0)
print(f"  Macro F1: {emo_macro:.4f}")

print("\nConfusion Matrix (Emotion) — rows=true, cols=pred:")
print(
    pd.DataFrame(
        confusion_matrix(test_df["emotion_label"], test_emo_preds, labels=[0, 1, 2]),
        index=LABEL_NAMES,
        columns=LABEL_NAMES,
    ).to_string()
)

print("\n--- Classification Reports ---")
print("\nUrgency:")
print(classification_report(
    test_df["urgency_label"], test_urg_preds,
    target_names=LABEL_NAMES, zero_division=0,
))
print("Emotion:")
print(classification_report(
    test_df["emotion_label"], test_emo_preds,
    target_names=LABEL_NAMES, zero_division=0,
))

# ── Save results ─────────────────────────────────────────────────────────────
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

results = {
    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    "run_id": timestamp,
    "model": f"Sentence-BERT ({SBERT_MODEL}) + Logistic Regression",
    "sbert_model": SBERT_MODEL,
    "embedding_dim": int(X_train.shape[1]),
    "lr_C": 1.0,
    "lr_solver": "lbfgs",
    # Validation
    "val_urgency_macro_f1": round(float(val_urg_f1), 4),
    "val_emotion_macro_f1": round(float(val_emo_f1), 4),
    # Test — urgency
    "test_urgency_macro_f1": round(float(urg_macro), 4),
    "test_urgency_f1_low":   round(float(urg_f1_per_class[0]), 4),
    "test_urgency_f1_medium": round(float(urg_f1_per_class[1]), 4),
    "test_urgency_f1_high":  round(float(urg_f1_per_class[2]), 4),
    # Test — emotion
    "test_emotion_macro_f1": round(float(emo_macro), 4),
    "test_emotion_f1_low":   round(float(emo_f1_per_class[0]), 4),
    "test_emotion_f1_medium": round(float(emo_f1_per_class[1]), 4),
    "test_emotion_f1_high":  round(float(emo_f1_per_class[2]), 4),
}

results_path = os.path.join(OUTPUT_DIR, f"sbert_baseline_results_{timestamp}.json")
with open(results_path, "w") as f:
    json.dump(results, f, indent=2)

print(f"\nResults saved to '{results_path}'")
