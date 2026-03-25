"""TF-IDF + Logistic Regression baseline for urgency and emotion classification.

Uses the same data split (70/15/15, stratified on urgency x emotion cell)
and the same evaluation metrics as the DeBERTa fine-tuned model so results
are directly comparable.
"""

import json
import os
import sys
from datetime import datetime

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.model_selection import train_test_split

# ── Config ──────────────────────────────────────────────────────────────────
CSV_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "data", "telecoms_complaints.csv"
)
LABEL_MAP = {"Low": 0, "Medium": 1, "High": 2}
LABEL_NAMES = ["Low", "Medium", "High"]
OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))

# ── Data ────────────────────────────────────────────────────────────────────
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

# ── TF-IDF ──────────────────────────────────────────────────────────────────
vectorizer = TfidfVectorizer(
    max_features=20_000,
    ngram_range=(1, 2),
    sublinear_tf=True,
    min_df=2,
)

X_train = vectorizer.fit_transform(train_df["complaint_text"])
X_val = vectorizer.transform(val_df["complaint_text"])
X_test = vectorizer.transform(test_df["complaint_text"])

print(f"TF-IDF vocabulary size: {len(vectorizer.vocabulary_)}")

# ── Train two independent Logistic Regression models ────────────────────────
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

# ── Evaluate on validation set ──────────────────────────────────────────────
val_urg_preds = urgency_clf.predict(X_val)
val_emo_preds = emotion_clf.predict(X_val)

val_urg_f1 = f1_score(val_df["urgency_label"], val_urg_preds, average="macro", zero_division=0)
val_emo_f1 = f1_score(val_df["emotion_label"], val_emo_preds, average="macro", zero_division=0)

print(f"\nValidation — Urgency Macro F1: {val_urg_f1:.4f}")
print(f"Validation — Emotion Macro F1: {val_emo_f1:.4f}")

# ── Evaluate on test set ────────────────────────────────────────────────────
test_urg_preds = urgency_clf.predict(X_test)
test_emo_preds = emotion_clf.predict(X_test)

print("\n" + "=" * 60)
print("TEST RESULTS")
print("=" * 60)

urg_f1_per_class = f1_score(
    test_df["urgency_label"], test_urg_preds, average=None, zero_division=0
)
emo_f1_per_class = f1_score(
    test_df["emotion_label"], test_emo_preds, average=None, zero_division=0
)

print("\n--- Urgency Head ---")
for i, name in enumerate(LABEL_NAMES):
    print(f"  F1 [{name}]: {urg_f1_per_class[i]:.4f}")
urg_macro = f1_score(
    test_df["urgency_label"], test_urg_preds, average="macro", zero_division=0
)
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
emo_macro = f1_score(
    test_df["emotion_label"], test_emo_preds, average="macro", zero_division=0
)
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

# ── Save results ────────────────────────────────────────────────────────────
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

results = {
    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    "run_id": timestamp,
    "model": "TF-IDF + Logistic Regression",
    "tfidf_max_features": 20_000,
    "tfidf_ngram_range": [1, 2],
    "lr_C": 1.0,
    "lr_solver": "lbfgs",
    # Validation
    "val_urgency_macro_f1": round(float(val_urg_f1), 4),
    "val_emotion_macro_f1": round(float(val_emo_f1), 4),
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

results_path = os.path.join(OUTPUT_DIR, f"baseline_results_{timestamp}.json")
with open(results_path, "w") as f:
    json.dump(results, f, indent=2)

print(f"\nResults saved to '{results_path}'")
