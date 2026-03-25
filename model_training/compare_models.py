"""Compare all three models on the held-out test set.

Runs in one command — no prior training needed for baselines, loads the
fine-tuned DeBERTa weights from model_output/.

Outputs (saved to model_training/results/):
  - test_predictions_<timestamp>.csv  — full test set with every model's predictions
  - metrics_summary_<timestamp>.json  — per-class and macro F1 for all three models
"""

import json
import os
import sys
from datetime import datetime

import pandas as pd
import torch
import torch.nn as nn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from transformers import AutoConfig, AutoModel, AutoTokenizer

try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "sentence-transformers", "-q"])
    from sentence_transformers import SentenceTransformer

# ── Config ───────────────────────────────────────────────────────────────────
CSV_PATH    = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "telecoms_complaints.csv")
MODEL_DIR   = os.path.join(os.path.dirname(os.path.abspath(__file__)), "model_output")
RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
LABEL_MAP   = {"Low": 0, "Medium": 1, "High": 2}
LABEL_NAMES = ["Low", "Medium", "High"]
MAX_LENGTH  = 192
DEVICE      = torch.device("cuda" if torch.cuda.is_available() else "cpu")

os.makedirs(RESULTS_DIR, exist_ok=True)

# ── Shared data split (identical to train_deberta.py) ────────────────────────
print("Loading dataset...")
df = pd.read_csv(CSV_PATH)
df["urgency_label"] = df["intended_urgency"].map(LABEL_MAP)
df["emotion_label"] = df["intended_emotion"].map(LABEL_MAP)
df["strat_key"] = df["urgency_label"].astype(str) + "_" + df["emotion_label"].astype(str)

train_df, temp_df = train_test_split(df, test_size=0.30, stratify=df["strat_key"], random_state=42)
val_df,   test_df = train_test_split(temp_df, test_size=0.50, stratify=temp_df["strat_key"], random_state=42)

print(f"Train: {len(train_df)} | Val: {len(val_df)} | Test: {len(test_df)}")

# Output dataframe — start with original test set columns
out_df = test_df[["complaint_text", "intended_urgency", "intended_emotion"]].copy().reset_index(drop=True)

metrics = {}  # will hold F1 results per model

# ═══════════════════════════════════════════════════════════════════════════════
# MODEL 1 — TF-IDF + Logistic Regression
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("MODEL 1: TF-IDF + Logistic Regression")
print("=" * 60)

vectorizer = TfidfVectorizer(max_features=20_000, ngram_range=(1, 2), sublinear_tf=True, min_df=2)
X_train_tfidf = vectorizer.fit_transform(train_df["complaint_text"])
X_test_tfidf  = vectorizer.transform(test_df["complaint_text"])

lr_params = {"max_iter": 1000, "solver": "lbfgs", "C": 1.0, "random_state": 42}

tfidf_urg_clf = LogisticRegression(**lr_params)
tfidf_urg_clf.fit(X_train_tfidf, train_df["urgency_label"])

tfidf_emo_clf = LogisticRegression(**lr_params)
tfidf_emo_clf.fit(X_train_tfidf, train_df["emotion_label"])

tfidf_urg_preds = tfidf_urg_clf.predict(X_test_tfidf)
tfidf_emo_preds = tfidf_emo_clf.predict(X_test_tfidf)

out_df["tfidf_urgency_pred"] = [LABEL_NAMES[p] for p in tfidf_urg_preds]
out_df["tfidf_emotion_pred"] = [LABEL_NAMES[p] for p in tfidf_emo_preds]

urg_f1 = f1_score(test_df["urgency_label"], tfidf_urg_preds, average=None, zero_division=0)
emo_f1 = f1_score(test_df["emotion_label"], tfidf_emo_preds, average=None, zero_division=0)
metrics["tfidf_lr"] = {
    "urgency_macro_f1":  round(float(f1_score(test_df["urgency_label"], tfidf_urg_preds, average="macro", zero_division=0)), 4),
    "urgency_f1_low":    round(float(urg_f1[0]), 4),
    "urgency_f1_medium": round(float(urg_f1[1]), 4),
    "urgency_f1_high":   round(float(urg_f1[2]), 4),
    "emotion_macro_f1":  round(float(f1_score(test_df["emotion_label"], tfidf_emo_preds, average="macro", zero_division=0)), 4),
    "emotion_f1_low":    round(float(emo_f1[0]), 4),
    "emotion_f1_medium": round(float(emo_f1[1]), 4),
    "emotion_f1_high":   round(float(emo_f1[2]), 4),
}
print(f"  Urgency Macro F1: {metrics['tfidf_lr']['urgency_macro_f1']:.4f}")
print(f"  Emotion Macro F1: {metrics['tfidf_lr']['emotion_macro_f1']:.4f}")

# ═══════════════════════════════════════════════════════════════════════════════
# MODEL 2 — Sentence-BERT (frozen) + Logistic Regression
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("MODEL 2: Sentence-BERT + Logistic Regression")
print("=" * 60)

sbert = SentenceTransformer("all-MiniLM-L6-v2")
print("Encoding train set...")
X_train_sbert = sbert.encode(train_df["complaint_text"].tolist(), show_progress_bar=True, batch_size=64)
print("Encoding test set...")
X_test_sbert  = sbert.encode(test_df["complaint_text"].tolist(),  show_progress_bar=True, batch_size=64)

sbert_urg_clf = LogisticRegression(**lr_params)
sbert_urg_clf.fit(X_train_sbert, train_df["urgency_label"])

sbert_emo_clf = LogisticRegression(**lr_params)
sbert_emo_clf.fit(X_train_sbert, train_df["emotion_label"])

sbert_urg_preds = sbert_urg_clf.predict(X_test_sbert)
sbert_emo_preds = sbert_emo_clf.predict(X_test_sbert)

out_df["sbert_urgency_pred"] = [LABEL_NAMES[p] for p in sbert_urg_preds]
out_df["sbert_emotion_pred"] = [LABEL_NAMES[p] for p in sbert_emo_preds]

urg_f1 = f1_score(test_df["urgency_label"], sbert_urg_preds, average=None, zero_division=0)
emo_f1 = f1_score(test_df["emotion_label"], sbert_emo_preds, average=None, zero_division=0)
metrics["sbert_lr"] = {
    "urgency_macro_f1":  round(float(f1_score(test_df["urgency_label"], sbert_urg_preds, average="macro", zero_division=0)), 4),
    "urgency_f1_low":    round(float(urg_f1[0]), 4),
    "urgency_f1_medium": round(float(urg_f1[1]), 4),
    "urgency_f1_high":   round(float(urg_f1[2]), 4),
    "emotion_macro_f1":  round(float(f1_score(test_df["emotion_label"], sbert_emo_preds, average="macro", zero_division=0)), 4),
    "emotion_f1_low":    round(float(emo_f1[0]), 4),
    "emotion_f1_medium": round(float(emo_f1[1]), 4),
    "emotion_f1_high":   round(float(emo_f1[2]), 4),
}
print(f"  Urgency Macro F1: {metrics['sbert_lr']['urgency_macro_f1']:.4f}")
print(f"  Emotion Macro F1: {metrics['sbert_lr']['emotion_macro_f1']:.4f}")

# ═══════════════════════════════════════════════════════════════════════════════
# MODEL 3 — Fine-tuned DeBERTa-v3-base (loads from model_output/)
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("MODEL 3: Fine-tuned DeBERTa-v3-base")
print("=" * 60)

class DeBERTaMultiHead(nn.Module):
    def __init__(self, model_dir, num_classes=3):
        super().__init__()
        config            = AutoConfig.from_pretrained(model_dir)
        self.backbone     = AutoModel.from_config(config)
        hidden_size       = config.hidden_size
        self.urgency_head = nn.Linear(hidden_size, num_classes)
        self.emotion_head = nn.Linear(hidden_size, num_classes)

    def forward(self, input_ids, attention_mask, token_type_ids=None):
        out = self.backbone(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        cls = out.last_hidden_state[:, 0, :]
        return self.urgency_head(cls), self.emotion_head(cls)

print(f"Loading model from '{MODEL_DIR}' ...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
deberta   = DeBERTaMultiHead(MODEL_DIR).to(DEVICE)
deberta.load_state_dict(torch.load(os.path.join(MODEL_DIR, "model_weights.pt"), map_location=DEVICE))
deberta.eval()
print(f"Running inference on {len(test_df)} test samples...")

deb_urg_preds, deb_emo_preds = [], []
texts = test_df["complaint_text"].tolist()
BATCH = 32

for i in range(0, len(texts), BATCH):
    batch_texts = texts[i: i + BATCH]
    enc = tokenizer(
        batch_texts,
        max_length=MAX_LENGTH,
        padding="max_length",
        truncation=True,
        return_tensors="pt",
    )
    input_ids      = enc["input_ids"].to(DEVICE)
    attention_mask = enc["attention_mask"].to(DEVICE)
    token_type_ids = enc.get("token_type_ids", torch.zeros_like(input_ids)).to(DEVICE)

    with torch.no_grad():
        urg_logits, emo_logits = deberta(input_ids, attention_mask, token_type_ids)

    deb_urg_preds.extend(urg_logits.argmax(dim=-1).cpu().tolist())
    deb_emo_preds.extend(emo_logits.argmax(dim=-1).cpu().tolist())

out_df["deberta_urgency_pred"] = [LABEL_NAMES[p] for p in deb_urg_preds]
out_df["deberta_emotion_pred"] = [LABEL_NAMES[p] for p in deb_emo_preds]

urg_true = test_df["urgency_label"].tolist()
emo_true = test_df["emotion_label"].tolist()
urg_f1 = f1_score(urg_true, deb_urg_preds, average=None, zero_division=0)
emo_f1 = f1_score(emo_true, deb_emo_preds, average=None, zero_division=0)
metrics["deberta_finetuned"] = {
    "urgency_macro_f1":  round(float(f1_score(urg_true, deb_urg_preds, average="macro", zero_division=0)), 4),
    "urgency_f1_low":    round(float(urg_f1[0]), 4),
    "urgency_f1_medium": round(float(urg_f1[1]), 4),
    "urgency_f1_high":   round(float(urg_f1[2]), 4),
    "emotion_macro_f1":  round(float(f1_score(emo_true, deb_emo_preds, average="macro", zero_division=0)), 4),
    "emotion_f1_low":    round(float(emo_f1[0]), 4),
    "emotion_f1_medium": round(float(emo_f1[1]), 4),
    "emotion_f1_high":   round(float(emo_f1[2]), 4),
}
print(f"  Urgency Macro F1: {metrics['deberta_finetuned']['urgency_macro_f1']:.4f}")
print(f"  Emotion Macro F1: {metrics['deberta_finetuned']['emotion_macro_f1']:.4f}")

# ═══════════════════════════════════════════════════════════════════════════════
# Summary table
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("COMPARISON SUMMARY")
print("=" * 60)
print(f"\n{'Model':<30} {'Urgency F1':>12} {'Emotion F1':>12}")
print("-" * 56)
labels = {
    "tfidf_lr":          "TF-IDF + LR",
    "sbert_lr":          "Sentence-BERT + LR",
    "deberta_finetuned": "Fine-tuned DeBERTa",
}
for key, label in labels.items():
    print(f"{label:<30} {metrics[key]['urgency_macro_f1']:>12.4f} {metrics[key]['emotion_macro_f1']:>12.4f}")

# ═══════════════════════════════════════════════════════════════════════════════
# Save outputs
# ═══════════════════════════════════════════════════════════════════════════════
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

csv_path = os.path.join(RESULTS_DIR, f"test_predictions_{timestamp}.csv")
out_df.to_csv(csv_path, index=False, encoding="utf-8-sig")
print(f"\nTest predictions saved to '{csv_path}'")

summary = {
    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    "test_set_size": len(test_df),
    "models": metrics,
}
json_path = os.path.join(RESULTS_DIR, f"metrics_summary_{timestamp}.json")
with open(json_path, "w") as f:
    json.dump(summary, f, indent=2)
print(f"Metrics summary saved to '{json_path}'")
