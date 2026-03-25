import os
import csv
import json
import subprocess
import sys
from datetime import datetime

def install(pkg):
    subprocess.check_call([sys.executable, "-m", "pip", "install", pkg, "-q"])

for pkg in ["transformers", "torch", "pandas", "scikit-learn", "sentencepiece", "tqdm"]:
    try:
        __import__(pkg if pkg != "scikit-learn" else "sklearn")
    except ImportError:
        install(pkg)

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel, get_linear_schedule_with_warmup
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, confusion_matrix
from tqdm import tqdm

# ── Config ──────────────────────────────────────────────────────────────────
CSV_PATH      = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "telecoms_complaints.csv")
MODEL_NAME    = "microsoft/deberta-v3-base"
OUTPUT_DIR    = os.path.join(os.path.dirname(os.path.abspath(__file__)), "model_output")
MAX_LENGTH    = 192
BATCH_SIZE    = 16
LR            = 2e-5
EPOCHS        = 10
PATIENCE      = 3
LABEL_MAP     = {"Low": 0, "Medium": 1, "High": 2}
LABEL_NAMES   = ["Low", "Medium", "High"]
DEVICE        = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"Using device: {DEVICE}")

# ── Data ─────────────────────────────────────────────────────────────────────
df = pd.read_csv(CSV_PATH)
df["urgency_label"]  = df["intended_urgency"].map(LABEL_MAP)
df["emotion_label"]  = df["intended_emotion"].map(LABEL_MAP)

# Stratify on combined label key
df["strat_key"] = df["urgency_label"].astype(str) + "_" + df["emotion_label"].astype(str)

train_df, temp_df = train_test_split(df, test_size=0.30, stratify=df["strat_key"], random_state=42)
val_df,   test_df = train_test_split(temp_df, test_size=0.50, stratify=temp_df["strat_key"], random_state=42)

print(f"Train: {len(train_df)} | Val: {len(val_df)} | Test: {len(test_df)}")

# ── Tokeniser ────────────────────────────────────────────────────────────────
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

# ── Dataset ──────────────────────────────────────────────────────────────────
def tokenize_df(dataframe):
    """Pre-tokenize an entire dataframe once and return a ready Dataset."""
    print(f"  Tokenizing {len(dataframe)} samples...", flush=True)
    enc = tokenizer(
        dataframe["complaint_text"].tolist(),
        max_length=MAX_LENGTH,
        padding="max_length",
        truncation=True,
        return_tensors="pt",
    )
    n = len(dataframe)
    token_type_ids = enc.get("token_type_ids", torch.zeros(n, MAX_LENGTH, dtype=torch.long))
    return torch.utils.data.TensorDataset(
        enc["input_ids"],
        enc["attention_mask"],
        token_type_ids,
        torch.tensor(dataframe["urgency_label"].tolist(), dtype=torch.long),
        torch.tensor(dataframe["emotion_label"].tolist(),  dtype=torch.long),
    )

print("Pre-tokenizing datasets...")
train_ds = tokenize_df(train_df)
val_ds   = tokenize_df(val_df)
test_ds  = tokenize_df(test_df)
print("Tokenization complete.", flush=True)

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,  num_workers=0)
val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
test_loader  = DataLoader(test_ds,  batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

# ── Model ────────────────────────────────────────────────────────────────────
class DeBERTaMultiHead(nn.Module):
    def __init__(self, model_name, num_classes=3):
        super().__init__()
        self.backbone      = AutoModel.from_pretrained(model_name)
        hidden_size        = self.backbone.config.hidden_size
        self.urgency_head  = nn.Linear(hidden_size, num_classes)
        self.emotion_head  = nn.Linear(hidden_size, num_classes)

    def forward(self, input_ids, attention_mask, token_type_ids=None):
        outputs = self.backbone(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids if token_type_ids is not None else None,
        )
        # [CLS] token representation
        cls_output = outputs.last_hidden_state[:, 0, :]
        return self.urgency_head(cls_output), self.emotion_head(cls_output)

model = DeBERTaMultiHead(MODEL_NAME).to(DEVICE)

# Class-weighted loss for urgency.
# Frequency-based weights alone downweight Medium (most frequent) despite it being
# the structurally hardest class (overlaps with both Low and High scenarios).
# Manual weights override this: Medium is boosted above its frequency-based value.
# Low=1.0, Medium=1.5, High=1.2  — tune Medium upward if it stays underperforming.
urgency_weights = torch.tensor([1.0, 1.5, 1.2], dtype=torch.float).to(DEVICE)
print(f"Urgency class weights: {urgency_weights.cpu().tolist()}")

urg_criterion = nn.CrossEntropyLoss(weight=urgency_weights)
emo_criterion = nn.CrossEntropyLoss()

optimizer  = torch.optim.AdamW(model.parameters(), lr=LR)
total_steps = len(train_loader) * EPOCHS
warmup_steps = int(0.1 * total_steps)   # 10% warmup
scheduler  = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps)
scaler     = torch.amp.GradScaler("cuda", enabled=DEVICE.type == "cuda")

# ── Helpers ──────────────────────────────────────────────────────────────────
def run_epoch(loader, train=True):
    model.train() if train else model.eval()
    total_loss = 0.0
    all_urg_preds, all_urg_labels = [], []
    all_emo_preds, all_emo_labels = [], []

    ctx = torch.enable_grad() if train else torch.no_grad()
    with ctx:
        for batch in tqdm(loader, desc="train" if train else "eval ", leave=False):
            input_ids, attention_mask, token_type_ids, urg_labels, emo_labels = batch
            input_ids      = input_ids.to(DEVICE)
            attention_mask = attention_mask.to(DEVICE)
            token_type_ids = token_type_ids.to(DEVICE)
            urg_labels     = urg_labels.to(DEVICE)
            emo_labels     = emo_labels.to(DEVICE)

            with torch.amp.autocast("cuda", enabled=DEVICE.type == "cuda"):
                urg_logits, emo_logits = model(input_ids, attention_mask, token_type_ids)
                loss = urg_criterion(urg_logits, urg_labels) + emo_criterion(emo_logits, emo_labels)

            if train:
                optimizer.zero_grad()
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scale_before = scaler.get_scale()
                scaler.step(optimizer)
                scaler.update()
                if scaler.get_scale() == scale_before:  # optimizer stepped (no inf/nan)
                    scheduler.step()

            total_loss += loss.item()
            all_urg_preds.extend(urg_logits.argmax(dim=-1).cpu().tolist())
            all_urg_labels.extend(urg_labels.cpu().tolist())
            all_emo_preds.extend(emo_logits.argmax(dim=-1).cpu().tolist())
            all_emo_labels.extend(emo_labels.cpu().tolist())

    avg_loss  = total_loss / len(loader)
    urg_f1    = f1_score(all_urg_labels, all_urg_preds, average="macro", zero_division=0)
    emo_f1    = f1_score(all_emo_labels, all_emo_preds, average="macro", zero_division=0)
    return avg_loss, urg_f1, emo_f1

# ── Training loop ─────────────────────────────────────────────────────────────
best_val_urg_f1  = -1.0       # early stopping now tracks urgency F1 (higher = better)
best_val_loss    = float("inf")
patience_counter = 0
best_model_state = None
best_epoch       = 1
best_val_emo_f1  = 0.0
epoch_history    = []

for epoch in range(1, EPOCHS + 1):
    train_loss, train_urg_f1, train_emo_f1 = run_epoch(train_loader, train=True)
    val_loss,   val_urg_f1,   val_emo_f1   = run_epoch(val_loader,   train=False)

    epoch_history.append({
        "epoch": epoch,
        "train_loss": round(train_loss, 4),
        "val_loss":   round(val_loss,   4),
        "val_urgency_f1": round(val_urg_f1, 4),
        "val_emotion_f1": round(val_emo_f1, 4),
    })

    print(
        f"Epoch {epoch}/{EPOCHS} | "
        f"Train Loss: {train_loss:.4f} | "
        f"Val Loss: {val_loss:.4f} | "
        f"Val Urgency F1: {val_urg_f1:.4f} | "
        f"Val Emotion F1: {val_emo_f1:.4f}"
    )

    if val_urg_f1 > best_val_urg_f1:
        best_val_urg_f1  = val_urg_f1
        best_val_loss    = val_loss
        best_epoch       = epoch
        best_val_emo_f1  = val_emo_f1
        patience_counter = 0
        best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        print(f"  -> New best model saved (val urgency F1 {best_val_urg_f1:.4f})")
    else:
        patience_counter += 1
        print(f"  -> No improvement. Patience {patience_counter}/{PATIENCE}")
        if patience_counter >= PATIENCE:
            print("Early stopping triggered.")
            break

# Restore best weights
model.load_state_dict(best_model_state)

# ── Test evaluation ───────────────────────────────────────────────────────────
model.eval()
all_urg_preds, all_urg_labels = [], []
all_emo_preds, all_emo_labels = [], []

with torch.no_grad():
    for batch in test_loader:
        input_ids, attention_mask, token_type_ids, urg_labels, emo_labels = batch
        input_ids      = input_ids.to(DEVICE)
        attention_mask = attention_mask.to(DEVICE)
        token_type_ids = token_type_ids.to(DEVICE)

        urg_logits, emo_logits = model(input_ids, attention_mask, token_type_ids)
        all_urg_preds.extend(urg_logits.argmax(dim=-1).cpu().tolist())
        all_urg_labels.extend(urg_labels.tolist())
        all_emo_preds.extend(emo_logits.argmax(dim=-1).cpu().tolist())
        all_emo_labels.extend(emo_labels.tolist())

print("\n" + "="*60)
print("TEST RESULTS")
print("="*60)

urg_f1_per_class = f1_score(all_urg_labels, all_urg_preds, average=None, zero_division=0)
emo_f1_per_class = f1_score(all_emo_labels, all_emo_preds, average=None, zero_division=0)

print("\n--- Urgency Head ---")
for i, name in enumerate(LABEL_NAMES):
    print(f"  F1 [{name}]: {urg_f1_per_class[i]:.4f}")
print(f"  Macro F1: {f1_score(all_urg_labels, all_urg_preds, average='macro', zero_division=0):.4f}")
print("\nConfusion Matrix (Urgency) — rows=true, cols=pred:")
print(pd.DataFrame(
    confusion_matrix(all_urg_labels, all_urg_preds, labels=[0,1,2]),
    index=LABEL_NAMES, columns=LABEL_NAMES
).to_string())

print("\n--- Emotion Head ---")
for i, name in enumerate(LABEL_NAMES):
    print(f"  F1 [{name}]: {emo_f1_per_class[i]:.4f}")
print(f"  Macro F1: {f1_score(all_emo_labels, all_emo_preds, average='macro', zero_division=0):.4f}")
print("\nConfusion Matrix (Emotion) — rows=true, cols=pred:")
print(pd.DataFrame(
    confusion_matrix(all_emo_labels, all_emo_preds, labels=[0,1,2]),
    index=LABEL_NAMES, columns=LABEL_NAMES
).to_string())

# ── Save ──────────────────────────────────────────────────────────────────────
os.makedirs(OUTPUT_DIR, exist_ok=True)
tokenizer.save_pretrained(OUTPUT_DIR)
torch.save(model.state_dict(), os.path.join(OUTPUT_DIR, "model_weights.pt"))

# Also save the backbone config so the model can be reconstructed
model.backbone.config.save_pretrained(OUTPUT_DIR)

print(f"\nModel and tokenizer saved to '{OUTPUT_DIR}/'")

# ── Experiment log ────────────────────────────────────────────────────────────
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
logs_dir  = "logs"
os.makedirs(logs_dir, exist_ok=True)

urg_macro = round(float(f1_score(all_urg_labels, all_urg_preds, average="macro", zero_division=0)), 4)
emo_macro = round(float(f1_score(all_emo_labels, all_emo_preds, average="macro", zero_division=0)), 4)

log_entry = {
    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    "run_id": timestamp,
    "notes": "",                          # fill in manually after each run
    # Hyperparameters
    "model": MODEL_NAME,
    "max_length": MAX_LENGTH,
    "batch_size": BATCH_SIZE,
    "lr": LR,
    "best_epoch": best_epoch,
    "epochs_max": EPOCHS,
    "early_stopping_patience": PATIENCE,
    "optimizer": "AdamW",
    # Validation at best epoch
    "best_val_loss":       round(best_val_loss,   4),
    "best_val_urgency_f1": round(best_val_urg_f1, 4),
    "best_val_emotion_f1": round(best_val_emo_f1, 4),
    # Test results — urgency
    "test_urgency_macro_f1":   urg_macro,
    "test_urgency_f1_low":     round(float(urg_f1_per_class[0]), 4),
    "test_urgency_f1_medium":  round(float(urg_f1_per_class[1]), 4),
    "test_urgency_f1_high":    round(float(urg_f1_per_class[2]), 4),
    # Test results — emotion
    "test_emotion_macro_f1":   emo_macro,
    "test_emotion_f1_low":     round(float(emo_f1_per_class[0]), 4),
    "test_emotion_f1_medium":  round(float(emo_f1_per_class[1]), 4),
    "test_emotion_f1_high":    round(float(emo_f1_per_class[2]), 4),
    # Full epoch-by-epoch history
    "epoch_history": epoch_history,
}

# Per-run detailed JSON file
run_log_path = os.path.join(logs_dir, f"run_{timestamp}.json")
with open(run_log_path, "w") as f:
    json.dump(log_entry, f, indent=2)

# Rolling summary CSV (one row per run, easy to compare)
summary_path = os.path.join(logs_dir, "summary.csv")
summary_fields = [
    "timestamp", "run_id", "notes", "model", "max_length", "batch_size", "lr",
    "best_epoch", "best_val_loss", "best_val_urgency_f1", "best_val_emotion_f1",
    "test_urgency_macro_f1", "test_urgency_f1_low", "test_urgency_f1_medium", "test_urgency_f1_high",
    "test_emotion_macro_f1", "test_emotion_f1_low",  "test_emotion_f1_medium",  "test_emotion_f1_high",
]
write_header = not os.path.exists(summary_path)
with open(summary_path, "a", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=summary_fields, extrasaction="ignore")
    if write_header:
        writer.writeheader()
    writer.writerow(log_entry)

print(f"\nRun log saved to '{run_log_path}'")
print(f"Summary updated at '{summary_path}'")
