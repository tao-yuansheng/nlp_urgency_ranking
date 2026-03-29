"""
plot_deberta_errors.py
Generates a grouped bar chart of DeBERTa misclassification error types
for Urgency and Emotion in a single panel, using an academic style (Times New Roman).
"""

import os
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from itertools import product

# ── Paths ────────────────────────────────────────────────────────────────────
SCRIPT_DIR  = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(SCRIPT_DIR)
CSV_PATH    = os.path.join(
    PROJECT_DIR, "model_training", "results", "test_predictions_20260327_121322.csv"
)
OUT_PATH    = os.path.join(SCRIPT_DIR, "deberta_error_distribution.png")

# ── Load data ─────────────────────────────────────────────────────────────────
df = pd.read_csv(CSV_PATH)

LABELS = ["Low", "Medium", "High"]

def get_error_counts(true_col: str, pred_col: str) -> dict:
    """Return a dict of 'True→Pred' counts for all misclassification pairs."""
    errors = df[df[true_col] != df[pred_col]]
    counts = {}
    for true_lbl, pred_lbl in product(LABELS, LABELS):
        if true_lbl == pred_lbl:
            continue
        key = f"{true_lbl}→{pred_lbl}"
        counts[key] = int(((errors[true_col] == true_lbl) & (errors[pred_col] == pred_lbl)).sum())
    # Keep only error types that actually occurred
    return {k: v for k, v in counts.items() if v > 0}

urgency_errors = get_error_counts("intended_urgency", "deberta_urgency_pred")
emotion_errors = get_error_counts("intended_emotion", "deberta_emotion_pred")

# ── Consistent x-axis: all possible misclassification pairs ──────────────────
all_keys = [
    f"{t}→{p}"
    for t, p in product(LABELS, LABELS)
    if t != p
]
# Only include keys that appear in at least one of the two tasks
visible_keys = [k for k in all_keys if urgency_errors.get(k, 0) > 0 or emotion_errors.get(k, 0) > 0]

urgency_vals = [urgency_errors.get(k, 0) for k in visible_keys]
emotion_vals = [emotion_errors.get(k, 0) for k in visible_keys]

# ── Matplotlib style ──────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family":      "serif",
    "font.serif":       ["Times New Roman", "Times", "DejaVu Serif"],
    "font.size":        13,
    "axes.titlesize":   15,
    "axes.labelsize":   13,
    "xtick.labelsize":  12,
    "ytick.labelsize":  12,
    "axes.spines.top":  False,
    "axes.spines.right":False,
    "axes.linewidth":   0.8,
    "grid.linewidth":   0.5,
    "grid.alpha":       0.4,
})

# ── Plot ──────────────────────────────────────────────────────────────────────
import numpy as np

COLOURS = ["#2c6fad", "#c0392b"]   # professional blue / red
BAR_WIDTH = 0.35
x = np.arange(len(visible_keys))

fig, ax = plt.subplots(figsize=(13, 5))

bars_u = ax.bar(x - BAR_WIDTH / 2, urgency_vals, BAR_WIDTH,
                label="Urgency", color=COLOURS[0], edgecolor="white", linewidth=0.5)
bars_e = ax.bar(x + BAR_WIDTH / 2, emotion_vals,  BAR_WIDTH,
                label="Emotion Intensity", color=COLOURS[1], edgecolor="white", linewidth=0.5)

# Value labels on top of each bar
for bar, v in list(zip(bars_u, urgency_vals)) + list(zip(bars_e, emotion_vals)):
    if v > 0:
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.4,
            str(v),
            ha="center", va="bottom",
            fontsize=11,
        )

ax.set_xticks(x)
ax.set_xticklabels(visible_keys, rotation=35, ha="right")
ax.set_xlabel("Error Type  (True Class → Predicted Class)", labelpad=8)
ax.set_ylabel("Error Count", labelpad=8)
ax.set_title("DeBERTa Misclassification Error Distribution", fontsize=15, fontweight="bold", pad=12)
ax.yaxis.set_major_locator(ticker.MaxNLocator(integer=True))
ax.grid(axis="y", linestyle="--")
ax.set_axisbelow(True)
ax.legend(frameon=True, framealpha=0.9, edgecolor="grey", fontsize=12)

fig.tight_layout()
fig.savefig(OUT_PATH, dpi=300, bbox_inches="tight")
print(f"Saved: {OUT_PATH}")
