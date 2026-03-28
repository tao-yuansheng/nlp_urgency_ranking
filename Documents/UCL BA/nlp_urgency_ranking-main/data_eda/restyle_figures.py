"""
Regenerate all EDA figures with FT-style academic palette (Times New Roman, muted colours).
Reference style: 1_2_joint_distribution.png
"""

import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.ticker as mticker
import seaborn as sns
from pathlib import Path
from scipy.stats import chi2_contingency
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import chi2 as sklearn_chi2
from sklearn.preprocessing import LabelEncoder
from sklearn.manifold import TSNE
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics import silhouette_score

# ── Reproducibility ──────────────────────────────────────────────────────────
SEED = 42
np.random.seed(SEED)

# ── Paths ─────────────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = ROOT / "data" / "telecoms_complaints.csv"
FIG_DIR = Path(__file__).resolve().parent / "figures"
FIG_DIR.mkdir(exist_ok=True)

# ── Style constants ───────────────────────────────────────────────────────────
BLUE   = "#4878A8"
RED    = "#C05050"
GREEN  = "#5A9E6F"
GREY   = "#7A7A7A"
LGREY  = "#B0B0B0"

URGENCY_PAL   = {"Low": GREEN, "Medium": BLUE, "High": RED}
EMOTION_PAL   = {"Low": "#6B9BD2", "Medium": "#9B7CB8", "High": RED}
URGENCY_ORDER = ["Low", "Medium", "High"]
EMOTION_ORDER = ["Low", "Medium", "High"]

GRID_ALPHA = 0.25
GRID_COLOR = "#CCCCCC"
EDGE_COLOR = "#333333"
EDGE_LW    = 0.5

def apply_global_style():
    plt.rcParams.update({
        "figure.facecolor":   "white",
        "axes.facecolor":     "white",
        "font.family":        "serif",
        "font.serif":         ["Times New Roman"],
        "axes.titleweight":   "bold",
        "axes.titlesize":     11,
        "axes.labelsize":     10,
        "xtick.labelsize":    9,
        "ytick.labelsize":    9,
        "legend.fontsize":    9,
        "figure.dpi":         100,
        "savefig.dpi":        300,
        "axes.grid":          False,
        "grid.alpha":         GRID_ALPHA,
        "grid.color":         GRID_COLOR,
        "axes.spines.top":    False,
        "axes.spines.right":  False,
    })

def enable_ygrid(ax):
    """Light horizontal grid lines only — for bar charts, boxplots, histograms."""
    ax.yaxis.grid(True, alpha=GRID_ALPHA, color=GRID_COLOR, linewidth=0.5)
    ax.xaxis.grid(False)
    ax.set_axisbelow(True)

def enable_xgrid(ax):
    """Light vertical grid lines only — for horizontal bar charts."""
    ax.xaxis.grid(True, alpha=GRID_ALPHA, color=GRID_COLOR, linewidth=0.5)
    ax.yaxis.grid(False)
    ax.set_axisbelow(True)

def no_grid(ax):
    """Disable all grid lines — for heatmaps, scatter plots."""
    ax.grid(False)

def set_tnr(ax, title=None, xlabel=None, ylabel=None):
    """Apply Times New Roman to all text elements on an axes."""
    for item in ([ax.title, ax.xaxis.label, ax.yaxis.label]
                 + ax.get_xticklabels() + ax.get_yticklabels()):
        item.set_fontfamily("Times New Roman")
    if title:
        ax.set_title(title, fontfamily="Times New Roman",
                     fontweight="bold", fontsize=11)
    if xlabel:
        ax.set_xlabel(xlabel, fontfamily="Times New Roman")
    if ylabel:
        ax.set_ylabel(ylabel, fontfamily="Times New Roman")

def fig_title(fig, text, **kwargs):
    """Suptitle in the reference style."""
    kw = dict(fontsize=12, fontweight="bold",
              fontfamily="Times New Roman", y=1.02)
    kw.update(kwargs)
    fig.suptitle(text, **kw)


# ── Load data ─────────────────────────────────────────────────────────────────
print("Loading data …")
df = pd.read_csv(DATA_PATH)
print(f"  {df.shape[0]:,} rows × {df.shape[1]} columns")

# Derived columns (same logic as notebook Cell 26 / 30)
df["word_count"]       = df["complaint_text"].str.split().str.len()
df["char_count"]       = df["complaint_text"].str.len()
df["sentence_count"]   = df["complaint_text"].str.count(r"[.!?]+\s")
SUBWORD_RATIO          = 1.3
MAX_LENGTH             = 192
df["approx_token_count"] = (df["word_count"] * SUBWORD_RATIO + 2).astype(int)
df["truncated"]        = df["approx_token_count"] > MAX_LENGTH

apply_global_style()

# ─────────────────────────────────────────────────────────────────────────────
# CELL 5 → 1_1_class_distributions.png
# ─────────────────────────────────────────────────────────────────────────────
print("Generating 1_1 …")

fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

# Urgency
urg_counts = df["intended_urgency"].value_counts().reindex(URGENCY_ORDER)
urg_pct    = urg_counts / len(df) * 100
bars = axes[0].bar(URGENCY_ORDER, urg_counts,
                   color=[URGENCY_PAL[u] for u in URGENCY_ORDER],
                   edgecolor=EDGE_COLOR, linewidth=EDGE_LW)
for bar, pct in zip(bars, urg_pct):
    axes[0].text(bar.get_x() + bar.get_width() / 2,
                 bar.get_height() + 30,
                 f"{pct:.1f}%", ha="center", va="bottom",
                 fontweight="bold", fontfamily="Times New Roman", fontsize=9)
design_targets = {"Low": 0.35, "Medium": 0.40, "High": 0.25}
for level, target in design_targets.items():
    idx = URGENCY_ORDER.index(level)
    axes[0].hlines(target * len(df), idx - 0.4, idx + 0.4,
                   colors="black", linestyles="dashed",
                   linewidth=1.5,
                   label="Design target" if idx == 0 else "")
axes[0].legend(loc="upper right")
set_tnr(axes[0], title="Urgency Distribution",
        ylabel="Count", xlabel="")
enable_ygrid(axes[0])

# Emotion
emo_counts = df["intended_emotion"].value_counts().reindex(EMOTION_ORDER)
emo_pct    = emo_counts / len(df) * 100
bars = axes[1].bar(EMOTION_ORDER, emo_counts,
                   color=[EMOTION_PAL[e] for e in EMOTION_ORDER],
                   edgecolor=EDGE_COLOR, linewidth=EDGE_LW)
for bar, pct in zip(bars, emo_pct):
    axes[1].text(bar.get_x() + bar.get_width() / 2,
                 bar.get_height() + 30,
                 f"{pct:.1f}%", ha="center", va="bottom",
                 fontweight="bold", fontfamily="Times New Roman", fontsize=9)
axes[1].axhline(len(df) / 3, color="black", linestyle="dashed",
                linewidth=1.5, label="Design target (33.3%)")
axes[1].legend(loc="upper right")
set_tnr(axes[1], title="Emotion Distribution",
        ylabel="Count", xlabel="")
enable_ygrid(axes[1])

fig_title(fig, "Marginal Class Distributions vs. Design Targets")
fig.tight_layout()
fig.savefig(FIG_DIR / "1_1_class_distributions.png", bbox_inches="tight")
plt.close(fig)
print("  saved 1_1_class_distributions.png")


# ─────────────────────────────────────────────────────────────────────────────
# CELL 9 → 1_3_metadata_coverage.png
# ─────────────────────────────────────────────────────────────────────────────
print("Generating 1_3 …")

metadata_cols = ["scenario", "style", "profile", "history"]
# Muted sequential palette — same length as longest series; we'll slice per subplot
BASE_BLUE_PALETTE = [
    "#C6D9EE", "#A8C4DF", "#8BAFD0", "#6E9AC1",
    "#5085B2", "#3C72A0", "#2E5F8E", "#204D7C",
    "#163F6A", "#0D3158", "#082548",
]

fig, axes = plt.subplots(2, 2, figsize=(16, 12))
axes = axes.ravel()

for idx, col in enumerate(metadata_cols):
    counts       = df[col].value_counts().sort_values(ascending=True)
    short_labels = [s[:45] + "…" if len(s) > 48 else s for s in counts.index]
    n            = len(counts)
    # Use the last `n` entries of the palette (darkest → lightest bottom→top)
    palette      = BASE_BLUE_PALETTE[-n:] if n <= len(BASE_BLUE_PALETTE) \
                   else BASE_BLUE_PALETTE * (n // len(BASE_BLUE_PALETTE) + 1)
    palette      = palette[:n]

    bars = axes[idx].barh(range(len(counts)), counts.values,
                          color=palette,
                          edgecolor=EDGE_COLOR, linewidth=EDGE_LW)
    axes[idx].set_yticks(range(len(counts)))
    axes[idx].set_yticklabels(short_labels, fontsize=8,
                              fontfamily="Times New Roman")
    axes[idx].set_xlabel("Count", fontfamily="Times New Roman")
    set_tnr(axes[idx], title=f"{col.title()} Distribution")
    enable_xgrid(axes[idx])

    for bar, val in zip(bars, counts.values):
        axes[idx].text(val + 5, bar.get_y() + bar.get_height() / 2,
                       str(val), va="center", fontsize=8,
                       fontfamily="Times New Roman")

fig_title(fig, "Metadata Coverage Across All Dimensions", y=1.01)
fig.tight_layout()
fig.savefig(FIG_DIR / "1_3_metadata_coverage.png", bbox_inches="tight")
plt.close(fig)
print("  saved 1_3_metadata_coverage.png")


# ─────────────────────────────────────────────────────────────────────────────
# CELL 11 → 1_4_scenario_urgency_affinity.png
# ─────────────────────────────────────────────────────────────────────────────
print("Generating 1_4 …")

SCENARIO_URGENCY = {
    "Difficulty Cancelling Service":       ["Low", "Medium"],
    "Fraud & Scams":                       ["Medium", "High"],
    "Overcharging & Incorrect Billing":    ["Low", "Medium", "High"],
    "Poor Network Coverage":               ["Low", "Medium"],
    "3G Shutdown Impact":                  ["Medium", "High"],
    "Auto-Renewal Without Consent":        ["Low", "Medium"],
    "Billing After Cancellation":          ["Medium", "High"],
    "High Early Termination Fees":         ["Medium", "High"],
    "Ineffective AI / Chatbot Support":    ["Low", "Medium"],
    "Unfulfilled Fix Promises":            ["Medium", "High"],
    "Long Call-Waiting Times":             ["Low", "Medium"],
    "Wrong Sale Due to Agent Mistake":     ["Medium", "High"],
    "Loyalty Penalty":                     ["Low", "Medium"],
    "Mid-Contract Price Increase":         ["Low", "Medium"],
    "Complete Service Outage":             ["High"],
    "Faulty Hardware / Handset Issues":    ["Low", "Medium", "High"],
    "Hidden Fees & Charges":               ["Low", "Medium"],
    "Lack of Progress Updates":            ["Low", "Medium"],
    "Poor Complaint Handling":             ["Medium", "High"],
    "Slow Broadband Speeds":               ["Low", "Medium"],
}

ct = pd.crosstab(df["scenario"], df["intended_urgency"])
ct = ct.reindex(columns=URGENCY_ORDER, fill_value=0).sort_index()

forbidden_mask = pd.DataFrame(False, index=ct.index, columns=ct.columns)
for scenario in ct.index:
    allowed = SCENARIO_URGENCY.get(scenario, URGENCY_ORDER)
    for urg in URGENCY_ORDER:
        if urg not in allowed:
            forbidden_mask.loc[scenario, urg] = True

# Row-normalise
ct_pct = ct.div(ct.sum(axis=1), axis=0) * 100

fig, ax = plt.subplots(figsize=(8, 9))
ax.spines["top"].set_visible(True)
ax.spines["right"].set_visible(True)
no_grid(ax)

sns.heatmap(ct_pct, annot=True, fmt=".1f", cmap="Blues",
            linewidths=1.5, linecolor="white",
            cbar_kws={"label": "% within scenario"},
            ax=ax, vmin=0,
            annot_kws={"fontfamily": "Times New Roman", "fontsize": 9})

# Red X on forbidden cells — use plain "X" (Times New Roman safe)
for row_i, scenario in enumerate(ct.index):
    for col_j, urg in enumerate(URGENCY_ORDER):
        if forbidden_mask.loc[scenario, urg]:
            ax.text(col_j + 0.5, row_i + 0.5, "X",
                    ha="center", va="center",
                    fontsize=18, color=RED, fontweight="bold",
                    fontfamily="Times New Roman", zorder=10)

ax.set_title("Scenario x Urgency Affinity\n(red X = forbidden combination)",
             fontweight="bold", fontsize=12, pad=14,
             fontfamily="Times New Roman")
ax.set_ylabel("Scenario", fontfamily="Times New Roman")
ax.set_xlabel("Urgency", fontfamily="Times New Roman")
for lbl in ax.get_xticklabels() + ax.get_yticklabels():
    lbl.set_fontfamily("Times New Roman")

fig.tight_layout()
fig.savefig(FIG_DIR / "1_4_scenario_urgency_affinity.png", bbox_inches="tight")
plt.close(fig)
print("  saved 1_4_scenario_urgency_affinity.png")


# ─────────────────────────────────────────────────────────────────────────────
# CELL 13 → 1_5_style_emotion_independence.png
# ─────────────────────────────────────────────────────────────────────────────
print("Generating 1_5 …")

ct_style_emo = pd.crosstab(df["style"], df["intended_emotion"])
ct_style_emo = ct_style_emo.reindex(columns=EMOTION_ORDER)
ct_norm_se   = ct_style_emo.div(ct_style_emo.sum(axis=1), axis=0) * 100

fig, ax = plt.subplots(figsize=(8, 5))
ax.spines["top"].set_visible(True)
ax.spines["right"].set_visible(True)
no_grid(ax)
sns.heatmap(ct_norm_se, annot=True, fmt=".1f", cmap="Blues",
            linewidths=1.5, linecolor="white",
            cbar_kws={"label": "% within style"},
            ax=ax, vmin=0)

val_max_se = ct_norm_se.values.max()
for i, style in enumerate(ct_norm_se.index):
    for j, emo in enumerate(EMOTION_ORDER):
        val = ct_norm_se.loc[style, emo]
        text_color = "white" if val / val_max_se > 0.55 else "#1a1a2e"
        ax.texts[i * len(EMOTION_ORDER) + j].set_color(text_color)
        ax.texts[i * len(EMOTION_ORDER) + j].set_fontfamily("Times New Roman")

ax.set_title("Style × Emotion (row-normalised %)\nIdeal = 33.3% per cell",
             fontweight="bold", fontsize=12, pad=14,
             fontfamily="Times New Roman")
ax.set_ylabel("", fontfamily="Times New Roman")
ax.set_xlabel("Emotion", fontfamily="Times New Roman")
for lbl in ax.get_xticklabels() + ax.get_yticklabels():
    lbl.set_fontfamily("Times New Roman")

fig.tight_layout()
fig.savefig(FIG_DIR / "1_5_style_emotion_independence.png", bbox_inches="tight")
plt.close(fig)
print("  saved 1_5_style_emotion_independence.png")


# ─────────────────────────────────────────────────────────────────────────────
# CELL 15 → 1_6_profile_urgency_independence.png
# ─────────────────────────────────────────────────────────────────────────────
print("Generating 1_6 …")

ct_prof_urg = pd.crosstab(df["profile"], df["intended_urgency"])
ct_prof_urg = ct_prof_urg.reindex(columns=URGENCY_ORDER)
ct_norm_pu  = ct_prof_urg.div(ct_prof_urg.sum(axis=1), axis=0) * 100
short_labels_pu = [p[:40] + "…" if len(p) > 43 else p
                   for p in ct_norm_pu.index]

fig, ax = plt.subplots(figsize=(9, 5))
ax.spines["top"].set_visible(True)
ax.spines["right"].set_visible(True)
no_grid(ax)
sns.heatmap(ct_norm_pu, annot=True, fmt=".1f", cmap="Blues",
            linewidths=1.5, linecolor="white",
            cbar_kws={"label": "% within profile"},
            ax=ax, vmin=0,
            yticklabels=short_labels_pu)

val_max_pu = ct_norm_pu.values.max()
for i, profile in enumerate(ct_norm_pu.index):
    for j, urg in enumerate(URGENCY_ORDER):
        val = ct_norm_pu.loc[profile, urg]
        text_color = "white" if val / val_max_pu > 0.55 else "#1a1a2e"
        ax.texts[i * len(URGENCY_ORDER) + j].set_color(text_color)
        ax.texts[i * len(URGENCY_ORDER) + j].set_fontfamily("Times New Roman")

ax.set_title("Profile × Urgency (row-normalised %)\n"
             "Design target: Low 35%, Med 40%, High 25%",
             fontweight="bold", fontsize=12, pad=14,
             fontfamily="Times New Roman")
ax.set_ylabel("", fontfamily="Times New Roman")
ax.set_xlabel("Urgency", fontfamily="Times New Roman")
for lbl in ax.get_xticklabels() + ax.get_yticklabels():
    lbl.set_fontfamily("Times New Roman")

fig.tight_layout()
fig.savefig(FIG_DIR / "1_6_profile_urgency_independence.png", bbox_inches="tight")
plt.close(fig)
print("  saved 1_6_profile_urgency_independence.png")


# ─────────────────────────────────────────────────────────────────────────────
# TF-IDF (shared, used by 2_1, 2_2, 4_1, 4_2)
# ─────────────────────────────────────────────────────────────────────────────
print("Computing TF-IDF …")
vectorizer = TfidfVectorizer(max_features=10_000, ngram_range=(1, 2),
                              min_df=5, sublinear_tf=True,
                              stop_words="english")
X_tfidf       = vectorizer.fit_transform(df["complaint_text"])
feature_names = np.array(vectorizer.get_feature_names_out())


# ─────────────────────────────────────────────────────────────────────────────
# CELL 19 → 2_1_discriminative_terms.png
# ─────────────────────────────────────────────────────────────────────────────
print("Generating 2_1 …")

def top_chi2_terms(X, labels, class_order, feature_names, n=15):
    le = LabelEncoder()
    y  = le.fit_transform(labels)
    results = {}
    for cls_name in class_order:
        cls_idx  = le.transform([cls_name])[0]
        y_binary = (y == cls_idx).astype(int)
        scores, _ = sklearn_chi2(X, y_binary)
        top_idx  = np.argsort(scores)[-n:][::-1]
        results[cls_name] = list(zip(feature_names[top_idx], scores[top_idx]))
    return results

urg_terms = top_chi2_terms(X_tfidf, df["intended_urgency"],
                           URGENCY_ORDER, feature_names, n=15)
emo_terms = top_chi2_terms(X_tfidf, df["intended_emotion"],
                           EMOTION_ORDER, feature_names, n=15)

fig, axes = plt.subplots(2, 3, figsize=(18, 10))

for col_idx, (terms_dict, label_name, palette, order) in enumerate([
    (urg_terms, "Urgency", URGENCY_PAL, URGENCY_ORDER),
    (emo_terms, "Emotion", EMOTION_PAL, EMOTION_ORDER),
]):
    for row_idx, level in enumerate(order):
        ax    = axes[col_idx, row_idx]
        terms = terms_dict[level][:12]
        words = [t[0] for t in terms][::-1]
        scores = [t[1] for t in terms][::-1]
        ax.barh(range(len(words)), scores,
                color=palette[level],
                edgecolor=EDGE_COLOR, linewidth=EDGE_LW, alpha=0.85)
        ax.set_yticks(range(len(words)))
        ax.set_yticklabels(words, fontsize=9,
                           fontfamily="Times New Roman")
        set_tnr(ax, title=f"{label_name} = {level}",
                xlabel="chi-sq score")
        enable_xgrid(ax)

fig_title(fig,
          "Most Discriminative Terms per Class "
          "(TF-IDF + Chi-Squared Feature Selection)",
          y=1.02)
fig.tight_layout()
fig.savefig(FIG_DIR / "2_1_discriminative_terms.png", bbox_inches="tight")
plt.close(fig)
print("  saved 2_1_discriminative_terms.png")


# ─────────────────────────────────────────────────────────────────────────────
# CELL 21 → 2_2_vocabulary_overlap.png
# ─────────────────────────────────────────────────────────────────────────────
print("Generating 2_2 …")

def get_top_vocab(texts, n=500):
    vec = TfidfVectorizer(max_features=n, stop_words="english",
                          sublinear_tf=True)
    vec.fit(texts)
    return set(vec.get_feature_names_out())

def jaccard(set_a, set_b):
    return len(set_a & set_b) / len(set_a | set_b)

urg_vocabs = {level: get_top_vocab(
    df[df["intended_urgency"] == level]["complaint_text"])
    for level in URGENCY_ORDER}
emo_vocabs = {level: get_top_vocab(
    df[df["intended_emotion"] == level]["complaint_text"])
    for level in EMOTION_ORDER}

fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

for ax, vocabs, title, order in [
    (axes[0], urg_vocabs, "Urgency", URGENCY_ORDER),
    (axes[1], emo_vocabs, "Emotion", EMOTION_ORDER),
]:
    ax.spines["top"].set_visible(True)
    ax.spines["right"].set_visible(True)
    no_grid(ax)
    sim_matrix = pd.DataFrame(
        [[jaccard(vocabs[a], vocabs[b]) for b in order] for a in order],
        index=order, columns=order)
    sns.heatmap(sim_matrix, annot=True, fmt=".3f", cmap="Blues",
                vmin=0.3, vmax=1.0,
                linewidths=1.5, linecolor="white",
                ax=ax, square=True)
    ax.set_title(f"{title} — Vocab Jaccard Similarity\n"
                 "(top 500 TF-IDF terms)",
                 fontweight="bold", fontsize=11,
                 fontfamily="Times New Roman")
    for lbl in ax.get_xticklabels() + ax.get_yticklabels():
        lbl.set_fontfamily("Times New Roman")

    val_max_jac = sim_matrix.values.max()
    for txt in ax.texts:
        # parse value from annotation text
        try:
            v = float(txt.get_text())
            txt.set_color("white" if v / val_max_jac > 0.75 else "#1a1a2e")
        except ValueError:
            pass
        txt.set_fontfamily("Times New Roman")

fig.tight_layout()
fig.savefig(FIG_DIR / "2_2_vocabulary_overlap.png", bbox_inches="tight")
plt.close(fig)
print("  saved 2_2_vocabulary_overlap.png")


# ─────────────────────────────────────────────────────────────────────────────
# CELL 26 → 3_1_text_length_by_class.png
# ─────────────────────────────────────────────────────────────────────────────
print("Generating 3_1 …")

# Seaborn boxplot colour handling via palette dict
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

sns.boxplot(data=df, x="intended_urgency", y="word_count",
            order=URGENCY_ORDER,
            palette=URGENCY_PAL, ax=axes[0, 0],
            fliersize=2, linewidth=1.2,
            flierprops=dict(markeredgecolor=GREY, markerfacecolor=LGREY,
                            marker=".", markersize=3))
set_tnr(axes[0, 0], title="Word Count by Urgency",
        ylabel="Word count", xlabel="")
enable_ygrid(axes[0, 0])

sns.boxplot(data=df, x="intended_emotion", y="word_count",
            order=EMOTION_ORDER,
            palette=EMOTION_PAL, ax=axes[0, 1],
            fliersize=2, linewidth=1.2,
            flierprops=dict(markeredgecolor=GREY, markerfacecolor=LGREY,
                            marker=".", markersize=3))
set_tnr(axes[0, 1], title="Word Count by Emotion",
        ylabel="Word count", xlabel="")
enable_ygrid(axes[0, 1])

sns.boxplot(data=df, x="intended_urgency", y="sentence_count",
            order=URGENCY_ORDER,
            palette=URGENCY_PAL, ax=axes[1, 0],
            fliersize=2, linewidth=1.2,
            flierprops=dict(markeredgecolor=GREY, markerfacecolor=LGREY,
                            marker=".", markersize=3))
set_tnr(axes[1, 0], title="Sentence Count by Urgency",
        ylabel="Sentence count", xlabel="")
enable_ygrid(axes[1, 0])

sns.boxplot(data=df, x="intended_emotion", y="sentence_count",
            order=EMOTION_ORDER,
            palette=EMOTION_PAL, ax=axes[1, 1],
            fliersize=2, linewidth=1.2,
            flierprops=dict(markeredgecolor=GREY, markerfacecolor=LGREY,
                            marker=".", markersize=3))
set_tnr(axes[1, 1], title="Sentence Count by Emotion",
        ylabel="Sentence count", xlabel="")
enable_ygrid(axes[1, 1])

fig_title(fig, "Text Length Distribution by Class", y=1.02)
fig.tight_layout()
fig.savefig(FIG_DIR / "3_1_text_length_by_class.png", bbox_inches="tight")
plt.close(fig)
print("  saved 3_1_text_length_by_class.png")


# ─────────────────────────────────────────────────────────────────────────────
# CELL 30 → 3_3_token_truncation.png
# ─────────────────────────────────────────────────────────────────────────────
print("Generating 3_3 …")

trunc_by_urg = (df.groupby("intended_urgency")["truncated"]
                  .mean().reindex(URGENCY_ORDER) * 100)
trunc_by_emo = (df.groupby("intended_emotion")["truncated"]
                  .mean().reindex(EMOTION_ORDER) * 100)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Histogram
axes[0].hist(df["approx_token_count"], bins=50,
             color=BLUE, edgecolor="white", alpha=0.85)
axes[0].axvline(MAX_LENGTH, color=RED, linestyle="--",
                linewidth=2, label=f"max_length={MAX_LENGTH}")
axes[0].legend(prop={"family": "Times New Roman"})
set_tnr(axes[0], title="Token Length Distribution (approx.)",
        xlabel="Approx. token count", ylabel="Frequency")
enable_ygrid(axes[0])

# Grouped bar
x     = np.arange(3)
width = 0.35
bars1 = axes[1].bar(x - width / 2, trunc_by_urg.values, width,
                    color=[URGENCY_PAL[u] for u in URGENCY_ORDER],
                    edgecolor=EDGE_COLOR, linewidth=EDGE_LW,
                    label="Urgency")
bars2 = axes[1].bar(x + width / 2, trunc_by_emo.values, width,
                    color=[EMOTION_PAL[e] for e in EMOTION_ORDER],
                    edgecolor=EDGE_COLOR, linewidth=EDGE_LW,
                    label="Emotion")

# Urgency legend patches
urg_patches = [mpatches.Patch(color=URGENCY_PAL[u],
                               label=f"Urg {u}") for u in URGENCY_ORDER]
emo_patches = [mpatches.Patch(color=EMOTION_PAL[e],
                               label=f"Emo {e}") for e in EMOTION_ORDER]
axes[1].legend(handles=urg_patches + emo_patches,
               ncol=2, loc="upper center",
               bbox_to_anchor=(0.5, -0.15),
               prop={"family": "Times New Roman"})

axes[1].set_xticks(x)
axes[1].set_xticklabels(["Low", "Medium", "High"],
                        fontfamily="Times New Roman")
set_tnr(axes[1], title="Truncation Rate by Class",
        ylabel="Truncation rate (%)", xlabel="Level")
enable_ygrid(axes[1])

for bars in [bars1, bars2]:
    for bar in bars:
        h = bar.get_height()
        axes[1].text(bar.get_x() + bar.get_width() / 2,
                     h + 0.3, f"{h:.1f}%",
                     ha="center", va="bottom", fontsize=11,
                     fontfamily="Times New Roman")

fig_title(fig, "Token Length Distribution & Truncation Analysis")
fig.tight_layout()
fig.savefig(FIG_DIR / "3_3_token_truncation.png", bbox_inches="tight")
plt.close(fig)
print("  saved 3_3_token_truncation.png")


# ─────────────────────────────────────────────────────────────────────────────
# SVD + t-SNE (shared for 4_1, 4_2)
# ─────────────────────────────────────────────────────────────────────────────
print("Computing SVD (50-d) …")
svd   = TruncatedSVD(n_components=50, random_state=SEED)
X_svd = svd.fit_transform(X_tfidf)
print(f"  SVD explained variance: {svd.explained_variance_ratio_.sum():.1%}")

print("Computing t-SNE (this may take ~30 s) …")
tsne  = TSNE(n_components=2, perplexity=30, random_state=SEED,
             n_iter=1000, init="pca")
X_2d  = tsne.fit_transform(X_svd)
df["tsne_x"] = X_2d[:, 0]
df["tsne_y"] = X_2d[:, 1]
print("  t-SNE done.")


# ─────────────────────────────────────────────────────────────────────────────
# CELL 34 → 4_1_tsne_tfidf.png
# ─────────────────────────────────────────────────────────────────────────────
print("Generating 4_1 …")

# Muted scenario palette (academic Set2-inspired)
SCENARIO_MUTED = [
    "#5B8DB8",  # blue
    "#7AAF7A",  # green
    "#C07070",  # rose
    "#A07AB0",  # mauve
    "#C9A055",  # amber
]

fig, axes = plt.subplots(1, 3, figsize=(20, 6))

# By urgency
for level in URGENCY_ORDER:
    mask = df["intended_urgency"] == level
    axes[0].scatter(df.loc[mask, "tsne_x"], df.loc[mask, "tsne_y"],
                    c=URGENCY_PAL[level], label=level,
                    s=6, alpha=0.35, linewidths=0)
axes[0].legend(markerscale=3, loc="best",
               prop={"family": "Times New Roman"})
set_tnr(axes[0], title="Coloured by Urgency",
        xlabel="t-SNE 1", ylabel="t-SNE 2")
no_grid(axes[0])

# By emotion
for level in EMOTION_ORDER:
    mask = df["intended_emotion"] == level
    axes[1].scatter(df.loc[mask, "tsne_x"], df.loc[mask, "tsne_y"],
                    c=EMOTION_PAL[level], label=level,
                    s=6, alpha=0.35, linewidths=0)
axes[1].legend(markerscale=3, loc="best",
               prop={"family": "Times New Roman"})
set_tnr(axes[1], title="Coloured by Emotion",
        xlabel="t-SNE 1")
no_grid(axes[1])

# By scenario (top 5)
top_scenarios  = df["scenario"].value_counts().head(5).index.tolist()
scenario_pal   = dict(zip(top_scenarios, SCENARIO_MUTED[:len(top_scenarios)]))
for scenario in top_scenarios:
    mask  = df["scenario"] == scenario
    short = scenario[:25] + "…" if len(scenario) > 28 else scenario
    axes[2].scatter(df.loc[mask, "tsne_x"], df.loc[mask, "tsne_y"],
                    c=[scenario_pal[scenario]], label=short,
                    s=6, alpha=0.35, linewidths=0)
axes[2].legend(markerscale=3, fontsize=7, loc="best",
               prop={"family": "Times New Roman"})
set_tnr(axes[2], title="Coloured by Scenario (top 5)",
        xlabel="t-SNE 1")
no_grid(axes[2])

fig_title(fig,
          "t-SNE Projection of TF-IDF Representations "
          "(pre-fine-tuning)",
          y=1.03)
fig.tight_layout()
fig.savefig(FIG_DIR / "4_1_tsne_tfidf.png", bbox_inches="tight")
plt.close(fig)
print("  saved 4_1_tsne_tfidf.png")


# ─────────────────────────────────────────────────────────────────────────────
# CELL 36 → 4_2_silhouette_by_label.png
# ─────────────────────────────────────────────────────────────────────────────
print("Generating 4_2 …")

le           = LabelEncoder()
results_sil  = {}
for col in ["intended_urgency", "intended_emotion",
            "scenario", "style", "profile"]:
    labels = le.fit_transform(df[col])
    score  = silhouette_score(X_svd, labels, metric="cosine",
                              sample_size=2000, random_state=SEED)
    results_sil[col] = score

labels_sorted  = sorted(results_sil, key=lambda k: results_sil[k],
                         reverse=True)
scores_sorted  = [results_sil[k] for k in labels_sorted]
display_names  = [l.replace("intended_", "").title()
                  for l in labels_sorted]
max_score      = max(scores_sorted)

colors = [RED if s == max_score else BLUE for s in scores_sorted]

fig, ax = plt.subplots(figsize=(9, 5))
bars = ax.barh(range(len(labels_sorted)), scores_sorted,
               color=colors,
               edgecolor=EDGE_COLOR, linewidth=EDGE_LW)
ax.set_yticks(range(len(labels_sorted)))
ax.set_yticklabels(display_names, fontfamily="Times New Roman")
set_tnr(ax,
        title="Cluster Separability by Label Type\n"
              "(higher = more separated in TF-IDF space)",
        xlabel="Silhouette Score (cosine)")
enable_xgrid(ax)

for bar, score in zip(bars, scores_sorted):
    ax.text(score + 0.003,
            bar.get_y() + bar.get_height() / 2,
            f"{score:.4f}", va="center", fontsize=10,
            fontfamily="Times New Roman")

fig.tight_layout()
fig.savefig(FIG_DIR / "4_2_silhouette_by_label.png", bbox_inches="tight")
plt.close(fig)
print("  saved 4_2_silhouette_by_label.png")

print("\nAll 11 figures regenerated successfully.")
