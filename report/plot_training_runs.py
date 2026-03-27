"""
Visualize test macro F1 scores across training runs.
Reads JSON log files from model_training/logs/, groups them into 3 dataset
sections (v1, v2, no-version), deduplicates within each section by
hyperparameters (keeping the first occurrence), then plots a line chart.
Output is saved to the same report/ directory.
"""

import json
import re
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.ticker as ticker
import numpy as np

# ── Paths ──────────────────────────────────────────────────────────────────────
SCRIPT_DIR = Path(__file__).parent
LOGS_DIR = SCRIPT_DIR.parent / "model_training" / "logs"
OUTPUT_FILE = SCRIPT_DIR / "training_runs_f1.png"

# ── Hardcoded x-axis labels (run_id → display label) ──────────────────────────
RUN_LABELS = {
    "20260324_baseline": "Run 1",
    "20260324_182448":   "Run 2",
    "20260324_204809":   "Run 3",
    "20260324_214353":   "Run 4",
    "20260325_110045":   "Run 5",
    "20260325_155145":   "Run 6",
    "20260325_160828":   "Run 7",
    "20260327_114738":   "Run 8",
}

# ── Academic style globals ─────────────────────────────────────────────────────
FONT_FAMILY = "serif"
FONT_SIZE_BASE = 9
FONT_SIZE_TITLE = 11
FONT_SIZE_LABEL = 10
FONT_SIZE_TICK = 8.5
FONT_SIZE_ANNOT = 7.5
FONT_SIZE_LEGEND = 8.5
FONT_SIZE_SECTION = 8.5

COLOR_URGENCY = "#2166AC"   # deep steel blue
COLOR_EMOTION = "#B2182B"   # dark crimson

# Very subtle section backgrounds — distinct but not distracting
SECTION_FILL = {
    "v1": "#e8f0f8",
    "v2": "#f0f4ec",
    "v3": "#f8f2e8",
}
SECTION_EDGE = {
    "v1": "#b0c4de",
    "v2": "#a8c8a0",
    "v3": "#d4b896",
}
SECTION_LABELS = {
    "v1": "Dataset v1",
    "v2": "Dataset v2",
    "v3": "Dataset v3",
}


# ── Data loading ───────────────────────────────────────────────────────────────
def get_section(filename: str) -> str:
    name = Path(filename).stem
    if re.match(r"run_v1_", name):
        return "v1"
    if re.match(r"run_v2_", name):
        return "v2"
    return "v3"


def lr_key(data: dict):
    if "lr_backbone" in data and "lr_heads" in data:
        return ("split", data["lr_backbone"], data["lr_heads"])
    return ("single", data.get("lr"))


def hyperparams_key(data: dict) -> tuple:
    # Include stopping criterion so runs that differ only on that aren't deduped
    stopping = "combined_f1" if "best_val_combined_f1" in data else "val_loss"
    return (
        data.get("model"),
        data.get("max_length"),
        data.get("batch_size"),
        lr_key(data),
        stopping,
    )


def load_runs() -> list[dict]:
    all_data = []
    for f in LOGS_DIR.glob("*.json"):
        with open(f, encoding="utf-8") as fh:
            data = json.load(fh)
        data["_section"] = get_section(f.name)
        data["_file"] = f.name
        all_data.append(data)

    all_data.sort(key=lambda d: d.get("timestamp", ""))

    sections_order = ["v1", "v2", "v3"]
    grouped: dict[str, list[dict]] = {s: [] for s in sections_order}
    for data in all_data:
        grouped[data["_section"]].append(data)

    runs = []
    for section in sections_order:
        seen_keys: set = set()
        for data in grouped[section]:
            key = hyperparams_key(data)
            if key in seen_keys:
                print(f"  [dedup] Skipping duplicate: {data['_file']}")
                continue
            seen_keys.add(key)
            runs.append(data)

    return runs


def make_label(data: dict) -> str:
    return RUN_LABELS.get(data.get("run_id", ""), data.get("run_id", ""))


# ── Smart annotation placement ─────────────────────────────────────────────────
def compute_offsets(urgency_vals, emotion_vals,
                    intra_threshold=0.06, inter_threshold=0.02, x_nudge=10):
    """
    Returns (u_offsets, e_offsets), each a list of [x_off, y_off] per run.

    y-placement rule:
      When urgency and emotion at the SAME run are within intra_threshold,
      the labels would collide (urgency-above vs emotion-below leaves them
      squashed in the middle).  Fix: flip to urgency-below / emotion-above.

    x-nudge rule:
      When adjacent runs have the SAME y-side label for the same metric and
      their values are within inter_threshold, nudge them apart horizontally.
    """
    n = len(urgency_vals)
    u_off = [[0, 0] for _ in range(n)]
    e_off = [[0, 0] for _ in range(n)]

    for i in range(n):
        diff = abs(emotion_vals[i] - urgency_vals[i])
        if diff < intra_threshold:
            u_off[i][1] = -14   # urgency below
            e_off[i][1] = +10   # emotion above
        else:
            u_off[i][1] = +8    # urgency above
            e_off[i][1] = -12   # emotion below

    def apply_x_nudge(offsets, vals):
        for i in range(n):
            y_i = offsets[i][1]
            left_close  = (i > 0     and offsets[i-1][1] == y_i
                           and abs(vals[i] - vals[i-1]) < inter_threshold)
            right_close = (i < n - 1 and offsets[i+1][1] == y_i
                           and abs(vals[i] - vals[i+1]) < inter_threshold)
            if left_close and right_close:
                offsets[i][0] = -x_nudge if i % 2 == 0 else +x_nudge
            elif left_close:
                offsets[i][0] = +x_nudge
            elif right_close:
                offsets[i][0] = -x_nudge

    apply_x_nudge(u_off, urgency_vals)
    apply_x_nudge(e_off, emotion_vals)

    # Steep-line avoidance: when a label sits in the path of a sharply rising
    # adjacent segment, push it sideways away from that segment.
    # steep_thresh is the minimum |Δy| (in F1 units) to count as steep.
    steep_thresh = 0.10
    line_nudge   = 14

    for i in range(n):
        for offsets, vals in ((u_off, urgency_vals), (e_off, emotion_vals)):
            y_off = offsets[i][1]
            # Label is ABOVE the point → vulnerable to a steep rise on the right
            if y_off > 0 and i < n - 1:
                if vals[i + 1] - vals[i] > steep_thresh:
                    offsets[i][0] = -line_nudge   # push left, away from rising line
            # Label is BELOW the point → vulnerable to a steep rise arriving from the left
            if y_off < 0 and i > 0:
                if vals[i] - vals[i - 1] > steep_thresh:
                    offsets[i][0] = +line_nudge   # push right, away from arriving line

    return u_off, e_off


def place_annotation(ax, xi, yi, text, color, x_off, y_off):
    ax.annotate(
        text, (xi, yi),
        textcoords="offset points",
        xytext=(x_off, y_off),
        ha="center",
        va="bottom" if y_off > 0 else "top",
        fontsize=FONT_SIZE_ANNOT,
        color=color,
        fontfamily=FONT_FAMILY,
    )


# ── Section boundary helper ────────────────────────────────────────────────────
def compute_boundaries(runs: list[dict]) -> list[tuple[int, int, str]]:
    boundaries = []
    current_sec = runs[0]["_section"]
    start = 0
    for i, r in enumerate(runs):
        if r["_section"] != current_sec:
            boundaries.append((start, i, current_sec))
            start = i
            current_sec = r["_section"]
    boundaries.append((start, len(runs), current_sec))
    return boundaries


# ── Main plot ──────────────────────────────────────────────────────────────────
def plot(runs: list[dict]):
    urgency = [r["test_urgency_macro_f1"] for r in runs]
    emotion = [r["test_emotion_macro_f1"] for r in runs]
    overall = [(u + e) / 2 for u, e in zip(urgency, emotion)]
    labels  = [make_label(r) for r in runs]
    x       = np.arange(len(runs))

    boundaries = compute_boundaries(runs)

    # ── rcParams: academic / paper style ──────────────────────────────────────
    matplotlib.rcParams.update({
        "font.family":          FONT_FAMILY,
        "font.size":            FONT_SIZE_BASE,
        "axes.titlesize":       FONT_SIZE_TITLE,
        "axes.labelsize":       FONT_SIZE_LABEL,
        "xtick.labelsize":      FONT_SIZE_TICK,
        "ytick.labelsize":      FONT_SIZE_TICK,
        "legend.fontsize":      FONT_SIZE_LEGEND,
        "axes.linewidth":       0.8,
        "axes.edgecolor":       "#333333",
        "grid.linewidth":       0.5,
        "grid.color":           "#cccccc",
        "grid.linestyle":       ":",
        "xtick.direction":      "in",
        "ytick.direction":      "in",
        "xtick.major.size":     3.5,
        "ytick.major.size":     3.5,
        "xtick.minor.visible":  False,
        "ytick.minor.visible":  False,
        "figure.dpi":           150,
        "savefig.dpi":          300,
        "figure.facecolor":     "white",
        "axes.facecolor":       "white",
        "axes.spines.top":      False,
        "axes.spines.right":    False,
    })

    fig, ax = plt.subplots(figsize=(10, 4.2))

    # ── Section background bands ───────────────────────────────────────────────
    for seg_s, seg_e, sec in boundaries:
        ax.axvspan(
            seg_s - 0.5, seg_e - 0.5,
            color=SECTION_FILL[sec],
            alpha=1.0,
            zorder=0,
        )

    # ── Grid (behind everything) ───────────────────────────────────────────────
    ax.set_axisbelow(True)
    ax.yaxis.grid(True)

    # ── Lines and markers ──────────────────────────────────────────────────────
    ax.plot(x, overall, marker="", linewidth=1.2, linestyle="--",
            color="#999999", label="Overall macro F1 (avg.)",
            alpha=0.7, zorder=2)
    ax.plot(x, urgency, marker="o", markersize=5.5, linewidth=1.5,
            color=COLOR_URGENCY, label="Urgency macro F1",
            markerfacecolor="white", markeredgewidth=1.4, zorder=4)
    ax.plot(x, emotion, marker="s", markersize=5.5, linewidth=1.5,
            color=COLOR_EMOTION, label="Emotion macro F1",
            markerfacecolor="white", markeredgewidth=1.4, zorder=4)

    # ── Value annotations (smart placement) ───────────────────────────────────
    u_off, e_off = compute_offsets(urgency, emotion)
    for xi in range(len(runs)):
        place_annotation(ax, xi, urgency[xi], f"{urgency[xi]:.3f}",
                         COLOR_URGENCY, u_off[xi][0], u_off[xi][1])
        place_annotation(ax, xi, emotion[xi], f"{emotion[xi]:.3f}",
                         COLOR_EMOTION, e_off[xi][0], e_off[xi][1])

    # ── Section divider lines ──────────────────────────────────────────────────
    for seg_s, seg_e, sec in boundaries[:-1]:
        ax.axvline(seg_e - 0.5, color="#999999", linestyle="--",
                   linewidth=0.8, zorder=3)

    # ── Y axis range ──────────────────────────────────────────────────────────
    all_f1 = urgency + emotion + overall
    ylo = max(0.0, min(all_f1) - 0.06)
    yhi = min(1.0, max(all_f1) + 0.10)
    ax.set_ylim(ylo, yhi)
    ax.yaxis.set_major_locator(ticker.MultipleLocator(0.05))
    ax.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.2f"))

    # ── Selected run marker ───────────────────────────────────────────────────
    SELECTED_LABEL = "Run 8"
    selected_idx = next(i for i, r in enumerate(runs)
                        if RUN_LABELS.get(r.get("run_id", "")) == SELECTED_LABEL)
    ax.axvline(selected_idx, color="#555555", linestyle=":",
               linewidth=1.0, alpha=0.6, zorder=3)
    ax.annotate(
        "Selected",
        xy=(selected_idx, ax.get_ylim()[1]),
        xytext=(6, -4),
        textcoords="offset points",
        va="top", ha="left",
        fontsize=7.5, fontstyle="italic", color="#555555",
        fontfamily=FONT_FAMILY,
    )

    # ── X axis ────────────────────────────────────────────────────────────────
    ax.set_xticks(x)
    tick_labels = ax.set_xticklabels(labels, rotation=25, ha="right")
    tick_labels[selected_idx].set_fontweight("bold")
    ax.set_xlim(-0.5, len(runs) - 0.5)

    # ── Section labels (top of each band, inside plot) ─────────────────────────
    yhi_actual = ax.get_ylim()[1]
    for seg_s, seg_e, sec in boundaries:
        mid = (seg_s + seg_e - 1) / 2
        ax.text(
            mid, yhi_actual - 0.004,
            SECTION_LABELS[sec],
            ha="center", va="top",
            fontsize=FONT_SIZE_SECTION,
            fontstyle="italic",
            color="#444444",
        )

    # ── Axis labels and title ──────────────────────────────────────────────────
    ax.set_xlabel("Training Run", labelpad=6)
    ax.set_ylabel("Test Macro F1", labelpad=6)
    ax.set_title("Test Macro F1 Score Across Training Runs", pad=10,
                 fontweight="bold")

    # ── Legend ────────────────────────────────────────────────────────────────
    line_handles, line_lbls = ax.get_legend_handles_labels()
    section_patches = [
        mpatches.Patch(facecolor=SECTION_FILL[s], edgecolor=SECTION_EDGE[s],
                       linewidth=0.8, label=SECTION_LABELS[s])
        for s in ["v1", "v2", "v3"]
    ]
    ax.legend(
        handles=line_handles + section_patches,
        loc="lower right",
        frameon=True,
        framealpha=0.92,
        edgecolor="#cccccc",
        handlelength=1.8,
        handletextpad=0.5,
        borderpad=0.6,
        labelspacing=0.4,
    )

    plt.tight_layout(pad=0.8)
    fig.savefig(OUTPUT_FILE, bbox_inches="tight")
    print(f"Saved to: {OUTPUT_FILE}")
    plt.close(fig)
    matplotlib.rcParams.update(matplotlib.rcParamsDefault)


if __name__ == "__main__":
    print("Loading runs...")
    runs = load_runs()
    print(f"Plotting {len(runs)} runs after deduplication.")
    for r in runs:
        print(f"  [{r['_section']}] {r['run_id']}")
    plot(runs)
