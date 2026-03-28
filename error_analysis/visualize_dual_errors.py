import os
import json
from collections import Counter
import matplotlib.pyplot as plt
import seaborn as sns

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
URGENCY_JSONL = os.path.join(SCRIPT_DIR, "data", "icp_pairs.jsonl")
EMOTION_JSONL = os.path.join(SCRIPT_DIR, "data", "emotion_icp_pairs.jsonl")
OUT_IMG = os.path.join(SCRIPT_DIR, "results", "dual_error_histogram.png")

def load_and_count(file_path):
    directions = []
    if os.path.exists(file_path):
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    try:
                        data = json.loads(line)
                        t = data.get("true_label", "Unknown").title()
                        p = data.get("error_prediction", "Unknown").title()
                        directions.append(f"True: {t} \u2192 Pred: {p}")
                    except json.JSONDecodeError:
                        pass
    return Counter(directions)

def generate_dual_visualization():
    urg_counts = load_and_count(URGENCY_JSONL)
    emo_counts = load_and_count(EMOTION_JSONL)
    
    # Slice to User's previously requested limits (4 for urg, 5 for emo)
    urg_top = urg_counts.most_common()[:4]
    emo_top = emo_counts.most_common()[:5]

    # --- The Economist Style ---
    bg_color = '#d5e4eb'
    grid_color = '#ffffff'
    bar_urg_color = '#006BA2' # Economist Classic Blue
    bar_emo_color = '#E3120B' # Economist Red
    text_color = '#333333'

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.patch.set_facecolor(bg_color)
    
    # Helper to apply style to each axis
    def apply_economist_style(ax, title):
        ax.set_facecolor(bg_color)
        ax.grid(axis='y', color=grid_color, linestyle='-', linewidth=1.5, zorder=0)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_visible(False)
        ax.spines['bottom'].set_color(text_color)
        ax.spines['bottom'].set_linewidth(1.5)
        ax.tick_params(axis='both', which='both', length=0, colors=text_color, labelsize=11)
        ax.set_axisbelow(True) # Put grid behind bars
        
        # Left-aligned bold title (Economist signature)
        ax.set_title(title, fontsize=14, fontweight='bold', loc='left', pad=15, color=text_color)
        ax.set_ylabel('') # The Economist rarely uses explicit axis labels if obvious
        ax.set_xlabel('')

    # Plot Urgency
    if urg_top:
        apply_economist_style(axes[0], 'Urgency Head: Error Distribution')
        lbls = [x[0] for x in urg_top]
        cnts = [x[1] for x in urg_top]
        
        axes[0].bar(lbls, cnts, color=bar_urg_color, zorder=3, width=0.6)
        axes[0].set_xticklabels(lbls, rotation=45, ha='right')
        
        # Bold annotations on top of bars
        for i, count in enumerate(cnts):
            axes[0].text(i, count + max(cnts)*0.02, str(count), ha='center', va='bottom', 
                         fontsize=11, fontweight='bold', color=text_color)

    # Plot Emotion
    if emo_top:
        apply_economist_style(axes[1], 'Emotion Head: Error Distribution')
        lbls = [x[0] for x in emo_top]
        cnts = [x[1] for x in emo_top]
        
        axes[1].bar(lbls, cnts, color=bar_emo_color, zorder=3, width=0.6)
        axes[1].set_xticklabels(lbls, rotation=45, ha='right')
        
        # Bold annotations on top of bars
        for i, count in enumerate(cnts):
            axes[1].text(i, count + max(cnts)*0.02, str(count), ha='center', va='bottom', 
                         fontsize=11, fontweight='bold', color=text_color)

    # Add red line at top left of figure (Classic Economist branding line)
    fig.add_artist(plt.Line2D([0.05, 0.1], [0.98, 0.98], color='#E3120B', linewidth=4, transform=fig.transFigure))

    plt.tight_layout(rect=[0.05, 0, 1, 0.95])
    os.makedirs(os.path.dirname(OUT_IMG), exist_ok=True)
    plt.savefig(OUT_IMG, dpi=300, bbox_inches='tight', facecolor=fig.get_facecolor(), edgecolor='none')
    print(f"Dual visualization successfully saved to {OUT_IMG}")

if __name__ == "__main__":
    generate_dual_visualization()
