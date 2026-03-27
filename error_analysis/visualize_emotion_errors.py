import os
import json
from collections import Counter
import matplotlib.pyplot as plt
import seaborn as sns

# Paths
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
IN_JSONL = os.path.join(SCRIPT_DIR, "data", "emotion_icp_pairs.jsonl")
OUT_IMG = os.path.join(SCRIPT_DIR, "results", "emotion_error_histogram.png")
OUT_MD = os.path.join(SCRIPT_DIR, "results", "emotion_error_distribution.md")

def generate_visualization():
    print(f"Loading data from {IN_JSONL}")
    if not os.path.exists(IN_JSONL):
        print("Data file not found. Please ensure emotion_icp_pairs.jsonl exists.")
        return

    # Extract directions
    directions = []
    with open(IN_JSONL, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                try:
                    data = json.loads(line)
                    true_label = data.get("true_label", "Unknown").title()
                    pred_label = data.get("error_prediction", "Unknown").title()
                    
                    # Create direction string
                    direction_str = f"True: {true_label} \u2192 Pred: {pred_label}"
                    directions.append(direction_str)
                except json.JSONDecodeError:
                    pass

    if not directions:
        print("No valid error pairs found to visualize.")
        return

    # Count frequencies
    direction_counts = Counter(directions)
    
    # Sort for plotting (highest frequency first)
    sorted_directions = direction_counts.most_common()
    labels = [item[0] for item in sorted_directions]
    counts = [item[1] for item in sorted_directions]

    print(f"Found {len(labels)} unique error directions among {len(directions)} total errors.")

    # Visualization Setup
    plt.figure(figsize=(10, 6))
    
    # Create seaborn barplot
    sns.set_theme(style="whitegrid")
    ax = sns.barplot(x=labels, y=counts, palette="viridis")
    
    # Formatting
    plt.title('DeBERTa Emotion Model: Error Direction Distribution', fontsize=16, pad=15)
    plt.xlabel('Error Direction', fontsize=12, labelpad=10)
    plt.ylabel('Frequency (Count)', fontsize=12, labelpad=10)
    
    # Annotate bars with exact counts
    for i, count in enumerate(counts):
        ax.text(i, count + 0.5, str(count), ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    # Rotate x labels so they fit nicely
    plt.xticks(rotation=45, ha='right')
    
    # Adjust layout to prevent label cutoff
    plt.tight_layout()
    
    # Save Output
    os.makedirs(os.path.dirname(OUT_IMG), exist_ok=True)
    plt.savefig(OUT_IMG, dpi=300, bbox_inches='tight')
    print(f"Success! Histogram saved to: {OUT_IMG}")
    
    # Generate Markdown Table
    print(f"Generating markdown table to {OUT_MD}")
    with open(OUT_MD, "w", encoding="utf-8") as f:
        f.write("# Emotion Model: Error Distribution Summary\n\n")
        f.write("This table summarizes the distribution of classification mistakes among the 111 extracted Incorrect-Correct Pairs (ICPs).\n\n")
        f.write("| Error Direction | Frequency (Count) |\n")
        f.write("| :--- | :--- |\n")
        for label, count in zip(labels, counts):
            f.write(f"| {label} | {count} |\n")
    print("Done.")

if __name__ == "__main__":
    generate_visualization()
