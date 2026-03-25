"""
generate_icps.py
Finds matching correct predictions for each incorrect prediction based on a strict 
fallback hierarchy (Scenario + Style -> Scenario Only -> Urgency Only) to enable
Contrastive Error Analysis.
"""

import os
import json
import pandas as pd

# ── Config ──────────────────────────────────────────────────────────────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
EVAL_CSV = os.path.join(SCRIPT_DIR, "data", "all_eval_predictions.csv")
ORIGINAL_CSV = os.path.join(SCRIPT_DIR, "..", "nlp_urgency_ranking", "data", "telecoms_complaints.csv")
OUT_JSONL = os.path.join(SCRIPT_DIR, "data", "icp_pairs.jsonl")

def load_data():
    if not os.path.exists(EVAL_CSV):
        raise FileNotFoundError(f"Missing {EVAL_CSV}")
    if not os.path.exists(ORIGINAL_CSV):
        raise FileNotFoundError(f"Missing {ORIGINAL_CSV}")
        
    eval_df = pd.read_csv(EVAL_CSV)
    original_df = pd.read_csv(ORIGINAL_CSV)
    
    # Merge to get scenario and style
    # We rename 'complaint_text' in original to 'text' for the merge
    original_sub = original_df[['complaint_text', 'scenario', 'style']].rename(columns={'complaint_text': 'text'})
    
    # Drop duplicates in original just in case text is identical across rows
    original_sub = original_sub.drop_duplicates(subset=['text'])
    
    merged_df = pd.merge(eval_df, original_sub, on='text', how='inner')
    
    if len(merged_df) != len(eval_df):
        print(f"Warning: Merged DataFrame has {len(merged_df)} rows, expected {len(eval_df)}. Some texts might not have matched exactly.")
        
    return merged_df

def match_incorrect_to_correct(incorrect_row, correct_df):
    """
    Implements the strict fallback hierarchy:
    1: Perfect (ground_truth_urgency + scenario + style)
    2: Partial (ground_truth_urgency + scenario)
    3: Fallback (ground_truth_urgency)
    """
    g_urgency = incorrect_row["ground_truth_urgency"]
    scenario = incorrect_row["scenario"]
    style = incorrect_row["style"]
    
    # Attempt 1: Perfect Match
    perfect = correct_df[
        (correct_df["ground_truth_urgency"] == g_urgency) & 
        (correct_df["scenario"] == scenario) & 
        (correct_df["style"] == style)
    ]
    if not perfect.empty:
        return perfect.iloc[0], "Perfect: Scenario + Style"
        
    # Attempt 2: Partial Match
    partial = correct_df[
        (correct_df["ground_truth_urgency"] == g_urgency) & 
        (correct_df["scenario"] == scenario)
    ]
    if not partial.empty:
        return partial.iloc[0], "Partial: Scenario Only"
        
    # Attempt 3: Fallback Match
    fallback = correct_df[
        (correct_df["ground_truth_urgency"] == g_urgency)
    ]
    if not fallback.empty:
        return fallback.iloc[0], "Fallback: Urgency Only"
        
    return None, "No Match Found"

if __name__ == "__main__":
    print("Loading datasets and merging metadata...")
    df = load_data()
    
    # Split into correct vs incorrect
    correct_mask = df["predicted_urgency"] == df["ground_truth_urgency"]
    correct_df = df[correct_mask].copy()
    incorrect_df = df[~correct_mask].copy()
    
    print(f"Total Evaluated: {len(df)}")
    print(f"Correct Predictions Pool: {len(correct_df)}")
    print(f"Incorrect Predictions to Match: {len(incorrect_df)}")
    
    results = []
    stats = {
        "Perfect: Scenario + Style": 0,
        "Partial: Scenario Only": 0,
        "Fallback: Urgency Only": 0,
        "No Match Found": 0
    }
    
    print("Matching Incorrect-Correct Pairs (ICPs)...")
    for _, incorrect_row in incorrect_df.iterrows():
        match_row, match_level = match_incorrect_to_correct(incorrect_row, correct_df)
        
        stats[match_level] += 1
        
        if match_row is not None:
            # Build the JSON object
            results.append({
                "true_label":       incorrect_row["ground_truth_urgency"],
                "metadata_matched": match_level,
                "error_text":       incorrect_row["text"],
                "error_prediction": incorrect_row["predicted_urgency"],
                "correct_text":     match_row["text"]
            })
            
    # Save to JSONL
    print(f"\nSaving {len(results)} ICPs to {OUT_JSONL}...")
    os.makedirs(os.path.dirname(OUT_JSONL), exist_ok=True)
    with open(OUT_JSONL, "w", encoding="utf-8") as f:
        for res in results:
            f.write(json.dumps(res, ensure_ascii=False) + "\n")
            
    print("\n" + "="*50)
    print("MATCHING STATISTICS")
    print("="*50)
    for k, v in stats.items():
        print(f"  {k:25s} : {v}")
    print("="*50)
    print("Done.")
