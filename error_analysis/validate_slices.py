"""
validate_slices.py - Validate the specific taxonomy categories against the full dataset
to check base rates and prove systemic weaknesses.
"""

import os
import json
import re
import pandas as pd

# ── Config ──────────────────────────────────────────────────────────────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_CSV = os.path.join(SCRIPT_DIR, "data", "all_eval_predictions.csv")
TAXONOMY_JSON = os.path.join(SCRIPT_DIR, "results", "final_taxonomy.json")

# Mapping the 3 major categories to specific text keywords based on their descriptions
# We use regex \b to match whole words where appropriate.
CATEGORY_KEYWORDS = {
    # Category 1: Underestimation of Overall Severity & Urgency
    # (Failing to synthesize cumulative factors, duration, business impact, vulnerability)
    "Underestimation of Overall Severity & Urgency": [
        r"\bweeks?\b", r"\bmonths?\b", r"\byears?\b", 
        r"\bagain\b", r"\brepeatedly\b", r"\bmultiple times\b", 
        r"\bstill broken\b", r"\bwaiting\b",
        r"\blosing money\b", r"\bbusiness\b", r"\bwork\b", r"\bjob\b",
        r"\bvulnerable\b", r"\belderly\b", r"\bdisabled\b", r"\bmedical\b"
    ],
    
    # Category 2: Overestimation & Miscalibration of Severity/Urgency Thresholds
    # (Over-indexing on subjective/emotional language or non-critical urgency words)
    "Overestimation & Miscalibration of Severity/Urgency Thresholds": [
        r"\burgent\b", r"\bemergency\b", r"\bdanger\b", 
        r"\bhealth\b", r"\bsafety\b", r"\bimmediate\b", r"\bimmediately\b",
        r"\bunacceptable\b", r"\bdisgusting\b", r"\bappalling\b", r"\bfurious\b", r"\bangry\b"
    ],
    
    # Category 3: Failure to Accurately Process Specific Information Cues
    # (Ignoring explicit formal threats, escalation bodies)
    "Failure to Accurately Process Specific Information Cues or Adhere to Instructions": [
        r"\bombudsman\b", r"\blawyer\b", r"\bsue\b", r"\blegal\b", 
        r"\bcourt\b", r"\btrading standards\b", r"\bregulator\b", 
        r"\bcancel\b", r"\bleaving\b", r"\bcompensation\b", r"\bofcom\b"
    ]
}

def load_data():
    if not os.path.exists(DATA_CSV):
        raise FileNotFoundError(f"Missing predictions dataset: {DATA_CSV}")
    return pd.read_csv(DATA_CSV)

if __name__ == "__main__":
    if not os.path.exists(DATA_CSV):
        print(f"ERROR: {DATA_CSV} not found. Please run extract_errors_only.py first.")
        exit(1)
        
    df = pd.read_csv(DATA_CSV)
    
    # Load taxonomy to verify categories match
    if not os.path.exists(TAXONOMY_JSON):
        print(f"ERROR: {TAXONOMY_JSON} not found.")
        exit(1)
        
    with open(TAXONOMY_JSON, "r", encoding="utf-8") as f:
        taxonomy = json.load(f)
        
    with open('report.txt', 'w', encoding='utf-8') as out:
        def log(msg):
            print(msg)
            out.write(msg + '\n')
            
        log("\n" + "="*70)
        log("SYSTEMIC WEAKNESS VALIDATION REPORT")
        log("="*70)
        
        # Global Baseline
        total_rows = len(df)
        global_correct = (df["predicted_urgency"] == df["ground_truth_urgency"]).sum()
        global_acc = global_correct / total_rows * 100
        log(f"\nGLOBAL BASELINE (Full Test Set)")
        log(f"  Total Samples: {total_rows}")
        log(f"  Baseline Accuracy: {global_acc:.1f}%\n")
        log("-" * 70)
        
        for tax_cat in taxonomy["taxonomy"]:
            cat_name = tax_cat["category_name"]
            
            # Check if we have keywords mapping
            if cat_name not in CATEGORY_KEYWORDS:
                log(f"Warning: No keyword mapping found for {cat_name}")
                continue
                
            keywords = CATEGORY_KEYWORDS[cat_name]
            # Compile regex (case insensitive)
            pattern = re.compile("|".join(keywords), re.IGNORECASE)
            
            # Slicing: create a boolean mask for rows matching the keywords
            mask = df["text"].astype(str).apply(lambda x: bool(pattern.search(x)))
            slice_df = df[mask]
            
            slice_total = len(slice_df)
            if slice_total == 0:
                log(f"\nCATEGORY: {cat_name}")
                log("  No rows matched the keywords for this category.")
                continue
                
            slice_correct = (slice_df["predicted_urgency"] == slice_df["ground_truth_urgency"]).sum()
            slice_incorrect = slice_total - slice_correct
            slice_acc = slice_correct / slice_total * 100
            
            # Determine if it's a systemic weakness
            acc_drop = global_acc - slice_acc
            status = "SYSTEMIC WEAKNESS" if acc_drop > 5 else "NORMAL VARIANCE"
            if acc_drop > 15:
                status = "SEVERE SYSTEMIC WEAKNESS"
                
            log(f"\nCATEGORY: {cat_name}")
            clean_keywords = [k.replace(r'\b', '').replace('\\b', '') for k in keywords[:5]]
            log(f"  Keywords used: {', '.join(clean_keywords)} ...")
            log(f"  Matches in Dataset: {slice_total} rows ({(slice_total/total_rows)*100:.1f}% of total)")
            log(f"  - Correct Predictions:   {slice_correct}")
            log(f"  - Incorrect Predictions: {slice_incorrect}")
            log(f"  Slice Accuracy:          {slice_acc:.1f}%  (Baseline: {global_acc:.1f}%)")
            log(f"  Accuracy Drop:           {acc_drop:+.1f}%")
            log(f"  => {status}")
            
        log("\n" + "="*70)
        log("Validation Complete.")
