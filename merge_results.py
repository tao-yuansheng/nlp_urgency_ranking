import pandas as pd
import json
import os
import re

input_file = os.path.join(os.path.dirname(__file__), "telecoms_complaints.csv")
temp_file = os.path.join(os.path.dirname(__file__), "evaluations_temp.jsonl")
output_file = os.path.join(os.path.dirname(__file__), "telecoms_complaints_evaluated.xlsx")

df = pd.read_csv(input_file)
eval_df = df.drop(columns=["intended_urgency", "intended_emotion"], errors='ignore')

# Regex to remove illegal excel characters
ILLEGAL_CHARACTERS_RE = re.compile(r'[\000-\010]|[\013-\014]|[\016-\037]')

evaluations_dict = {}
with open(temp_file, "r") as f:
    for line in f:
        try:
            ev = json.loads(line)
            # Clean reasoning text
            if "reasoning" in ev and ev["reasoning"]:
                ev["reasoning"] = ILLEGAL_CHARACTERS_RE.sub(r'', ev["reasoning"])
            evaluations_dict[ev["id"]] = ev
        except Exception:
            pass

all_evaluations = list(evaluations_dict.values())
print(f"Total Unique Evaluated IDs: {len(all_evaluations)}")

evals_df = pd.DataFrame(all_evaluations)
if not evals_df.empty:
    final_df = pd.merge(eval_df, evals_df, on="id", how="left")
    
    # Strip illegal characters from all text columns
    for col in final_df.select_dtypes(['object', 'string']).columns:
        final_df[col] = final_df[col].apply(lambda x: ILLEGAL_CHARACTERS_RE.sub(r'', str(x)) if pd.notnull(x) else x)
        
    final_df.to_excel(output_file, index=False)
    print(f"Successfully saved to {output_file}")
else:
    print("No evaluations generated.")
