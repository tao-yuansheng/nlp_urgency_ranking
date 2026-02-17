"""Generate 225 synthetic telecoms complaints using OpenAI GPT-5-mini."""

import json
import os
import sys

import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI

from prompts import (
    EMOTION_DEFINITIONS,
    SYSTEM_PROMPTS,
    URGENCY_DEFINITIONS,
)
from taxonomy import GRID, build_assignments

load_dotenv()


def _build_user_prompt(cell_assignments: list[dict]) -> str:
    """Build a user prompt that requests multiple complaints for one cell."""
    n = len(cell_assignments)
    urgency = cell_assignments[0]["urgency"]
    emotion = cell_assignments[0]["emotion"]

    # Detect divergent urgency/emotion combos and add a clarification
    level_order = {"Low": 0, "Medium": 1, "High": 2}
    divergence_note = ""
    if abs(level_order[urgency] - level_order[emotion]) >= 2:
        if level_order[urgency] > level_order[emotion]:
            divergence_note = (
                "\nIMPORTANT: The urgency and emotion levels are intentionally "
                "different. The customer is describing a severe, high-impact issue "
                "but writing in a calm, composed, factual manner — they are not "
                "emotional despite the seriousness. Do NOT let the severity of the "
                "issue bleed into the tone. Keep the writing measured and neutral.\n\n"
            )
        else:
            divergence_note = (
                "\nIMPORTANT: The urgency and emotion levels are intentionally "
                "different. The customer is highly emotional and upset, but the "
                "underlying issue is relatively minor. The emotional intensity "
                "is genuine — some customers feel strongly about issues that "
                "others might consider small. Write the complaint with authentic "
                "high emotion while keeping the actual problem minor in scope.\n\n"
            )

    header = (
        f"Generate exactly {n} distinct customer complaints. "
        f"All complaints share the same urgency and emotion level:\n\n"
        f"Urgency: {urgency} — {URGENCY_DEFINITIONS[urgency]}\n"
        f"Emotion: {emotion} — {EMOTION_DEFINITIONS[emotion]}\n\n"
        f"{divergence_note}"
        f"Each complaint has a specific scenario, writing style, and "
        f"communication channel listed below. Make every complaint sound "
        f"unique — vary phrasing, length, and details.\n\n"
        f"For realism, include where appropriate:\n"
        f"- Account or reference numbers (e.g., ACC-7291834, REF-20240315)\n"
        f"- Specific dates of incidents and prior contacts\n"
        f"- Names of staff spoken to previously\n"
        f"- Prior contact history ('I have already called three times')\n"
        f"- Greetings and sign-offs appropriate to the channel\n"
        f"- Channel-appropriate formatting (emails are structured with "
        f"paragraphs; live chats are short and immediate; online forms tend "
        f"to be concise and structured; social media posts are brief and "
        f"public-facing, possibly with @mentions or hashtags)\n\n"
    )

    items = []
    for i, a in enumerate(cell_assignments, 1):
        items.append(
            f"Complaint {i}:\n"
            f"  Scenario: {a['scenario']}\n"
            f"  Style: {a['style']}\n"
            f"  Channel: {a['channel']}\n"
        )

    footer = (
        '\nReturn a JSON object with a single key "complaints" containing a '
        f"list of exactly {n} strings, each being one complaint text. "
        "Output only the complaint text in each string — no labels, metadata, "
        "or preamble."
    )

    return header + "\n".join(items) + footer


def generate_all(seed: int = 42) -> pd.DataFrame:
    """Generate all 225 complaints and return a DataFrame."""
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("Error: OPENAI_API_KEY not set. Add it to .env or environment.")
        sys.exit(1)

    client = OpenAI(api_key=api_key)
    assignments = build_assignments(seed)

    # Group assignments by cell
    cells: dict[tuple[str, str], list[dict]] = {}
    for a in assignments:
        key = (a["urgency"], a["emotion"])
        cells.setdefault(key, []).append(a)

    all_rows: list[dict] = []
    complaint_id = 1

    for cell_idx, (urgency, emotion, _count) in enumerate(GRID):
        cell_assignments = cells[(urgency, emotion)]
        system_prompt_idx = cell_assignments[0]["system_prompt_idx"]
        system_prompt = SYSTEM_PROMPTS[system_prompt_idx]
        user_prompt = _build_user_prompt(cell_assignments)

        print(f"Generating cell {cell_idx + 1}/9: "
              f"{urgency} urgency x {emotion} emotion "
              f"({len(cell_assignments)} complaints)...")

        response = client.chat.completions.create(
            model="gpt-5-mini",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=1.0,
            response_format={"type": "json_object"},
        )

        raw = response.choices[0].message.content
        data = json.loads(raw)
        complaints = data.get("complaints", [])

        if len(complaints) != len(cell_assignments):
            print(f"  Warning: expected {len(cell_assignments)} complaints, "
                  f"got {len(complaints)}. Padding/trimming.")
            # Pad with empty or trim
            while len(complaints) < len(cell_assignments):
                complaints.append("[GENERATION FAILED — re-run needed]")
            complaints = complaints[: len(cell_assignments)]

        for a, text in zip(cell_assignments, complaints):
            all_rows.append({
                "id": complaint_id,
                "complaint_text": text.strip(),
                "intended_urgency": a["urgency"],
                "intended_emotion": a["emotion"],
                "scenario": a["scenario"],
                "style": a["style"],
                "channel": a["channel"],
            })
            complaint_id += 1

    df = pd.DataFrame(all_rows)
    return df


def main() -> None:
    df = generate_all()

    output_path = os.path.join(os.path.dirname(__file__), "telecoms_complaints_sample.csv")
    df.to_csv(output_path, index=False, encoding="utf-8-sig")
    print(f"\nSaved {len(df)} complaints to {output_path}")

    # Summary
    print("\n--- Distribution Summary ---")
    print("\nScenario counts:")
    print(df["scenario"].value_counts().to_string())
    print("\nStyle counts:")
    print(df["style"].value_counts().to_string())
    print("\nChannel counts:")
    print(df["channel"].value_counts().to_string())
    print("\nUrgency x Emotion counts:")
    print(df.groupby(["intended_urgency", "intended_emotion"]).size().to_string())


if __name__ == "__main__":
    main()
