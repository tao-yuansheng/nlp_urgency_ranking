"""Urgency × Emotion grid, scenario/style assignment, and distribution logic."""

import random
from prompts import SCENARIOS, STYLES

# ---------------------------------------------------------------------------
# Grid definition: (urgency, emotion) → count
# ---------------------------------------------------------------------------
GRID = [
    ("Low",    "Low",    12),
    ("Low",    "Medium", 11),
    ("Low",    "High",   11),
    ("Medium", "Low",    11),
    ("Medium", "Medium", 11),
    ("Medium", "High",   11),
    ("High",   "Low",    11),
    ("High",   "Medium", 11),
    ("High",   "High",   11),
]


def build_assignments(seed: int = 42) -> list[dict]:
    """Return 100 complaint assignments satisfying all distribution constraints.

    Guarantees:
    - Each scenario is used >= 10 times (8 scenarios, 100 complaints).
    - Each style is used >= 16 times (5 styles, 100 complaints).
    - No duplicate (scenario, style) pair within the same cell.
    """
    rng = random.Random(seed)
    total = sum(count for *_, count in GRID)  # 50

    # --- Build scenario and style pools with minimum guarantees -----------
    # Minimum fills: 8 scenarios × 10 = 80, need 20 more
    scenario_pool = SCENARIOS * 10  # 80 guaranteed
    extras_scenario = rng.choices(SCENARIOS, k=total - len(scenario_pool))
    scenario_pool += extras_scenario
    rng.shuffle(scenario_pool)

    # Minimum fills: 5 styles × 16 = 80, need 20 more
    style_pool = STYLES * 16  # 80 guaranteed
    extras_style = rng.choices(STYLES, k=total - len(style_pool))
    style_pool += extras_style
    rng.shuffle(style_pool)

    # --- Assign to cells, avoiding duplicate (scenario, style) per cell ---
    assignments: list[dict] = []
    s_idx = 0  # pointer into scenario_pool
    t_idx = 0  # pointer into style_pool

    for cell_idx, (urgency, emotion, count) in enumerate(GRID):
        used_pairs: set[tuple[str, str]] = set()
        cell_assignments: list[dict] = []

        for _ in range(count):
            # Try to find a non-duplicate pair
            scenario = scenario_pool[s_idx % len(scenario_pool)]
            style = style_pool[t_idx % len(style_pool)]
            attempts = 0
            while (scenario, style) in used_pairs and attempts < 50:
                # Swap style to resolve collision
                t_idx += 1
                style = style_pool[t_idx % len(style_pool)]
                attempts += 1

            used_pairs.add((scenario, style))
            cell_assignments.append({
                "urgency": urgency,
                "emotion": emotion,
                "scenario": scenario,
                "style": style,
                "system_prompt_idx": cell_idx % 3,
            })
            s_idx += 1
            t_idx += 1

        assignments.extend(cell_assignments)

    # --- Validate constraints ------------------------------------------------
    from collections import Counter

    scenario_counts = Counter(a["scenario"] for a in assignments)
    style_counts = Counter(a["style"] for a in assignments)

    for sc, cnt in scenario_counts.items():
        if cnt < 10:
            raise ValueError(f"Scenario '{sc}' only assigned {cnt} times (<10)")
    for st, cnt in style_counts.items():
        if cnt < 16:
            raise ValueError(f"Style '{st}' only assigned {cnt} times (<16)")

    assert len(assignments) == total, f"Expected {total}, got {len(assignments)}"
    return assignments


if __name__ == "__main__":
    from collections import Counter

    assignments = build_assignments()
    print(f"Total assignments: {len(assignments)}\n")

    print("Scenario distribution:")
    for sc, cnt in sorted(Counter(a["scenario"] for a in assignments).items()):
        print(f"  {sc}: {cnt}")

    print("\nStyle distribution:")
    for st, cnt in sorted(Counter(a["style"] for a in assignments).items()):
        print(f"  {st}: {cnt}")

    print("\nPer-cell breakdown:")
    for urg, emo, count in GRID:
        cell = [a for a in assignments if a["urgency"] == urg and a["emotion"] == emo]
        pairs = [(a["scenario"], a["style"]) for a in cell]
        print(f"  {urg} urgency × {emo} emotion: {len(cell)} complaints, "
              f"unique pairs: {len(set(pairs))}/{len(pairs)}")
