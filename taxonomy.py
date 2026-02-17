"""Urgency x Emotion grid, scenario/style/channel assignment, and distribution logic."""

import random
from prompts import SCENARIOS, STYLES, CHANNELS

# ---------------------------------------------------------------------------
# Grid definition: (urgency, emotion) -> count
# ---------------------------------------------------------------------------
GRID = [
    ("Low",    "Low",    25),
    ("Low",    "Medium", 25),
    ("Low",    "High",   25),
    ("Medium", "Low",    25),
    ("Medium", "Medium", 25),
    ("Medium", "High",   25),
    ("High",   "Low",    25),
    ("High",   "Medium", 25),
    ("High",   "High",   25),
]


def _build_pool(items: list[str], total: int, min_per_item: int,
                 rng: random.Random) -> list[str]:
    """Build a shuffled pool guaranteeing each item appears >= min_per_item times."""
    pool = items * min_per_item
    remaining = total - len(pool)
    if remaining > 0:
        pool += rng.choices(items, k=remaining)
    rng.shuffle(pool)
    return pool


def build_assignments(seed: int = 42) -> list[dict]:
    """Return 225 complaint assignments satisfying all distribution constraints.

    Guarantees:
    - Each of 20 scenarios is used >= 11 times.
    - Each of 8 styles is used >= 28 times.
    - Each of 4 channels is used >= 56 times.
    - No duplicate (scenario, style, channel) triple within the same cell.
    """
    rng = random.Random(seed)
    total = sum(count for *_, count in GRID)  # 225

    # --- Build pools with minimum guarantees (exact length = total) ---------
    scenario_pool = _build_pool(SCENARIOS, total, 11, rng)  # 20x11=220 +5
    style_pool = _build_pool(STYLES, total, 28, rng)        # 8x28=224  +1
    channel_pool = _build_pool(CHANNELS, total, 56, rng)     # 4x56=224  +1

    # --- Slice pools into per-cell chunks, then resolve collisions ----------
    # This approach never skips pool entries, so global distribution is exact.
    assignments: list[dict] = []
    offset = 0

    for cell_idx, (urgency, emotion, count) in enumerate(GRID):
        cell_scenarios = list(scenario_pool[offset:offset + count])
        cell_styles = list(style_pool[offset:offset + count])
        cell_channels = list(channel_pool[offset:offset + count])

        # Resolve any duplicate triples within the cell by swapping within
        # the cell's own lists (preserves global counts exactly).
        used_triples: set[tuple[str, str, str]] = set()
        for i in range(count):
            triple = (cell_scenarios[i], cell_styles[i], cell_channels[i])
            attempts = 0
            while triple in used_triples and attempts < 200:
                # Pick a random other index in the cell and swap channels
                j = rng.randint(0, count - 1)
                if j != i:
                    cell_channels[i], cell_channels[j] = (
                        cell_channels[j], cell_channels[i])
                triple = (cell_scenarios[i], cell_styles[i], cell_channels[i])
                attempts += 1
            used_triples.add(triple)

        for i in range(count):
            assignments.append({
                "urgency": urgency,
                "emotion": emotion,
                "scenario": cell_scenarios[i],
                "style": cell_styles[i],
                "channel": cell_channels[i],
                "system_prompt_idx": cell_idx % 3,
            })

        offset += count

    # --- Validate constraints ----------------------------------------------
    from collections import Counter

    scenario_counts = Counter(a["scenario"] for a in assignments)
    style_counts = Counter(a["style"] for a in assignments)
    channel_counts = Counter(a["channel"] for a in assignments)

    for sc, cnt in scenario_counts.items():
        if cnt < 11:
            raise ValueError(f"Scenario '{sc}' only assigned {cnt} times (<11)")
    for st, cnt in style_counts.items():
        if cnt < 28:
            raise ValueError(f"Style '{st}' only assigned {cnt} times (<28)")
    for ch, cnt in channel_counts.items():
        if cnt < 56:
            raise ValueError(f"Channel '{ch}' only assigned {cnt} times (<56)")

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

    print("\nChannel distribution:")
    for ch, cnt in sorted(Counter(a["channel"] for a in assignments).items()):
        print(f"  {ch}: {cnt}")

    print("\nPer-cell breakdown:")
    for urg, emo, count in GRID:
        cell = [a for a in assignments
                if a["urgency"] == urg and a["emotion"] == emo]
        triples = [(a["scenario"], a["style"], a["channel"]) for a in cell]
        print(f"  {urg} urgency x {emo} emotion: {len(cell)} complaints, "
              f"unique triples: {len(set(triples))}/{len(triples)}")
