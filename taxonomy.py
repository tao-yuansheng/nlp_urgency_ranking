"""Urgency x Emotion grid, assignment logic, and distribution constraints."""

import random
from prompts import (
    SCENARIOS, STYLES, CUSTOMER_PROFILES, COMPLAINT_HISTORY,
    SCENARIO_URGENCY,
)

# ---------------------------------------------------------------------------
# Grid layout and urgency weights
# ---------------------------------------------------------------------------
URGENCY_LEVELS = ["Low", "Medium", "High"]
EMOTION_LEVELS = ["Low", "Medium", "High"]
URGENCY_WEIGHTS = {"Low": 0.35, "Medium": 0.40, "High": 0.25}


def _build_grid(total: int) -> list[tuple[str, str, int]]:
    """Build the grid with urgency-skewed counts (35/40/25%)."""
    grid: list[tuple[str, str, int]] = []
    allocated = 0

    for urg in URGENCY_LEVELS:
        urg_total = round(total * URGENCY_WEIGHTS[urg])
        # Split this urgency's total evenly across 3 emotion levels
        base, remainder = divmod(urg_total, len(EMOTION_LEVELS))
        for emo_idx, emo in enumerate(EMOTION_LEVELS):
            count = base + (1 if emo_idx < remainder else 0)
            grid.append((urg, emo, count))
            allocated += count

    # Fix any rounding drift so total is exact
    drift = total - allocated
    if drift != 0:
        # Adjust the largest cell
        largest_idx = max(range(len(grid)), key=lambda i: grid[i][2])
        urg, emo, count = grid[largest_idx]
        grid[largest_idx] = (urg, emo, count + drift)

    return grid


def _build_pool(items: list[str], total: int, min_per_item: int,
                 rng: random.Random) -> list[str]:
    """Build a shuffled pool guaranteeing each item appears >= min_per_item times."""
    pool = items * min_per_item
    remaining = total - len(pool)
    if remaining > 0:
        pool += rng.choices(items, k=remaining)
    rng.shuffle(pool)
    return pool


def _scenarios_for_urgency(urgency: str) -> list[str]:
    """Return the list of scenarios allowed at the given urgency level."""
    return [sc for sc in SCENARIOS if urgency in SCENARIO_URGENCY[sc]]


def build_assignments(total: int = 5000, seed: int = 42) -> list[dict]:
    """Return `total` complaint assignments with scenario-urgency affinity.

    Distribution: Low 35%, Medium 40%, High 25%.

    Guarantees:
    - Each scenario only appears at its allowed urgency levels.
    - Within each urgency level, scenarios are distributed as evenly as possible.
    - Global style, profile, and history distributions are balanced.
    - No duplicate (scenario, style, profile, history) tuple within the same cell.
    """
    rng = random.Random(seed)
    grid = _build_grid(total)

    # --- Per-urgency scenario pools -----------------------------------------
    # Group cells by urgency to compute per-urgency totals
    urgency_totals: dict[str, int] = {}
    for urg, _emo, count in grid:
        urgency_totals[urg] = urgency_totals.get(urg, 0) + count

    scenario_pools: dict[str, list[str]] = {}
    for urg in URGENCY_LEVELS:
        allowed = _scenarios_for_urgency(urg)
        urg_total = urgency_totals[urg]
        min_per = urg_total // len(allowed)
        scenario_pools[urg] = _build_pool(allowed, urg_total, min_per, rng)

    # --- Global pools for style, profile, history ---------------------------
    style_pool = _build_pool(STYLES, total, total // len(STYLES), rng)
    profile_pool = _build_pool(
        CUSTOMER_PROFILES, total, total // len(CUSTOMER_PROFILES), rng)
    history_pool = _build_pool(
        COMPLAINT_HISTORY, total, total // len(COMPLAINT_HISTORY), rng)

    # --- Assign per cell ----------------------------------------------------
    assignments: list[dict] = []
    global_offset = 0
    scenario_offsets: dict[str, int] = {urg: 0 for urg in URGENCY_LEVELS}

    for cell_idx, (urgency, emotion, count) in enumerate(grid):
        # Scenario: draw from per-urgency pool
        sc_off = scenario_offsets[urgency]
        cell_scenarios = list(
            scenario_pools[urgency][sc_off:sc_off + count])
        scenario_offsets[urgency] = sc_off + count

        # Style, profile, history: draw from global pools
        cell_styles = list(style_pool[global_offset:global_offset + count])
        cell_profiles = list(profile_pool[global_offset:global_offset + count])
        cell_histories = list(
            history_pool[global_offset:global_offset + count])

        # Resolve duplicate tuples by swapping profiles and histories
        used_tuples: set[tuple[str, str, str, str]] = set()
        for i in range(count):
            t = (cell_scenarios[i], cell_styles[i],
                 cell_profiles[i], cell_histories[i])
            attempts = 0
            while t in used_tuples and attempts < 300:
                j = rng.randint(0, count - 1)
                if j != i:
                    if attempts % 2 == 0:
                        cell_profiles[i], cell_profiles[j] = (
                            cell_profiles[j], cell_profiles[i])
                    else:
                        cell_histories[i], cell_histories[j] = (
                            cell_histories[j], cell_histories[i])
                t = (cell_scenarios[i], cell_styles[i],
                     cell_profiles[i], cell_histories[i])
                attempts += 1
            used_tuples.add(t)

        for i in range(count):
            assignments.append({
                "urgency": urgency,
                "emotion": emotion,
                "scenario": cell_scenarios[i],
                "style": cell_styles[i],
                "profile": cell_profiles[i],
                "history": cell_histories[i],
                "system_prompt_idx": cell_idx % 3,
            })

        global_offset += count

    # --- Validate constraints -----------------------------------------------
    from collections import Counter

    # Check scenario-urgency affinity
    for a in assignments:
        allowed = SCENARIO_URGENCY[a["scenario"]]
        if a["urgency"] not in allowed:
            raise ValueError(
                f"Scenario '{a['scenario']}' assigned to urgency "
                f"'{a['urgency']}' but only allowed at {allowed}"
            )

    # Check per-urgency scenario minimums
    for urg in URGENCY_LEVELS:
        allowed = _scenarios_for_urgency(urg)
        urg_assignments = [a for a in assignments if a["urgency"] == urg]
        counts = Counter(a["scenario"] for a in urg_assignments)
        min_expected = len(urg_assignments) // len(allowed)
        for sc in allowed:
            cnt = counts.get(sc, 0)
            if cnt < min_expected:
                raise ValueError(
                    f"Scenario '{sc}' at {urg} urgency: {cnt} times "
                    f"(< {min_expected})"
                )

    # Check global axis minimums
    for axis_name, key, items in [
        ("Style", "style", STYLES),
        ("Profile", "profile", CUSTOMER_PROFILES),
        ("History", "history", COMPLAINT_HISTORY),
    ]:
        counts = Counter(a[key] for a in assignments)
        min_expected = total // len(items)
        for item, cnt in counts.items():
            if cnt < min_expected:
                raise ValueError(
                    f"{axis_name} '{item}' only assigned {cnt} times "
                    f"(< {min_expected})"
                )

    assert len(assignments) == total, f"Expected {total}, got {len(assignments)}"
    return assignments


if __name__ == "__main__":
    import argparse
    from collections import Counter

    parser = argparse.ArgumentParser()
    parser.add_argument("--total", type=int, default=5000)
    args = parser.parse_args()

    assignments = build_assignments(total=args.total)
    print(f"Total assignments: {len(assignments)}\n")

    # Urgency distribution
    urg_counts = Counter(a["urgency"] for a in assignments)
    print("Urgency distribution:")
    for urg in URGENCY_LEVELS:
        pct = 100 * urg_counts[urg] / len(assignments)
        print(f"  {urg}: {urg_counts[urg]} ({pct:.1f}%)")
    print()

    # Per-urgency scenario breakdown
    for urg in URGENCY_LEVELS:
        urg_assigns = [a for a in assignments if a["urgency"] == urg]
        print(f"Scenarios at {urg} urgency ({len(urg_assigns)} total):")
        for sc, cnt in sorted(Counter(a["scenario"] for a in urg_assigns).items()):
            print(f"  {sc}: {cnt}")
        print()

    # Global axes
    for axis_name, key in [
        ("Style", "style"),
        ("Profile", "profile"),
        ("History", "history"),
    ]:
        print(f"{axis_name} distribution:")
        for item, cnt in sorted(Counter(a[key] for a in assignments).items()):
            print(f"  {item}: {cnt}")
        print()

    # Per-cell breakdown
    print("Per-cell breakdown:")
    grid = _build_grid(args.total)
    for urg, emo, count in grid:
        cell = [a for a in assignments
                if a["urgency"] == urg and a["emotion"] == emo]
        tuples = [(a["scenario"], a["style"], a["profile"], a["history"])
                  for a in cell]
        print(f"  {urg} urgency x {emo} emotion: {len(cell)} complaints, "
              f"unique tuples: {len(set(tuples))}/{len(tuples)}")
