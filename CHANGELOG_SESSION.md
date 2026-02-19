# Session Change Log
**Session started:** 2026-02-19

| # | Action | File(s) | Description |
|---|--------|---------|-------------|
| 1 | Create | `CHANGELOG_SESSION.md` | Initialized session change log |
| 2 | Edit | `prompts.py` | Replaced 20 scenarios with curated UK telecoms complaint list |
| 3 | Edit | `prompts.py` | Removed `CHANNELS` list |
| 4 | Edit | `prompts.py` | Cleaned 3 system prompts — removed account/reference numbers, channel-specific formatting |
| 5 | Edit | `taxonomy.py` | Removed channel import, pool, assignment, collision resolution, and validation |
| 6 | Edit | `taxonomy.py` | Simplified dedup from (scenario, style, channel) triples to (scenario, style) pairs |
| 7 | Edit | `generate_complaints.py` | Removed channel from user prompt, output dict, and summary stats |
| 8 | Edit | `generate_complaints.py` | Removed account reference and channel formatting bullet points from prompt |
| 9 | Edit | `generate_complaints.py` | Added `_generate_batch()` helper with retry logic (up to 2 retries per batch) |
| 10 | Edit | `generate_complaints.py` | Refactored `generate_all()` to split cells into batches of 15 instead of 25-per-call |
| 11 | Edit | `prompts.py` | Added `CUSTOMER_PROFILES` (8 personas) and `COMPLAINT_HISTORY` (4 depths) |
| 12 | Edit | `prompts.py` | Added length variety instruction to all 3 system prompts |
| 13 | Rewrite | `taxonomy.py` | Made grid configurable via `total` param (default 5000), added profile/history pools, 4-tuple dedup |
| 14 | Edit | `generate_complaints.py` | Added profile/history to user prompt spec and output dict |
| 15 | Edit | `generate_complaints.py` | Added `--total` and `--seed` CLI args via argparse, updated `main()` |
| 16 | Edit | `generate_complaints.py` | Added diversity instruction to prompt header |
| 17 | Edit | `prompts.py` | Added `SCENARIO_URGENCY` dict mapping each scenario to allowed urgency levels (Low=12, Med=18, High=10) |
| 18 | Rewrite | `taxonomy.py` | Skewed urgency distribution (35/40/25%), per-urgency scenario pools, affinity validation |
| 19 | Create | `scenario_urgency_affinity.csv` | Human-readable reference table of scenario-to-urgency mappings |
