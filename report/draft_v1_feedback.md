# Critical Evaluation: MSIN0221_Group_Assignment_v1.docx

## Overall Impression

The draft is substantially stronger than a typical student NLP report. The framing is clear, the writing is academic without being bloated, and the dataset versioning narrative is genuinely novel and well-argued. The structure is coherent from introduction through to conclusion. That said, there are several specific issues worth addressing before submission.

---

## Section-by-Section Critique

### Section 1 — Introduction

The opening example is excellent and immediately motivates the problem. The literature review is integrated well rather than dumped in a standalone section.

Issues:
- The contributions list says "three contributions" but the second contribution says "we train and compare three classification architectures" — there are four models (TF-IDF, SBERT, Haiku, DeBERTa). Fix the count or revise the description.
- "The paper used an inter-model benchmark technique; we implemented an intra-model data-matching technique" switches voice mid-sentence. Rewrite to stay consistent ("we").
- The ICP citation (Ashury-Tahan et al., 2026) is a 2026 arXiv paper. Verify it is accessible and correctly cited — it may raise examiner questions.

---

### Section 2 — Data

Well-written overall. Two issues:

- The table in 2.2 has inconsistent column widths in the rendered docx — the Scenario row breaks oddly. Minor formatting fix needed.
- The claim "No two complaints within the same grid cell share an identical combination of all four characteristics" is a strong guarantee. Confirm this is enforced in code. If it is, add a brief parenthetical confirming it. If not, soften the language.

---

### Section 3 — Methodology

- The comment left in 3.3 ("could be repetitive with introduction?") is valid. The disentangled attention explanation partially overlaps with the intro's contributions list. Fix: trim the intro's DeBERTa reference to just the citation and model name; keep all technical detail in 3.3 only.
- The early stopping paragraph in the docx omits the comparison between urgency-only and combined F1 monitoring (0.827 vs. 0.830). This detail justifies the design choice empirically rather than just asserting it. It should be reinstated from the methodology markdown file.

---

### Section 4 — Results

4.1 is strong. One factual check required:

- The text states TF-IDF outperforms DeBERTa on urgency "by 1.6 percentage points (0.819 vs. 0.803)". The actual metrics JSON shows TF-IDF urgency = 0.8192 and DeBERTa urgency = 0.8071, a gap of 1.2 pp, not 1.6. Verify against the results file before submission.

4.2 is clean. The comment "update when possible" on Figure X just needs the figure number filled in.

4.3 is solid. The loss-vs-F1 divergence observation is a strong empirical justification for the early stopping choice.

4.4 references "Figure 5" in the caption but the figures are sequentially numbered 1–4 before this point. Either the figure numbering is wrong or a figure is missing. Check and correct.

---

### Section 5 — Contrastive Error Analysis

This section changes register noticeably. Sections 1–4 read as a polished conference-style paper; Section 5 reads closer to a student report. Specific issues:

- **Inconsistent error counts**: Section 5.1 reports 155 urgency errors and 112 emotion errors, but Section 4.4 reports 151 and 107 respectively. These must be reconciled before submission — an examiner will notice immediately.
- **Informal headers**: "Top 3 Urgency Blindspots" and "Top 3 Emotion Blindspots" are informal. Rename to "Principal urgency failure modes" and "Principal emotion failure modes" or similar.
- **Table formatting**: The two blindspot tables have inconsistent column widths and right-column text wraps mid-cell. Formatting fix needed.
- **Adversarial test (5.4)**: 10 samples is too small to make strong claims from. The current framing ("preliminary support") is appropriately hedged, but "The model passed 6/10 tests" reads as weak validation. Consider framing this as a qualitative illustration of the blindspots rather than a pass/fail evaluation.

---

### Section 6 — Limitations

- Written as four separate paragraphs each introducing a new limitation in isolation, which reads as a checklist. Restructure into flowing prose that groups related limitations (e.g., synthetic text quality + noise gap; design constraints + real-world edge cases).
- "GPT-5o-mini" in the first sentence is a typo. The rest of the paper uses "GPT-5-mini". Fix for consistency.

---

### Section 7 — Conclusion

Well-written and appropriately scoped. No major issues. One suggestion: the final sentence on future work could be one level more specific by naming the particular blindspot categories (e.g., formal legal threats, cumulative issue histories) identified in Section 5 rather than describing them generically.

---

### References

The reference list is largely correct. One potential gap: Li et al. (2023) appears in the team's literature review table (screenshot shared separately) but does not appear in the reference list. Either it was removed from the body text and the table is outdated, or it needs to be added. Verify.

---

## Priority Fixes Before Submission

| # | Issue | Location |
|---|-------|----------|
| 1 | Reconcile error counts: 155/112 (§5.1) vs 151/107 (§4.4) | §4.4 and §5.1 |
| 2 | Fix urgency gap figure: stated 1.6pp, actual ~1.2pp | §4.1 |
| 3 | Fix figure numbering — "Figure 5" appears out of sequence | §4.4 |
| 4 | Fix voice switch in contributions paragraph | §1 |
| 5 | Fix "GPT-5o-mini" typo | §6 |
| 6 | Reinstate early stopping comparison sentence (0.827 vs 0.830) | §3.3 |
| 7 | Rename informal blindspot headers | §5.2, §5.3 |
| 8 | Restructure Limitations into flowing prose | §6 |
