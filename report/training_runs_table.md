# Training Run Hyperparameters

All runs use DeBERTa-v3-base (`microsoft/deberta-v3-base`), batch size 16, AdamW optimiser, and early stopping on validation loss unless noted otherwise. "Changes" describes what was modified relative to the immediately preceding run.

| Run | Dataset | `max_length` | LR (backbone) | LR (heads) | Max Epochs | Weight Decay | Changes from previous run |
|:---:|:-------:|:------------:|:-------------:|:----------:|:----------:|:------------:|:--------------------------|
| Run 1 | v1 | 128 | 2×10⁻⁵ | — | 5 | — | **Baseline.** Unified LR, mixed-precision (fp16), no class weights, no LR scheduler, patience = 2. |
| Run 2 | v2 | 128 | 2×10⁻⁵ | — | 5 | — | Dataset updated to v2; all other settings identical to Run 1. |
| Run 3 | v2 | 192 | 2×10⁻⁵ | — | 10 | — | `max_length` increased 128 → 192; max epochs extended to 10; patience increased to 3. |
| Run 4 | v2 | 192 | 1×10⁻⁵ | — | 10 | — | Unified LR halved (2×10⁻⁵ → 1×10⁻⁵). |
| Run 5 | v3 | 192 | 2×10⁻⁵ | — | 10 | — | Dataset updated to v3; unified LR restored to 2×10⁻⁵. |
| Run 6 | v3 | 192 | 1×10⁻⁵ | 5×10⁻⁵ | 10 | — | Introduced **split learning rates**: lower LR for DeBERTa backbone, higher LR for task-specific classification heads. |
| Run 7 | v3 | 192 | 1×10⁻⁵ | 2×10⁻⁵ | 15 | 0.05 | Head LR reduced (5×10⁻⁵ → 2×10⁻⁵); weight decay (0.05) added; max epochs extended to 15. |
| **Run 8** ✓ | v3 | 192 | 2×10⁻⁵ | — | 10 | — | **Early stopping criterion changed from val loss to combined macro F1** (mean of urgency and emotion F1). Config otherwise identical to Run 5. Best overall mean F1. |

**Notes:**
- "—" in LR (heads) indicates a unified learning rate applied to all parameters.
- Runs 2–4 share the same dataset (v2) but explore different sequence length and LR configurations.
- Runs 5–7 use dataset v3 and progressively refine the optimisation strategy.
