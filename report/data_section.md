# 2. Data

## 2.1 Why Synthetic Data

No publicly available dataset provides dual ground-truth labels for both urgency and emotional intensity in customer complaints. Existing open complaint corpora, such as the Consumer Financial Protection Bureau dataset, are either unlabelled on these dimensions, domain-mismatched, or annotated for sentiment alone. Real-world telecoms complaints additionally carry privacy constraints that limit redistribution. Generating synthetic data allowed us to guarantee label correctness by design: rather than annotating text post-hoc and reconciling inter-annotator disagreement, we instructed the generation model to write each complaint to a specified urgency and emotion level, making the intended label the ground truth by construction.

## 2.2 Dataset Design and Generation

The dataset consists of 5,000 complaints structured as a 3×3 grid, with urgency on one axis and emotion on the other, yielding nine distinct cells. The urgency distribution reflects the expected skew in real complaint volumes, where the majority of customer contacts do not constitute operational emergencies. Emotion is distributed evenly within each urgency group. The dataset is split 70/15/15 into training (3,500), validation (750), and test (750) sets, stratified on the full urgency-by-emotion cell to preserve the intended class distribution across all three partitions.

| | Low Emotion | Medium Emotion | High Emotion | **Total** |
|---|---|---|---|---|
| **Low Urgency** (35%) | ~583 | ~583 | ~584 | 1,750 |
| **Medium Urgency** (40%) | ~667 | ~667 | ~666 | 2,000 |
| **High Urgency** (25%) | ~417 | ~417 | ~416 | 1,250 |
| **Total** | ~1,667 | ~1,667 | ~1,666 | **5,000** |

Each complaint is assigned four characteristics prior to generation, ensuring linguistic diversity within and across cells:

| Dimension | Options | Purpose |
|---|---|---|
| Scenario | 20 UK telecoms topics | Controls subject matter |
| Writing style | 8 styles (e.g. formal, sarcastic, legalistic) | Controls surface-level linguistic form |
| Customer profile | 8 personas (e.g. elderly, small business owner) | Controls voice and framing |
| Complaint history | 4 depths (first contact to escalation) | Controls implied severity context |

No two complaints within the same grid cell share an identical combination of all four characteristics, preventing the model from learning repetitive patterns. A scenario-urgency affinity map further enforces realism by restricting which scenarios may appear at each urgency level; for example, "Complete Service Outage" is restricted to High urgency only. The full affinity map is provided in the Appendix.

Complaints were generated using GPT-5-mini in batches of five, with up to ten batches running in parallel. Each batch prompt included urgency and emotion level definitions, a tone instruction specifying how emotion should manifest through word choice and sentence structure rather than through explicit vocabulary, and the four assigned characteristics for each complaint. Incomplete or malformed batches were retried automatically up to twice.

## 2.3 Dataset Versioning and Quality Validation

Model performance on held-out data was used as a diagnostic signal for dataset quality, with three successive versions produced before the dataset was finalised.

The first version applied explicit vocabulary instructions per emotion level and restricted which writing styles could appear at each emotion level. The fine-tuned classifier achieved an emotion macro F1 of 0.99 (macro F1 is defined in Section 3.4), which indicated data leakage rather than genuine classification. The model had learned a vocabulary lookup table and a style-to-emotion mapping, neither of which required any understanding of the complaint text.

The second version removed both artefacts. All eight writing styles were permitted at all emotion levels, and vocabulary instructions were replaced with structural tone guidance. Intentional ambiguity was introduced at class boundaries so that Medium emotion, in particular, could not be resolved by any single lexical signal. Emotion macro F1 fell to 0.87, a more credible result. However, urgency macro F1 plateaued at 0.62 across all hyperparameter configurations, which pointed to a residual data quality problem rather than a modelling limitation. Human inspection of a sample of second-version complaints further revealed instances of label-complaint mismatch, where the generated text did not faithfully reflect its assigned urgency or emotion level, providing direct qualitative confirmation of the quantitative signal.

The third and final version addressed this by upgrading the generation model from GPT-4o-mini to GPT-5-mini and reducing batch size from 15 to 5 complaints per API call. With larger batches, the generation model exhibited context saturation: writing styles lost distinctiveness across the batch, and Medium urgency complaints became linguistically indistinguishable from adjacent classes. Smaller batches under a stronger model produced more consistent label-to-text fidelity. A manual inspection of a random sample confirmed the qualitative improvement. Distinguishing Medium from adjacent classes remained genuinely difficult even for human reviewers, a difficulty that is reflected in Medium achieving the lowest per-class F1 across all models in Section 4.

This iterative process reflects a broader principle of synthetic data generation: when the training signal is itself machine-produced, dataset quality engineering can be as consequential as model architecture selection, and model performance serves as a more reliable diagnostic than manual inspection alone.
