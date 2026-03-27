# 2. Data

## 2.1 Why Synthetic Data

No publicly available dataset provides dual ground-truth labels for both urgency and emotional intensity in customer complaints. Existing open complaint corpora, such as the Consumer Financial Protection Bureau dataset, are either unlabelled on these dimensions, domain-mismatched, or annotated for a single dimension only. Those that do include emotional annotation rarely capture operational severity simultaneously, and those labelled for severity tend to treat emotional expression as secondary or absent altogether. The two dimensions are seldom framed as independent classification targets in publicly available resources. Real-world telecoms complaints additionally carry privacy constraints that limit redistribution. Generating synthetic data allowed us to guarantee label correctness by design: rather than annotating text post-hoc and reconciling inter-annotator disagreement, we instructed the generation model to write each complaint to a specified urgency and emotion level, making the intended label the ground truth by construction.

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

Complaints were generated using GPT-5-mini in batches of five, with up to ten batches running in parallel. Each batch prompt included urgency and emotion level definitions, a tone instruction specifying how emotion should manifest through word choice and sentence structure rather than through explicit vocabulary, and the four assigned characteristics for each complaint. To prevent stylistic convergence across 5,000 complaints, three distinct system prompt templates were rotated across batches. Each template conveyed the same core generation instruction but varied in framing and persona guidance, ensuring that surface-level phrasing patterns in the generated text could not be traced back to a single prompt formulation. 

## 2.3 Dataset Versioning and Quality Validation

Model performance on held-out data was used as a diagnostic signal for dataset quality, with three successive versions produced before the dataset was finalised.

The first version applied explicit vocabulary instructions per emotion level and restricted which writing styles could appear at each emotion level. This introduced data leakage: the model learned a vocabulary lookup table and a style-to-emotion mapping rather than any genuine understanding of the complaint text, producing unrealistically high emotion classification performance.

The second version removed both artefacts by permitting all writing styles at all emotion levels and replacing vocabulary instructions with structural tone guidance. Intentional ambiguity was introduced at class boundaries so that no single lexical signal could resolve the emotion class. Despite this, urgency classification plateaued across all hyperparameter configurations, and human inspection of a sample of complaints revealed further instances of label-complaint mismatch, where the generated text did not faithfully reflect its assigned labels.

The third and final version addressed this by upgrading the generation model from GPT-4o-mini to GPT-5-mini and reducing batch size from 15 to 5 complaints per API call. With larger batches, the generation model exhibited context saturation, causing writing styles to lose distinctiveness and adjacent urgency classes to become linguistically indistinguishable. Smaller batches under a stronger model produced more consistent label-to-text fidelity. The quantitative impact of these changes is reported in Section 4.

This iterative process reflects a broader principle of synthetic data generation: when the training signal is itself machine-produced, dataset quality engineering can be as consequential as model architecture selection, and model performance serves as a more reliable diagnostic than manual inspection alone.
