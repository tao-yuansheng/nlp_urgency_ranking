---
title: "Decoupling Urgency and Emotion in Automated Telecom Complaint Triage"
authors: "Liqiang Cheng, Olena Brylinska, Yuansheng Tao, Yuyi Song, Yuzhi Zhao"
course: "MSIN0221 Group Assignment — Group 16"
---

Decoupling Urgency and Emotion in Automated Telecom Complaint Triage


# Introduction

Customer-facing complaint handling in the telecommunications sector presents a triage challenge that is difficult to automate. When a support agent receives an inbound complaint, they must make two simultaneous judgements: how objectively serious is the underlying issue, and how emotionally escalated is the customer expressing that issue? These are not the same questions. A customer who writes *"I would like to **flag**, at your earliest convenience, that our office has had no **connectivity** for seventy-two hours and we are losing approximately three thousand pounds per day"* is describing a catastrophic situation in perfectly calm prose. A customer who writes *"THIS IS ABSOLUTELY DISGUSTING I CANNOT BELIEVE YOU CHANGED THE APP FONT"* is emotionally escalated about something trivial. Misrouting these two cases, by deprioritising the first because the language is polite or over-escalating the second because the language is aggressive, is precisely the failure mode that automated triage must avoid. Research also shows that even the best available language models often mix up closely related emotion categories; for example, treating Medium intensity as High, when they have not been fine-tuned for the specific domain in question (Zhang et al., 2024).

The case for automating complaint triage with natural language processing is well established. Vairetti et al. (2024) demonstrate that transformer-based deep learning models can automatically label and prioritise large volumes of complaints with high accuracy, achieving 92.1% accuracy on a dataset of over 44,000 complaints from a Chilean occupational safety organisation using a BERT-based architecture combined with multicriteria decision-making. More recently, Roumeliotis et al. (2025) evaluated fourteen large language models, including GPT-4o, Claude, and Gemini, on zero-shot consumer complaint classification, finding that while reasoning models offer improved contextual understanding, they incur significantly higher processing time and inference cost compared to fine-tuned specialist models. These findings establish the viability of NLP for complaint handling while highlighting a tension between model expressiveness and operational cost. A related challenge concerns the availability of labelled training data for such tasks. Long et al. (2024) propose a generic framework for LLM-based synthetic dataset generation using conditional prompting, demonstrating that LLMs can produce diverse, task-relevant training examples at low cost. However, Tan et al. (2024) caution that LLM-generated annotations carry inherent quality risks, including label-text deception, where the generated label does not truthfully reflect the content of the generated text, a limitation that motivates careful quality control in any synthetically labelled pipeline.

A key gap in existing literature motivates the present work. Existing complaint prioritisation frameworks treat urgency as a composite label that conflates objective issue severity with the emotional tone of the customer expressing it. Vairetti et al. (2024), derive urgency labels from a weighted combination of multiple complaint factors without separating emotional intensity as an independent dimension. This design choice means that the relationship between emotional expression and objective urgency is never examined as a potential confound, and models trained under such frameworks have no mechanism to distinguish a calm customer describing a catastrophic outage from an emotionally escalated customer complaining about something trivial. This problem is further illustrated by Athira et al. (2025), who find that different model architectures perform best on different complaint analysis dimensions, with no single architecture achieving uniform high performance across sentiment, emotion, and severity classification simultaneously. Their finding suggests that these dimensions require distinct modelling approaches rather than treatment as a unified composite label, yet no prior dataset has been explicitly constructed to enforce this separation at the data generation stage. Against this background, this paper asks: How effectively can different NLP models classify urgency and emotional intensity independently in synthetic UK telecom complaints, and what error patterns emerge when these dimensions are disentangled by design?

This paper presents three contributions that directly address this gap. First, we introduce a 5,000-complaint synthetic dataset of UK telecoms complaints with fine-grained orthogonal control over multiple dimensions, in which urgency and emotional intensity are deliberately decoupled by design. Labels are generated using a conditional prompting pipeline with quality controls informed by the risks identified by Tan et al. (2024). Second, we train and compare four classification architectures: a TF-IDF with Logistic Regression baseline, a frozen Sentence-BERT encoder with Logistic Regression (Reimers and Gurevych, 2019), Few-shot Claude Haiku 4.5, and a fine-tuned DeBERTa-v3-base model. Third, we conduct a per-instance causal error analysis augmented by Informative Correct Predictions (ICP), a concept introduced by Ashury-Tahan et al. (2026) in an inter-model benchmark context. We adapt this technique to an intra-model data-matching approach, producing a taxonomy of failure modes that identifies the specific linguistic patterns responsible for each class of mistake.


# Data


## Why Synthetic Data

No publicly available dataset provides dual ground-truth labels for both urgency and emotional intensity in customer complaints. Existing open complaint corpora, such as the Consumer Financial Protection Bureau dataset, are either unlabelled on these dimensions, domain-mismatched, or annotated for a single dimension only. The two dimensions are seldom framed as independent classification targets in publicly available resources. Real-world telecoms complaints additionally carry privacy constraints that limit redistribution. Generating synthetic data allowed us to guarantee label correctness by design, that is rather than annotating text post-hoc and reconciling inter-annotator disagreement, we instructed the generation model to write each complaint to a specified urgency and emotion level, making the intended label the ground truth by construction.


## Dataset Design and Generation

The dataset consists of 5,000 complaints structured as a 3×3 grid, with urgency on one axis and emotion on the other. The urgency distribution reflects a realistic skew, where critical operational contacts are assumed to be less frequent than routine ones. Emotion is distributed evenly within each urgency group. The dataset is split 70/15/15 into training (3,500), validation (750), and test (750) sets, stratified on the full urgency-by-emotion cell to preserve the intended class distribution across all three partitions.


![Joint Urgency x Emotion Distribution (%)](1_2_joint_distribution.png)
*Joint Urgency x Emotion Distribution (%)*

Each complaint is assigned four characteristics prior to generation, ensuring linguistic diversity within and across cells:


| Dimension | Options | Purpose |
| --- | --- | --- |
| Scenario | 20 UK telecoms topics | Controls subject matter |
| Writing style | 8 styles (e.g. formal, sarcastic, legalistic) | Controls surface-level linguistic form |
| Customer profile | 8 personas (e.g. elderly, small business owner) | Controls voice and framing |
| Complaint history | 4 depths (first contact to escalation) | Controls implied severity context |


**Table: Synthetic Data creation rules**

No two complaints within the same grid cell share an identical combination of all four characteristics, preventing the model from learning repetitive patterns. A scenario-urgency affinity map further enforces realism by restricting which scenarios may appear at each urgency level. The full affinity map is provided in the Appendix.

Complaints were generated using GPT-5-mini in batches of five, with up to ten batches running in parallel. Each batch prompt included urgency and emotion level definitions, a tone instruction specifying how emotion should manifest through word choice and sentence structure rather than through explicit vocabulary, and the four assigned characteristics for each complaint. To prevent stylistic convergence across 5,000 complaints, three distinct system prompt templates were rotated across batches. Each template conveyed the same core generation instruction but varied in framing and persona guidance, ensuring that surface-level phrasing patterns in the generated text could not be traced back to a single prompt formulation.


## Dataset Versioning and Quality Validation

Model performance on held-out data was used as a diagnostic signal for dataset quality, with three successive versions produced before the dataset was finalised.

The first version applied explicit vocabulary instructions per emotion level and restricted which writing styles could appear at each emotion level. This introduced data leakage, where  the model learned a vocabulary lookup table and a style-to-emotion mapping rather than any genuine understanding of the complaint text, producing unrealistically high emotion classification performance.

The second version removed both artefacts by permitting all writing styles at all emotion levels and replacing vocabulary instructions with structural tone guidance. Intentional ambiguity was introduced at class boundaries so that no single lexical signal could resolve the emotion class. Despite this, urgency classification plateaued across all hyperparameter configurations, and human inspection of a sample of complaints revealed further instances of label-complaint mismatch, where the generated text did not faithfully reflect its assigned labels.

The third and final version addressed this by upgrading the generation model from GPT-4o-mini to GPT-5-mini and reducing batch size from 15 to 5 complaints per API call. With larger batches, the generation model exhibited context saturation, causing writing styles to lose distinctiveness and adjacent urgency classes to become linguistically indistinguishable. Smaller batches under a stronger model produced more consistent label-to-text fidelity. The quantitative impact of these changes is reported in Section 4.

This iterative process reflects a broader principle of synthetic data generation. When the training signal is itself machine-produced, dataset quality engineering can be as consequential as model architecture selection, and model performance serves as a more reliable diagnostic than manual inspection alone.


# Methodology


## Task Formulation

The task is formulated as dual multi-class classification. Given a customer complaint text, the model must independently predict an urgency level and an emotion level, each drawn from the set {Low, Medium, High}. The two dimensions are treated as independent classification targets. Urgency captures the operational severity of the issue described, while emotion captures the affective intensity of the language used. 


## Baselines

Three baseline models of increasing representational capacity were evaluated to contextualise the performance of the fine-tuned model, each probing a distinct question about where the classification signal resides. All baselines were evaluated on the same stratified test split of 750 complaints.

The first baseline, few-shot Claude Haiku 4.5, establishes whether a general-purpose large language model can perform the task from definitions and labelled examples alone, without any parameter updates. Nine labelled examples were provided in the system prompt, one for each urgency-by-emotion cell, sampled deterministically from the training set using a fixed seed. Complaints were batched in groups of ten per API call and the model was prompted to return structured JSON classifications at temperature 0, with no chain-of-thought reasoning elicited.

Where the LLM baseline relies entirely on in-context learning, the second baseline grounds classification in surface-level lexical features. Complaint texts were vectorised using TF-IDF with unigram and bigram features, a maximum vocabulary of 20,000 terms, and sublinear term frequency scaling. Two independent logistic regression classifiers were then trained, one per target dimension, using L2 regularisation with C=1.0 and the LBFGS solver. This approach provides a reference for how much of the classification signal is accessible through word frequency patterns alone, without any contextual language understanding.

The third baseline extends this by replacing sparse lexical features with dense semantic representations, while still holding the encoder fixed. Complaint texts were encoded using all-MiniLM-L6-v2, a sentence transformer producing 384-dimensional embeddings (Reimers and Gurevych, 2019), with all encoder weights frozen. The same logistic regression configuration was applied on top of these embeddings. By keeping the encoder frozen, this baseline isolates the contribution of general-purpose semantic representations from that of task-specific adaptation, establishing a direct lower bound on what fine-tuning adds over off-the-shelf embeddings.


## Fine-tuned DeBERTa-v3-base

DeBERTa-v3-base was selected over standard BERT for its disentangled attention mechanism (He et al., 2021), with the v3 variant further improving pre-training efficiency through an ELECTRA-style replaced token detection objective (He et al., 2023), which decouples content and position embeddings into separate vectors that interact through dedicated attention matrices. This architecture enables the model to distinguish between what a word means and where it appears in the sequence, a property particularly relevant for complaint classification where positional cues such as escalation language appearing at the end of a message or urgency markers embedded in subordinate clauses carry diagnostic value.

The model uses a shared encoder with two independent linear classification heads, following the parallel multi-task architecture described by Chen et al. (2024). The [CLS] token representation from the final hidden layer (768 dimensions) is passed to both an urgency head and an emotion head, each projecting to three output classes. Both heads are trained jointly by summing their respective cross-entropy losses. The urgency head applies class weights of [1.0, 1.5, 1.2] for Low, Medium, and High respectively. Medium receives the highest weight not because of class frequency (it is in fact the most frequent class at 40%) but because it structurally overlaps with both adjacent classes, making it the hardest to distinguish. The emotion head uses unweighted cross-entropy, as emotion classes are evenly distributed and boundary ambiguity was already addressed through the dataset design described in Section 2.

Early stopping monitors the combined validation macro F1, defined as the mean of validation urgency and emotion macro F1. Monitoring validation loss alone did not reliably track improvements in classification quality across both tasks. Training halts after three consecutive epochs with no improvement in the combined F1, and the model weights from the best epoch are restored.


| Parameter | Value |
| --- | --- |
| Base model | microsoft/deberta-v3-base |
| Max sequence length | 192 tokens |
| Batch size | 16 |
| Optimiser | AdamW |
| Learning rate | 2e-5 |
| LR schedule | Linear warmup (10% of total steps) with linear decay |
| Gradient clipping | Max norm 1.0 |
| Max epochs | 10 |
| Early stopping patience | 3 (on validation combined macro F1) |
| Urgency class weights | [1.0, 1.5, 1.2] (Low / Medium / High) |
| Emotion class weights | None (uniform) |
| Mixed precision | FP16 on CUDA |


**Table: Training Model Set up**


## Evaluation Metrics

All models are evaluated using macro-averaged F1 score, defined as the unweighted mean of per-class F1 scores across the three classes. Macro averaging weights each class equally regardless of its frequency in the test set, which is preferable to accuracy or micro-averaged F1 in the presence of class imbalance. Macro F1 penalises models that perform well on frequent classes but poorly on rare ones, providing a more balanced assessment of classification quality across the full label space.

Per-class F1 scores are additionally reported for each model and each dimension, enabling fine-grained comparison of where models succeed and fail across the urgency and emotion spectra.


# Results & Evaluation


## Overall Model Comparison


![Test Set Macro F1 by Model](fig1_model_comparison.png)
*Test Set Macro F1 by Model*

The fine-tuned DeBERTa-v3-base achieves the highest combined performance with a mean macro F1 of 0.830 (urgency 0.803, emotion 0.857), followed by TF-IDF with Logistic Regression at 0.792, frozen Sentence-BERT with Logistic Regression at 0.661, and few-shot Claude Haiku 4.5 at 0.516. TF-IDF outperforms DeBERTa on urgency by 1.6 percentage points (0.819 vs. 0.803), suggesting that urgency is partly solvable by lexical features alone, scenario-specific n-grams like "outage" or "disconnected" provide strong class signal without contextual understanding. DeBERTa's advantage concentrates on emotion, where it gains 9.3 percentage points over TF-IDF (0.857 vs. 0.764), as emotion classification requires interpreting tone, register, and pragmatic intent that bag-of-words features cannot capture.

The frozen SBERT baseline underperforms TF-IDF on both dimensions by over 12 percentage points. This reflects a domain gap: all-MiniLM-L6-v2 optimises for paraphrase detection, compressing complaints into a semantic space that discards the fine-grained lexical signals TF-IDF preserves. Few-shot Haiku 4.5 scores lowest, consistent with Roumeliotis et al. (2025), who observe that general-purpose LLMs struggle with fine-grained ordinal distinctions without task-specific training.


![Per-Class F1 Scores on Test Set](fig2_per_class_f1.png)
*Per-Class F1 Scores on Test Set*

A class-level breakdown of F1 scores across models and dimensions further clarifies this pattern. Medium is the worst-performing class across every model and both dimensions. This is consistent with Lu et al. (2024), who demonstrate that classifiers, large language models included, systematically concentrate confusion at middle categories, where decision boundaries between adjacent classes are least separable. DeBERTa's urgency Medium F1 (0.734) trails Low (0.835) and High (0.839) by approximately 10 percentage points. This is consistent with the dataset design: Medium was deliberately constructed to overlap with adjacent classes (Section 2.3).


## Dataset Versioning with DeBERTa-v3


![Test Macro F1 Score Across Training Runs](training_runs_f1.png)
*Test Macro F1 Score Across Training Runs*

Figure 4 traces DeBERTa-v3-base test performance across eight training runs spanning three dataset versions, illustrating how classification quality evolved as a function of data quality rather than model configuration.

The single run on Dataset v1 produced an emotion macro F1 of 0.991, an implausibly high result that confirmed the data leakage described in Section 2.3. The model had effectively learned a vocabulary lookup table rather than any genuine emotional understanding, rendering the result uninformative as a measure of classification capability.

Dataset v2 corrected the leakage artefacts, and emotion macro F1 fell to a credible range of 0.835 to 0.866 across three runs with varying hyperparameters. However, urgency macro F1 remained invariant across those same runs, spanning only 0.603 to 0.622 regardless of changes to learning rate, class weights, and sequence length. The narrow range of urgency scores across three independently configured runs is the key diagnostic: if the ceiling were a modelling problem, hyperparameter changes would have produced some variation. The fact that they did not pointed unambiguously to a data quality constraint.

The transition to Dataset v3 produced an immediate and substantial improvement in urgency macro F1, rising from 0.615 on the final v2 run to 0.801 on the first v3 run, an increase of 18.6 percentage points. This step change, achieved without any modification to the model architecture or training configuration, confirms that the v2 urgency ceiling was caused by generation quality rather than model capacity. The finding aligns with Iskander et al. (2024), who empirically demonstrate that models trained on smaller but higher-quality synthetic datasets consistently outperform those trained on larger unvalidated sets. Across four runs on Dataset v3, urgency macro F1 remained stable between 0.801 and 0.811, and emotion macro F1 between 0.830 and 0.857, indicating that the model had reached a consistent performance level bounded by the v3 dataset.

Run 8 was selected as the final model on the basis of achieving the highest combined macro F1, with an urgency score of 0.802 and an emotion score of 0.857. Details of each run is presented in the Appendix.


## Training Dynamics – Run 8


![DeBERTa Training Dynamics](fig3_training_dynamics.png)
*DeBERTa Training Dynamics*

The model was selected at Epoch 6 by early stopping on validation combined F1. Validation loss diverges from training loss after epoch 3, rising from 0.957 to 1.745, yet validation F1 continues improving until epoch 6 (combined F1 of 0.830). Had we used loss-based early stopping, the model would have been selected at epoch 3 (combined F1 of 0.813), forfeiting 1.7 percentage points. The emotion head converges by epoch 3, while the urgency head continues climbing until epoch 6, which is consistent with the fact that urgency requires longer training to synthesise situational details distributed across complaint texts.


## Confusion Structure


![DeBERTa Confusion Matrices](fig5_confusion_matrices.png)
*DeBERTa Confusion Matrices*

DeBERTa makes 151 urgency errors (20.1%) and 107 emotion errors (14.3%). For urgency, 97.4% of mistakes are single-step confusions between adjacent classes, which suggests that the model largely preserves the ordinal structure of the labels. The main asymmetry, however, is downward error: the most common urgency misclassification is Medium predicted as Low (64 cases, 42.4% of urgency errors), and lower-than-true predictions are more frequent than higher-than-true ones overall. This is also reflected in the class-wise metrics. Low urgency has high recall (89.7%) but lower precision (78.1%), indicating that the Low class absorbs a substantial number of cases from higher true classes. Medium shows the weakest recall (67.7%), largely because many Medium cases are pushed down into Low, while High remains comparatively stable with precision of 82.1% and recall of 85.6%. Hence, these results suggest a systematic tendency to under-predict urgency rather than a pattern of random error. For emotion, the dominant error goes in the opposite direction: Medium is most often predicted as High (36 cases, 33.6% of emotion errors), suggesting a mild upward shift in that head. These error patterns are examined in more detail in the next section.


# Contrastive Error Analysis Methodology

To evaluate the DeBERTa model's double-head performance beyond standard aggregate metrics, we implemented a Contrastive Error Analysis pipeline. Rather than looking at aggregate failure rates, this method systematically matches incorrect predictions with successful predictions that share the exact same metadata (ground truth label, scenario, and communication style). These matched 'Informative-Correct Pairs' (ICPs) allow us to isolate specific linguistic, structural, or contextual features, enabling us to identify the underlying algorithmic blindspots confusing the model.


## Dual-Head Accuracy Comparison

Consistent with the performance gap reported in Section 4.1, the urgency head produced substantially more classification errors than the emotion head. In practice, often the severity of the issue is defined by company internal documents. Emotion, even though nuanced, often has more apparent vocabulary triggers.


![DeBERTa Misclassification Error Distribution (Urgency vs Emotion).](deberta_error_distribution.png)
*DeBERTa Misclassification Error Distribution (Urgency vs Emotion).*


## Urgency Head Analysis


| Algorithmic Blindspot | Description |
| --- | --- |
| Emotional & Vulnerability Over-indexing | The model frequently misinterprets expressions of emotional distress or personal vulnerability as 'High' urgency, while failing to adequately weigh explicit statements of patience or willingness to wait. |
| Contextual Misinterpretation of Deadlines | The model over-prioritizes explicit deadlines or external escalation threats (e.g., Ombudsman, legal action) without sufficiently contextualizing the actual impact or immediacy of the threat. |
| Failure to Synthesize Problem History | The model struggles to weigh the cumulative effect of repeated service failures or prolonged unresolved issues, over-indexing on single recent events instead of persistent crises. |


**Table: Top 3 Urgency Blindspots**


## Emotion Head Analysis


| Algorithmic Blindspot | Description |
| --- | --- |
| Misinterpreting Cumulative Impact | The model struggles to infer emotion (Low, Medium, High) from the cumulative weight of repeated issues or severe financial strain without explicit high-impact angry keywords. |
| Over-reliance on Overt Aggressive Language | The model over-indexes on explicit personal frustration or aggressive 'power phrases', misclassifying intensity when the tone is strictly formal but legally threatening. |
| Under-recognition of Internal Escalation | The model fails to identify 'Medium' or 'High' emotional severity in initial contacts due to critical system failures without explicit threats. |


**Table: Top 3 Emotion Blindspots**


## Adversarial Stress Testing

To validate the blindspots identified through contrastive analysis, we constructed 10 targeted adversarial cases, each designed to isolate a specific failure mode. These cases serve as qualitative illustrations, each either confirming or challenging the diagnosed blindspot. Notably, the four cases that produced incorrect predictions mapped precisely onto Blindspots 1, 2 and 3, providing targeted evidence that the contrastive analysis correctly identified the model's failure modes.

For example, the adversarial test 'Passive-Aggressive Legal Threat' (True Emotion: High, Predicted: Low) failed exactly as predicted by Emotion Blindspot Over-reliance on Overt Aggressive Language. It lacked explicit angry words but carried extreme threat. Similarly, the 'Sarcastic Praise' test failed because the model misinterpreted the cumulative impact (Emotion Blindspot #1) when masked by mathematically positive words. These complementary results confirm the robustness of our passive error analysis. While 10 adversarial cases are insufficient for quantitative claims about model robustness, the value of this analysis lies in the specificity of the failures as each incorrect prediction corresponded to a previously identified blindspot, suggesting the contrastive analysis has diagnostic validity.


# Limitations of the Approach

The model is trained exclusively on machine-generated text, meaning it learns linguistic patterns, vocabulary, and rhetorical structures characteristic of large language models rather than genuine human writing. Real-world complaints tend to be considerably more chaotic, incorporating severe spelling errors, SMS abbreviations, and stream-of-consciousness formatting that the synthetic corpus does not capture. Although adversarial testing partially mitigates this, it cannot fully replicate the noise present in live production environments.

The dataset is constructed around a fixed set of scenarios, writing styles, and customer profiles, with strict logical rules governing label assignments, for instance, a scenario classified as "Complete Service Outage" cannot be assigned Low urgency. While this design ensures internal consistency, real customers routinely produce contradictory complaints that violate such boundaries, and the full range of real-world telecommunications situations extends considerably beyond the controlled grid used here.

The model is evaluated exclusively on a held-out partition of the same synthetic dataset used for training, meaning that reported performance reflects generalisation within a controlled generative distribution rather than to real-world data. Without validation on genuine telecoms complaints, it remains uncertain whether the classification boundaries learned from synthetic text transfer to the broader linguistic variation present in live production environments. This represents the most consequential constraint on the external validity of the reported results.


# Conclusion

This paper examined whether urgency and emotional intensity, two dimensions often conflated in automated complaint triage, can be separated through deliberate dataset design and task-specific modelling. The results suggest they can be decoupled to a practically useful extent within a controlled synthetic setting, while also revealing where this separation remains difficult.

Among the four evaluated models, the fine-tuned DeBERTa-v3-base achieved the strongest balanced performance, reaching a combined macro F1 of 0.830 (urgency 0.803, emotion 0.857). TF-IDF marginally outperformed DeBERTa on urgency alone, indicating that part of the urgency signal is recoverable from lexical cues and scenario-specific wording, whereas DeBERTa's clear advantage on emotion suggests that emotional intensity depends more heavily on contextual and pragmatic interpretation. The poor performance of frozen Sentence-BERT further indicates that general-purpose semantic embeddings are insufficient for this task.

The dataset versioning process was central to the final outcome. The 18.6 percentage-point improvement in urgency macro F1 between Dataset v2 and v3 strongly suggests that generation quality was a major constraint, reinforcing the principle that in synthetic pipelines, improving the training signal can matter at least as much as model selection.

The contrastive error analysis identified recurring blind spots in both heads: for urgency, cumulative complaint histories, vulnerability cues, and deadline-related language; for emotion, legalistic but severe threats, cumulative frustration without aggressive wording, and understated internal escalation. The adversarial stress test was consistent with these diagnosed blind spots, providing preliminary support for the ICP-based methodology.

The dual-head classification framework developed in this paper provides the technical foundation for an automated complaint prioritisation system in telecoms customer service. By independently predicting objective urgency and emotional intensity, the model generates two operationally distinct signals: urgency determines queue position, ensuring the most critical issues reach appropriately skilled agents first, while emotional intensity provides agents with advance context about the customer's affective state, enabling more effective communication before the interaction begins. In the UK telecoms context, this carries direct regulatory relevance, as Ofcom requires prioritised handling of vulnerable customer complaints, a requirement the emotion head can help operationalise systematically at scale.

These findings should be interpreted within the constraints of a fully synthetic dataset that does not capture the noise and inconsistency of real customer communications, and a 192-token limit that may not fully represent longer complaints. Validation on real telecom data is the most important next step, alongside a data augmentation strategy that injects adversarial edge cases targeting the identified blind spots, such as formal legal threats conveying high emotion, to recalibrate the model's decision boundaries.


## References

Ashury-Tahan, S., Mai, Y., Bandel, E., Shmueli-Scheuer, M. and Choshen, L. (2026). ErrorMap and ErrorAtlas: Charting the Failure Landscape of Large Language Models. [online] arXiv.org. Available at: https://arxiv.org/abs/2601.15812 [Accessed 25 Mar. 2026].
Athira, Adith M. and Gupta, D. (2025). Effective Complaint Detection in Financial Services through Complaint, Severity, Emotion and Sentiment Analysis. Procedia Computer Science, 258, pp.2220–2231. doi:https://doi.org/10.1016/j.procs.2025.04.472.
Chen, S., Zhang, Y. and Yang, Q. (2024) 'Multi-task learning in natural language processing: an overview', ACM Computing Surveys, 56(12), Article 295, pp. 1–32. Available at: https://doi.org/10.1145/3663363
He, P., Liu, X., Gao, J. and Chen, W. (2021). DeBERTa: Decoding-enhanced BERT with Disentangled Attention. [online] doi:https://doi.org/10.48550/arxiv.2006.03654.
He, P., Gao, J. and Chen, W. (2023) 'DeBERTaV3: Improving DeBERTa using ELECTRA-style pre-training with gradient-disentangled embedding sharing', International Conference on Learning Representations (ICLR). Available at: https://arxiv.org/abs/2111.09543.
Iskander, S., Tolmach, S., Shapira, O., Cohen, N. and Karnin, Z. (2024) 'Quality matters: Evaluating synthetic data for tool-using LLMs', in Al-Onaizan, Y., Bansal, M. and Chen, Y.-N. (eds.) Proceedings of the 2024 Conference on Empirical Methods in Natural Language Processing. Miami, Florida: Association for Computational Linguistics, pp. 4958–4976. doi: 10.18653/v1/2024.emnlp-main.285.
Long, L., Wang, R., Xiao, R., Zhao, J., Ding, X., Chen, G. and Wang, H. (2024). On LLMs-Driven Synthetic Data Generation, Curation, and Evaluation: A Survey. Findings of the Association for Computational Linguistics: ACL 2024, [online] pp.11065–11082. doi:https://doi.org/10.18653/v1/2024.findings-acl.658.
Lu, Z., Tian, J., Wei, W., Qu, X., Cheng, Y., Xie, W. and Chen, D. (2024) 'Mitigating boundary ambiguity and inherent bias for text classification in the era of large language models', in Ku, L.-W., Martins, A. and Srikumar, V. (eds.) Findings of the Association for Computational Linguistics: ACL 2024. Bangkok, Thailand: Association for Computational Linguistics, pp. 7841–7864. doi: 10.18653/v1/2024.findings-acl.467.
Reimers, N. and Gurevych, I. (2019). Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks. Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing and the 9th International Joint Conference on Natural Language Processing (EMNLP-IJCNLP). [online] doi:https://doi.org/10.18653/v1/d19-1410.
Roumeliotis, K.I., Tselikas, N.D. and Nasiopoulos, D.K. (2025). Think Before You Classify: The Rise of Reasoning Large Language Models for Consumer Complaint Detection and Classification. Electronics, [online] 14(6), pp.1070–1070. doi:https://doi.org/10.3390/electronics14061070.
Tan, Z., Li, D., Wang, S., Beigi, A., Jiang, B., Bhattacharjee, A., Karami, M., Li, J., Cheng, L. and Liu, H. (2024). Large Language Models for Data Annotation: A Survey. [online] arXiv.org. doi:https://doi.org/10.48550/arXiv.2402.13446.
Törnberg, P. (2024) Fine-tuned 'small' LLMs (still) significantly outperform zero-shot generative AI models in text classification. [online] arXiv.org. Available at: https://arxiv.org/abs/2406.08660 [Accessed 27 March 2026].
Vairetti, C., Aránguiz, I., Maldonado, S., Karmy, J.P. and Leal, A. (2024). Analytics-driven complaint prioritisation via deep learning and multicriteria decision-making. European Journal of Operational Research, [online] 312(3), pp.1108–1118. doi:https://doi.org/10.1016/j.ejor.2023.08.027.
Zhang, W., Deng, Y., Liu, B., Pan, S.J. and Bing, L. (2024) 'Sentiment analysis in the era of large language models: A reality check', in Findings of the Association for Computational Linguistics: NAACL 2024. Mexico City: Association for Computational Linguistics, pp. 2937–2958. doi: 10.18653/v1/2024.findings-naacl.246.