# 3. Methodology

## 3.1 Task Formulation

The task is formulated as dual multi-class classification: given a customer complaint text, the model must independently predict an urgency level and an emotion level, each drawn from the set {Low, Medium, High}. The two dimensions are treated as independent classification targets. Urgency captures the operational severity of the issue described, while emotion captures the affective intensity of the language used. A customer may describe a catastrophic service outage in a composed tone (High urgency, Low emotion) or express fury over a trivial billing discrepancy (Low urgency, High emotion). The model must therefore learn to decouple these two signals rather than relying on their co-occurrence.

## 3.2 Baselines

Three baseline models of increasing complexity were evaluated to contextualise the performance of the fine-tuned model. All baselines were evaluated on the same stratified test split of 750 complaints.

**Few-shot Claude Haiku 4.5.** The first baseline uses a large language model with no task-specific training. Nine labelled examples were provided in the system prompt, one for each urgency-by-emotion cell, sampled deterministically from the training set using a fixed seed. Complaints were batched in groups of ten per API call, and the model was prompted to return structured JSON classifications at temperature 0.0. No chain-of-thought reasoning was elicited. This baseline establishes whether a general-purpose LLM can perform the task from definitions and examples alone, without any parameter updates.

**TF-IDF with Logistic Regression.** The second baseline represents a standard non-neural approach. Complaint texts were vectorised using term frequency-inverse document frequency with unigram and bigram features, a maximum vocabulary of 20,000 terms, and sublinear term frequency scaling. Two independent logistic regression classifiers were trained, one per target dimension, using L2 regularisation with C=1.0 and the LBFGS solver. This baseline isolates the contribution of surface-level lexical features, providing a reference for how much of the classification signal is accessible without any contextual language understanding.

**Frozen Sentence-BERT with Logistic Regression.** The third baseline replaces sparse lexical features with dense semantic representations. Complaint texts were encoded using all-MiniLM-L6-v2, a pre-trained sentence transformer producing 384-dimensional embeddings, with all encoder weights frozen. Two independent logistic regression classifiers were again trained on top of these embeddings using the same configuration as the TF-IDF baseline. By keeping the encoder frozen, this baseline isolates the value of general-purpose semantic representations without task-specific adaptation, establishing a lower bound on what fine-tuning adds.

## 3.3 Fine-tuned DeBERTa-v3-base

DeBERTa-v3-base was selected over standard BERT for its disentangled attention mechanism (He et al., 2021), which decouples content and position embeddings into separate vectors that interact through dedicated attention matrices. This architecture enables the model to distinguish between what a word means and where it appears in the sequence, a property particularly relevant for complaint classification where positional cues such as escalation language appearing at the end of a message or urgency markers embedded in subordinate clauses carry diagnostic value.

The model uses a shared encoder with two independent linear classification heads. The [CLS] token representation from the final hidden layer (768 dimensions) is passed to both an urgency head and an emotion head, each projecting to three output classes. Both heads are trained jointly by summing their respective cross-entropy losses. The urgency head applies class weights of [1.0, 1.5, 1.2] for Low, Medium, and High respectively. Medium receives the highest weight not because of class frequency (it is in fact the most frequent class at 40%) but because it structurally overlaps with both adjacent classes, making it the hardest to distinguish. The emotion head uses unweighted cross-entropy, as emotion classes are evenly distributed and boundary ambiguity was already addressed through the dataset design described in Section 2.

Early stopping monitors the combined validation macro F1, defined as the mean of validation urgency and emotion macro F1. Monitoring validation loss alone did not reliably track improvements in classification quality across both tasks. Monitoring urgency macro F1 alone was also trialled — urgency being consistently the harder task across all dataset versions — but switching to the combined metric yielded a marginally higher test performance (overall mean F1 of 0.830 vs. 0.827), suggesting that the combined criterion better prevents premature stopping when one task plateaus while the other continues to improve. Training halts after three consecutive epochs with no improvement in the combined F1, and the model weights from the best epoch are restored.

| Parameter | Value |
|---|---|
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

## 3.4 Evaluation Metrics

All models are evaluated using macro-averaged F1 score, defined as the unweighted mean of per-class F1 scores across the three classes. Per-class F1 is the harmonic mean of precision and recall for that class. Macro averaging weights each class equally regardless of its frequency in the test set, which is preferable to accuracy or micro-averaged F1 in the presence of class imbalance: accuracy would be inflated by correct predictions on the majority class (Medium urgency at 40%), while micro-averaging would similarly dilute the contribution of minority classes. Macro F1 penalises models that perform well on frequent classes but poorly on rare ones, providing a more balanced assessment of classification quality across the full label space.

Per-class F1 scores are additionally reported for each model and each dimension, enabling fine-grained comparison of where models succeed and fail across the urgency and emotion spectra.
