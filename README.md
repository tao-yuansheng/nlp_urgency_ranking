# Telecoms Complaint Classifier

An NLP pipeline that fine-tunes **DeBERTa-v3-base** to simultaneously classify UK telecoms customer complaints by **urgency** and **emotion** — two independent 3-class labels (Low / Medium / High).

Built as part of the MSIN0221 group project at UCL.

---

## Results

| Head | Macro F1 | Low F1 | Medium F1 | High F1 |
|---|---|---|---|---|
| Urgency | **0.80** | 0.82 | 0.74 | 0.84 |
| Emotion | **0.85** | 0.89 | 0.82 | 0.84 |

Adversarial test (10 hand-crafted edge cases): **6 / 10**

---

## Fine-tuned Model

The trained model is hosted on HuggingFace Hub:

**[yuansheng-tao/emotion_urgency_classifier](https://huggingface.co/yuansheng-tao/emotion_urgency_classifier)**

Download it without re-training:

```bash
python model_training/download_model.py
```

---

## Quick Start

### Option A — Use the pre-trained model

```bash
pip install -r requirements.txt
python model_training/download_model.py
python model_training/adversarial_test.py   # optional: run evaluation
```

### Option B — Train from scratch

```bash
pip install -r requirements.txt

# 1. Generate the dataset (requires OpenAI API key — see data_generation/README.md)
cp .env.example .env          # add your OPENAI_API_KEY
python data_generation/generate_complaints.py

# 2. Train
python model_training/train_deberta.py
```

---

## Repository Structure

```
Project/
├── data/
│   └── telecoms_complaints.csv          # 5,000 synthetic labelled complaints
├── data_generation/                     # Synthetic dataset pipeline
│   └── README.md                        # Full data generation documentation
├── model_training/                      # Fine-tuning & evaluation
│   ├── train_deberta.py                 # Fine-tune DeBERTa-v3-base
│   ├── baseline_tfidf_lr.py             # TF-IDF + Logistic Regression baseline
│   ├── baseline_sbert_lr.py             # Sentence-BERT (frozen) + LR baseline
│   ├── compare_models.py                # Run all three models and save predictions
│   ├── adversarial_test.py              # 10-item edge-case evaluation
│   ├── download_model.py                # Download fine-tuned model from HuggingFace
│   └── README.md                        # Training, download, and results documentation
├── requirements.txt
└── .env.example                         # API key template
```

- [data_generation/README.md](data_generation/README.md) — how the dataset is built (GPT-5-mini, taxonomy, affinity map)
- [model_training/README.md](model_training/README.md) — model architecture, training config, results across dataset versions

---

## Requirements

```bash
pip install -r requirements.txt
```

Key dependencies: `torch`, `transformers`, `pandas`, `scikit-learn`, `huggingface_hub`, `openai`, `python-dotenv`
