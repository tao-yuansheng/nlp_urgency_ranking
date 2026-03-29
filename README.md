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
├── data/                                # Dataset files
│   ├── telecoms_complaints.csv          # 5,000 synthetic labelled complaints (v3, GPT-5-mini)
│   ├── telecoms_complaints-4o-mini.csv  # Alternative generation run
│   └── archive/                         # Older dataset versions
├── data_generation/                     # Synthetic dataset pipeline
│   ├── generate_complaints.py           # Main generation script (OpenAI API)
│   ├── prompts.py                       # Taxonomy definitions and system prompts
│   ├── taxonomy.py                      # Dataset planning and distribution logic
│   ├── scenario_urgency_affinity.csv    # Affinity map constraining realistic combinations
│   └── README.md                        # Full data generation documentation
├── data_eda/                            # Exploratory data analysis
│   ├── eda_synthetic_complaints.ipynb   # EDA notebook
│   ├── restyle_figures.py               # Figure styling utilities
│   └── figures/                         # 14 EDA visualisations (class dist., vocabulary, t-SNE, etc.)
├── model_training/                      # Fine-tuning & evaluation
│   ├── train_deberta.py                 # Fine-tune DeBERTa-v3-base (dual-head)
│   ├── baseline_tfidf_lr.py             # TF-IDF + Logistic Regression baseline
│   ├── baseline_sbert_lr.py             # Sentence-BERT (frozen) + LR baseline
│   ├── baseline_llm_haiku.py            # LLM baseline (Claude Haiku)
│   ├── compare_models.py                # Run all models and save predictions
│   ├── adversarial_test.py              # 10-item edge-case evaluation
│   ├── download_model.py                # Download fine-tuned model from HuggingFace
│   ├── logs/                            # JSON training run logs (per dataset version)
│   ├── results/                         # Test predictions, metrics, and baseline results
│   ├── model_output/                    # Pre-trained model weights (local cache)
│   └── README.md                        # Training, download, and results documentation
├── error_analysis/                      # Systematic error investigation
│   ├── run_stage1.py                    # Stage 1: ErrorMap classification
│   ├── run_stage2.py                    # Stage 2: deeper error analysis
│   ├── run_icp_analysis.py              # Inline contrastive pair analysis
│   ├── generate_icps.py                 # Generate ICP pairs
│   ├── visualize_errors.py              # Error visualisation scripts
│   ├── logbook.md                       # Session-by-session analysis log
│   └── results/                         # Error histograms, contrastive reports (PDF/PNG)
├── results_evaluation/                  # Model evaluation summary
│   ├── generate_figures.ipynb           # Results visualisation notebook
│   └── fig[1-6]_*.png                   # Comparison figures (model, per-class F1, confusion, etc.)
├── report/                              # Intermediate report drafts
│   ├── data_section.md                  # Data section draft
│   ├── methodology_section.md           # Methodology section draft
│   ├── training_runs_table.md           # Training runs summary table
│   ├── plot_deberta_errors.py           # Error distribution plotting script
│   ├── plot_training_runs.py            # Training runs plotting script
│   └── figures/                         # Plots generated for the report
├── final_report_latex/                  # Final deliverable
│   ├── main.tex                         # LaTeX source (ACL format)
│   ├── main_v20.pdf                     # Final submitted PDF
│   ├── figures/                         # Report figures (fig01–fig12)
│   └── versions/                        # Archived PDF build history
├── docs/                                # Project documents
│   ├── Group 16 - MSIN0221 - Proposal.docx
│   ├── MSIN0221_Group_Assignment_v1.docx
│   ├── MSIN0221_Group_Assignment_v2.docx
│   ├── CHANGELOG_SESSION.md
│   ├── CHANGES_2026-03-24.md
│   └── CHANGES_2026-03-25.md
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
