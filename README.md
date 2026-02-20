# NLP Urgency Ranking — Synthetic Complaint Generator

This project generates synthetic telecoms customer complaints using OpenAI's GPT-4o-mini. Each complaint is labelled with an intended urgency level and emotion level, forming a balanced dataset for NLP urgency-ranking research.

## Setup

1. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

2. **Configure your API key**
   - Copy the example environment file:
     ```bash
     cp .env.example .env
     ```
   - Open `.env` and replace the placeholder with your actual OpenAI API key:
     ```
     OPENAI_API_KEY=sk-your-actual-key-here
     ```

3. **Run the generator**
   ```bash
   python generate_complaints.py
   ```
   By default this generates 5,000 complaints. Use `--total` and `--seed` to customise:
   ```bash
   python generate_complaints.py --total 100 --seed 42
   ```

## Pipeline Overview

The generator uses a **3x3 urgency x emotion grid** with a skewed urgency distribution (35% Low / 40% Medium / 25% High). Each complaint is assigned a unique combination of:

- **Scenario** (20 UK telecoms complaint topics) — constrained by a scenario-urgency affinity map
- **Writing style** (8 styles, e.g. formal, sarcastic, legalistic)
- **Customer profile** (8 personas, e.g. elderly customer, small business owner)
- **Complaint history** (4 depths, from first contact to escalation)

This gives 5,120 unique combinations, ensuring diversity at scale. Three rotating system prompts further reduce stylistic uniformity.

## File Descriptions

| File | Description |
|------|-------------|
| `generate_complaints.py` | Main script. Calls the OpenAI API in batches of 15, with retry logic, and saves the output as a CSV. |
| `prompts.py` | Urgency/emotion definitions, 20 scenarios, scenario-urgency affinity map, writing styles, customer profiles, complaint history depths, and 3 rotating system prompts. |
| `taxonomy.py` | 3x3 grid with skewed distribution, pool-based assignment with per-urgency scenario constraints, 4-tuple dedup, and validation. Run standalone to verify distributions. |
| `scenario_urgency_affinity.csv` | Human-readable reference of which scenarios are allowed at each urgency level. |
| `requirements.txt` | Python dependencies: `openai`, `pandas`, `python-dotenv`. |
| `.env.example` | Template for the `.env` file. Copy this to `.env` and add your API key. |

## Dataset Scale & Diversity Justification

The design supports generating **5,000–10,000 complaints without meaningful repetition**, for three independent reasons:

### 1. Combinatorial space exceeds the target range

The four metadata dimensions produce a large unique-tuple space:

| Dimension | Count |
|-----------|-------|
| Scenarios | 20 |
| Writing styles | 8 |
| Customer profiles | 8 |
| Complaint history depths | 4 |
| **Total unique 4-tuples** | **5,120** |

At 5,000 entries the dataset uses ~98% of the available unique combinations. At 10,000 entries the generator reuses 4-tuples, but linguistic diversity (see point 3) prevents these from producing duplicate complaint texts.

### 2. Affinity constraints create realistic within-cell variety

Scenarios and styles are not freely combined — each is restricted to compatible urgency/emotion levels via affinity maps:

- **10–18 scenarios** are allowed per urgency level (not all 20)
- **6–8 styles** are allowed per emotion level (not all 8)
- Within each of the 9 grid cells, the generator enforces **no duplicate (scenario, style, profile, history) tuples**, so every complaint within a cell has a distinct metadata signature

This means the 9 grid cells each draw from their own constrained sub-space, avoiding the clustering that a flat random draw would produce.

### 3. Three independent sources of linguistic diversity

Even when two complaints share the same 4-tuple metadata, their text will differ because:

- **Temperature = 1.0** — maximum LLM sampling randomness on every API call
- **3 rotating system prompts** — different framing instructions per cell cycle (complaint-writing assistant / real customer simulation / generator mode)
- **CRITICAL tone instructions** — emotion-level instructions enforce distinct vocabulary and sentence structure at Low / Medium / High, rather than relying on punctuation or capitalization

Together these ensure that 10,000 complaints remain linguistically varied even where metadata overlaps.

## Output Format

The generated CSV (`telecoms_complaints.csv`) contains:

| Column | Description |
|--------|-------------|
| `id` | Complaint number |
| `complaint_text` | The generated complaint message |
| `intended_urgency` | Low, Medium, or High |
| `intended_emotion` | Low, Medium, or High |
| `scenario` | Complaint topic (e.g. Fraud & Scams, Slow Broadband Speeds) |
| `style` | Writing style (e.g. formal professional, terse and minimal) |
| `profile` | Customer persona (e.g. elderly customer, student on a tight budget) |
| `history` | Complaint history depth (e.g. first contact, escalation) |