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
   This will generate complaints and save them to `telecoms_complaints_sample.csv`.

## File Descriptions

| File | Description |
|------|-------------|
| `generate_complaints.py` | Main script. Calls the OpenAI API to generate complaints across all urgency/emotion cells and saves the output as a CSV. |
| `prompts.py` | Contains urgency and emotion level definitions, complaint scenarios, writing styles, and the rotating system prompt templates sent to the model. |
| `taxonomy.py` | Defines the 3×3 urgency × emotion grid and the logic for assigning scenarios and styles to each cell while enforcing distribution constraints. |
| `requirements.txt` | Python dependencies: `openai`, `pandas`, `python-dotenv`. |
| `.env.example` | Template for the `.env` file. Copy this to `.env` and add your API key. |
| `.env` | **Not tracked by git.** Holds your `OPENAI_API_KEY`. |
| `telecoms_complaints_sample.csv` | Generated output — a CSV of complaints with columns: `id`, `complaint_text`, `intended_urgency`, `intended_emotion`, `scenario`, `style`. |

## Output Format

The generated CSV contains the following columns:

- **id** — Complaint number
- **complaint_text** — The generated complaint message
- **intended_urgency** — Low, Medium, or High
- **intended_emotion** — Low, Medium, or High
- **scenario** — The complaint topic (e.g., billing overcharge, network outage)
- **style** — The writing style (e.g., formal professional, casual conversational)
