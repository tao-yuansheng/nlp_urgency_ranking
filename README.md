# NLP Urgency Ranking — Synthetic Complaint Generator

## What is this?

This tool generates realistic-sounding fake (synthetic) customer complaints about telecoms services in the UK. Each complaint is pre-labelled with two things:

- **Urgency level** — how urgently the issue needs resolving (Low, Medium, or High)
- **Emotion level** — how emotionally the customer is writing (Low = calm and factual, High = angry and distressed)

We use OpenAI's GPT-4o-mini to write the complaint text, and we carefully control what each complaint is about, who is writing it, and how it sounds. The result is a dataset of 5,000 complaints ready to use for training NLP models to automatically detect urgency and emotion.

## Why synthetic data?

Rather than collecting and manually labelling thousands of real complaints — which is slow, expensive, and raises privacy concerns — we instruct the AI to write complaints with specific labels built in from the start. The labels are guaranteed correct by design: we tell the AI exactly what urgency level and emotion level to use when writing each complaint, so there is no ambiguity in the ground truth.

---

## How the generator works — step by step

Before any complaint is written, the generator plans the entire dataset. The process has four stages.

### Step 1 — Plan the distribution

The generator creates a **3×3 grid** with urgency on one axis and emotion on the other. This gives 9 combinations (called cells). Every complaint belongs to exactly one cell.

|  | Low Emotion | Medium Emotion | High Emotion |
|---|---|---|---|
| **Low Urgency** | Calm note about a minor issue | Frustrated note about a minor issue | Angry note about a minor issue |
| **Medium Urgency** | Calm report of slow broadband | Frustrated report of slow broadband | Furious report of slow broadband |
| **High Urgency** | Calm report of a complete outage | Frustrated report of a complete outage | Furious report of a complete outage |

The split across urgency levels is intentionally uneven to reflect reality — most complaints are not emergencies:

| Urgency | Share | Count (out of 5,000) |
|---|---|---|
| Low | 35% | 1,750 |
| Medium | 40% | 2,000 |
| High | 25% | 1,250 |

Emotion is distributed evenly within each urgency group (roughly one third per emotion level).

### Step 2 — Assign characteristics to each complaint

Every complaint is given four characteristics before the AI writes it:

1. **Scenario** — what the complaint is about (1 of 20 UK telecoms topics)
2. **Writing style** — how the text is written (1 of 8 styles)
3. **Customer profile** — who is writing (1 of 8 personas)
4. **Complaint history** — how many times they have contacted before (1 of 4 depths)

The generator ensures that no two complaints in the same grid cell share the same combination of all four characteristics, so every complaint has a unique setup.

**Key constraint — affinity mapping:**
Not every scenario fits every urgency level. For example, "Complete Service Outage" should never appear as a Low urgency complaint — that would be unrealistic. We enforce this with a scenario–urgency affinity map (see the full table below).

Similarly, not every writing style fits every emotion level. "Passive-aggressive / sarcastic" does not make sense for a calm, Low emotion complaint, so it is restricted to Medium and High emotion only.

### Step 3 — Write the complaints

Once all 5,000 complaint specifications are planned, GPT-4o-mini writes the actual text. Complaints are sent in **batches of 15** at a time, with up to **10 batches running in parallel** to speed things up.

Each batch includes instructions telling the AI:
- The urgency and emotion level definitions for that batch
- A **CRITICAL TONE instruction** specifying exactly how emotional the writing must sound
- The specific scenario, style, profile, and history for each of the 15 complaints
- Instructions to vary phrasing, length, and detail so complaints do not sound repetitive

If the AI returns fewer complaints than requested, the generator automatically retries (up to 2 times per batch).

### Step 4 — Save the output

All complaints are assembled and saved to `telecoms_complaints.csv` with their labels attached.

---

## The 4 Complaint Dimensions

### Urgency levels

| Level | What it means | Typical example |
|---|---|---|
| **Low** | Minor inconvenience with no immediate impact; can wait days or weeks | A small unexplained charge on a bill |
| **Medium** | Noticeable disruption or partial service loss; needs attention within days | Slow broadband affecting daily work |
| **High** | Severe, time-critical issue; complete loss of service or regulatory risk | A business with no internet access at all |

### Emotion levels

| Level | What it means | Example language |
|---|---|---|
| **Low** | Calm, factual, unemotional — reads like an incident report | *"I am writing to report an issue with my account..."* |
| **Medium** | Clearly frustrated and losing patience, but still coherent | *"I am frankly disappointed and fed up with the lack of response..."* |
| **High** | Genuinely angry, distressed, or desperate; threatens to escalate | *"This is absolutely appalling. I am livid and will contact Ofcom immediately..."* |

### Writing styles (8 total)

| Style | Description |
|---|---|
| Formal professional | Structured, business-like language |
| Casual conversational | Relaxed, everyday language |
| Passive-aggressive / sarcastic | Ironic or subtly cutting tone |
| Verbose and detailed | Long, thorough, highly detailed account |
| Terse and minimal | Very brief and straight to the point |
| Narrative / storytelling | Tells the story of events in chronological order |
| Legalistic / rights-aware | References consumer rights, Ofcom regulations, or legal options |
| Polite but firm | Courteous in tone but clear and assertive about expectations |

> **Note:** Not all styles are available at all emotion levels. Sarcasm and passive-aggression are restricted to Medium and High emotion. Formal professional and polite but firm are restricted to Low and Medium emotion.

### Customer profiles (8 total)

| # | Profile |
|---|---|
| 1 | Young professional — tech-savvy and impatient |
| 2 | Elderly customer — not confident with technology |
| 3 | Small business owner relying on the service |
| 4 | Parent managing a family plan |
| 5 | Student on a tight budget |
| 6 | Long-term loyal customer (10+ years) |
| 7 | Recently switched from another provider |
| 8 | Vulnerable customer with a disability or health condition |

### Complaint history depths (4 total)

| Depth | Meaning |
|---|---|
| First contact | Raising the issue for the first time |
| Second attempt | Contacted once before with no resolution |
| Repeat complainer | Contacted 3–5 times over several weeks |
| Escalation | Exhausted normal channels; requesting a manager or the ombudsman |

---

## Scenario–Urgency Affinity Map

Not all complaint topics are appropriate at every urgency level. The table below shows which urgency levels each scenario can appear at. A blank cell means that combination is not allowed and will never appear in the dataset.

| Scenario | Low | Medium | High |
|---|:---:|:---:|:---:|
| Difficulty Cancelling Service | ✓ | ✓ | |
| Fraud & Scams | | ✓ | ✓ |
| Overcharging & Incorrect Billing | ✓ | ✓ | ✓ |
| Poor Network Coverage | ✓ | ✓ | |
| 3G Shutdown Impact | | ✓ | ✓ |
| Auto-Renewal Without Consent | ✓ | ✓ | |
| Billing After Cancellation | | ✓ | ✓ |
| High Early Termination Fees | | ✓ | ✓ |
| Ineffective AI / Chatbot Support | ✓ | ✓ | |
| Unfulfilled Fix Promises | | ✓ | ✓ |
| Long Call-Waiting Times | ✓ | ✓ | |
| Wrong Sale Due to Agent Mistake | | ✓ | ✓ |
| Loyalty Penalty | ✓ | ✓ | |
| Mid-Contract Price Increase | ✓ | ✓ | |
| Complete Service Outage | | | ✓ |
| Faulty Hardware / Handset Issues | ✓ | ✓ | ✓ |
| Hidden Fees & Charges | ✓ | ✓ | |
| Lack of Progress Updates | ✓ | ✓ | |
| Poor Complaint Handling | | ✓ | ✓ |
| Slow Broadband Speeds | ✓ | ✓ | |

**Counts:** Low urgency — 12 eligible scenarios | Medium urgency — 18 eligible scenarios | High urgency — 10 eligible scenarios

---

## Dataset Scale & Diversity Justification

The design supports generating **5,000–10,000 complaints without meaningful repetition**, for three independent reasons.

### 1. The combination space is large enough

The four dimensions together produce a large number of possible unique combinations:

| Dimension | Options |
|---|---|
| Scenarios | 20 |
| Writing styles | 8 |
| Customer profiles | 8 |
| Complaint history depths | 4 |
| **Total unique combinations** | **5,120** |

At 5,000 entries the dataset uses ~98% of all available unique setups. Even at 10,000 entries, where some setups repeat, the actual complaint texts will still differ because of linguistic randomness (see point 3).

### 2. Affinity constraints keep combinations realistic

Because scenarios are restricted to compatible urgency levels, and styles to compatible emotion levels, each of the 9 grid cells draws from its own smaller pool of valid combinations — rather than a flat random draw across everything. This prevents unrealistic pairings (such as a calm, factual complaint about a complete service outage) and ensures that each cell's complaints feel coherent.

### 3. Three sources of linguistic variety

Even when two complaints share the same four-part setup, their text will differ because:

- **Temperature = 1.0** — the AI is set to maximum randomness on every call, so it never produces identical phrasing twice
- **3 rotating system prompts** — the AI receives a different framing instruction with each batch cycle (complaint-writing assistant / imagined real customer / complaint generator mode), which changes how it approaches the writing
- **CRITICAL tone instructions** — each batch includes an explicit instruction specifying the emotional register for that emotion level, enforcing different vocabulary and sentence structures at Low, Medium, and High

---

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
   (`--seed` controls the assignment randomness so results are reproducible. The AI writing itself still varies each run.)

---

## Output Format

The generated file `telecoms_complaints.csv` contains one row per complaint with these columns:

| Column | Description |
|---|---|
| `id` | Complaint number (1 to 5,000) |
| `complaint_text` | The full generated complaint message |
| `intended_urgency` | Low, Medium, or High — the urgency label |
| `intended_emotion` | Low, Medium, or High — the emotion label |
| `scenario` | What the complaint is about (e.g. Fraud & Scams) |
| `style` | How it is written (e.g. terse and minimal) |
| `profile` | Who is writing it (e.g. elderly customer) |
| `history` | How many times they have contacted before (e.g. first contact) |

---

## File Descriptions

| File | What it does |
|---|---|
| `generate_complaints.py` | Main script — calls the OpenAI API in parallel batches and saves the output CSV |
| `prompts.py` | Defines all the labels, scenarios, styles, profiles, history depths, affinity maps, and the 3 system prompts used to instruct the AI |
| `taxonomy.py` | Plans the full dataset before generation — builds the grid, distributes all 5,000 assignments, enforces affinity rules, and validates the result |
| `scenario_urgency_affinity.csv` | The affinity map from the table above, in CSV format for reference |
| `requirements.txt` | Python package dependencies |
| `.env.example` | Template for the API key file |
