"""System prompt templates, level definitions, scenarios, and styles."""

# ---------------------------------------------------------------------------
# Urgency definitions
# ---------------------------------------------------------------------------
URGENCY_DEFINITIONS = {
    "Low": (
        "Minor inconvenience, no financial or service impact, can wait for "
        "resolution (e.g., a small billing query, cosmetic app issue)."
    ),
    "Medium": (
        "Noticeable disruption but not critical, moderate financial impact or "
        "partial service loss (e.g., intermittent connectivity, unexpected "
        "charge under £50)."
    ),
    "High": (
        "Severe impact — complete service loss, significant financial harm, "
        "legal/regulatory implications, or safety concern (e.g., total network "
        "outage for days, hundreds overcharged, number ported without consent)."
    ),
}

# ---------------------------------------------------------------------------
# Emotion definitions
# ---------------------------------------------------------------------------
EMOTION_DEFINITIONS = {
    "Low": (
        "Calm, factual, neutral tone — reporting an issue without emotional "
        "language. No exclamation marks, no capitalised words for emphasis, "
        "no strong adjectives. Plain and matter-of-fact."
    ),
    "Medium": (
        "Noticeably frustrated or disappointed. Use some emotional language "
        "like 'really frustrating', 'quite disappointed'. Occasional "
        "exclamation marks are acceptable but not excessive. Tone is firm "
        "but still measured."
    ),
    "High": (
        "Angry, distressed, or exasperated. You MUST use textual markers of "
        "strong emotion: CAPITALISED WORDS for emphasis (e.g., 'I am DONE', "
        "'STILL no response'), multiple exclamation marks (!! or !!!), bold "
        "demands for escalation ('I want a manager NOW!!!'), and emotionally "
        "charged language ('absolutely disgusting', 'unacceptable', "
        "'I cannot believe'). The complaint should visually look intense."
    ),
}

# ---------------------------------------------------------------------------
# Scenarios
# ---------------------------------------------------------------------------
SCENARIOS = [
    "Billing overcharge / unexpected fees",
    "Network outage / poor coverage",
    "Contract dispute / early termination fee",
    "Slow or unreliable internet speed",
    "Rude or unhelpful customer service",
    "Porting / number transfer issues",
    "Roaming charges / international billing",
    "Service cancellation difficulty",
]

# ---------------------------------------------------------------------------
# Writing styles
# ---------------------------------------------------------------------------
STYLES = [
    "Formal professional",
    "Casual conversational",
    "Angry and blunt",
    "Passive-aggressive / sarcastic",
    "Non-native English",
]

# ---------------------------------------------------------------------------
# Three rotating system prompt templates
# ---------------------------------------------------------------------------
SYSTEM_PROMPTS = [
    # Template A
    (
        "You are a complaint-writing assistant. Your task is to produce "
        "realistic customer complaints addressed to a telecoms provider, as "
        "if submitted via email, online form, or live chat. Write only the "
        "complaint text — no labels, headings, metadata, or preamble. "
        "Vary the length naturally; some complaints should be short and "
        "others longer. Use realistic but varied customer names, or omit "
        "the name entirely. Never use placeholder names like 'John Doe'. "
        "IMPORTANT: When the emotion level is High, you must reflect anger "
        "through text formatting — use CAPITALISED WORDS, multiple "
        "exclamation marks (!! or !!!), and emotionally charged phrases. "
        "Do not write high-emotion complaints in calm, grammatically "
        "perfect prose."
    ),
    # Template B
    (
        "Imagine you are different real customers contacting a "
        "telecommunications company to complain. For each complaint you "
        "generate, output nothing but the raw complaint message — no "
        "titles, no tags, no explanations. The complaints should feel "
        "authentic: varying in length, tone, and detail. If a name is "
        "used, make it sound genuine and diverse — avoid generic "
        "placeholders. When emotion is high, write like a genuinely angry "
        "customer: CAPS for emphasis, exclamation marks!!!, short punchy "
        "sentences, demands for managers. Don't sanitise the anger into "
        "polished language."
    ),
    # Template C
    (
        "Act as a generator of customer complaint messages for a telecoms "
        "company. Each output must read like a genuine message a customer "
        "would send through a support channel. Provide only the complaint "
        "body — do not include any metadata, labels, or framing text. "
        "Let the complaints differ naturally in length and specificity. "
        "Use believable names when appropriate; never use obvious filler "
        "names. For high-emotion complaints, make the text visually intense "
        "— CAPITALISE key words, use !!! liberally, include raw frustrated "
        "language. The reader should immediately feel the customer's anger "
        "just from scanning the text."
    ),
]
