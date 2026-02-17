"""System prompt templates, level definitions, scenarios, styles, and channels."""

# ---------------------------------------------------------------------------
# Urgency definitions
# ---------------------------------------------------------------------------
URGENCY_DEFINITIONS = {
    "Low": (
        "Minor inconvenience with no immediate impact on essential services or "
        "finances. The customer can wait days or weeks for resolution without "
        "meaningful disruption to their daily life. Examples: cosmetic app "
        "glitch, small billing query, curiosity about a plan feature."
    ),
    "Medium": (
        "Noticeable disruption to service or finances that requires attention "
        "within days. The customer is experiencing partial service degradation, "
        "an unexpected charge, or a process that is stalled. They can still "
        "function but the issue is causing ongoing inconvenience. Examples: "
        "intermittent connectivity, disputed charge on a bill, delayed number "
        "port."
    ),
    "High": (
        "Severe, time-critical impact — complete loss of essential service, "
        "significant financial harm, risk of regulatory or legal consequences, "
        "or a vulnerability/safety concern. The customer needs immediate "
        "resolution and may face escalating harm if the issue persists. "
        "Examples: total service outage lasting days, unauthorised account "
        "changes, data breach exposure, vulnerable customer unable to contact "
        "emergency services."
    ),
}

# ---------------------------------------------------------------------------
# Emotion definitions
# ---------------------------------------------------------------------------
EMOTION_DEFINITIONS = {
    "Low": (
        "Calm, measured, and factual. The customer reports the issue in a "
        "neutral, informational tone as if describing it to a colleague. "
        "Language is precise, unemotional, and solution-oriented. There is no "
        "frustration, blame, or urgency in the wording. The complaint reads "
        "like a matter-of-fact report."
    ),
    "Medium": (
        "Noticeably frustrated or disappointed. The customer uses language "
        "that signals irritation: words like 'frustrating', 'disappointed', "
        "'unacceptable', 'fed up'. Their tone is firm and they may express "
        "dissatisfaction with how the situation has been handled, but they "
        "remain coherent and reasonably measured. They want resolution and "
        "are losing patience."
    ),
    "High": (
        "Highly emotional — angry, distressed, desperate, or exasperated. "
        "The customer's language reflects intense feeling: strong adjectives "
        "('appalling', 'disgraceful', 'absolutely livid'), expressions of "
        "desperation or helplessness, threats to leave the provider or "
        "escalate to a regulator, repeated emphasis on how long the problem "
        "has persisted, and a sense that trust has been broken. The emotional "
        "intensity should come through vocabulary and sentence structure, not "
        "primarily through formatting. Some customers may use emphatic "
        "capitalisation or exclamation marks, but others may express intense "
        "anger in controlled, cutting prose."
    ),
}

# ---------------------------------------------------------------------------
# Scenarios (20)
# ---------------------------------------------------------------------------
SCENARIOS = [
    # Original 8
    "Billing overcharge / unexpected fees",
    "Network outage / poor coverage",
    "Contract dispute / early termination fee",
    "Slow or unreliable internet speed",
    "Rude or unhelpful customer service",
    "Porting / number transfer issues",
    "Roaming charges / international billing",
    "Service cancellation difficulty",
    # New 12
    "Data breach / privacy concern",
    "Accessibility needs not met",
    "Vulnerable customer mistreatment",
    "Multiple unresolved issues over time",
    "Device / handset fault or warranty dispute",
    "SIM card / eSIM activation problem",
    "Broadband installation delay or failure",
    "Customer loyalty not rewarded / retention offer dispute",
    "Direct debit or payment processing error",
    "Mis-selling / misleading sales practices",
    "Service downgrade without consent",
    "Complaint handling process failure",
]

# ---------------------------------------------------------------------------
# Writing styles (8) — intentionally orthogonal to emotion
# ---------------------------------------------------------------------------
STYLES = [
    "Formal professional",
    "Casual conversational",
    "Passive-aggressive / sarcastic",
    "Verbose and detailed",
    "Terse and minimal",
    "Narrative / storytelling",
    "Legalistic / rights-aware",
    "Polite but firm",
]

# ---------------------------------------------------------------------------
# Communication channels (4)
# ---------------------------------------------------------------------------
CHANNELS = [
    "email",
    "live chat",
    "online form",
    "social media",
]

# ---------------------------------------------------------------------------
# Three rotating system prompt templates
# ---------------------------------------------------------------------------
SYSTEM_PROMPTS = [
    # Template A
    (
        "You are a complaint-writing assistant. Your task is to produce "
        "realistic customer complaints addressed to a UK telecoms provider. "
        "Write only the complaint text — no labels, headings, metadata, or "
        "preamble. Vary the length naturally; some complaints should be short "
        "and others longer. Use realistic but varied customer names, or omit "
        "the name entirely. Never use placeholder names like 'John Doe'. "
        "Include realistic details: reference numbers (e.g., 'REF-20240917'), "
        "dates of prior contacts, account numbers, and specific amounts where "
        "relevant. Mention prior interactions ('I already called twice last "
        "week'). Match the communication channel — emails should have "
        "greetings and sign-offs, live chats should be informal and "
        "immediate, online forms should be structured, and social media posts "
        "should be concise and public-facing. Let emotional intensity come "
        "through naturally via word choice, sentence rhythm, and tone — not "
        "primarily through formatting tricks."
    ),
    # Template B
    (
        "Imagine you are different real customers contacting a UK "
        "telecommunications company to complain. For each complaint you "
        "generate, output nothing but the raw complaint message — no titles, "
        "no tags, no explanations. The complaints should feel authentic: "
        "varying in length, tone, and detail. If a name is used, make it "
        "sound genuine and diverse — avoid generic placeholders. Ground each "
        "complaint in specific details: dates, reference numbers, prior call "
        "history, specific amounts or plan names. Format the message "
        "appropriately for the specified communication channel. When emotion "
        "is high, write like a genuinely upset customer — the intensity "
        "should come from what they say and how they say it, not from "
        "excessive punctuation alone. Some angry customers write in clipped, "
        "controlled fury; others ramble; others threaten to go to Ofcom."
    ),
    # Template C
    (
        "Act as a generator of customer complaint messages for a UK telecoms "
        "company. Each output must read like a genuine message a customer "
        "would send through the specified support channel. Provide only the "
        "complaint body — do not include any metadata, labels, or framing "
        "text. Let the complaints differ naturally in length and specificity. "
        "Use believable names when appropriate; never use obvious filler "
        "names. Make each complaint feel grounded: include account details, "
        "dates, reference numbers, names of staff spoken to, and relevant "
        "history. Adapt the format to the channel (email vs chat vs form vs "
        "social media). Express emotion through authentic language — the "
        "reader should sense the customer's emotional state from their word "
        "choices, the rhythm of their sentences, and what they choose to "
        "emphasise, rather than from typographic conventions alone."
    ),
]
