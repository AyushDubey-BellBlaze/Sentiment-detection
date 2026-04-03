# rule_based.py - Simple keyword-based sentiment analysis

POSITIVE_WORDS = {
    "amazing", "great", "excellent", "good", "wonderful", "fantastic",
    "love", "best", "happy", "brilliant", "awesome", "outstanding",
    "perfect", "superb", "delightful", "enjoyed", "pleasant", "nice"
}

NEGATIVE_WORDS = {
    "bad", "terrible", "awful", "horrible", "worst", "hate", "poor",
    "disappointing", "boring", "dull", "waste", "ugly", "unpleasant",
    "pathetic", "disgusting", "mediocre", "failed", "useless"
}

def analyze(text: str) -> dict:
    words = text.lower().split()
    pos = sum(1 for w in words if w.strip(".,!?") in POSITIVE_WORDS)
    neg = sum(1 for w in words if w.strip(".,!?") in NEGATIVE_WORDS)

    if pos > neg:
        label, score = "POSITIVE", round(0.55 + 0.1 * (pos - neg), 2)
    elif neg > pos:
        label, score = "NEGATIVE", round(0.55 + 0.1 * (neg - pos), 2)
    else:
        label, score = "NEUTRAL", 0.50

    return {"label": label, "confidence": min(score, 0.99)}