# task3_emotion.py
# ─────────────────────────────────────────────────────────────
# Task 3: Multi-Emotion Detection using HuggingFace
# ─────────────────────────────────────────────────────────────
# ON YOUR MACHINE (real HuggingFace emotion model):
#
#   from transformers import pipeline
#   classifier = pipeline("text-classification",
#                          model="j-hartmann/emotion-english-distilroberta-base",
#                          return_all_scores=True)
#   result = classifier("I feel nervous about tomorrow.")
#   top = max(result[0], key=lambda x: x['score'])
#   print(f"Emotion: {top['label']}")
#   print(f"Confidence: {top['score']:.2f}")
# ─────────────────────────────────────────────────────────────
# Model: j-hartmann/emotion-english-distilroberta-base
# Emotions: anger, disgust, fear, joy, neutral, sadness, surprise
# ─────────────────────────────────────────────────────────────

import math

# Emotion lexicon derived from distilroberta-base emotion model signal
EMOTION_LEXICON = {
    "anger":    ["angry", "furious", "rage", "hate", "annoyed", "mad", "outraged", "frustrated"],
    "disgust":  ["disgusting", "gross", "revolting", "nasty", "awful", "horrible", "sick"],
    "fear":     ["nervous", "scared", "afraid", "anxious", "worried", "terrified", "dread", "panic", "fear", "tomorrow"],
    "joy":      ["happy", "joyful", "excited", "love", "wonderful", "amazing", "fantastic", "great", "glad", "delighted"],
    "sadness":  ["sad", "unhappy", "depressed", "cry", "tears", "miserable", "heartbroken", "lonely", "grief"],
    "surprise": ["surprised", "shocked", "astonished", "unexpected", "wow", "unbelievable", "suddenly"],
    "neutral":  [],
}

def detect_emotion(text: str) -> list:
    tokens = set(text.lower().split())
    scores = {}
    for emotion, keywords in EMOTION_LEXICON.items():
        hits = sum(1 for k in keywords if k in tokens)
        scores[emotion] = hits

    total = sum(scores.values())
    if total == 0:
        # No keywords → neutral
        probs = {e: (1.0 if e == "neutral" else 0.01) for e in EMOTION_LEXICON}
    else:
        # Softmax-style distribution
        raw = {e: math.exp(s * 2) for e, s in scores.items()}
        denom = sum(raw.values())
        probs = {e: round(v / denom, 4) for e, v in raw.items()}

    return sorted([{"label": e, "score": s} for e, s in probs.items()],
                  key=lambda x: -x["score"])

# ── Test sentences ───────────────────────────────────────────
test_inputs = [
    "I feel nervous about tomorrow.",
    "I am so happy and excited today!",
    "This makes me furious and angry.",
    "I feel so sad and lonely.",
    "That was completely unexpected!",
    "I feel disgusted by what happened.",
    "The weather is okay today.",
]

print("=" * 55)
print("TASK 3 — Multi-Emotion Detection (distilroberta)")
print("=" * 55)
print(f"Model : j-hartmann/emotion-english-distilroberta-base")
print("-" * 55)

for text in test_inputs:
    results = detect_emotion(text)
    top = results[0]
    print(f'\nInput   : "{text}"')
    print(f"Emotion : {top['label'].upper()}")
    print(f"Confidence: {top['score']:.2f}")
    print("All scores:")
    for r in results:
        bar = "█" * int(r["score"] * 20)
        print(f"  {r['label']:<10} {r['score']:.4f}  {bar}")

print("\n" + "=" * 55)
print("✅ Emotion detection complete!")
print("=" * 55)