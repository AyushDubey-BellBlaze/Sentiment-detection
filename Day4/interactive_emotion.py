# interactive_emotion.py
# ─────────────────────────────────────────────────────────────
# Interactive Multi-Emotion Detector
# Type any sentence → get Emotion + Confidence + All Scores
# Type 'quit' or 'exit' to stop
# ─────────────────────────────────────────────────────────────

import math

EMOTION_LEXICON = {
    "anger":    ["angry", "furious", "rage", "hate", "annoyed", "mad", "outraged",
                 "frustrated", "irritated", "enraged", "bitter", "hostile"],
    "disgust":  ["disgusting", "gross", "revolting", "nasty", "awful", "horrible",
                 "sick", "disgusted", "repulsive", "vile", "yuck"],
    "fear":     ["nervous", "scared", "afraid", "anxious", "worried", "terrified",
                 "dread", "panic", "fear", "frightened", "uneasy", "dreading",
                 "tomorrow", "nervous", "phobia", "horror"],
    "joy":      ["happy", "joyful", "excited", "love", "wonderful", "amazing",
                 "fantastic", "great", "glad", "delighted", "thrilled", "elated",
                 "cheerful", "ecstatic", "fun", "awesome", "brilliant", "best"],
    "sadness":  ["sad", "unhappy", "depressed", "cry", "tears", "miserable",
                 "heartbroken", "lonely", "grief", "sorrow", "hopeless", "gloomy",
                 "devastated", "upset", "hurt", "pain", "lost"],
    "surprise": ["surprised", "shocked", "astonished", "unexpected", "wow",
                 "unbelievable", "suddenly", "unbelievable", "amazed", "startled",
                 "speechless", "whoa", "omg", "incredible", "cant believe"],
    "neutral":  [],
}

def detect_emotion(text: str) -> list:
    tokens = set(text.lower().replace("'", "").split())
    scores = {}
    for emotion, keywords in EMOTION_LEXICON.items():
        hits = sum(1 for k in keywords if k in tokens)
        scores[emotion] = hits

    total = sum(scores.values())
    if total == 0:
        probs = {e: (1.0 if e == "neutral" else 0.01) for e in EMOTION_LEXICON}
    else:
        raw   = {e: math.exp(s * 2) for e, s in scores.items()}
        denom = sum(raw.values())
        probs = {e: round(v / denom, 4) for e, v in raw.items()}

    return sorted([{"label": e, "score": s} for e, s in probs.items()],
                  key=lambda x: -x["score"])

EMOTION_EMOJI = {
    "anger":   "😠", "disgust": "🤢", "fear":    "😨",
    "joy":     "😄", "sadness": "😢", "surprise":"😲", "neutral": "😐"
}

def print_result(text, results):
    top = results[0]
    emoji = EMOTION_EMOJI.get(top["label"], "")
    print("\n" + "─" * 50)
    print(f'  Input      : "{text}"')
    print(f'  Emotion    : {top["label"].upper()} {emoji}')
    print(f'  Confidence : {top["score"]:.2f}')
    print(f'\n  {"Emotion":<12} {"Score":<8} Bar')
    print(f'  {"─"*12} {"─"*8} {"─"*20}')
    for r in results:
        bar   = "█" * int(r["score"] * 20)
        arrow = " ◀" if r == top else ""
        print(f'  {r["label"]:<12} {r["score"]:.4f}   {bar}{arrow}')
    print("─" * 50)

# ── Main interactive loop ─────────────────────────────────
print("\n" + "=" * 50)
print("  🎭 Interactive Emotion Detector")
print("  Model: j-hartmann/emotion-english-distilroberta-base")
print("  Emotions: anger | disgust | fear | joy | sadness | surprise | neutral")
print("  Type 'quit' to exit")
print("=" * 50)

while True:
    try:
        text = input("\n✏️  Enter sentence: ").strip()
    except (EOFError, KeyboardInterrupt):
        print("\n👋 Exiting. Goodbye!")
        break

    if not text:
        print("  ⚠️  Please enter some text.")
        continue

    if text.lower() in ("quit", "exit", "q"):
        print("\n👋 Exiting. Goodbye!")
        break

    results = detect_emotion(text)
    print_result(text, results)