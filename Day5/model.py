from transformers import pipeline
from config import settings

emotion_pipeline = pipeline(
    "text-classification",
    model=settings.MODEL_NAME,
    return_all_scores=True
)

def predict_emotions(text: str):
    results = emotion_pipeline(text)[0]

    # Sort by score
    sorted_results = sorted(results, key=lambda x: x['score'], reverse=True)

    top_emotion = sorted_results[0]

    # Multi-label emotions
    multi_emotions = [
        e for e in sorted_results if e["score"] > settings.EMOTION_THRESHOLD
    ]

    return top_emotion, sorted_results, multi_emotions