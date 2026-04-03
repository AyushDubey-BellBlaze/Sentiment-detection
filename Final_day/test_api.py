"""
test_api.py — Test the Emotion Analysis API using requests.
Make sure the API is running first: uvicorn main:app --reload
"""

import requests
import json

BASE_URL = "http://127.0.0.1:8000"

test_sentences = [
    "I love this workshop so much!",
    "I'm terrified of the dark",
    "I can't believe they betrayed my trust",
    "I'm so bored and nothing is interesting",
    "I feel guilty about what I said to her",
    "This is the most exciting day ever!",
    "I miss the good old days with my family",
    "I'm grateful for all the help you gave me",
    "I don't understand anything that's happening",
    "I feel so alone and nobody cares",
    "I'm embarrassed by what I did yesterday",
    "I'm in awe of this beautiful sunset",
    "I feel proud of finishing this project",
    "I envy their success and opportunities",
    "The grief of losing him is unbearable"
]

def test_analyze(text):
    response = requests.post(
        f"{BASE_URL}/analyze",
        json={"text": text}
    )
    if response.status_code == 200:
        data = response.json()
        print(f"\nInput   : {data['input_text']}")
        print(f"Top 3   :")
        for i, pred in enumerate(data["top_3_predictions"], 1):
            bar = "█" * int(pred["confidence"] * 30)
            print(f"  {i}. {pred['emotion']:15s}  {pred['confidence']:.4f}  {bar}")
        print(f"Total emotions analyzed: {data['total_emotions']}")
    else:
        print(f"Error: {response.status_code} - {response.text}")

def test_health():
    r = requests.get(f"{BASE_URL}/")
    print("Health Check:", r.json())

def test_list_emotions():
    r = requests.get(f"{BASE_URL}/emotions")
    data = r.json()
    print(f"\nTotal Emotions Supported: {data['total']}")
    print("Emotions:", ", ".join(data["emotions"]))

if __name__ == "__main__":
    print("=" * 55)
    print("     Emotion Analysis API — Test Suite")
    print("=" * 55)

    test_health()
    test_list_emotions()

    print("\n--- Running Sentence Tests ---")
    for sentence in test_sentences:
        test_analyze(sentence)
        print("-" * 55)