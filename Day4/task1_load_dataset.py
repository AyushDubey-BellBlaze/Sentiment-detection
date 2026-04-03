# task1_load_dataset.py
# ─────────────────────────────────────────────────────────────
# Task 1: Load IMDB Dataset from HuggingFace
# ─────────────────────────────────────────────────────────────
# HOW IT WORKS ON YOUR MACHINE:
#   from datasets import load_dataset
#   dataset = load_dataset("imdb")
#
# This script simulates the exact same DatasetDict structure
# so you can run and understand the output locally too.
# ─────────────────────────────────────────────────────────────

from datasets import Dataset, DatasetDict
import pandas as pd

# ── Simulate IMDB dataset (mirrors real HuggingFace structure) ──
train_data = {
    "text": [
        "This movie was absolutely fantastic! The acting was superb.",
        "Terrible film. Waste of time and money. Boring from start to finish.",
        "One of the best movies I have ever seen. Highly recommended!",
        "Awful screenplay, bad direction. I fell asleep halfway through.",
        "A masterpiece of storytelling. Beautiful cinematography.",
        "Completely disappointing. The plot made no sense at all.",
        "Great performances by the entire cast. A must watch.",
        "Horrible movie. The worst I have seen this year.",
        "Brilliant direction and an amazing script. Loved every minute.",
        "Poor acting and a dull story. Would not recommend.",
    ],
    "label": [1, 0, 1, 0, 1, 0, 1, 0, 1, 0]  # 1=positive, 0=negative
}

test_data = {
    "text": [
        "I thought it would be bad, but it was amazing.",
        "Not worth watching. Very disappointing overall.",
        "Fantastic visuals and a gripping storyline!",
        "Mediocre at best. Nothing special about this film.",
    ],
    "label": [1, 0, 1, 0]
}

# Create HuggingFace-style DatasetDict
dataset = DatasetDict({
    "train": Dataset.from_dict(train_data),
    "test":  Dataset.from_dict(test_data),
})

# ── Output ──────────────────────────────────────────────────
print("=" * 55)
print("TASK 1 — IMDB Dataset Loaded")
print("=" * 55)
print(dataset)
print(f"\nTrain samples : {len(dataset['train'])}")
print(f"Test  samples : {len(dataset['test'])}")
print(f"\nLabel mapping : 0 = NEGATIVE  |  1 = POSITIVE")
print("\nSample entry from train set:")
print(f"  Text  : {dataset['train'][0]['text']}")
print(f"  Label : {'POSITIVE' if dataset['train'][0]['label'] == 1 else 'NEGATIVE'}")
print("=" * 55)
print("✅ Dataset ready for training!")