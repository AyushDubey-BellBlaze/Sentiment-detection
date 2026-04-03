# ml_model.py - Traditional ML sentiment analysis using TF-IDF + Logistic Regression

from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np

# Small training dataset
TRAIN_TEXTS = [
    "I love this product", "This is amazing", "Great experience",
    "Wonderful service", "Excellent quality", "Fantastic result",
    "Best purchase ever", "Really happy with this", "Outstanding performance",
    "I hate this", "Terrible experience", "This is awful",
    "Worst product ever", "Very disappointing", "Horrible quality",
    "Complete waste of money", "Really bad service", "Poor performance",
    "It was okay", "Not great not bad", "Average experience"
]
TRAIN_LABELS = [1]*9 + [0]*9 + [2]*3  # 1=pos, 0=neg, 2=neutral

LABEL_MAP = {1: "POSITIVE", 0: "NEGATIVE", 2: "NEUTRAL"}

vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(TRAIN_TEXTS)
clf = LogisticRegression(max_iter=1000, random_state=42)
clf.fit(X, TRAIN_LABELS)

def analyze(text: str) -> dict:
    vec = vectorizer.transform([text])
    pred = clf.predict(vec)[0]
    proba = clf.predict_proba(vec)[0]
    confidence = round(float(np.max(proba)), 2)
    return {"label": LABEL_MAP[pred], "confidence": confidence}