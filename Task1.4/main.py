# main.py

import pandas as pd
import string
from nltk.corpus import stopwords
import nltk

# Download NLTK data (run once)
nltk.download('stopwords')

# Task 3: Preprocessing function
def preprocess_text(text):
    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    tokens = text.split()
    stop_words = set(stopwords.words('english'))
    filtered_words = [word for word in tokens if word not in stop_words]
    return filtered_words

# Load dataset
df = pd.read_csv("data.csv")
print("Dataset Loaded:")
print(df)

# Import Task 4 separately
from sentiment import rule_based_sentiment

# Apply Task 4 on each row
df['tokens'] = df['text'].apply(preprocess_text)
df['prediction'] = df['tokens'].apply(rule_based_sentiment)

print("\nDataset with Sentiment Prediction:")
print(df[['text', 'tokens', 'prediction']])