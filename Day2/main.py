# main.py

from data_loader import load_imdb_data
from feature_extractor import extract_tfidf_features

# ─────────────────────────────────────────
# Task 1: Load Dataset
# ─────────────────────────────────────────
(X_train, X_test, y_train, y_test), target_names = load_imdb_data(
    path=r"C:\Users\Lenovo\OneDrive - BELLBLAZE TECHNOLOGIES PRIVATE LIMITED\Desktop\aclImdb_v1\aclImdb\train"
)

print(f"Training samples: {len(X_train)}")
print(f"Test samples: {len(X_test)}")
print(f"Labels: {target_names}")

# Step 2: Print dataset info
print("\n=== Dataset Info ===")
print(f"Number of training samples: {len(X_train)}")
print(f"Number of testing samples: {len(X_test)}")
print(f"Target labels: {target_names}")

# Step 3: Print first 5 training samples
print("\n=== First 5 training samples ===")
for i in range(5):
    label = target_names[y_train[i]]
    sentiment = "😊 Positive" if y_train[i] == 1 else "😡 Negative" if y_train[i] == 0 else "😐 Neutral"
    print(f"Label: {label} ({sentiment})")
    print(f"Text: {X_train[i][:200]}...")
    print("-" * 50)

# ─────────────────────────────────────────
# Task 2: Convert Text to TF-IDF Features
# ─────────────────────────────────────────
print("\n=== Task 2: TF-IDF Feature Extraction ===")

X_train_tfidf, X_test_tfidf, vectorizer = extract_tfidf_features(X_train, X_test)

print(f"Training matrix shape : {X_train_tfidf.shape}")
print(f"Testing matrix shape  : {X_test_tfidf.shape}")
print(f"Total features (vocab): {len(vectorizer.get_feature_names_out())}")
print(f"Matrix type           : {type(X_train_tfidf)}")
print(f"Sample feature names  : {vectorizer.get_feature_names_out()[100:110]}")