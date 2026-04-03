# feature_extractor.py

from sklearn.feature_extraction.text import TfidfVectorizer

def extract_tfidf_features(X_train, X_test):
    """
    Convert raw text reviews into TF-IDF feature vectors.
    """
    vectorizer = TfidfVectorizer(
        max_features=10000,
        ngram_range=(1, 2),
        stop_words='english',
        strip_accents='unicode',
        lowercase=True,
        min_df=2,
    )

    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)

    return X_train_tfidf, X_test_tfidf, vectorizer