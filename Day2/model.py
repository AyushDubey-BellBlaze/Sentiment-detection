# model.py
from sklearn.linear_model import LogisticRegression

def train_model(X_train_vec, y_train):
    """
    Train Logistic Regression classifier.
    Returns the trained model.
    """
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train_vec, y_train)
    return model