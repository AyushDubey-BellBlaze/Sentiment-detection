# evaluate.py
from sklearn.metrics import classification_report, accuracy_score

def evaluate_model(model, X_test_vec, y_test):
    """
    Prints classification report and accuracy
    """
    y_pred = model.predict(X_test_vec)
    print("Classification Report:\n")
    print(classification_report(y_test, y_pred))
    print("Accuracy:", round(accuracy_score(y_test, y_pred), 2))