# data_loader.py

from sklearn.datasets import load_files
from sklearn.model_selection import train_test_split
import numpy as np

def load_imdb_data(path, test_size=0.2, random_state=42):
    """
    Load IMDB dataset with 3 classes: neg, pos, unsup (neutral)
    """
    data = load_files(
        path,
        categories=['pos', 'neg', 'unsup'],
        encoding='utf-8'
        # 'errors' parameter removed — not supported in your sklearn version
    )

    X = data.data
    y = data.target
    target_names = ['neg', 'pos', 'neutral']

    # Remap labels: pos=1, neg=0, unsup(neutral)=2
    label_map = {
        list(data.target_names).index('neg'): 0,
        list(data.target_names).index('pos'): 1,
        list(data.target_names).index('unsup'): 2,
    }
    y = np.array([label_map[label] for label in y])

    # Split into train/test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    return (X_train, X_test, y_train, y_test), target_names