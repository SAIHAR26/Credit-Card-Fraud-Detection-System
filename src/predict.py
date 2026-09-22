"""Prediction helpers for saved fraud detection models."""

from pathlib import Path

import joblib


def load_model(model_path):
    """Load a saved model artifact from disk."""
    path = Path(model_path)
    if not path.exists():
        raise FileNotFoundError(f"Model file not found: {path}")
    return joblib.load(path)


def predict_transaction(model, transaction_features):
    """Predict whether a transaction is legitimate or fraudulent."""
    return model.predict(transaction_features)
