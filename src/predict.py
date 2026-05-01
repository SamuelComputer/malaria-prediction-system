"""
Prediction utilities for the malaria prediction system.
"""

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler

from src.preprocessing import preprocess_single

LABEL_MAP = {0: "Malaria Negative", 1: "Malaria Positive"}


def predict_single(
    patient: dict,
    model: RandomForestClassifier,
    scaler: StandardScaler,
) -> dict:
    """Predict whether a single patient has malaria.

    Parameters
    ----------
    patient:
        Dictionary with patient feature values.
    model:
        Fitted :class:`~sklearn.ensemble.RandomForestClassifier`.
    scaler:
        Fitted :class:`~sklearn.preprocessing.StandardScaler` used during
        training.

    Returns
    -------
    dict
        ``{"prediction": int, "probability": float, "label": str}``
    """
    X = preprocess_single(patient, scaler)
    prediction = int(model.predict(X)[0])
    probability = float(model.predict_proba(X)[0][1])
    return {
        "prediction": prediction,
        "probability": round(probability, 4),
        "label": LABEL_MAP[prediction],
    }


def predict_batch(
    patients: list[dict],
    model: RandomForestClassifier,
    scaler: StandardScaler,
) -> list[dict]:
    """Predict malaria for a list of patients.

    Parameters
    ----------
    patients:
        List of patient feature dictionaries.
    model:
        Fitted :class:`~sklearn.ensemble.RandomForestClassifier`.
    scaler:
        Fitted :class:`~sklearn.preprocessing.StandardScaler`.

    Returns
    -------
    list[dict]
        One result dictionary per patient (same order as *patients*).
    """
    return [predict_single(p, model, scaler) for p in patients]
