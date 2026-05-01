"""
Model training, evaluation, and persistence for the malaria prediction system.
"""

import os
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    classification_report,
)
from sklearn.preprocessing import StandardScaler

from src.preprocessing import preprocess_dataframe

MODELS_DIR = os.path.join(os.path.dirname(__file__), "..", "models")
MODEL_PATH = os.path.join(MODELS_DIR, "malaria_model.joblib")
SCALER_PATH = os.path.join(MODELS_DIR, "scaler.joblib")


def train(
    df: pd.DataFrame,
    test_size: float = 0.2,
    random_state: int = 42,
    n_estimators: int = 200,
    max_depth: int | None = None,
) -> tuple[RandomForestClassifier, StandardScaler, dict]:
    """Train a Random Forest classifier on *df*.

    Parameters
    ----------
    df:
        DataFrame that must contain all feature columns plus a
        ``malaria_positive`` target column.
    test_size:
        Fraction of data to use as the held-out test set.
    random_state:
        Seed for reproducibility.
    n_estimators:
        Number of trees in the forest.
    max_depth:
        Maximum depth of each tree (``None`` = grow until leaves are pure).

    Returns
    -------
    model : RandomForestClassifier
        Fitted model.
    scaler : StandardScaler
        Fitted scaler (used to normalise features).
    metrics : dict
        Evaluation metrics on the held-out test set.
    """
    y = df["malaria_positive"].values

    X, scaler = preprocess_dataframe(df, fit_scaler=True)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    model = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        random_state=random_state,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)

    metrics = evaluate(model, X_test, y_test)
    return model, scaler, metrics


def evaluate(
    model: RandomForestClassifier,
    X_test: np.ndarray,
    y_test: np.ndarray,
) -> dict:
    """Compute evaluation metrics on a test set.

    Parameters
    ----------
    model:
        Fitted :class:`~sklearn.ensemble.RandomForestClassifier`.
    X_test:
        Feature matrix for the test set.
    y_test:
        True labels for the test set.

    Returns
    -------
    dict
        Dictionary with keys ``accuracy``, ``precision``, ``recall``,
        ``f1``, ``roc_auc``, and ``report``.
    """
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    return {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "precision": float(precision_score(y_test, y_pred, zero_division=0)),
        "recall": float(recall_score(y_test, y_pred, zero_division=0)),
        "f1": float(f1_score(y_test, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_test, y_prob)),
        "report": classification_report(y_test, y_pred),
    }


def save_model(model: RandomForestClassifier, scaler: StandardScaler) -> None:
    """Persist *model* and *scaler* to disk.

    Parameters
    ----------
    model:
        Fitted :class:`~sklearn.ensemble.RandomForestClassifier`.
    scaler:
        Fitted :class:`~sklearn.preprocessing.StandardScaler`.
    """
    os.makedirs(MODELS_DIR, exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    joblib.dump(scaler, SCALER_PATH)


def load_model() -> tuple[RandomForestClassifier, StandardScaler]:
    """Load model and scaler from disk.

    Returns
    -------
    model : RandomForestClassifier
    scaler : StandardScaler

    Raises
    ------
    FileNotFoundError
        When either the model or the scaler file is missing.
    """
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            f"Model not found at {MODEL_PATH}. Run train.py first."
        )
    if not os.path.exists(SCALER_PATH):
        raise FileNotFoundError(
            f"Scaler not found at {SCALER_PATH}. Run train.py first."
        )
    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    return model, scaler
