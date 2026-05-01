"""
Data preprocessing and feature engineering for the malaria prediction model.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder


SYMPTOM_FEATURES = [
    "has_fever",
    "has_chills",
    "has_headache",
    "has_nausea",
    "has_fatigue",
    "has_sweating",
    "has_muscle_pain",
]

NUMERIC_FEATURES = ["age", "body_temperature", "rainfall_mm"]

CATEGORICAL_FEATURES = ["gender", "region", "season"]

ALL_FEATURES = NUMERIC_FEATURES + SYMPTOM_FEATURES + CATEGORICAL_FEATURES

REGION_MAP = {"tropical": 2, "subtropical": 1, "temperate": 0}
SEASON_MAP = {"wet": 1, "dry": 0}
GENDER_MAP = {"M": 1, "F": 0}


def encode_categoricals(df: pd.DataFrame) -> pd.DataFrame:
    """Encode categorical columns to numeric values.

    Parameters
    ----------
    df:
        Input DataFrame containing raw categorical columns.

    Returns
    -------
    pd.DataFrame
        Copy of *df* with categorical columns replaced by numeric codes.
    """
    df = df.copy()
    df["gender"] = df["gender"].str.upper().map(GENDER_MAP)
    df["region"] = df["region"].str.lower().map(REGION_MAP)
    df["season"] = df["season"].str.lower().map(SEASON_MAP)
    return df


def validate_input(data: dict) -> None:
    """Raise *ValueError* if required fields are missing or have invalid values.

    Parameters
    ----------
    data:
        Dictionary with a single patient's feature values.
    """
    missing = [f for f in ALL_FEATURES if f not in data]
    if missing:
        raise ValueError(f"Missing required fields: {missing}")

    if data["gender"].upper() not in GENDER_MAP:
        raise ValueError(f"gender must be one of {list(GENDER_MAP.keys())}")

    if data["region"].lower() not in REGION_MAP:
        raise ValueError(f"region must be one of {list(REGION_MAP.keys())}")

    if data["season"].lower() not in SEASON_MAP:
        raise ValueError(f"season must be one of {list(SEASON_MAP.keys())}")

    if not (0 <= data["age"] <= 120):
        raise ValueError("age must be between 0 and 120")

    if not (30.0 <= data["body_temperature"] <= 45.0):
        raise ValueError("body_temperature must be between 30.0 and 45.0 °C")

    for symptom in SYMPTOM_FEATURES:
        if data[symptom] not in (0, 1):
            raise ValueError(f"{symptom} must be 0 or 1")


def preprocess_dataframe(
    df: pd.DataFrame,
    scaler: StandardScaler | None = None,
    fit_scaler: bool = True,
) -> tuple[np.ndarray, StandardScaler]:
    """Full preprocessing pipeline for a DataFrame.

    Encodes categoricals, engineers features, and scales numeric columns.

    Parameters
    ----------
    df:
        DataFrame containing all feature columns and optionally a
        ``malaria_positive`` target column.
    scaler:
        Pre-fitted :class:`~sklearn.preprocessing.StandardScaler`.
        When *None* and ``fit_scaler=True`` a new scaler is fitted.
    fit_scaler:
        When *True* the scaler is fitted on *df* (training mode).
        When *False* the provided *scaler* is used as-is (inference mode).

    Returns
    -------
    X : np.ndarray
        Feature matrix ready for model input.
    scaler : StandardScaler
        Fitted scaler (same object as *scaler* if provided and not re-fitted).
    """
    df = encode_categoricals(df)

    # Derived feature: symptom burden count
    df["symptom_count"] = df[SYMPTOM_FEATURES].sum(axis=1)

    feature_cols = NUMERIC_FEATURES + SYMPTOM_FEATURES + ["gender", "region", "season", "symptom_count"]

    X = df[feature_cols].values.astype(float)

    if scaler is None:
        scaler = StandardScaler()

    if fit_scaler:
        X = scaler.fit_transform(X)
    else:
        X = scaler.transform(X)

    return X, scaler


def preprocess_single(patient: dict, scaler: StandardScaler) -> np.ndarray:
    """Preprocess a single patient record for inference.

    Parameters
    ----------
    patient:
        Dictionary with feature values (see :data:`ALL_FEATURES`).
    scaler:
        Fitted :class:`~sklearn.preprocessing.StandardScaler` from training.

    Returns
    -------
    np.ndarray
        2-D array of shape ``(1, n_features)`` ready for model ``predict``.
    """
    validate_input(patient)
    df = pd.DataFrame([patient])
    X, _ = preprocess_dataframe(df, scaler=scaler, fit_scaler=False)
    return X
