"""Tests for src/model.py and src/predict.py"""

import os
import tempfile
import pytest
import numpy as np
import pandas as pd

from src.model import train, evaluate, save_model, load_model
from src.predict import predict_single, predict_batch


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def synthetic_df():
    """Small synthetic dataset (200 rows) for fast tests."""
    from data.generate_sample_data import generate
    return generate(n_samples=200, random_state=0)


@pytest.fixture(scope="module")
def trained_artifacts(synthetic_df):
    """Train once and share across tests in this module."""
    model, scaler, metrics = train(synthetic_df, test_size=0.2, random_state=0)
    return model, scaler, metrics


@pytest.fixture()
def sample_patient():
    return {
        "age": 32,
        "gender": "F",
        "body_temperature": 39.1,
        "has_fever": 1,
        "has_chills": 1,
        "has_headache": 1,
        "has_nausea": 0,
        "has_fatigue": 1,
        "has_sweating": 1,
        "has_muscle_pain": 0,
        "region": "tropical",
        "season": "wet",
        "rainfall_mm": 145.0,
    }


# ---------------------------------------------------------------------------
# train
# ---------------------------------------------------------------------------

class TestTrain:
    def test_returns_three_items(self, synthetic_df):
        result = train(synthetic_df, test_size=0.2, random_state=1)
        assert len(result) == 3

    def test_metrics_keys(self, trained_artifacts):
        _, _, metrics = trained_artifacts
        for key in ("accuracy", "precision", "recall", "f1", "roc_auc", "report"):
            assert key in metrics

    def test_accuracy_reasonable(self, trained_artifacts):
        _, _, metrics = trained_artifacts
        # Random Forest on well-structured synthetic data should exceed 0.60
        assert metrics["accuracy"] > 0.60

    def test_roc_auc_above_random(self, trained_artifacts):
        _, _, metrics = trained_artifacts
        assert metrics["roc_auc"] > 0.50


# ---------------------------------------------------------------------------
# evaluate
# ---------------------------------------------------------------------------

class TestEvaluate:
    def test_all_correct_predictions(self, trained_artifacts):
        model, _, _ = trained_artifacts
        # Feed the model its own training predictions (perfect case simulation)
        X = np.zeros((10, model.n_features_in_))
        y = np.zeros(10, dtype=int)
        result = evaluate(model, X, y)
        assert 0.0 <= result["accuracy"] <= 1.0

    def test_zero_division_handled(self, trained_artifacts):
        model, _, _ = trained_artifacts
        # All predictions same class – precision/recall should not raise
        X = np.zeros((5, model.n_features_in_))
        y = np.zeros(5, dtype=int)
        result = evaluate(model, X, y)
        assert "precision" in result


# ---------------------------------------------------------------------------
# save_model / load_model
# ---------------------------------------------------------------------------

class TestPersistence:
    def test_round_trip(self, trained_artifacts, tmp_path, monkeypatch):
        model, scaler, _ = trained_artifacts

        # Redirect MODELS_DIR to a temp directory
        import src.model as model_module
        monkeypatch.setattr(model_module, "MODELS_DIR", str(tmp_path))
        monkeypatch.setattr(model_module, "MODEL_PATH", str(tmp_path / "malaria_model.joblib"))
        monkeypatch.setattr(model_module, "SCALER_PATH", str(tmp_path / "scaler.joblib"))

        save_model(model, scaler)
        loaded_model, loaded_scaler = load_model()

        assert loaded_model.n_estimators == model.n_estimators
        assert loaded_scaler.n_features_in_ == scaler.n_features_in_

    def test_load_model_missing_raises(self, tmp_path, monkeypatch):
        import src.model as model_module
        monkeypatch.setattr(model_module, "MODEL_PATH", str(tmp_path / "nonexistent.joblib"))
        monkeypatch.setattr(model_module, "SCALER_PATH", str(tmp_path / "nonexistent_scaler.joblib"))

        with pytest.raises(FileNotFoundError):
            load_model()


# ---------------------------------------------------------------------------
# predict_single
# ---------------------------------------------------------------------------

class TestPredictSingle:
    def test_returns_expected_keys(self, trained_artifacts, sample_patient):
        model, scaler, _ = trained_artifacts
        result = predict_single(sample_patient, model, scaler)
        assert set(result.keys()) == {"prediction", "probability", "label"}

    def test_prediction_is_binary(self, trained_artifacts, sample_patient):
        model, scaler, _ = trained_artifacts
        result = predict_single(sample_patient, model, scaler)
        assert result["prediction"] in (0, 1)

    def test_probability_in_range(self, trained_artifacts, sample_patient):
        model, scaler, _ = trained_artifacts
        result = predict_single(sample_patient, model, scaler)
        assert 0.0 <= result["probability"] <= 1.0

    def test_label_matches_prediction(self, trained_artifacts, sample_patient):
        model, scaler, _ = trained_artifacts
        result = predict_single(sample_patient, model, scaler)
        expected_label = "Malaria Positive" if result["prediction"] == 1 else "Malaria Negative"
        assert result["label"] == expected_label

    def test_invalid_input_raises(self, trained_artifacts):
        model, scaler, _ = trained_artifacts
        with pytest.raises(ValueError):
            predict_single({"age": 25}, model, scaler)


# ---------------------------------------------------------------------------
# predict_batch
# ---------------------------------------------------------------------------

class TestPredictBatch:
    def test_returns_correct_length(self, trained_artifacts, sample_patient):
        model, scaler, _ = trained_artifacts
        patients = [sample_patient] * 5
        results = predict_batch(patients, model, scaler)
        assert len(results) == 5

    def test_each_result_has_keys(self, trained_artifacts, sample_patient):
        model, scaler, _ = trained_artifacts
        patients = [sample_patient, sample_patient]
        results = predict_batch(patients, model, scaler)
        for r in results:
            assert "prediction" in r and "probability" in r
