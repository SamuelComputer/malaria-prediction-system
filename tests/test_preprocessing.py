"""Tests for src/preprocessing.py"""

import pytest
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from src.preprocessing import (
    encode_categoricals,
    validate_input,
    preprocess_dataframe,
    preprocess_single,
    GENDER_MAP,
    REGION_MAP,
    SEASON_MAP,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def sample_patient():
    return {
        "age": 28,
        "gender": "M",
        "body_temperature": 38.5,
        "has_fever": 1,
        "has_chills": 1,
        "has_headache": 0,
        "has_nausea": 0,
        "has_fatigue": 1,
        "has_sweating": 1,
        "has_muscle_pain": 0,
        "region": "tropical",
        "season": "wet",
        "rainfall_mm": 130.0,
    }


@pytest.fixture()
def sample_df(sample_patient):
    rows = [dict(sample_patient, malaria_positive=1) for _ in range(30)]
    rows += [
        dict(
            sample_patient,
            has_fever=0,
            has_chills=0,
            body_temperature=36.7,
            malaria_positive=0,
        )
        for _ in range(30)
    ]
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# encode_categoricals
# ---------------------------------------------------------------------------

class TestEncodeCategoricals:
    def test_gender_mapping(self):
        df = pd.DataFrame({"gender": ["M", "F", "m", "f"], "region": ["tropical"] * 4, "season": ["wet"] * 4})
        out = encode_categoricals(df)
        assert list(out["gender"]) == [GENDER_MAP["M"], GENDER_MAP["F"], GENDER_MAP["M"], GENDER_MAP["F"]]

    def test_region_mapping(self):
        df = pd.DataFrame({"gender": ["M"] * 3, "region": ["tropical", "subtropical", "temperate"], "season": ["wet"] * 3})
        out = encode_categoricals(df)
        assert out["region"].tolist() == [REGION_MAP["tropical"], REGION_MAP["subtropical"], REGION_MAP["temperate"]]

    def test_season_mapping(self):
        df = pd.DataFrame({"gender": ["M"] * 2, "region": ["tropical"] * 2, "season": ["wet", "dry"]})
        out = encode_categoricals(df)
        assert out["season"].tolist() == [SEASON_MAP["wet"], SEASON_MAP["dry"]]

    def test_does_not_mutate_original(self, sample_patient):
        df = pd.DataFrame([sample_patient])
        _ = encode_categoricals(df)
        assert df["gender"].iloc[0] == "M"


# ---------------------------------------------------------------------------
# validate_input
# ---------------------------------------------------------------------------

class TestValidateInput:
    def test_valid_patient_passes(self, sample_patient):
        validate_input(sample_patient)  # should not raise

    def test_missing_field_raises(self, sample_patient):
        del sample_patient["age"]
        with pytest.raises(ValueError, match="Missing required fields"):
            validate_input(sample_patient)

    def test_invalid_gender_raises(self, sample_patient):
        sample_patient["gender"] = "X"
        with pytest.raises(ValueError, match="gender"):
            validate_input(sample_patient)

    def test_invalid_region_raises(self, sample_patient):
        sample_patient["region"] = "arctic"
        with pytest.raises(ValueError, match="region"):
            validate_input(sample_patient)

    def test_invalid_season_raises(self, sample_patient):
        sample_patient["season"] = "spring"
        with pytest.raises(ValueError, match="season"):
            validate_input(sample_patient)

    def test_age_out_of_range_raises(self, sample_patient):
        sample_patient["age"] = 200
        with pytest.raises(ValueError, match="age"):
            validate_input(sample_patient)

    def test_temperature_out_of_range_raises(self, sample_patient):
        sample_patient["body_temperature"] = 20.0
        with pytest.raises(ValueError, match="body_temperature"):
            validate_input(sample_patient)

    def test_invalid_symptom_value_raises(self, sample_patient):
        sample_patient["has_fever"] = 2
        with pytest.raises(ValueError, match="has_fever"):
            validate_input(sample_patient)


# ---------------------------------------------------------------------------
# preprocess_dataframe
# ---------------------------------------------------------------------------

class TestPreprocessDataframe:
    def test_returns_correct_shapes(self, sample_df):
        X, scaler = preprocess_dataframe(sample_df)
        assert X.ndim == 2
        assert X.shape[0] == len(sample_df)
        assert isinstance(scaler, StandardScaler)

    def test_scaler_reuse_in_inference_mode(self, sample_df):
        X_train, scaler = preprocess_dataframe(sample_df, fit_scaler=True)
        X_infer, _ = preprocess_dataframe(sample_df, scaler=scaler, fit_scaler=False)
        # Same data → same result
        np.testing.assert_array_almost_equal(X_train, X_infer)

    def test_symptom_count_feature(self, sample_df):
        # With fit_scaler, we can't directly check the raw symptom_count,
        # but we can verify the output has the expected number of columns.
        X, _ = preprocess_dataframe(sample_df)
        # 3 numeric + 7 symptoms + 3 categorical + 1 symptom_count = 14 features
        assert X.shape[1] == 14


# ---------------------------------------------------------------------------
# preprocess_single
# ---------------------------------------------------------------------------

class TestPreprocessSingle:
    def test_output_shape(self, sample_patient, sample_df):
        _, scaler = preprocess_dataframe(sample_df)
        X = preprocess_single(sample_patient, scaler)
        assert X.shape == (1, 14)

    def test_raises_on_invalid_input(self, sample_df):
        _, scaler = preprocess_dataframe(sample_df)
        bad_patient = {"age": 30}  # missing many fields
        with pytest.raises(ValueError):
            preprocess_single(bad_patient, scaler)
