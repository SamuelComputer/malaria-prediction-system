"""
Generate a synthetic malaria training dataset and save it to data/malaria_data.csv.

Usage
-----
    python data/generate_sample_data.py
"""

import os
import numpy as np
import pandas as pd

OUTPUT_PATH = os.path.join(os.path.dirname(__file__), "malaria_data.csv")
RANDOM_STATE = 42
N_SAMPLES = 2000


def generate(n_samples: int = N_SAMPLES, random_state: int = RANDOM_STATE) -> pd.DataFrame:
    """Generate a synthetic malaria dataset.

    Positive cases are constructed so that symptomatic, tropical/wet-season
    patients with elevated temperature are more likely to be positive,
    mirroring real-world epidemiology.

    Parameters
    ----------
    n_samples:
        Total number of records to generate.
    random_state:
        NumPy random seed for reproducibility.

    Returns
    -------
    pd.DataFrame
        Dataset with features and a ``malaria_positive`` target column.
    """
    rng = np.random.default_rng(random_state)

    age = rng.integers(1, 80, size=n_samples)
    gender = rng.choice(["M", "F"], size=n_samples)
    region = rng.choice(["tropical", "subtropical", "temperate"], size=n_samples, p=[0.5, 0.3, 0.2])
    season = rng.choice(["wet", "dry"], size=n_samples, p=[0.55, 0.45])

    # Rainfall depends on region and season
    rainfall_base = {"tropical": 150, "subtropical": 80, "temperate": 40}
    season_mult = {"wet": 1.6, "dry": 0.5}
    rainfall_mm = np.array(
        [
            max(0.0, rng.normal(rainfall_base[r] * season_mult[s], 30))
            for r, s in zip(region, season)
        ]
    )

    # Symptoms more prevalent in malaria-endemic settings
    region_risk = {"tropical": 0.55, "subtropical": 0.30, "temperate": 0.10}
    base_risk = np.array([region_risk[r] for r in region])
    base_risk += (season == "wet") * 0.10

    malaria_positive = (rng.random(n_samples) < base_risk).astype(int)

    # Symptoms correlated with positive status
    def symptom(base_p_pos: float, base_p_neg: float) -> np.ndarray:
        p = np.where(malaria_positive == 1, base_p_pos, base_p_neg)
        return (rng.random(n_samples) < p).astype(int)

    has_fever = symptom(0.90, 0.20)
    has_chills = symptom(0.80, 0.15)
    has_headache = symptom(0.75, 0.30)
    has_nausea = symptom(0.60, 0.20)
    has_fatigue = symptom(0.85, 0.35)
    has_sweating = symptom(0.70, 0.15)
    has_muscle_pain = symptom(0.55, 0.25)

    # Temperature elevated in positive cases
    body_temperature = np.where(
        malaria_positive == 1,
        rng.normal(38.8, 0.6, n_samples),
        rng.normal(36.8, 0.4, n_samples),
    ).clip(35.0, 42.0)

    df = pd.DataFrame(
        {
            "age": age,
            "gender": gender,
            "body_temperature": body_temperature.round(1),
            "has_fever": has_fever,
            "has_chills": has_chills,
            "has_headache": has_headache,
            "has_nausea": has_nausea,
            "has_fatigue": has_fatigue,
            "has_sweating": has_sweating,
            "has_muscle_pain": has_muscle_pain,
            "region": region,
            "season": season,
            "rainfall_mm": rainfall_mm.round(1),
            "malaria_positive": malaria_positive,
        }
    )
    return df


if __name__ == "__main__":
    df = generate()
    df.to_csv(OUTPUT_PATH, index=False)
    pos = df["malaria_positive"].sum()
    print(f"Dataset saved to {OUTPUT_PATH}")
    print(f"  Total records : {len(df)}")
    print(f"  Positive cases: {pos} ({100 * pos / len(df):.1f}%)")
    print(f"  Negative cases: {len(df) - pos} ({100 * (1 - pos / len(df)):.1f}%)")
