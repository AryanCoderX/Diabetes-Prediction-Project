"""Input preparation for the bundled diabetes classifier."""

from __future__ import annotations

import pandas as pd

FEATURES = (
    "Pregnancies",
    "Glucose",
    "BloodPressure",
    "SkinThickness",
    "Insulin",
    "BMI",
    "DiabetesPedigreeFunction",
    "Age",
)

ZERO_AS_MISSING = ("Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI")

# Means used by the legacy estimator included with this app. Newer pipeline
# models carry their own fitted imputer and do not use these constants.
TRAINING_MEANS = {
    "Glucose": 121.686762778506,
    "BloodPressure": 72.405184174625,
    "SkinThickness": 29.153419593346,
    "Insulin": 155.548223350254,
    "BMI": 32.457463672391,
}


def prepare_model_input(patient: pd.DataFrame, model: object) -> pd.DataFrame:
    """Match inference preprocessing to the saved estimator or pipeline."""
    prepared = patient.loc[:, list(FEATURES)].copy()
    for column in ZERO_AS_MISSING:
        prepared[column] = prepared[column].astype("float64").replace(0, float("nan"))

    steps = getattr(model, "named_steps", {})
    if "imputer" not in steps:
        prepared = prepared.fillna(TRAINING_MEANS)
    return prepared
