"""
src.preprocessing
~~~~~~~~~~~~~~~~~
Data loading, inspection, and preprocessing pipeline for the Diabetes Prediction System.

Pipeline overview:
  - Numerical features (Age, BMI, Glucose, Blood_Pressure, Cholesterol):
      median imputation -> StandardScaler
  - Binary integer features (Family_History, Heart_Disease, High_BP, High_Chol):
      most-frequent imputation (no scaling needed)
  - Categorical features (Sex, Physical_Activity, Diet_Quality, Smoking_Status):
      most-frequent imputation -> OneHotEncoder

All transformations are fitted ONLY on training data to prevent data leakage.
"""
from __future__ import annotations

import pandas as pd
import numpy as np
from pathlib import Path

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
TARGET_COLUMN = "Diabetes_binary"

# Numerical features that need median imputation + StandardScaler
NUMERICAL_FEATURES = ["Age", "BMI", "Glucose", "Blood_Pressure", "Cholesterol"]

# Binary 0/1 integer features -- only most-frequent imputation, no scaling
BINARY_FEATURES = ["Family_History", "Heart_Disease", "High_BP", "High_Chol"]

# Categorical string features requiring OneHotEncoding
CATEGORICAL_FEATURES = ["Sex", "Physical_Activity", "Diet_Quality", "Smoking_Status"]

# All feature columns in a reproducible fixed order (target excluded)
FEATURE_COLUMNS = NUMERICAL_FEATURES + BINARY_FEATURES + CATEGORICAL_FEATURES

RANDOM_STATE = 42
TEST_SIZE = 0.20


def load_dataset(data_path) -> pd.DataFrame:
    """Load the diabetes CSV dataset and return a DataFrame."""
    path = Path(data_path)
    if not path.exists():
        raise FileNotFoundError(
            f"Dataset not found at \'{path}\'. "
            "Ensure \'diabetes_binary_health.csv\' is present in the data/ directory."
        )
    return pd.read_csv(path)


def inspect_dataset(df: pd.DataFrame) -> dict:
    """
    Compute and return a comprehensive summary of dataset properties.
    Includes shape, missing values, duplicates, column type lists,
    and class distribution for the target column.
    """
    summary = {
        "shape": df.shape,
        "columns": list(df.columns),
        "missing_values": df.isnull().sum().to_dict(),
        "missing_pct": (df.isnull().sum() / len(df) * 100).round(2).to_dict(),
        "duplicate_rows": int(df.duplicated().sum()),
        "numerical_cols": list(df.select_dtypes(include=[np.number]).columns),
        "categorical_cols": list(df.select_dtypes(include=["object"]).columns),
    }
    if TARGET_COLUMN in df.columns:
        vc = df[TARGET_COLUMN].value_counts().sort_index()
        summary["class_distribution"] = vc.to_dict()
        summary["class_balance_pct"] = (vc / len(df) * 100).round(2).to_dict()
    return summary


def build_preprocessor() -> ColumnTransformer:
    """
    Build and return an unfitted sklearn ColumnTransformer.

    Transformers:
      \'num\' : median imputation + StandardScaler  (numerical)
      \'bin\' : most-frequent imputation            (binary integer)
      \'cat\' : most-frequent imputation + OHE      (categorical)
    """
    numerical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    binary_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
    ])
    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numerical_pipeline, NUMERICAL_FEATURES),
            ("bin", binary_pipeline, BINARY_FEATURES),
            ("cat", categorical_pipeline, CATEGORICAL_FEATURES),
        ],
        remainder="drop",
    )
    return preprocessor


def get_feature_names_out(preprocessor: ColumnTransformer) -> list:
    """Extract ordered feature names from a FITTED ColumnTransformer."""
    names = list(NUMERICAL_FEATURES) + list(BINARY_FEATURES)
    cat_t = preprocessor.named_transformers_["cat"]
    ohe = cat_t.named_steps["encoder"]
    names.extend(ohe.get_feature_names_out(CATEGORICAL_FEATURES).tolist())
    return names


def prepare_features(df: pd.DataFrame):
    """Split a DataFrame into X (features) and y (target). Validates columns."""
    missing_cols = [c for c in FEATURE_COLUMNS if c not in df.columns]
    if missing_cols:
        raise ValueError(f"Dataset is missing required columns: {missing_cols}")
    if TARGET_COLUMN not in df.columns:
        raise ValueError(f"Target column \'{TARGET_COLUMN}\' not found.")
    X = df[FEATURE_COLUMNS].copy()
    y = df[TARGET_COLUMN].copy()
    return X, y
