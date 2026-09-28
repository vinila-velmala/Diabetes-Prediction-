"""
src.train_model
~~~~~~~~~~~~~~~
Trains and saves the Random Forest diabetes prediction pipeline.

Workflow:
  1. Load data/diabetes_binary_health.csv
  2. Inspect dataset properties
  3. Prepare features and target
  4. 80/20 stratified train-test split
  5. Build preprocessing + RandomForestClassifier pipeline
  6. Fit pipeline on training data (no data leakage)
  7. Save complete pipeline to models/diabetes_model.pkl

Usage:
    python src/train_model.py
"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.preprocessing import (
    load_dataset, inspect_dataset, build_preprocessor,
    prepare_features, RANDOM_STATE, TEST_SIZE,
)

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
MODEL_SAVE_PATH = Path("models") / "diabetes_model.pkl"
DATA_PATH = Path("data") / "diabetes_binary_health.csv"

RF_PARAMS = {
    "n_estimators": 200,
    "max_depth": 12,
    "min_samples_split": 5,
    "min_samples_leaf": 2,
    "max_features": "sqrt",
    "class_weight": "balanced",
    "random_state": RANDOM_STATE,
    "n_jobs": -1,
}


def build_pipeline() -> Pipeline:
    """Construct the preprocessing + RandomForest pipeline."""
    preprocessor = build_preprocessor()
    classifier = RandomForestClassifier(**RF_PARAMS)
    return Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", classifier),
    ])


def train():
    """Full training workflow. Returns (pipeline, X_train, X_test, y_train, y_test)."""
    print("=" * 60)
    print("  Diabetes Prediction -- Random Forest Training")
    print("=" * 60)

    print("\n[1/5] Loading dataset...")
    df = load_dataset(DATA_PATH)

    print("[2/5] Inspecting dataset...")
    summary = inspect_dataset(df)
    shape = summary["shape"]
    dups = summary["duplicate_rows"]
    mv = {k: v for k, v in summary["missing_values"].items() if v > 0}
    cls = summary.get("class_distribution", {})
    print(f"  Shape        : {shape}")
    print(f"  Duplicates   : {dups}")
    print(f"  Missing vals : {mv if mv else chr(39) + chr(39)}")
    print(f"  Class dist   : {cls}")

    print("[3/5] Preparing features and target...")
    X, y = prepare_features(df)

    print("[4/5] Splitting data (80/20 stratified)...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y,
    )
    print(f"  Train size   : {len(X_train)}")
    print(f"  Test size    : {len(X_test)}")

    print("[5/5] Building and fitting pipeline...")
    pipeline = build_pipeline()
    pipeline.fit(X_train, y_train)

    MODEL_SAVE_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, MODEL_SAVE_PATH)
    print(f"\n  Model saved  : {MODEL_SAVE_PATH}")
    print("=" * 60)
    print("  Training complete.")
    print("=" * 60)
    return pipeline, X_train, X_test, y_train, y_test


if __name__ == "__main__":
    from src.evaluate_model import evaluate
    pipeline, X_train, X_test, y_train, y_test = train()
    evaluate(pipeline, X_test, y_test, save_plots=True)
