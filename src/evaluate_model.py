"""
src.evaluate_model
~~~~~~~~~~~~~~~~~~
Model evaluation for the Diabetes Prediction System.

Computes and displays:
  - Accuracy, Precision, Recall, F1-Score
  - ROC-AUC score
  - Confusion matrix
  - Full classification report
  - ROC curve
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report, roc_curve,
)

PLOTS_DIR = Path("models") / "plots"


def evaluate(pipeline, X_test, y_test, save_plots: bool = True) -> dict:
    """Evaluate pipeline on test set; return dict of metrics."""
    y_pred = pipeline.predict(X_test)
    y_prob = pipeline.predict_proba(X_test)[:, 1]

    metrics = {
        "accuracy":  round(accuracy_score(y_test, y_pred), 4),
        "precision": round(precision_score(y_test, y_pred, zero_division=0), 4),
        "recall":    round(recall_score(y_test, y_pred, zero_division=0), 4),
        "f1_score":  round(f1_score(y_test, y_pred, zero_division=0), 4),
        "roc_auc":   round(roc_auc_score(y_test, y_prob), 4),
    }

    print("\n" + "=" * 60)
    print("  Model Evaluation Results (Test Set)")
    print("=" * 60)
    for metric, value in metrics.items():
        print(f"  {metric.upper():<12}: {value:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=["Non-Diabetic", "Diabetic"]))

    if save_plots:
        PLOTS_DIR.mkdir(parents=True, exist_ok=True)
        _plot_confusion_matrix(y_test, y_pred)
        _plot_roc_curve(y_test, y_prob)

    return metrics


def _plot_confusion_matrix(y_test, y_pred):
    """Save a styled confusion matrix heatmap."""
    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=["Non-Diabetic", "Diabetic"],
        yticklabels=["Non-Diabetic", "Diabetic"],
        ax=ax,
    )
    ax.set_title("Confusion Matrix", fontsize=14, fontweight="bold", pad=15)
    ax.set_xlabel("Predicted Label", fontsize=11)
    ax.set_ylabel("True Label", fontsize=11)
    plt.tight_layout()
    path = PLOTS_DIR / "confusion_matrix.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Confusion matrix saved : {path}")


def _plot_roc_curve(y_test, y_prob):
    """Save the ROC curve with AUC annotation."""
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    auc_val = roc_auc_score(y_test, y_prob)
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(fpr, tpr, color="#4C72B0", lw=2, label=f"ROC Curve (AUC = {auc_val:.4f})")
    ax.plot([0, 1], [0, 1], color="gray", linestyle="--", lw=1, label="Random Classifier")
    ax.set_title("ROC Curve -- Random Forest", fontsize=14, fontweight="bold", pad=15)
    ax.set_xlabel("False Positive Rate", fontsize=11)
    ax.set_ylabel("True Positive Rate (Recall)", fontsize=11)
    ax.legend(loc="lower right")
    ax.grid(alpha=0.3)
    plt.tight_layout()
    path = PLOTS_DIR / "roc_curve.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  ROC curve saved        : {path}")
