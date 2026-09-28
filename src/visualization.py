"""
src.visualization
~~~~~~~~~~~~~~~~~
Healthcare analytics visualizations for the Diabetes Prediction System.

Generates and saves:
  1. Diabetes class distribution
  2-6. Feature distributions by diabetes status (Age, BMI, Glucose, BP, Cholesterol)
  7. Correlation heatmap
  8. Feature importance (from trained Random Forest)
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

PLOTS_DIR = Path("models") / "plots"
PALETTE = {0: "#4C72B0", 1: "#DD8452"}

sns.set_theme(style="whitegrid", font_scale=1.1)


def _save(fig, name: str) -> Path:
    """Save figure to PLOTS_DIR and close it."""
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    path = PLOTS_DIR / name
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")
    return path


def plot_class_distribution(df: pd.DataFrame) -> Path:
    """Bar chart of diabetes class distribution."""
    counts = df["Diabetes_binary"].value_counts().sort_index()
    labels = ["Non-Diabetic (0)", "Diabetic (1)"]
    colors = [PALETTE[0], PALETTE[1]]
    fig, ax = plt.subplots(figsize=(6, 4))
    bars = ax.bar(labels, counts.values, color=colors, edgecolor="white", linewidth=0.8)
    for bar, count in zip(bars, counts.values):
        pct = count / counts.sum() * 100
        ax.text(
            bar.get_x() + bar.get_width() / 2, bar.get_height() + 10,
            f"{count:,}\n({pct:.1f}%)",
            ha="center", va="bottom", fontsize=10,
        )
    ax.set_title("Diabetes Class Distribution", fontsize=14, fontweight="bold", pad=15)
    ax.set_ylabel("Number of Patients", fontsize=11)
    ax.set_ylim(0, counts.max() * 1.2)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    return _save(fig, "class_distribution.png")


def plot_feature_distribution(df: pd.DataFrame, feature: str, xlabel: str) -> Path:
    """KDE density plot for a numerical feature, grouped by diabetes status."""
    fig, ax = plt.subplots(figsize=(7, 4))
    for label, grp in df.groupby("Diabetes_binary"):
        lbl = "Diabetic" if label == 1 else "Non-Diabetic"
        grp[feature].dropna().plot.kde(ax=ax, label=lbl, color=PALETTE[label], linewidth=2)
    title = f"{xlabel} Distribution by Diabetes Status"
    ax.set_title(title, fontsize=13, fontweight="bold", pad=12)
    ax.set_xlabel(xlabel, fontsize=11)
    ax.set_ylabel("Density", fontsize=11)
    ax.legend(title="Diabetes Status")
    ax.grid(alpha=0.3)
    plt.tight_layout()
    return _save(fig, f"dist_{feature.lower()}.png")


def plot_correlation_heatmap(df: pd.DataFrame) -> Path:
    """Heatmap of Pearson correlations between health features and target."""
    cols = ["Age", "BMI", "Glucose", "Blood_Pressure", "Cholesterol",
            "Family_History", "Heart_Disease", "High_BP", "High_Chol", "Diabetes_binary"]
    corr_df = df[[c for c in cols if c in df.columns]].corr()
    mask = np.triu(np.ones_like(corr_df, dtype=bool))
    fig, ax = plt.subplots(figsize=(9, 7))
    sns.heatmap(
        corr_df, mask=mask, annot=True, fmt=".2f",
        cmap="coolwarm", center=0, vmin=-1, vmax=1,
        linewidths=0.5, ax=ax, annot_kws={"size": 9},
    )
    note = "(Note: correlation shows statistical association, not medical causation)"
    ax.set_title(f"Feature Correlation Heatmap\n{note}", fontsize=12, fontweight="bold", pad=15)
    plt.tight_layout()
    return _save(fig, "correlation_heatmap.png")


def plot_feature_importance(pipeline, feature_names: list) -> Path:
    """Horizontal bar chart of top 15 Random Forest feature importances."""
    rf = pipeline.named_steps["classifier"]
    importances = rf.feature_importances_
    indices = np.argsort(importances)[::-1][:15]
    fig, ax = plt.subplots(figsize=(8, 6))
    colors_bar = plt.cm.RdYlGn(np.linspace(0.2, 0.9, len(indices)))[::-1]
    ax.barh(
        [feature_names[i] for i in indices[::-1]],
        importances[indices[::-1]],
        color=colors_bar, edgecolor="white",
    )
    ax.set_title("Top 15 Feature Importances -- Random Forest",
                 fontsize=13, fontweight="bold", pad=12)
    ax.set_xlabel("Mean Decrease in Impurity (Importance)", fontsize=11)
    ax.grid(axis="x", alpha=0.3)
    plt.tight_layout()
    return _save(fig, "feature_importance.png")


def generate_all_visualizations(df: pd.DataFrame, pipeline, feature_names: list) -> dict:
    """Generate and save all analytics visualizations. Returns dict of paths."""
    print("\nGenerating visualizations...")
    paths = {}
    paths["class_distribution"] = plot_class_distribution(df)
    feature_map = [
        ("Age", "Age (years)"),
        ("BMI", "BMI (kg/m2)"),
        ("Glucose", "Glucose (mg/dL)"),
        ("Blood_Pressure", "Blood Pressure (mmHg)"),
        ("Cholesterol", "Cholesterol (mg/dL)"),
    ]
    for col, label in feature_map:
        if col in df.columns:
            paths[f"dist_{col}"] = plot_feature_distribution(df, col, label)
    paths["correlation_heatmap"] = plot_correlation_heatmap(df)
    paths["feature_importance"] = plot_feature_importance(pipeline, feature_names)
    print(f"  All {len(paths)} visualizations saved to {PLOTS_DIR}.")
    return paths
