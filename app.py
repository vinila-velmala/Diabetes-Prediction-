"""
app.py
~~~~~~
Streamlit application for the Diabetes Prediction & Healthcare Analytics System.

Features:
  - Interactive sidebar for entering patient health data
  - Random Forest prediction with probability/risk percentage
  - Dataset insights (shape, missing values, class distribution)
  - Model evaluation metrics (accuracy, precision, recall, F1, ROC-AUC)
  - Healthcare analytics visualizations
  - Feature importance chart

Medical Disclaimer:
  This application is for educational and research purposes only.
  It is NOT a substitute for professional medical diagnosis or advice.

Usage:
    streamlit run app.py
"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, roc_curve,
)
from sklearn.model_selection import train_test_split

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).parent))

from src.preprocessing import (
    load_dataset, inspect_dataset, prepare_features,
    get_feature_names_out, FEATURE_COLUMNS, RANDOM_STATE, TEST_SIZE,
)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
MODEL_PATH = Path("models") / "diabetes_model.pkl"
DATA_PATH = Path("data") / "diabetes_binary_health.csv"
PLOTS_DIR = Path("models") / "plots"

# ---------------------------------------------------------------------------
# Page configuration
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Diabetes Prediction & Healthcare Analytics",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Custom CSS
# ---------------------------------------------------------------------------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

/* Main header gradient */
.main-header {
    background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
    padding: 2.5rem 2rem;
    border-radius: 16px;
    margin-bottom: 1.5rem;
    text-align: center;
}
.main-header h1 {
    color: #ffffff;
    font-size: 2.4rem;
    font-weight: 700;
    margin: 0 0 0.5rem 0;
    letter-spacing: -0.5px;
}
.main-header p {
    color: #a8c7fa;
    font-size: 1.05rem;
    margin: 0;
}

/* Metric cards */
.metric-card {
    background: linear-gradient(135deg, #f8faff 0%, #eef2ff 100%);
    border: 1px solid #c7d7ff;
    border-radius: 12px;
    padding: 1.2rem 1rem;
    text-align: center;
}
.metric-card .label {
    font-size: 0.75rem;
    font-weight: 600;
    color: #6b7280;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    margin-bottom: 0.4rem;
}
.metric-card .value {
    font-size: 1.7rem;
    font-weight: 700;
    color: #1e3a8a;
}

/* Prediction result boxes */
.pred-diabetic {
    background: linear-gradient(135deg, #fee2e2 0%, #fecaca 100%);
    border: 2px solid #f87171;
    border-radius: 16px;
    padding: 1.8rem;
    text-align: center;
}
.pred-diabetic h2 {
    color: #991b1b;
    font-size: 1.9rem;
    margin: 0.5rem 0;
}

.pred-healthy {
    background: linear-gradient(135deg, #d1fae5 0%, #a7f3d0 100%);
    border: 2px solid #34d399;
    border-radius: 16px;
    padding: 1.8rem;
    text-align: center;
}
.pred-healthy h2 {
    color: #065f46;
    font-size: 1.9rem;
    margin: 0.5rem 0;
}

/* Disclaimer box */
.disclaimer {
    background: #fffbeb;
    border-left: 4px solid #f59e0b;
    border-radius: 8px;
    padding: 1rem 1.2rem;
    margin: 1rem 0;
    font-size: 0.85rem;
    color: #78350f;
}

/* Section header */
.section-header {
    font-size: 1.3rem;
    font-weight: 700;
    color: #1e3a8a;
    border-bottom: 2px solid #c7d7ff;
    padding-bottom: 0.5rem;
    margin-bottom: 1rem;
}

/* Sidebar styling */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #1a1a2e 0%, #16213e 100%);
}
[data-testid="stSidebar"] .st-emotion-cache-1kyxreq,
[data-testid="stSidebar"] label,
[data-testid="stSidebar"] .stSelectbox label,
[data-testid="stSidebar"] .stSlider label,
[data-testid="stSidebar"] .stNumberInput label,
[data-testid="stSidebar"] .stRadio label {
    color: #e2e8f0 !important;
}
[data-testid="stSidebar"] h2,
[data-testid="stSidebar"] h3 {
    color: #93c5fd !important;
}

.prob-bar-container {
    background: #e5e7eb;
    border-radius: 999px;
    height: 20px;
    width: 100%;
    overflow: hidden;
    margin-top: 8px;
}
.prob-bar-fill-high {
    background: linear-gradient(90deg, #f97316, #ef4444);
    height: 100%;
    border-radius: 999px;
    transition: width 0.5s ease;
}
.prob-bar-fill-low {
    background: linear-gradient(90deg, #34d399, #10b981);
    height: 100%;
    border-radius: 999px;
    transition: width 0.5s ease;
}
</style>
""", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Caching: load dataset and model
# ---------------------------------------------------------------------------
@st.cache_data
def load_data():
    """Load the diabetes dataset (cached)."""
    df = load_dataset(DATA_PATH)
    summary = inspect_dataset(df)
    return df, summary


@st.cache_resource
def load_model():
    """Load the trained pipeline from disk (cached)."""
    if not MODEL_PATH.exists():
        return None
    return joblib.load(MODEL_PATH)


@st.cache_data
def get_eval_metrics(_pipeline):
    """Compute and cache model evaluation metrics on the test set."""
    df = load_dataset(DATA_PATH)
    X, y = prepare_features(df)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y,
    )
    y_pred = _pipeline.predict(X_test)
    y_prob = _pipeline.predict_proba(X_test)[:, 1]
    metrics = {
        "Accuracy": round(accuracy_score(y_test, y_pred), 4),
        "Precision": round(precision_score(y_test, y_pred, zero_division=0), 4),
        "Recall": round(recall_score(y_test, y_pred, zero_division=0), 4),
        "F1-Score": round(f1_score(y_test, y_pred, zero_division=0), 4),
        "ROC-AUC": round(roc_auc_score(y_test, y_prob), 4),
    }
    cm = confusion_matrix(y_test, y_pred)
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    return metrics, cm, fpr, tpr, y_test, y_pred


@st.cache_data
def get_feature_names(_pipeline):
    return get_feature_names_out(_pipeline.named_steps["preprocessor"])


# ---------------------------------------------------------------------------
# Helper: plot confusion matrix (inline in Streamlit)
# ---------------------------------------------------------------------------
def make_cm_figure(cm):
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=["Non-Diabetic", "Diabetic"],
        yticklabels=["Non-Diabetic", "Diabetic"],
        ax=ax, linewidths=0.5,
    )
    ax.set_title("Confusion Matrix", fontsize=13, fontweight="bold", pad=12)
    ax.set_xlabel("Predicted Label", fontsize=10)
    ax.set_ylabel("True Label", fontsize=10)
    plt.tight_layout()
    return fig


def make_roc_figure(fpr, tpr, auc_val):
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.plot(fpr, tpr, color="#4C72B0", lw=2, label=f"ROC Curve (AUC = {auc_val:.4f})")
    ax.plot([0, 1], [0, 1], color="gray", linestyle="--", lw=1, label="Random Classifier")
    ax.set_title("ROC Curve — Random Forest", fontsize=13, fontweight="bold", pad=12)
    ax.set_xlabel("False Positive Rate", fontsize=10)
    ax.set_ylabel("True Positive Rate (Recall)", fontsize=10)
    ax.legend(loc="lower right", fontsize=9)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    return fig


def make_feature_importance_figure(pipeline, feature_names):
    rf = pipeline.named_steps["classifier"]
    importances = rf.feature_importances_
    indices = np.argsort(importances)[::-1][:15]
    fig, ax = plt.subplots(figsize=(8, 5))
    colors_bar = plt.cm.RdYlGn(np.linspace(0.2, 0.9, len(indices)))[::-1]
    ax.barh(
        [feature_names[i] for i in indices[::-1]],
        importances[indices[::-1]],
        color=colors_bar, edgecolor="white",
    )
    ax.set_title("Top 15 Feature Importances — Random Forest", fontsize=13, fontweight="bold", pad=12)
    ax.set_xlabel("Mean Decrease in Impurity", fontsize=10)
    ax.grid(axis="x", alpha=0.3)
    plt.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Sidebar: Patient Input
# ---------------------------------------------------------------------------
def render_sidebar():
    """Render the patient input form in the sidebar."""
    with st.sidebar:
        st.markdown("## 🩺 Patient Information")
        st.markdown("Enter the patient's health data below.")
        st.markdown("---")

        st.markdown("### 📋 Demographics")
        age = st.slider("Age (years)", min_value=18, max_value=90, value=45, step=1)
        sex = st.selectbox("Sex", ["Female", "Male"])

        st.markdown("### 💉 Clinical Measurements")
        bmi = st.slider("BMI (kg/m²)", min_value=10.0, max_value=60.0, value=27.0, step=0.1,
                        help="Body Mass Index. Normal: 18.5–24.9")
        glucose = st.slider("Glucose (mg/dL)", min_value=50.0, max_value=300.0, value=100.0, step=0.5,
                            help="Fasting blood glucose level. Normal: 70–99 mg/dL")
        blood_pressure = st.slider("Blood Pressure (mmHg)", min_value=60.0, max_value=200.0, value=120.0, step=1.0,
                                   help="Systolic blood pressure")
        cholesterol = st.slider("Cholesterol (mg/dL)", min_value=100.0, max_value=400.0, value=200.0, step=1.0,
                                help="Total cholesterol. Desirable: < 200 mg/dL")

        st.markdown("### 🏃 Lifestyle")
        physical_activity = st.selectbox("Physical Activity Level",
                                         ["Sedentary", "Moderate", "Active"])
        diet_quality = st.selectbox("Diet Quality", ["Poor", "Average", "Good"])
        smoking_status = st.selectbox("Smoking Status",
                                      ["Non-Smoker", "Former Smoker", "Current Smoker"])

        st.markdown("### 🏥 Medical History")
        family_history = st.radio("Family History of Diabetes", [0, 1],
                                  format_func=lambda x: "Yes" if x == 1 else "No")
        heart_disease = st.radio("Heart Disease", [0, 1],
                                 format_func=lambda x: "Yes" if x == 1 else "No")
        high_bp = st.radio("High Blood Pressure", [0, 1],
                           format_func=lambda x: "Yes" if x == 1 else "No")
        high_chol = st.radio("High Cholesterol", [0, 1],
                             format_func=lambda x: "Yes" if x == 1 else "No")

        st.markdown("---")
        predict_btn = st.button("🔍 Predict Diabetes Risk", use_container_width=True, type="primary")

    return {
        "Age": age, "BMI": bmi, "Glucose": glucose,
        "Blood_Pressure": blood_pressure, "Cholesterol": cholesterol,
        "Family_History": family_history, "Heart_Disease": heart_disease,
        "High_BP": high_bp, "High_Chol": high_chol,
        "Sex": sex, "Physical_Activity": physical_activity,
        "Diet_Quality": diet_quality, "Smoking_Status": smoking_status,
    }, predict_btn


# ---------------------------------------------------------------------------
# Main App Layout
# ---------------------------------------------------------------------------
def main():
    # Header
    st.markdown("""
    <div class="main-header">
        <h1>🩺 Diabetes Prediction & Healthcare Analytics</h1>
        <p>AI-powered diabetes risk assessment using Random Forest classification on real health data.</p>
    </div>
    """, unsafe_allow_html=True)

    # Disclaimer
    st.markdown("""
    <div class="disclaimer">
        ⚠️ <strong>Medical Disclaimer:</strong> This application is for <strong>educational and research purposes only</strong>.
        It is NOT a substitute for professional medical diagnosis, treatment, or advice.
        Always consult a qualified healthcare professional for medical decisions.
    </div>
    """, unsafe_allow_html=True)

    # Load data and model
    df, summary = load_data()
    pipeline = load_model()

    if pipeline is None:
        st.error(
            "⚠️ Trained model not found. Please run `python src/train_model.py` first."
        )
        return

    feature_names = get_feature_names(pipeline)

    # Sidebar inputs
    patient_inputs, predict_btn = render_sidebar()

    # -----------------------------------------------------------------------
    # Tabs
    # -----------------------------------------------------------------------
    tab1, tab2, tab3, tab4 = st.tabs([
        "🎯 Prediction", "📊 Dataset Insights", "🏥 Health Analytics", "📈 Model Evaluation"
    ])

    # ====================================================================
    # TAB 1: PREDICTION
    # ====================================================================
    with tab1:
        st.markdown('<div class="section-header">🎯 Diabetes Risk Prediction</div>', unsafe_allow_html=True)

        if predict_btn:
            try:
                # Build input dataframe in the exact column order expected
                input_df = pd.DataFrame([{
                    "Age": float(patient_inputs["Age"]),
                    "BMI": float(patient_inputs["BMI"]),
                    "Glucose": float(patient_inputs["Glucose"]),
                    "Blood_Pressure": float(patient_inputs["Blood_Pressure"]),
                    "Cholesterol": float(patient_inputs["Cholesterol"]),
                    "Family_History": int(patient_inputs["Family_History"]),
                    "Heart_Disease": int(patient_inputs["Heart_Disease"]),
                    "High_BP": int(patient_inputs["High_BP"]),
                    "High_Chol": int(patient_inputs["High_Chol"]),
                    "Sex": patient_inputs["Sex"],
                    "Physical_Activity": patient_inputs["Physical_Activity"],
                    "Diet_Quality": patient_inputs["Diet_Quality"],
                    "Smoking_Status": patient_inputs["Smoking_Status"],
                }])

                prediction = pipeline.predict(input_df)[0]
                probability = pipeline.predict_proba(input_df)[0]
                diabetes_prob = float(probability[1]) * 100
                non_diabetes_prob = float(probability[0]) * 100

                # Result display
                col1, col2, col3 = st.columns([1, 2, 1])
                with col2:
                    if prediction == 1:
                        st.markdown(f"""
                        <div class="pred-diabetic">
                            <div style="font-size:3rem">🔴</div>
                            <h2>⚠️ Diabetic Risk Detected</h2>
                            <p style="color:#7f1d1d; font-size:1rem; margin:0.5rem 0 0 0">
                                The model predicts a <strong>high probability</strong> of diabetes
                                based on the provided health indicators.
                            </p>
                        </div>
                        """, unsafe_allow_html=True)
                    else:
                        st.markdown(f"""
                        <div class="pred-healthy">
                            <div style="font-size:3rem">🟢</div>
                            <h2>✅ Non-Diabetic</h2>
                            <p style="color:#064e3b; font-size:1rem; margin:0.5rem 0 0 0">
                                The model predicts a <strong>low probability</strong> of diabetes
                                based on the provided health indicators.
                            </p>
                        </div>
                        """, unsafe_allow_html=True)

                st.markdown("---")

                # Probability display
                st.markdown("### 📊 Estimated Risk Probability")
                col_a, col_b = st.columns(2)
                with col_a:
                    bar_class = "prob-bar-fill-high" if diabetes_prob > 50 else "prob-bar-fill-low"
                    st.markdown(f"""
                    <div style="text-align:center; padding:1rem; background:#f8faff; border-radius:12px; border:1px solid #c7d7ff;">
                        <div style="font-size:0.8rem; font-weight:600; color:#6b7280; text-transform:uppercase; letter-spacing:0.05em;">Diabetic Probability</div>
                        <div style="font-size:2.8rem; font-weight:700; color:{'#dc2626' if diabetes_prob > 50 else '#059669'}; margin:0.3rem 0">{diabetes_prob:.1f}%</div>
                        <div class="prob-bar-container">
                            <div class="{bar_class}" style="width:{diabetes_prob:.1f}%"></div>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                with col_b:
                    st.markdown(f"""
                    <div style="text-align:center; padding:1rem; background:#f8faff; border-radius:12px; border:1px solid #c7d7ff;">
                        <div style="font-size:0.8rem; font-weight:600; color:#6b7280; text-transform:uppercase; letter-spacing:0.05em;">Non-Diabetic Probability</div>
                        <div style="font-size:2.8rem; font-weight:700; color:#059669; margin:0.3rem 0">{non_diabetes_prob:.1f}%</div>
                        <div class="prob-bar-container">
                            <div class="prob-bar-fill-low" style="width:{non_diabetes_prob:.1f}%"></div>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

                st.markdown("""
                <div class="disclaimer" style="margin-top:1.5rem">
                    ℹ️ The probability above represents the model's estimated likelihood based on patterns
                    learned from the training dataset. It is <strong>not a clinical diagnosis</strong>.
                    This model was trained for educational purposes only.
                </div>
                """, unsafe_allow_html=True)

                # Patient summary
                st.markdown("### 👤 Patient Input Summary")
                input_summary = pd.DataFrame([{
                    "Age": patient_inputs["Age"],
                    "BMI": patient_inputs["BMI"],
                    "Glucose (mg/dL)": patient_inputs["Glucose"],
                    "Blood Pressure (mmHg)": patient_inputs["Blood_Pressure"],
                    "Cholesterol (mg/dL)": patient_inputs["Cholesterol"],
                    "Sex": patient_inputs["Sex"],
                    "Physical Activity": patient_inputs["Physical_Activity"],
                    "Diet Quality": patient_inputs["Diet_Quality"],
                    "Smoking Status": patient_inputs["Smoking_Status"],
                    "Family History": "Yes" if patient_inputs["Family_History"] else "No",
                    "Heart Disease": "Yes" if patient_inputs["Heart_Disease"] else "No",
                    "High BP": "Yes" if patient_inputs["High_BP"] else "No",
                    "High Cholesterol": "Yes" if patient_inputs["High_Chol"] else "No",
                }]).T.rename(columns={0: "Value"})
                st.dataframe(input_summary, use_container_width=True)

            except Exception as e:
                st.error(f"Prediction error: {e}")
        else:
            st.info("👈 **Enter patient information in the sidebar**, then click **Predict Diabetes Risk** to get results.")
            st.markdown("#### How This Works")
            col1, col2, col3 = st.columns(3)
            with col1:
                st.markdown("""
                **1. Enter Patient Data**
                Fill in age, BMI, glucose, blood pressure, cholesterol, lifestyle factors, and medical history in the sidebar.
                """)
            with col2:
                st.markdown("""
                **2. AI Prediction**
                The Random Forest model processes the inputs through a preprocessing pipeline and outputs a diabetes probability.
                """)
            with col3:
                st.markdown("""
                **3. Review Results**
                See the predicted risk classification (Diabetic / Non-Diabetic) with an estimated probability percentage.
                """)

    # ====================================================================
    # TAB 2: DATASET INSIGHTS
    # ====================================================================
    with tab2:
        st.markdown('<div class="section-header">📊 Dataset Insights</div>', unsafe_allow_html=True)

        col1, col2, col3, col4 = st.columns(4)
        cols = [col1, col2, col3, col4]
        stat_items = [
            ("Total Records", f"{summary['shape'][0]:,}"),
            ("Features", f"{summary['shape'][1] - 1}"),
            ("Missing Values", f"{sum(v for v in summary['missing_values'].values() if v > 0):,}"),
            ("Duplicate Rows", f"{summary['duplicate_rows']}"),
        ]
        for col, (label, value) in zip(cols, stat_items):
            with col:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="label">{label}</div>
                    <div class="value">{value}</div>
                </div>
                """, unsafe_allow_html=True)

        st.markdown("---")
        col_left, col_right = st.columns(2)

        with col_left:
            st.markdown("#### 🎯 Target Variable: Diabetes_binary")
            class_dist = summary.get("class_distribution", {})
            class_pct = summary.get("class_balance_pct", {})
            dist_df = pd.DataFrame({
                "Class": ["Non-Diabetic (0)", "Diabetic (1)"],
                "Count": [class_dist.get(0, 0), class_dist.get(1, 0)],
                "Percentage (%)": [class_pct.get(0, 0), class_pct.get(1, 0)],
            })
            st.dataframe(dist_df, use_container_width=True, hide_index=True)

            st.markdown("#### 🔍 Missing Values Summary")
            mv_data = {k: v for k, v in summary["missing_values"].items() if v > 0}
            if mv_data:
                mv_df = pd.DataFrame({
                    "Column": list(mv_data.keys()),
                    "Missing Count": list(mv_data.values()),
                    "Missing (%)": [summary["missing_pct"][k] for k in mv_data.keys()],
                })
                st.dataframe(mv_df, use_container_width=True, hide_index=True)
                st.caption("Missing values are imputed: median for numerical, mode for categorical.")
            else:
                st.success("No missing values found in the dataset.")

        with col_right:
            st.markdown("#### 📋 Dataset Columns & Types")
            col_info = []
            for col_name in summary["columns"]:
                if col_name in summary["numerical_cols"]:
                    dtype_desc = "Numerical"
                elif col_name in summary["categorical_cols"]:
                    dtype_desc = "Categorical"
                else:
                    dtype_desc = "Binary Integer"
                col_info.append({"Column": col_name, "Type": dtype_desc,
                                  "Role": "Target" if col_name == "Diabetes_binary" else "Feature"})
            col_df = pd.DataFrame(col_info)
            st.dataframe(col_df, use_container_width=True, hide_index=True)

        st.markdown("---")
        st.markdown("#### 📈 Dataset Statistical Summary")
        st.dataframe(df.describe().round(2), use_container_width=True)

    # ====================================================================
    # TAB 3: HEALTH ANALYTICS
    # ====================================================================
    with tab3:
        st.markdown('<div class="section-header">🏥 Healthcare Analytics Visualizations</div>', unsafe_allow_html=True)
        st.markdown(
            "> **Note:** Visualizations show statistical associations between health variables and diabetes status. "
            "Correlation does **not** imply medical causation."
        )

        # Class distribution
        plot_path = PLOTS_DIR / "class_distribution.png"
        if plot_path.exists():
            col1, col2, col3 = st.columns([1, 2, 1])
            with col2:
                st.image(str(plot_path), caption="Diabetes Class Distribution", use_container_width=True)

        st.markdown("---")
        st.markdown("#### Feature Distributions by Diabetes Status")

        feature_plots = [
            ("dist_age.png", "Age Distribution"),
            ("dist_bmi.png", "BMI Distribution"),
            ("dist_glucose.png", "Glucose Distribution"),
            ("dist_blood_pressure.png", "Blood Pressure Distribution"),
            ("dist_cholesterol.png", "Cholesterol Distribution"),
        ]

        cols_row = st.columns(2)
        for idx, (fname, caption) in enumerate(feature_plots):
            fpath = PLOTS_DIR / fname
            if fpath.exists():
                with cols_row[idx % 2]:
                    st.image(str(fpath), caption=caption, use_container_width=True)

        st.markdown("---")
        st.markdown("#### Correlation Heatmap")
        corr_path = PLOTS_DIR / "correlation_heatmap.png"
        if corr_path.exists():
            col1, col2, col3 = st.columns([0.5, 3, 0.5])
            with col2:
                st.image(str(corr_path), caption="Feature Correlation Heatmap", use_container_width=True)
        st.caption(
            "🔍 Correlation values range from -1 (strong negative) to +1 (strong positive). "
            "Values near 0 indicate weak association. These are statistical patterns, not causal relationships."
        )

    # ====================================================================
    # TAB 4: MODEL EVALUATION
    # ====================================================================
    with tab4:
        st.markdown('<div class="section-header">📈 Model Evaluation</div>', unsafe_allow_html=True)

        metrics, cm, fpr, tpr, y_test, y_pred = get_eval_metrics(pipeline)
        auc_val = metrics["ROC-AUC"]

        # Metric cards
        metric_cols = st.columns(5)
        metric_items = [
            ("Accuracy", metrics["Accuracy"], "Overall correct predictions"),
            ("Precision", metrics["Precision"], "Of predicted diabetic, how many were correct"),
            ("Recall", metrics["Recall"], "Of actual diabetic cases, how many were detected"),
            ("F1-Score", metrics["F1-Score"], "Harmonic mean of precision and recall"),
            ("ROC-AUC", metrics["ROC-AUC"], "Area under the ROC curve"),
        ]
        for col, (name, val, tip) in zip(metric_cols, metric_items):
            with col:
                st.markdown(f"""
                <div class="metric-card" title="{tip}">
                    <div class="label">{name}</div>
                    <div class="value">{val:.4f}</div>
                </div>
                """, unsafe_allow_html=True)

        st.markdown("---")

        col1, col2 = st.columns(2)
        with col1:
            st.markdown("#### Confusion Matrix")
            st.pyplot(make_cm_figure(cm), use_container_width=True)
            tn, fp, fn, tp = cm.ravel()
            st.caption(
                f"TN={tn} (correctly identified non-diabetic), "
                f"TP={tp} (correctly identified diabetic), "
                f"FP={fp} (false alarms), "
                f"FN={fn} (missed diabetic cases)"
            )

        with col2:
            st.markdown("#### ROC Curve")
            st.pyplot(make_roc_figure(fpr, tpr, auc_val), use_container_width=True)
            st.caption(
                "The ROC curve shows the trade-off between True Positive Rate (sensitivity) "
                "and False Positive Rate at various classification thresholds. "
                "AUC closer to 1.0 indicates better discriminative performance."
            )

        st.markdown("---")
        st.markdown("#### Feature Importance")
        st.pyplot(make_feature_importance_figure(pipeline, feature_names), use_container_width=True)
        st.caption(
            "Feature importances are based on Mean Decrease in Impurity (MDI) from the Random Forest. "
            "Higher importance indicates greater influence on the model's predictions. "
            "This does not imply medical causation."
        )

        st.markdown("---")
        st.markdown("#### Model Information")
        col_a, col_b = st.columns(2)
        with col_a:
            st.markdown("""
            **Algorithm:** Random Forest Classifier  
            **Library:** scikit-learn  
            **n_estimators:** 200  
            **max_depth:** 12  
            **class_weight:** balanced  
            **random_state:** 42  
            """)
        with col_b:
            st.markdown("""
            **Dataset:** diabetes_binary_health.csv  
            **Total samples:** 3,500  
            **Train / Test split:** 80% / 20%  
            **Stratified split:** Yes  
            **Preprocessing:** Median imputation + StandardScaler + OneHotEncoder  
            """)

        st.info(
            "ℹ️ **Recall** is particularly relevant in a health screening context because failing "
            "to detect a diabetic patient (false negative) can be more harmful than a false alarm. "
            "However, this is a research/educational model, not a clinical screening tool."
        )


if __name__ == "__main__":
    main()
