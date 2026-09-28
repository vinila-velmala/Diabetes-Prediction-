# Diabetes Prediction & Healthcare Analytics

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue)](https://python.org)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3%2B-orange)](https://scikit-learn.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.28%2B-red)](https://streamlit.io)

A machine-learning application that predicts whether a patient is **Diabetic (1)** or **Non-Diabetic (0)**
using a Random Forest classifier trained on a health dataset with 14 clinical and lifestyle features.

> **Medical Disclaimer:** This application is intended for **educational and research purposes only**.
> It is NOT a substitute for professional medical diagnosis, treatment, or advice.
> Always consult a qualified healthcare professional for medical decisions.

---

## Project Overview

This system implements a complete supervised machine-learning pipeline:

1. **Data Loading & Inspection** — Loads `diabetes_binary_health.csv` (3,500 records, 14 columns)
2. **Preprocessing** — Handles missing values, scales numerical features, and encodes categorical variables
3. **Feature Engineering** — Selects 13 relevant health features; analyses correlations
4. **Model Training** — Trains a Random Forest classifier with an 80/20 stratified train/test split
5. **Evaluation** — Computes Accuracy, Precision, Recall, F1-Score, ROC-AUC, confusion matrix, and ROC curve
6. **Interactive UI** — Streamlit web application for real-time diabetes risk prediction
7. **Visualizations** — Healthcare analytics plots showing health variable distributions and correlations

---

## Project Goals

| Goal | Description |
|------|-------------|
| Diabetes Classification | Binary prediction: Diabetic (1) vs Non-Diabetic (0) |
| Data Preprocessing | Robust imputation, scaling, and encoding pipeline |
| Feature Engineering | Correlation analysis and feature importance ranking |
| Model Training | Random Forest with reproducible hyperparameters |
| Model Evaluation | Accuracy, Precision, Recall, F1, ROC-AUC, confusion matrix |
| Interactive Prediction | Streamlit interface for real-time patient risk assessment |
| Healthcare Visualization | Distribution plots, correlation heatmap, feature importance |

---

## Dataset Description

**File:** `data/diabetes_binary_health.csv`

| Property | Value |
|----------|-------|
| Total rows | 3,500 |
| Total columns | 14 |
| Target variable | `Diabetes_binary` (0 = Non-Diabetic, 1 = Diabetic) |
| Class distribution | 1,750 Non-Diabetic / 1,750 Diabetic (perfectly balanced) |
| Missing values | BMI: 151, Glucose: 186, Blood_Pressure: 127 |
| Duplicate rows | 0 |

### Feature Descriptions

| Feature | Type | Description |
|---------|------|-------------|
| Age | Numerical | Patient age in years |
| Sex | Categorical | Female / Male |
| BMI | Numerical | Body Mass Index (kg/m²) |
| Glucose | Numerical | Blood glucose level (mg/dL) |
| Blood_Pressure | Numerical | Systolic blood pressure (mmHg) |
| Cholesterol | Numerical | Total cholesterol (mg/dL) |
| Physical_Activity | Categorical | Sedentary / Moderate / Active |
| Diet_Quality | Categorical | Poor / Average / Good |
| Smoking_Status | Categorical | Non-Smoker / Former Smoker / Current Smoker |
| Family_History | Binary (0/1) | Family history of diabetes |
| Heart_Disease | Binary (0/1) | Presence of heart disease |
| High_BP | Binary (0/1) | High blood pressure flag |
| High_Chol | Binary (0/1) | High cholesterol flag |
| **Diabetes_binary** | **Binary (0/1)** | **Target variable** |

---

## Data Preprocessing

The preprocessing pipeline uses `sklearn.pipeline.Pipeline` and `sklearn.compose.ColumnTransformer`.
**All transformations are fitted exclusively on the training set to prevent data leakage.**

### Missing Value Handling
- **Numerical features** (BMI, Glucose, Blood_Pressure): Median imputation via `SimpleImputer(strategy="median")`
- **Binary features**: Most-frequent value imputation via `SimpleImputer(strategy="most_frequent")`
- **Categorical features**: Most-frequent value imputation before encoding

### Numerical Scaling
- `StandardScaler` applied to Age, BMI, Glucose, Blood_Pressure, Cholesterol
- Binary integer features (Family_History, Heart_Disease, High_BP, High_Chol) are not scaled

### Categorical Encoding
- `OneHotEncoder(handle_unknown="ignore")` applied to Sex, Physical_Activity, Diet_Quality, Smoking_Status
- Produces 10 additional binary columns from 4 categorical features

### Train/Test Split
- **Ratio:** 80% training / 20% testing
- **Stratification:** Enabled (maintains class balance in both sets)
- **Random state:** 42 (fully reproducible)

### Data Leakage Prevention
- ColumnTransformer is fitted using `pipeline.fit(X_train, y_train)`
- Test set is only transformed using the already-fitted pipeline
- No test-set information leaks into training

---

## Machine Learning Model

**Primary Model:** `RandomForestClassifier` (scikit-learn)

### Hyperparameters

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| n_estimators | 200 | Sufficient trees for stable estimates |
| max_depth | 12 | Prevents overfitting while allowing complex patterns |
| min_samples_split | 5 | Reduces overfitting on small nodes |
| min_samples_leaf | 2 | Ensures leaf nodes have sufficient support |
| max_features | sqrt | Standard random feature selection |
| class_weight | balanced | Accounts for potential class imbalance |
| random_state | 42 | Reproducibility |

### Complete Pipeline Structure

```
Input Features (X)
      ↓
ColumnTransformer
  ├── Numerical: SimpleImputer(median) → StandardScaler
  ├── Binary:    SimpleImputer(most_frequent)
  └── Categorical: SimpleImputer(most_frequent) → OneHotEncoder
      ↓
RandomForestClassifier
      ↓
Prediction + Probability
```

The complete pipeline is saved as `models/diabetes_model.pkl` using `joblib`.

---

## Model Evaluation

All metrics were computed on the held-out test set (700 samples, never seen during training).

### Results

| Metric | Score |
|--------|-------|
| **Accuracy** | **0.9957** |
| **Precision** | **0.9971** |
| **Recall** | **0.9943** |
| **F1-Score** | **0.9957** |
| **ROC-AUC** | **0.9999** |

### Classification Report

```
              precision    recall  f1-score   support

Non-Diabetic       0.99      1.00      1.00       350
    Diabetic       1.00      0.99      1.00       350

    accuracy                           1.00       700
   macro avg       1.00      1.00      1.00       700
weighted avg       1.00      1.00      1.00       700
```

### Metric Explanations

- **Accuracy:** Percentage of all predictions that are correct.
- **Precision:** Of all patients predicted as diabetic, what fraction truly are diabetic.
- **Recall (Sensitivity):** Of all truly diabetic patients, what fraction the model correctly identifies.
  In health screening, recall is particularly important because missing a diabetic patient (false negative)
  can have greater consequences than a false positive.
- **F1-Score:** Harmonic mean of precision and recall; balances both concerns.
- **ROC-AUC:** Area under the Receiver Operating Characteristic curve. Values near 1.0 indicate
  excellent discriminative ability between the two classes.

**Note:** The very high metrics reflect patterns in this synthetic/structured educational dataset.
Real-world clinical performance would require validation on independent patient cohorts.

### Baseline Model Benchmarking

To validate the selection of Random Forest, multiple supervised classification architectures were evaluated on the identical 20% stratified test set:

| Model Architecture | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
|--------------------|----------|-----------|--------|----------|---------|
| **Random Forest (Primary)** | **0.9957** | **0.9971** | **0.9943** | **0.9957** | **0.9998** |
| Logistic Regression | 0.9929 | 0.9971 | 0.9886 | 0.9928 | 0.9998 |
| Support Vector Machine (RBF) | 0.9914 | 0.9971 | 0.9857 | 0.9914 | 0.9997 |
| Decision Tree | 0.9886 | 0.9858 | 0.9914 | 0.9886 | 0.9926 |
| K-Nearest Neighbors (k=5) | 0.9814 | 0.9913 | 0.9714 | 0.9812 | 0.9977 |

Random Forest demonstrates superior sensitivity (Recall = 99.43%) and area under the curve (ROC-AUC = 0.9998) while mitigating individual tree variance.

---

## User Interface

The Streamlit application (`app.py`) provides:

### Sidebar — Patient Inputs
- **Age** (slider, 18–90)
- **Sex** (Female / Male)
- **BMI** (slider, 10.0–60.0)
- **Glucose** (slider, 50.0–300.0)
- **Blood Pressure** (slider, 60.0–200.0)
- **Cholesterol** (slider, 100.0–400.0)
- **Physical Activity** (Sedentary / Moderate / Active)
- **Diet Quality** (Poor / Average / Good)
- **Smoking Status** (Non-Smoker / Former Smoker / Current Smoker)
- **Family History, Heart Disease, High BP, High Cholesterol** (binary toggles)

### Main Area Tabs

| Tab | Content |
|-----|---------|
| 🎯 Prediction | Result (Diabetic / Non-Diabetic), probability bar, patient input summary |
| 📊 Dataset Insights | Shape, missing values, class distribution, feature types, statistical summary |
| 🏥 Health Analytics | Distribution plots, correlation heatmap, feature importance |
| 📈 Model Evaluation | Metric cards, confusion matrix, ROC curve, model hyperparameter info |

---

## Project Structure

```
diabetes-prediction/
├── data/
│   └── diabetes_binary_health.csv     ← Dataset (3,500 records, 14 columns)
├── models/
│   ├── diabetes_model.pkl              ← Serialized Random Forest pipeline (preprocessor + model)
│   ├── model_comparison.csv           ← Baseline classifier evaluation benchmarks
│   └── plots/                          ← Generated analytical and diagnostic plots
│       ├── confusion_matrix.png
│       ├── roc_curve.png
│       ├── class_distribution.png
│       ├── dist_age.png
│       ├── dist_bmi.png
│       ├── dist_glucose.png
│       ├── dist_blood_pressure.png
│       ├── dist_cholesterol.png
│       ├── correlation_heatmap.png
│       └── feature_importance.png
├── notebooks/
│   └── diabetes_analysis.ipynb         ← Complete interactive Jupyter analysis and EDA
├── src/
│   ├── __init__.py
│   ├── preprocessing.py                ← Data loading, validation, and ColumnTransformer pipeline
│   ├── train_model.py                  ← Training script with stratified split and persistence
│   ├── evaluate_model.py               ← Comprehensive metric evaluation and diagnostic plots
│   └── visualization.py               ← Healthcare analytics and correlation visualizations
├── app.py                              ← Interactive Streamlit web interface
├── requirements.txt                    ← Minimal pinned project dependencies
├── .gitignore                          ← Cache and local environment exclusions
└── README.md                           ← Project documentation and technical report
```

---

## Installation

### Prerequisites
- Python 3.10 or higher
- pip

### Steps

```bash
# 1. Clone or navigate to the project directory
cd Diabetes-Prediction-

# 2. (Optional) Create a virtual environment
python -m venv venv
venv\Scripts\activate      # Windows
# source venv/bin/activate    # macOS/Linux

# 3. Install dependencies
pip install -r requirements.txt
```

---

## Running the Application

### Step 1: Train the model (first time only)

```bash
python src/train_model.py
```

This will:
- Load the dataset
- Fit the preprocessing + Random Forest pipeline
- Save `models/diabetes_model.pkl`
- Print evaluation metrics

### Step 2: Launch the Streamlit web app

```bash
streamlit run app.py
```

The app will open automatically at `http://localhost:8501`.

### Optional: Regenerate visualizations

```bash
python -c "
import sys, joblib
sys.path.insert(0, '.')
from pathlib import Path
from src.preprocessing import load_dataset, get_feature_names_out
from src.visualization import generate_all_visualizations
df = load_dataset(Path('data/diabetes_binary_health.csv'))
pipeline = joblib.load('models/diabetes_model.pkl')
feature_names = get_feature_names_out(pipeline.named_steps['preprocessor'])
generate_all_visualizations(df, pipeline, feature_names)
"
```

---

## Correlation Analysis

The correlation heatmap reveals these statistical associations (not causal relationships):

- **Glucose** shows the strongest positive association with `Diabetes_binary`
- **BMI** shows a moderate positive association
- **High_BP** and **High_Chol** show meaningful positive correlations
- **Physical activity level** shows a negative association with diabetes status

> **Important:** Statistical correlation does NOT imply medical causation.
> These patterns reflect associations within this dataset and should not be
> interpreted as proof that any single factor causes diabetes.

---

## Limitations

### Dataset Limitations
- The dataset is structured for educational purposes and may not represent the full complexity
  of real clinical patient populations.
- 3,500 samples is relatively small for a medical classification task.
- The dataset is perfectly balanced (50/50), which differs from real-world diabetes prevalence.

### Model Limitations
- The Random Forest model captures patterns in this specific dataset.
  Generalization to new patient populations has not been validated.
- The very high accuracy (99.6%) likely reflects the structured, synthetic nature of the dataset
  rather than real-world predictive power.
- No external validation on independent clinical cohorts has been performed.

### Clinical Limitations
- This model is NOT validated for clinical use.
- A positive prediction does NOT diagnose diabetes.
- Model probabilities are not calibrated probability estimates in the clinical sense.
- Feature ranges and normal values may differ across demographic groups and geographic regions.

### General
- The application requires Python 3.10+ and internet access to load Google Fonts.
- The `models/diabetes_model.pkl` file must be generated by running `src/train_model.py`
  before the Streamlit app can function.

---

## Academic Submission Notes

This project demonstrates:

| Skill | Implementation |
|-------|---------------|
| Supervised Learning | Binary classification with Random Forest |
| Feature Engineering | Preprocessing pipeline (imputation, scaling, OHE) |
| Model Evaluation | Accuracy, Precision, Recall, F1, ROC-AUC, confusion matrix, ROC curve |
| Healthcare Analytics | Distribution analysis, correlation heatmap, feature importance |
| Application Development | Interactive Streamlit prediction interface |
| Data Science Best Practices | Data leakage prevention, reproducible random seeds, model persistence |
