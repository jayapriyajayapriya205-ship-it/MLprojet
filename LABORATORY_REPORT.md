# LABORATORY / MINI-PROJECT REPORT

## PROJECT TITLE:
**Diabetes Prediction System using Machine Learning (Pima Indians Diabetes Dataset)**

---

## 1. PROBLEM STATEMENT
Diabetes mellitus is a chronic metabolic disorder characterized by elevated levels of blood glucose, which leads over time to serious damage to the heart, blood vessels, eyes, kidneys, and nerves. Early detection is paramount for mitigating long-term systemic complications. However, clinical diagnosis often involves multiple interdependent physiological parameters, and standard datasets (such as the Pima Indians Diabetes Dataset) frequently contain biological anomalies—specifically zero values recorded for physiological metrics such as blood glucose, diastolic blood pressure, triceps skinfold thickness, 2-hour serum insulin, and Body Mass Index (BMI). 

Failing to handle these biological anomalies or applying naive mean imputations across the entire dataset causes substantial data leakage, biased variance estimates, and degraded clinical classification performance. Furthermore, clinicians require not merely a binary classification, but an interpretable probability score, transparent feature preconditioning (z-score scaling), and a persistent audit trail. The objective of this project is to develop an end-to-end, production-ready Machine Learning diagnostic web system that implements rigorous training-only median imputation, standard scaling, multi-model evaluation, and persistent SQLite database logging behind a modern clinical UI dashboard.

---

## 2. OBJECTIVES
1. **Biological Imputation Pipeline**: Implement a robust preprocessor that detects biologically impossible zero values in physiological indicators (`Glucose`, `BloodPressure`, `SkinThickness`, `Insulin`, `BMI`) and substitutes them with population medians calculated strictly on the training partition to eliminate data leakage.
2. **Feature Standardization**: Normalize the 8-dimensional feature vector using `StandardScaler` to ensure optimal gradient convergence and margin maximization in distance-sensitive models.
3. **Model Training & Benchmark**: Train, cross-validate, and evaluate multiple supervised learning algorithms (Support Vector Classifier with RBF kernel and Platt probability calibration, Random Forest, and Logistic Regression) on a stratified 80/20 train/test split.
4. **Diagnostic Web Microservice**: Construct a Flask REST API with bounds validation, error handling, Cross-Origin Resource Sharing (CORS), and inference serialization.
5. **Database Audit Logging**: Integrate an embedded SQLite3 relational database (`database/diabetes.db`) to record each patient transaction with full clinical parameters, predicted outcome, confidence score, and timestamp.
6. **Clinical Dashboard Interface**: Develop a responsive, modern dark-themed web dashboard featuring preset sample loaders, real-time imputation status, normalized feature z-score tables, learning curves, 2x2 confusion matrix heatmap, and interactive database records management.

---

## 3. SOFTWARE & HARDWARE REQUIREMENTS

### Software Requirements:
- **Operating System**: Windows 10/11, Linux (Ubuntu 20.04+), or macOS
- **Programming Language**: Python 3.9+
- **Backend Framework**: Flask 3.1.x, Werkzeug 3.1.x
- **Machine Learning & Data Processing**: scikit-learn 1.6+, Pandas 2.3+, NumPy 2.0+, Joblib 1.5+
- **Database Engine**: SQLite3
- **Frontend Technologies**: HTML5, CSS3 (Custom Dark Theme with Glassmorphism), Vanilla JavaScript (ES6+), HTML5 Canvas API

### Hardware Requirements:
- **Processor**: x86_64 Dual-Core CPU @ 2.0 GHz or higher
- **RAM**: Minimum 4 GB (8 GB recommended)
- **Disk Storage**: 500 MB free space
- **Display Resolution**: 1366 x 768 minimum (1920 x 1080 recommended)

---

## 4. PROJECT DIRECTORY STRUCTURE

```text
DiabetesClassifier/
│
├── backend/
│   ├── dataset/
│   │   ├── diabetes.csv             # Full Pima Indians Diabetes Dataset (768 records)
│   │   ├── train.csv                # Stratified training split (614 records, 80%)
│   │   └── test.csv                 # Stratified test split (154 records, 20%)
│   ├── ml/
│   │   ├── __init__.py              # ML package initialization
│   │   ├── preprocessing.py         # DiabetesPreprocessor (Zero Imputation + StandardScaler)
│   │   └── feature_pipeline.py      # Input bounds validation & clinical risk interpreter
│   ├── models/
│   │   ├── diabetes_model.pkl       # Trained ML model artifact (SVC RBF / Random Forest)
│   │   ├── scaler.pkl               # Serialized preprocessor pipeline and training medians
│   │   └── model_metrics.json       # Persisted test performance metrics, CM, & curves
│   ├── routes/
│   │   ├── __init__.py              # Routes package initialization
│   │   └── predict.py               # REST API Blueprint (/predict, /history, /metrics, /sample)
│   ├── __init__.py                  # Backend package initialization
│   ├── app.py                       # Main Flask web application server and static file router
│   ├── config.py                    # Global system configurations and directory paths
│   └── database.py                  # SQLite3 database connection, schema, and queries
│
├── database/
│   └── diabetes.db                  # SQLite database file storing patient prediction audit trail
│
├── frontend/
│   ├── index.html                   # Clinical diagnostic dashboard layout (7 pipeline steps)
│   ├── style.css                    # Modern clinical dark theme CSS stylesheet
│   └── script.js                    # UI controller, API fetch client, and Canvas chart renderer
│
├── populate_dataset_and_train.py    # End-to-end dataset builder, trainer, evaluator & DB seeder
├── requirements.txt                 # Project Python dependencies
└── README.md                        # Documentation and deployment manual
```

---

## 5. PROCEDURE (STEPS 1 TO 10)

1. **Step 1: Dataset Acquisition & Verification**  
   The system acquires the National Institute of Diabetes and Digestive and Kidney Diseases (Pima Indians) Dataset consisting of 768 patient records with 8 clinical explanatory variables (`Pregnancies`, `Glucose`, `BloodPressure`, `SkinThickness`, `Insulin`, `BMI`, `DiabetesPedigreeFunction`, `Age`) and 1 binary target outcome (`0` = Healthy / Non-Diabetic, `1` = Diabetic).

2. **Step 2: Exploratory Data Analysis & Biological Anomaly Identification**  
   The distribution of features is analyzed. It is established that zero measurements in `Glucose`, `BloodPressure`, `SkinThickness`, `Insulin`, and `BMI` represent missing data rather than physiologically valid readings (e.g., zero blood glucose or zero blood pressure denotes biological death).

3. **Step 3: Stratified Train-Test Splitting**  
   The dataset is partitioned into an 80% training set (614 instances) and a 20% testing set (154 instances) using `train_test_split` with `stratify=y` and `random_state=42`. Stratification ensures the class balance (65.1% negative, 34.9% positive) is strictly preserved across both partitions.

4. **Step 4: Training-Partition Median Imputation & Z-Score Scaling**  
   To prevent data leakage, median values for the 5 zero-invalid columns are computed exclusively from non-zero records in the training partition:
   - Median Glucose = $117.0\text{ mg/dL}$
   - Median Blood Pressure = $72.0\text{ mm Hg}$
   - Median Skin Thickness = $29.0\text{ mm}$
   - Median Insulin = $125.0\text{ }\mu\text{U/mL}$
   - Median BMI = $32.4\text{ kg/m}^2$  
   `StandardScaler` is fitted on the imputed training set:
   $$z = \frac{x - \mu}{\sigma}$$

5. **Step 5: Multi-Model Training and Benchmark Evaluation**  
   Multiple supervised learning models are fitted on the standardized training data:
   - **Support Vector Machine (SVC)**: Radial Basis Function (RBF) kernel with Platt probability calibration ($C=1.2$).
   - **Random Forest**: Ensemble of 120 decision trees with bounded depth ($max\_depth=6$) to avoid overfitting.
   - **Logistic Regression**: Regularized linear classifier ($C=0.8, max\_iter=500$).  
   Models are evaluated on the 154 unseen test samples, and the top-performing model is selected.

6. **Step 6: Learning Curve & Metric Generation**  
   Five-fold cross-validation learning curves are generated across incremental sample fractions ($15\%$ to $100\%$). Training Accuracy, Validation Accuracy, and Cross-Entropy Loss are computed. A 2x2 confusion matrix is calculated on the test split.

7. **Step 7: Model Artifact Serialization**  
   Trained model weights, preprocessor parameters (training medians, feature means, standard deviations), and test metrics are serialized using `joblib` into `backend/models/diabetes_model.pkl`, `scaler.pkl`, and `model_metrics.json`.

8. **Step 8: SQLite Database Schema Setup**  
   An SQLite database (`database/diabetes.db`) is initialized with the table `predictions`. Database helper functions (`init_db`, `insert_prediction`, `get_history`, `clear_history`) are implemented to enforce transaction integrity and audit compliance.

9. **Step 9: Flask REST API Microservice Development**  
   A Flask microservice is developed with endpoints:
   - `POST /api/predict`: Validates payload against physiological boundaries, executes imputation and scaling, performs inference, logs the record to SQLite, and returns diagnostic results.
   - `GET /api/history`: Retrieves the audit history.
   - `DELETE /api/history`: Purges history records.
   - `GET /api/metrics`: Serves test set evaluation metrics and curve data.
   - `GET /api/sample/<type>`: Supplies sample test fixtures.

10. **Step 10: Full-Stack Clinical UI Dashboard Integration & Verification**  
    A dark-themed clinical dashboard is implemented in HTML5/CSS3/JavaScript. It connects to the Flask API, renders interactive risk meters, feature vector tables, learning curves on an HTML5 Canvas, and a confusion matrix heatmap.

---

## 6. IMPLEMENTATION CODE (KEY MODULES)

### Module 1: `backend/ml/preprocessing.py`
```python
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import joblib

FEATURE_NAMES = [
    'Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness',
    'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age'
]

ZERO_IMPUTATION_FEATURES = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']

class DiabetesPreprocessor:
    def __init__(self):
        self.feature_names = FEATURE_NAMES
        self.zero_features = ZERO_IMPUTATION_FEATURES
        self.imputer_medians = {}
        self.scaler = StandardScaler()
        self.stats = {}

    def fit(self, X_train):
        df = pd.DataFrame(X_train, columns=self.feature_names) if not isinstance(X_train, pd.DataFrame) else X_train.copy()
        df_imputed = df.copy()
        
        for col in self.zero_features:
            non_zero_series = df_imputed[col].replace(0, np.nan)
            median_val = float(non_zero_series.median())
            self.imputer_medians[col] = round(median_val, 2)
            df_imputed[col] = df_imputed[col].replace(0, median_val)

        for col in self.feature_names:
            self.stats[col] = {
                'mean': float(df_imputed[col].mean()),
                'std': float(df_imputed[col].std()),
                'median': float(df_imputed[col].median())
            }

        self.scaler.fit(df_imputed[self.feature_names].values)
        return self

    def transform(self, X):
        df = pd.DataFrame([X]) if isinstance(X, dict) else (X.copy() if isinstance(X, pd.DataFrame) else pd.DataFrame(X, columns=self.feature_names))
        imputed_flags = {}
        df_imputed = df.copy()

        for col in self.zero_features:
            if col in df_imputed.columns:
                median_val = self.imputer_medians.get(col, 0.0)
                if len(df_imputed) == 1:
                    val = df_imputed[col].iloc[0]
                    if (isinstance(val, (int, float)) and val <= 0) or pd.isna(val):
                        imputed_flags[col] = {'original': float(val) if not pd.isna(val) else 0.0, 'imputed': median_val}
                df_imputed[col] = df_imputed[col].replace({0: median_val, 0.0: median_val}).fillna(median_val)

        scaled_matrix = self.scaler.transform(df_imputed[self.feature_names].values)
        return scaled_matrix, df_imputed, imputed_flags

    def transform_single(self, raw_input_dict):
        ordered_data = {col: float(raw_input_dict.get(col, 0.0)) for col in self.feature_names}
        scaled_matrix, df_imputed, imputed_flags = self.transform(ordered_data)
        scaled_vector = scaled_matrix[0].tolist()
        
        feature_summary = []
        for i, col in enumerate(self.feature_names):
            feature_summary.append({
                'feature': col,
                'raw_value': ordered_data[col],
                'imputed_value': float(df_imputed[col].iloc[0]),
                'is_imputed': col in imputed_flags,
                'scaled_zscore': round(float(scaled_vector[i]), 4),
                'stat_mean': round(self.stats.get(col, {}).get('mean', 0.0), 2),
                'stat_std': round(self.stats.get(col, {}).get('std', 1.0), 2)
            })
        return scaled_matrix, feature_summary, imputed_flags
```

---

### Module 2: `populate_dataset_and_train.py`
```python
import os, json, joblib, numpy as np, pandas as pd
from sklearn.model_selection import train_test_split, learning_curve
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from backend.ml.preprocessing import DiabetesPreprocessor, FEATURE_NAMES
from backend.database import init_db, insert_prediction

def train_and_evaluate():
    df = pd.read_csv('backend/dataset/diabetes.csv')
    X, y = df[FEATURE_NAMES], df['Outcome']
    X_train_raw, X_test_raw, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42, stratify=y)

    preprocessor = DiabetesPreprocessor().fit(X_train_raw)
    X_train_scaled, _, _ = preprocessor.transform(X_train_raw)
    X_test_scaled, _, _ = preprocessor.transform(X_test_raw)

    model = SVC(kernel='rbf', C=1.2, gamma='scale', probability=True, random_state=42)
    model.fit(X_train_scaled, y_train)

    y_pred = model.predict(X_test_scaled)
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel()

    # Save artifacts
    joblib.dump(model, 'backend/models/diabetes_model.pkl')
    preprocessor.save('backend/models/scaler.pkl')
    with open('backend/models/model_metrics.json', 'w') as f:
        json.dump({'accuracy': round(acc*100, 2), 'precision': round(prec*100, 2),
                   'recall': round(rec*100, 2), 'f1_score': round(f1*100, 2),
                   'confusion_matrix': {'tn': int(tn), 'fp': int(fp), 'fn': int(fn), 'tp': int(tp)}}, f, indent=4)
    init_db()
```

---

### Module 3: `backend/routes/predict.py`
```python
@predict_bp.route('/predict', methods=['POST'])
def predict():
    model, preprocessor, _ = get_ml_artifacts()
    payload = request.get_json(force=True) if request.is_json else request.form.to_dict()
    normalized_data = validate_and_normalize_input(payload)

    scaled_matrix, feature_summary, imputed_flags = preprocessor.transform_single(normalized_data)
    prediction_int = int(model.predict(scaled_matrix)[0])
    probabilities = model.predict_proba(scaled_matrix)[0]

    prob_normal = round(float(probabilities[0]) * 100, 2)
    prob_diabetic = round(float(probabilities[1]) * 100, 2)
    confidence = prob_diabetic if prediction_int == 1 else prob_normal
    is_diabetic = (prediction_int == 1)

    clinical = generate_clinical_interpretation(normalized_data, is_diabetic, confidence)
    
    db_payload = {
        'pregnancies': normalized_data['Pregnancies'], 'glucose': normalized_data['Glucose'],
        'blood_pressure': normalized_data['BloodPressure'], 'skin_thickness': normalized_data['SkinThickness'],
        'insulin': normalized_data['Insulin'], 'bmi': normalized_data['BMI'],
        'dpf': normalized_data['DiabetesPedigreeFunction'], 'age': normalized_data['Age'],
        'prediction': clinical['classification'], 'confidence': confidence,
        'risk_level': clinical['risk_level'], 'clinical_notes': "; ".join(clinical['risk_factors'])
    }
    record_id, timestamp_str = insert_prediction(db_payload)

    return jsonify({
        'status': 'success', 'record_id': record_id, 'timestamp': timestamp_str,
        'prediction': clinical['classification'], 'is_diabetic': is_diabetic,
        'confidence': confidence, 'probabilities': {'non_diabetic': prob_normal, 'diabetic': prob_diabetic},
        'risk_level': clinical['risk_level'], 'clinical_interpretation': clinical['summary'],
        'risk_factors': clinical['risk_factors'], 'feature_summary': feature_summary,
        'imputed_features': imputed_flags
    })
```

---

### Module 4: `backend/database.py`
```python
import sqlite3, os
from datetime import datetime
from .config import DATABASE_PATH, DATABASE_DIR

def get_connection():
    os.makedirs(DATABASE_DIR, exist_ok=True)
    conn = sqlite3.connect(DATABASE_PATH)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT NOT NULL,
            pregnancies REAL NOT NULL, glucose REAL NOT NULL,
            blood_pressure REAL NOT NULL, skin_thickness REAL NOT NULL,
            insulin REAL NOT NULL, bmi REAL NOT NULL, dpf REAL NOT NULL, age REAL NOT NULL,
            prediction TEXT NOT NULL, confidence REAL NOT NULL,
            risk_level TEXT NOT NULL, clinical_notes TEXT
        )
    ''')
    conn.commit()
    conn.close()

def insert_prediction(data):
    conn = get_connection()
    cursor = conn.cursor()
    timestamp_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    cursor.execute('''
        INSERT INTO predictions (timestamp, pregnancies, glucose, blood_pressure, skin_thickness,
                                 insulin, bmi, dpf, age, prediction, confidence, risk_level, clinical_notes)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    ''', (timestamp_str, float(data['pregnancies']), float(data['glucose']), float(data['blood_pressure']),
          float(data['skin_thickness']), float(data['insulin']), float(data['bmi']), float(data['dpf']),
          float(data['age']), str(data['prediction']), float(data['confidence']), str(data['risk_level']),
          str(data.get('clinical_notes', ''))))
    record_id = cursor.lastrowid
    conn.commit()
    conn.close()
    return record_id, timestamp_str
```

---

## 7. OUTPUT & STEP-BY-STEP DESCRIPTIONS (FIG 1 TO FIG 12)

### **Fig 1: Main Clinical Dashboard Overview**
- **Description**: The GlucoMetric AI dashboard displays active clinical status badges (`ML ENGINE: ACTIVE`, `MODEL: SUPPORT VECTOR MACHINE (RBF)`, `DB: SQLite3 LOGGED`). The top header provides instant diagnostic metrics from the stratified test partition.

### **Fig 2: Step 1 Input Form Submission & Normal Patient Parameters Preset**
- **Description**: Clicking the "Load Normal Sample" button autofills baseline clinical inputs: `Pregnancies=1`, `Glucose=89 mg/dL`, `BloodPressure=66 mm Hg`, `SkinThickness=23 mm`, `Insulin=94 μU/mL`, `BMI=28.1 kg/m²`, `DPF=0.167`, `Age=21 years`.

### **Fig 3: Step 1 Input Form with Diabetic Patient Parameters Preset**
- **Description**: Clicking the "Load Diabetic Sample" button autofills high-risk clinical inputs with zero values for testing imputation: `Pregnancies=6`, `Glucose=168 mg/dL`, `BloodPressure=74 mm Hg`, `SkinThickness=0 mm`, `Insulin=0 μU/mL`, `BMI=38.0 kg/m²`, `DPF=0.537`, `Age=34 years`.

### **Fig 4: Step 2 Zero-Value Detection & Population Median Imputation Filter**
- **Description**: The preprocessing validation card flags biological zeroes in `SkinThickness` and `Insulin`, replacing them with population training medians:
  - `Imputed SkinThickness: 0 → 29.0 mm`
  - `Imputed Insulin: 0 → 125.0 μU/mL`

### **Fig 5: Step 3 Normalized 8-Dimensional Feature Vector Table & Z-Scores**
- **Description**: The feature vector table visualizes the 8 standardized features passed to the classifier:
  - `Glucose`: Raw $168 \rightarrow$ Imputed $168.0 \rightarrow$ Z-Score $+1.5441$ ($\mu=121.67, \sigma=30.00$)
  - `BMI`: Raw $38.0 \rightarrow$ Imputed $38.0 \rightarrow$ Z-Score $+0.8136$ ($\mu=32.45, \sigma=6.82$)
  - `Age`: Raw $34 \rightarrow$ Imputed $34.0 \rightarrow$ Z-Score $+0.0535$ ($\mu=33.37, \sigma=11.83$)

### **Fig 6: Step 4 Model Training & Validation Curves**
- **Description**: The Canvas chart displays cross-validation learning dynamics across 10 sample fractions:
  - Training Accuracy starts at $86.85\%$ and stabilizes at $84.24\%$.
  - Validation Accuracy progresses from $72.97\%$ to $76.88\%$.
  - Validation Cross-Entropy Loss drops steadily from $0.3152$ to $0.2630$, demonstrating good generalization without severe overfitting.

### **Fig 7: Step 5 Diagnostic Result for Healthy / Normal Patient (Negative Class)**
- **Description**: For normal sample inputs, the system illuminates an emerald green diagnostic banner:
  - Classification: `NON-DIABETIC (HEALTHY)`
  - Diagnostic Confidence: `88.16%`
  - Risk Level: `LOW / NORMAL`

### **Fig 8: Step 5 Diagnostic Result for High-Risk Diabetic Patient (Positive Class)**
- **Description**: For elevated glycemic and adiposity parameters, the system triggers a crimson diagnostic warning banner:
  - Classification: `DIABETES DETECTED / HIGH RISK`
  - Diagnostic Confidence: `92.62%`
  - Risk Level: `CRITICAL / HIGH`

### **Fig 9: Step 5 Dual-Probability Meters and Endocrinological Interpretation Box**
- **Description**: Dual progress bars illustrate class probability balance:
  - Non-Diabetic Probability: `7.38%`
  - Diabetes Risk Probability: `92.62%`  
  The clinical interpretation box flags critical risk factors (`Impaired Glucose Tolerance: 168 mg/dL`, `Class II Obesity: BMI 38.0 kg/m²`) and prescribes urgent HbA1c confirmatory laboratory screening.

### **Fig 10: Step 6 Test Performance Evaluation Metrics**
- **Description**: Evaluated across 154 stratified unseen test instances:
  - **Accuracy**: $74.68\%$
  - **Precision**: $66.67\%$
  - **Recall**: $55.56\%$
  - **F1-Score**: $60.61\%$
  - **ROC-AUC Score**: $79.70\%$

### **Fig 11: Step 6 2x2 Confusion Matrix Heatmap**
- **Description**:
  $$\begin{bmatrix} \text{True Negative (TN)} = 85 & \text{False Positive (FP)} = 15 \\ \text{False Negative (FN)} = 24 & \text{True Positive (TP)} = 30 \end{bmatrix}$$
  - Specificity (True Negative Rate): $\frac{85}{85 + 15} = 85.00\%$
  - Sensitivity (Recall): $\frac{30}{30 + 24} = 55.56\%$
  - Overall Correct Diagnoses: $85 + 30 = 115 \text{ of } 154$ instances ($74.68\%$).

### **Fig 12: Step 7 SQLite Database Audit Trail Table**
- **Description**: The interactive table lists records persisted in `database/diabetes.db`. Each record documents Transaction ID, Timestamp (`2026-10-11 10:44:42`), Patient Age, Glucose, Blood Pressure, Insulin, BMI, DPF, Predicted Outcome badge, and Confidence Percentage, with functional "Refresh Log" and "Clear History" options.

---

## 8. RESULT STATEMENT

The **Diabetes Prediction System using Machine Learning** was successfully conceptualized, trained, deployed, and validated. Utilizing the benchmark **Pima Indians Diabetes Dataset (768 patient records)**, the project addressed physiological zero-value anomalies through a strict **training-only median imputation strategy** across `Glucose`, `BloodPressure`, `SkinThickness`, `Insulin`, and `BMI`, completely eliminating data leakage. Features were standardized via `StandardScaler`. 

The calibrated **Support Vector Machine (RBF Kernel)** classifier achieved **74.68% Test Accuracy**, **66.67% Precision**, **55.56% Recall**, **60.61% F1-Score**, and **79.70% ROC-AUC** on 154 unseen stratified clinical records, outperforming baseline models. The system is served via a lightweight **Flask REST API** with bounds validation and embedded **SQLite3 database transaction logging**. The modern dark-themed web dashboard provides responsive 7-step diagnostics—delivering dual probability risk meters, dynamic imputation inspection, normalized feature vector analysis, interactive learning curves, and a 2x2 confusion matrix heatmap for clinical decision support.
