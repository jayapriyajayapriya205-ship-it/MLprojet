import os
import json
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, learning_curve
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report, log_loss
)
import joblib

# Setup paths relative to script
BASE_DIR = os.path.abspath(os.path.dirname(__file__))
BACKEND_DIR = os.path.join(BASE_DIR, 'backend')
DATASET_DIR = os.path.join(BACKEND_DIR, 'dataset')
MODELS_DIR = os.path.join(BACKEND_DIR, 'models')
DATABASE_DIR = os.path.join(BASE_DIR, 'database')
DB_PATH = os.path.join(DATABASE_DIR, 'diabetes.db')

os.makedirs(DATASET_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(DATABASE_DIR, exist_ok=True)

from backend.ml.preprocessing import DiabetesPreprocessor, FEATURE_NAMES, ZERO_IMPUTATION_FEATURES
from backend.database import init_db, insert_prediction

COLUMN_NAMES = [
    'Pregnancies',
    'Glucose',
    'BloodPressure',
    'SkinThickness',
    'Insulin',
    'BMI',
    'DiabetesPedigreeFunction',
    'Age',
    'Outcome'
]

def load_or_fetch_dataset():
    csv_path = os.path.join(DATASET_DIR, 'diabetes.csv')
    local_candidate = os.path.join(BASE_DIR, 'test_pima.csv')
    parent_candidate = os.path.join(BASE_DIR, '..', 'test_pima.csv')

    df = None
    if os.path.exists(csv_path) and os.path.getsize(csv_path) > 1000:
        print(f"Loading existing dataset from: {csv_path}")
        df = pd.read_csv(csv_path)
    elif os.path.exists(local_candidate):
        print(f"Found local dataset file at {local_candidate}, processing headers...")
        df = pd.read_csv(local_candidate, header=None, names=COLUMN_NAMES)
        df.to_csv(csv_path, index=False)
    elif os.path.exists(parent_candidate):
        print(f"Found dataset at {parent_candidate}, processing headers...")
        df = pd.read_csv(parent_candidate, header=None, names=COLUMN_NAMES)
        df.to_csv(csv_path, index=False)
    else:
        print("Downloading standard Pima Indians Diabetes dataset...")
        import urllib.request
        url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv"
        try:
            urllib.request.urlretrieve(url, csv_path)
            df = pd.read_csv(csv_path, header=None, names=COLUMN_NAMES)
            df.to_csv(csv_path, index=False)
        except Exception as e:
            raise RuntimeError(f"Could not load or download dataset: {e}")

    # Ensure correct column names
    if list(df.columns) != COLUMN_NAMES:
        df.columns = COLUMN_NAMES
        df.to_csv(csv_path, index=False)

    print(f"Dataset ready: {df.shape[0]} records, {df.shape[1]} features.")
    print(f"Outcome Distribution: Negative (0) = {(df['Outcome']==0).sum()}, Positive (1) = {(df['Outcome']==1).sum()}")
    return df

def generate_training_curves(model, X_train, y_train, X_test, y_test):
    """
    Simulates training & validation progression across training set fractions (iterations)
    """
    train_sizes, train_scores, test_scores = learning_curve(
        model, X_train, y_train,
        cv=5,
        scoring='accuracy',
        train_sizes=np.linspace(0.15, 1.0, 10),
        random_state=42
    )

    train_acc_mean = np.mean(train_scores, axis=1)
    val_acc_mean = np.mean(test_scores, axis=1)

    # Derive loss proxy from accuracy (log-loss approximation)
    train_loss = [round(float(-np.log(np.clip(acc, 0.01, 0.99))), 4) for acc in train_acc_mean]
    val_loss = [round(float(-np.log(np.clip(acc, 0.01, 0.99))), 4) for acc in val_acc_mean]

    iterations = [int(i + 1) for i in range(len(train_sizes))]
    samples = [int(s) for s in train_sizes]

    return {
        'iterations': iterations,
        'sample_sizes': samples,
        'train_accuracy': [round(float(a) * 100, 2) for a in train_acc_mean],
        'val_accuracy': [round(float(a) * 100, 2) for a in val_acc_mean],
        'train_loss': train_loss,
        'val_loss': val_loss
    }

def train_and_evaluate():
    print("=" * 70)
    print(" DIABETES PREDICTION SYSTEM: MODEL TRAINING & PIPELINE PIPELINE ")
    print("=" * 70)

    # 1. Dataset loading
    df = load_or_fetch_dataset()

    # 2. Stratified Train / Test split
    X = df[FEATURE_NAMES]
    y = df['Outcome']
    X_train_raw, X_test_raw, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )

    # Save train & test splits
    train_df = pd.concat([X_train_raw, y_train], axis=1)
    test_df = pd.concat([X_test_raw, y_test], axis=1)
    train_path = os.path.join(DATASET_DIR, 'train.csv')
    test_path = os.path.join(DATASET_DIR, 'test.csv')
    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path, index=False)
    print(f"Saved Train Split ({len(train_df)} rows) to: {train_path}")
    print(f"Saved Test Split ({len(test_df)} rows) to: {test_path}")

    # 3. Fit Preprocessing Pipeline (Median Imputation + Standard Scaling)
    print("\nFitting Preprocessor (Median Imputation for zeroes + StandardScaler)...")
    preprocessor = DiabetesPreprocessor()
    preprocessor.fit(X_train_raw)

    print("Computed Training Medians for Physiological Zeroes:")
    for col, med in preprocessor.imputer_medians.items():
        print(f"  - {col:15s}: {med}")

    # Transform Train and Test sets
    X_train_scaled, _, _ = preprocessor.transform(X_train_raw)
    X_test_scaled, _, _ = preprocessor.transform(X_test_raw)

    # 4. Compare Models
    models = {
        'Random Forest': RandomForestClassifier(
            n_estimators=120,
            max_depth=6,
            min_samples_split=4,
            min_samples_leaf=2,
            random_state=42
        ),
        'Support Vector Machine (RBF)': SVC(
            kernel='rbf',
            C=1.2,
            gamma='scale',
            probability=True,
            random_state=42
        ),
        'Logistic Regression': LogisticRegression(
            C=0.8,
            max_iter=500,
            random_state=42
        )
    }

    best_name = None
    best_model = None
    best_f1 = -1.0
    results = {}

    print("\nModel Benchmark on Stratified Test Set (N=154):")
    print("-" * 70)
    for name, model in models.items():
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)
        y_prob = model.predict_proba(X_test_scaled)[:, 1]

        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, zero_division=0)
        rec = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        auc = roc_auc_score(y_test, y_prob)

        results[name] = {
            'accuracy': acc,
            'precision': prec,
            'recall': rec,
            'f1': f1,
            'auc': auc
        }

        print(f"[{name:28s}] Acc: {acc*100:5.2f}% | Prec: {prec*100:5.2f}% | Rec: {rec*100:5.2f}% | F1: {f1*100:5.2f}% | AUC: {auc*100:5.2f}%")

        # Select highest F1-score model
        if f1 > best_f1:
            best_f1 = f1
            best_name = name
            best_model = model

    print("-" * 70)
    print(f"Selected Best Performing Model: {best_name} (F1-score: {best_f1*100:.2f}%)")

    # 5. Final Evaluation Metrics with Best Model
    y_pred_best = best_model.predict(X_test_scaled)
    y_prob_best = best_model.predict_proba(X_test_scaled)[:, 1]

    cm = confusion_matrix(y_test, y_pred_best)
    tn, fp, fn, tp = cm.ravel()

    final_metrics = {
        'model_name': best_name,
        'test_samples': int(len(y_test)),
        'accuracy': round(float(accuracy_score(y_test, y_pred_best)) * 100, 2),
        'precision': round(float(precision_score(y_test, y_pred_best)) * 100, 2),
        'recall': round(float(recall_score(y_test, y_pred_best)) * 100, 2),
        'f1_score': round(float(f1_score(y_test, y_pred_best)) * 100, 2),
        'roc_auc': round(float(roc_auc_score(y_test, y_prob_best)) * 100, 2),
        'confusion_matrix': {
            'tn': int(tn),
            'fp': int(fp),
            'fn': int(fn),
            'tp': int(tp),
            'matrix': [[int(tn), int(fp)], [int(fn), int(tp)]]
        },
        'imputation_medians': preprocessor.imputer_medians,
        'feature_stats': preprocessor.stats
    }

    # 6. Generate Training vs Validation Curves
    print("\nGenerating Training vs Validation learning curves...")
    curves = generate_training_curves(best_model, X_train_scaled, y_train, X_test_scaled, y_test)
    final_metrics['training_curves'] = curves

    # 7. Save Model & Scaler Artifacts
    model_save_path = os.path.join(MODELS_DIR, 'diabetes_model.pkl')
    scaler_save_path = os.path.join(MODELS_DIR, 'scaler.pkl')
    metrics_save_path = os.path.join(MODELS_DIR, 'model_metrics.json')

    joblib.dump(best_model, model_save_path)
    preprocessor.save(scaler_save_path)
    with open(metrics_save_path, 'w', encoding='utf-8') as f:
        json.dump(final_metrics, f, indent=4)

    print(f"\nArtifacts successfully persisted:")
    print(f"  - Model:   {model_save_path}")
    print(f"  - Scaler:  {scaler_save_path}")
    print(f"  - Metrics: {metrics_save_path}")

    # 8. Initialize Database and Add Initial Seed History Records
    print("\nInitializing SQLite database...")
    init_db()

    # Seed with sample test cases
    sample_normal = {
        'pregnancies': 1, 'glucose': 85, 'blood_pressure': 66, 'skin_thickness': 29,
        'insulin': 0, 'bmi': 26.6, 'dpf': 0.351, 'age': 31,
        'prediction': 'NON-DIABETIC (HEALTHY)', 'confidence': 88.4,
        'risk_level': 'LOW / NORMAL',
        'clinical_notes': 'Normal glucose and healthy BMI baselines.'
    }
    sample_diabetic = {
        'pregnancies': 6, 'glucose': 148, 'blood_pressure': 72, 'skin_thickness': 35,
        'insulin': 0, 'bmi': 33.6, 'dpf': 0.627, 'age': 50,
        'prediction': 'DIABETES DETECTED / HIGH RISK', 'confidence': 84.7,
        'risk_level': 'CRITICAL / HIGH',
        'clinical_notes': 'Elevated glucose (148 mg/dL) and Obese BMI (33.6).'
    }
    insert_prediction(sample_normal)
    insert_prediction(sample_diabetic)
    print("Database initialized and populated with baseline clinical records.")

    print("=" * 70)
    print(" MODEL TRAINING AND PIPELINE POPULATION COMPLETE! ")
    print("=" * 70)

if __name__ == '__main__':
    train_and_evaluate()
