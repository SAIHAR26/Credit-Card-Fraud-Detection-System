import pandas as pd
import numpy as np
import joblib

from pathlib import Path

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix
)


# ============================================================
# PATHS
# ============================================================

PROCESSED_DIR = Path("data/processed")
MODEL_DIR = Path("models")

MODEL_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# LOAD DATA
# ============================================================

print("=" * 70)
print("LOADING PROCESSED DATA")
print("=" * 70)

train = pd.read_csv(
    PROCESSED_DIR / "fraud_train_processed.csv"
)

test = pd.read_csv(
    PROCESSED_DIR / "fraud_test_processed.csv"
)

print("Train:", train.shape)
print("Test :", test.shape)


# ============================================================
# PREPARE FEATURES
# ============================================================

TARGET = "is_fraud"

# Raw timestamp is not used directly by ML models
DROP_FEATURES = [
    "trans_date_trans_time"
]

X_train = train.drop(
    columns=[TARGET] + DROP_FEATURES
)

y_train = train[TARGET]

X_test = test.drop(
    columns=[TARGET] + DROP_FEATURES
)

y_test = test[TARGET]


# Make absolutely sure train/test columns match
X_test = X_test[X_train.columns]

print("\nNumber of features:", X_train.shape[1])

print("\nTraining class distribution:")
print(y_train.value_counts())

print("\nTest class distribution:")
print(y_test.value_counts())


# ============================================================
# LOGISTIC REGRESSION
# ============================================================

print("\n" + "=" * 70)
print("TRAINING LOGISTIC REGRESSION")
print("=" * 70)

# Scale numerical features for Logistic Regression
scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

logistic_model = LogisticRegression(
    class_weight="balanced",
    max_iter=1000,
    solver="liblinear",
    random_state=42
)

logistic_model.fit(
    X_train_scaled,
    y_train
)

logistic_probability = logistic_model.predict_proba(
    X_test_scaled
)[:, 1]

logistic_prediction = (
    logistic_probability >= 0.5
).astype(int)


# ============================================================
# RANDOM FOREST
# ============================================================

print("\n" + "=" * 70)
print("TRAINING RANDOM FOREST")
print("=" * 70)

random_forest = RandomForestClassifier(
    n_estimators=200,
    max_depth=20,
    min_samples_leaf=2,
    class_weight="balanced_subsample",
    n_jobs=-1,
    random_state=42
)

random_forest.fit(
    X_train,
    y_train
)

rf_probability = random_forest.predict_proba(
    X_test
)[:, 1]

rf_prediction = (
    rf_probability >= 0.5
).astype(int)


# ============================================================
# XGBOOST
# ============================================================

print("\n" + "=" * 70)
print("TRAINING XGBOOST")
print("=" * 70)

fraud_count = y_train.sum()
legitimate_count = len(y_train) - fraud_count

scale_pos_weight = (
    legitimate_count / fraud_count
)

print(
    f"XGBoost scale_pos_weight: "
    f"{scale_pos_weight:.2f}"
)

xgb_model = XGBClassifier(
    n_estimators=300,
    max_depth=8,
    learning_rate=0.08,
    subsample=0.8,
    colsample_bytree=0.8,
    objective="binary:logistic",
    eval_metric="aucpr",
    scale_pos_weight=scale_pos_weight,
    tree_method="hist",
    n_jobs=-1,
    random_state=42
)

xgb_model.fit(
    X_train,
    y_train
)

xgb_probability = xgb_model.predict_proba(
    X_test
)[:, 1]

xgb_prediction = (
    xgb_probability >= 0.5
).astype(int)


# ============================================================
# EVALUATION FUNCTION
# ============================================================

def evaluate_model(name, y_true, prediction, probability):

    precision = precision_score(
        y_true,
        prediction,
        zero_division=0
    )

    recall = recall_score(
        y_true,
        prediction,
        zero_division=0
    )

    f1 = f1_score(
        y_true,
        prediction,
        zero_division=0
    )

    roc_auc = roc_auc_score(
        y_true,
        probability
    )

    pr_auc = average_precision_score(
        y_true,
        probability
    )

    matrix = confusion_matrix(
        y_true,
        prediction
    )

    print("\n" + "-" * 60)
    print(name)
    print("-" * 60)

    print(f"Precision : {precision:.4f}")
    print(f"Recall    : {recall:.4f}")
    print(f"F1 Score  : {f1:.4f}")
    print(f"ROC-AUC   : {roc_auc:.4f}")
    print(f"PR-AUC    : {pr_auc:.4f}")

    print("\nConfusion Matrix:")
    print(matrix)

    return {
        "Model": name,
        "Precision": precision,
        "Recall": recall,
        "F1": f1,
        "ROC-AUC": roc_auc,
        "PR-AUC": pr_auc
    }


# ============================================================
# EVALUATE MODELS
# ============================================================

results = []

results.append(
    evaluate_model(
        "Logistic Regression",
        y_test,
        logistic_prediction,
        logistic_probability
    )
)

results.append(
    evaluate_model(
        "Random Forest",
        y_test,
        rf_prediction,
        rf_probability
    )
)

results.append(
    evaluate_model(
        "XGBoost",
        y_test,
        xgb_prediction,
        xgb_probability
    )
)


# ============================================================
# COMPARISON TABLE
# ============================================================

results_df = pd.DataFrame(results)

print("\n" + "=" * 70)
print("MODEL COMPARISON")
print("=" * 70)

print(
    results_df.to_string(index=False)
)


# ============================================================
# SAVE RESULTS
# ============================================================

results_df.to_csv(
    MODEL_DIR / "model_comparison.csv",
    index=False
)


# ============================================================
# SAVE MODELS
# ============================================================

joblib.dump(
    logistic_model,
    MODEL_DIR / "logistic_regression.pkl"
)

joblib.dump(
    scaler,
    MODEL_DIR / "logistic_scaler.pkl"
)

joblib.dump(
    random_forest,
    MODEL_DIR / "random_forest.pkl"
)

joblib.dump(
    xgb_model,
    MODEL_DIR / "xgboost.pkl"
)


# Save feature names
joblib.dump(
    list(X_train.columns),
    MODEL_DIR / "feature_columns.pkl"
)


print("\n" + "=" * 70)
print("TRAINING COMPLETE")
print("=" * 70)

print("\nSaved models:")
print("models/logistic_regression.pkl")
print("models/logistic_scaler.pkl")
print("models/random_forest.pkl")
print("models/xgboost.pkl")

print("\nSaved comparison:")
print("models/model_comparison.csv")