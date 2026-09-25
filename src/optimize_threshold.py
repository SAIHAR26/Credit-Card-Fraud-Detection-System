import pandas as pd
import numpy as np
import joblib

from pathlib import Path

from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)


# ============================================================
# PATHS
# ============================================================

PROCESSED_DIR = Path("data/processed")
MODEL_DIR = Path("models")

# ============================================================
# LOAD DATA
# ============================================================

print("=" * 70)
print("LOADING DATA AND XGBOOST MODEL")
print("=" * 70)

train = pd.read_csv(
    PROCESSED_DIR / "fraud_train_processed.csv"
)

test = pd.read_csv(
    PROCESSED_DIR / "fraud_test_processed.csv"
)

model = joblib.load(
    MODEL_DIR / "xgboost.pkl"
)

feature_columns = joblib.load(
    MODEL_DIR / "feature_columns.pkl"
)

# Remove raw timestamp
X_test = test.drop(
    columns=["is_fraud", "trans_date_trans_time"]
)

# Ensure exact feature order
X_test = X_test[feature_columns]

y_test = test["is_fraud"]

# ============================================================
# GET FRAUD PROBABILITIES
# ============================================================

print("Generating fraud probabilities...")

probabilities = model.predict_proba(X_test)[:, 1]

# ============================================================
# THRESHOLD TESTING
# ============================================================

thresholds = np.arange(
    0.10,
    0.91,
    0.05
)

results = []

for threshold in thresholds:

    predictions = (
        probabilities >= threshold
    ).astype(int)

    precision = precision_score(
        y_test,
        predictions,
        zero_division=0
    )

    recall = recall_score(
        y_test,
        predictions,
        zero_division=0
    )

    f1 = f1_score(
        y_test,
        predictions,
        zero_division=0
    )

    tn, fp, fn, tp = confusion_matrix(
        y_test,
        predictions
    ).ravel()

    results.append({
        "Threshold": round(threshold, 2),
        "Precision": precision,
        "Recall": recall,
        "F1": f1,
        "False_Positives": fp,
        "False_Negatives": fn,
        "True_Positives": tp,
        "True_Negatives": tn
    })


# ============================================================
# DISPLAY RESULTS
# ============================================================

results_df = pd.DataFrame(results)

print("\n" + "=" * 70)
print("THRESHOLD COMPARISON")
print("=" * 70)

print(
    results_df.to_string(index=False)
)


# ============================================================
# BEST F1 THRESHOLD
# ============================================================

best_row = results_df.loc[
    results_df["F1"].idxmax()
]

print("\n" + "=" * 70)
print("BEST F1 THRESHOLD")
print("=" * 70)

print(
    f"Threshold : {best_row['Threshold']:.2f}"
)

print(
    f"Precision : {best_row['Precision']:.4f}"
)

print(
    f"Recall    : {best_row['Recall']:.4f}"
)

print(
    f"F1 Score  : {best_row['F1']:.4f}"
)

print(
    f"False Positives : "
    f"{int(best_row['False_Positives'])}"
)

print(
    f"False Negatives : "
    f"{int(best_row['False_Negatives'])}"
)


# ============================================================
# SAVE RESULTS
# ============================================================

results_df.to_csv(
    MODEL_DIR / "threshold_comparison.csv",
    index=False
)

print("\nSaved:")
print("models/threshold_comparison.csv")

print("=" * 70)