import pandas as pd
import numpy as np
import joblib
import shap

from pathlib import Path


# ============================================================
# PATHS
# ============================================================

PROCESSED_DIR = Path("data/processed")
MODEL_DIR = Path("models")
REPORT_DIR = Path("reports")

REPORT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# LOAD DATA AND MODEL
# ============================================================

print("=" * 70)
print("LOADING XGBOOST MODEL AND TEST DATA")
print("=" * 70)

test = pd.read_csv(
    PROCESSED_DIR / "fraud_test_processed.csv"
)

model = joblib.load(
    MODEL_DIR / "xgboost.pkl"
)

feature_columns = joblib.load(
    MODEL_DIR / "feature_columns.pkl"
)


# ============================================================
# PREPARE TEST DATA
# ============================================================

X_test = test.drop(
    columns=["is_fraud", "trans_date_trans_time"]
)

X_test = X_test[feature_columns]

y_test = test["is_fraud"]

print("Test features:", X_test.shape)


# ============================================================
# SELECT SAMPLE
# ============================================================

# SHAP does not need all 555k transactions.
# Use a representative sample for efficient explanation.

sample_size = min(5000, len(X_test))

X_sample = X_test.sample(
    n=sample_size,
    random_state=42
)

print(f"SHAP sample size: {len(X_sample):,}")


# ============================================================
# CREATE SHAP EXPLAINER
# ============================================================

print("\nCreating SHAP TreeExplainer...")

explainer = shap.TreeExplainer(model)


# ============================================================
# CALCULATE SHAP VALUES
# ============================================================

print("Calculating SHAP values...")

shap_values = explainer.shap_values(
    X_sample
)


# ============================================================
# HANDLE SHAP OUTPUT
# ============================================================

# For binary XGBoost models, SHAP normally returns
# one value per feature for each transaction.

if isinstance(shap_values, list):
    shap_values = shap_values[1]

shap_values = np.asarray(shap_values)

print(
    "SHAP values shape:",
    shap_values.shape
)


# ============================================================
# GLOBAL FEATURE IMPORTANCE
# ============================================================

print("\nCalculating global feature importance...")

importance = np.abs(shap_values).mean(axis=0)

importance_df = pd.DataFrame({
    "Feature": X_sample.columns,
    "Mean_Absolute_SHAP": importance
})

importance_df = importance_df.sort_values(
    "Mean_Absolute_SHAP",
    ascending=False
)

print("\n" + "=" * 70)
print("GLOBAL SHAP FEATURE IMPORTANCE")
print("=" * 70)

print(
    importance_df.to_string(index=False)
)


# ============================================================
# SAVE GLOBAL IMPORTANCE
# ============================================================

importance_df.to_csv(
    REPORT_DIR / "shap_feature_importance.csv",
    index=False
)


# ============================================================
# SAVE SHAP VALUES
# ============================================================

shap_values_df = pd.DataFrame(
    shap_values,
    columns=X_sample.columns
)

shap_values_df.to_csv(
    REPORT_DIR / "shap_values.csv",
    index=False
)


# ============================================================
# SHAP SUMMARY PLOT
# ============================================================

print("\nCreating SHAP summary plot...")

shap.summary_plot(
    shap_values,
    X_sample,
    show=False
)

import matplotlib.pyplot as plt

plt.tight_layout()

plt.savefig(
    REPORT_DIR / "shap_summary.png",
    dpi=200,
    bbox_inches="tight"
)

plt.close()


# ============================================================
# FINISH
# ============================================================

print("\n" + "=" * 70)
print("SHAP ANALYSIS COMPLETE")
print("=" * 70)

print("\nSaved:")
print("reports/shap_feature_importance.csv")
print("reports/shap_values.csv")
print("reports/shap_summary.png")