import pandas as pd
import numpy as np
import joblib
import shap

from pathlib import Path


# ============================================================
# PATHS
# ============================================================

MODEL_DIR = Path("models")
PROCESSED_DIR = Path("data/processed")


# ============================================================
# LOAD MODEL
# ============================================================

model = joblib.load(
    MODEL_DIR / "xgboost.pkl"
)

feature_columns = joblib.load(
    MODEL_DIR / "feature_columns.pkl"
)

explainer = shap.TreeExplainer(model)

THRESHOLD = 0.25


# ============================================================
# RISK LEVEL
# ============================================================

def get_risk_level(probability):

    if probability < 0.25:
        return "LOW"

    elif probability < 0.60:
        return "MEDIUM"

    else:
        return "HIGH"


# ============================================================
# PREDICT TRANSACTION
# ============================================================

def predict_transaction(transaction):

    """
    transaction:
        Dictionary containing the processed feature values.

    Returns:
        Fraud probability
        Prediction
        Risk level
        SHAP explanations
    """

    # Convert input to DataFrame
    X = pd.DataFrame([transaction])

    # Ensure correct feature order
    X = X[feature_columns]

    # Fraud probability
    probability = model.predict_proba(X)[0, 1]

    # Fraud prediction using optimized threshold
    prediction = int(
        probability >= THRESHOLD
    )

    # Risk
    risk = get_risk_level(probability)

    # SHAP
    shap_values = explainer.shap_values(X)

    if isinstance(shap_values, list):
        shap_values = shap_values[1]

    shap_values = np.asarray(shap_values)[0]

    # Create explanation table
    explanation = pd.DataFrame({
        "feature": feature_columns,
        "shap_value": shap_values,
        "feature_value": X.iloc[0].values
    })

    # Sort by absolute SHAP impact
    explanation["absolute_impact"] = (
        explanation["shap_value"].abs()
    )

    explanation = explanation.sort_values(
        "absolute_impact",
        ascending=False
    )

    # Top 5 factors
    top_factors = explanation.head(5)

    return {
        "fraud_probability": float(probability),
        "prediction": prediction,
        "risk_level": risk,
        "top_factors": top_factors[
            ["feature", "shap_value", "feature_value"]
        ].to_dict("records")
    }


# ============================================================
# TEST WITH REAL TRANSACTION
# ============================================================

if __name__ == "__main__":

    print("=" * 70)
    print("TESTING PREDICTION ENGINE")
    print("=" * 70)

    test = pd.read_csv(
        PROCESSED_DIR / "fraud_test_processed.csv"
    )

    # Take one real transaction
    transaction = (
        test
        .drop(columns=["is_fraud", "trans_date_trans_time"])
        .iloc[0]
        .to_dict()
    )

    result = predict_transaction(
        transaction
    )

    print("\nFraud Probability:")
    print(
        f"{result['fraud_probability'] * 100:.2f}%"
    )

    print("\nPrediction:")

    if result["prediction"] == 1:
        print("FRAUD")
    else:
        print("LEGITIMATE")

    print("\nRisk Level:")
    print(result["risk_level"])

    print("\nTop Contributing Factors:")

    for factor in result["top_factors"]:

        direction = (
            "increased"
            if factor["shap_value"] > 0
            else "decreased"
        )

        print(
            f"- {factor['feature']}: "
            f"{factor['feature_value']} "
            f"({direction} fraud score)"
        )

    print("\n" + "=" * 70)
    print("PREDICTION ENGINE TEST COMPLETE")
    print("=" * 70)