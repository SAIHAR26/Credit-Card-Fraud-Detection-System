import joblib
import pandas as pd

from pathlib import Path


# ============================================================
# PATHS
# ============================================================

MODEL_DIR = Path("models")
PROCESSED_DIR = Path("data/processed")

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
# FORMAT FEATURE NAMES
# ============================================================

def format_feature_name(feature):

    feature_names = {
        "amt": "transaction amount",
        "category": "transaction category",
        "previous_amount": "previous transaction amount",
        "hour": "transaction time",
        "time_since_previous_hours": "time since previous transaction",
        "previous_transaction_count": "previous transaction count",
        "amount_deviation": "amount deviation from historical behavior",
        "age": "cardholder age",
        "city_pop": "city population",
        "historical_avg_amount": "historical average transaction amount",
        "day": "day of month",
        "state": "transaction state",
        "merchant": "merchant",
        "merchant_distance_km": "merchant distance",
        "month": "month",
        "day_of_week": "day of week",
        "gender": "gender",
        "is_weekend": "weekend indicator"
    }

    return feature_names.get(
        feature,
        feature.replace("_", " ")
    )


# ============================================================
# CREATE NATURAL FACTOR LIST
# ============================================================

def create_factor_text(features):

    formatted = [
        format_feature_name(feature)
        for feature in features
    ]

    if len(formatted) == 1:
        return formatted[0]

    if len(formatted) == 2:
        return f"{formatted[0]} and {formatted[1]}"

    if len(formatted) == 3:
        return (
            f"{formatted[0]}, "
            f"{formatted[1]}, and "
            f"{formatted[2]}"
        )

    return (
        f"{', '.join(formatted[:-1])}, "
        f"and {formatted[-1]}"
    )


# ============================================================
# GENERATE SUMMARY
# ============================================================

def generate_summary(
    probability,
    risk_level,
    top_factors
):

    probability_percent = probability * 100

    positive_factors = [
        factor["feature"]
        for factor in top_factors
        if factor["shap_value"] > 0
    ]

    negative_factors = [
        factor["feature"]
        for factor in top_factors
        if factor["shap_value"] < 0
    ]

    summary = (
        f"The model estimates a fraud probability of "
        f"{probability_percent:.4f}% for this transaction. "
        f"It is currently classified as {risk_level} risk. "
    )

    if positive_factors and negative_factors:
        summary += (
            f"{create_factor_text(positive_factors[:3])} "
            f"influenced the fraud score upward, while "
            f"{create_factor_text(negative_factors[:3])} "
            f"lowered the score."
        )

    elif positive_factors:
        summary += (
            f"{create_factor_text(positive_factors[:3])} "
            f"influenced the fraud score upward."
        )

    elif negative_factors:
        summary += (
            f"{create_factor_text(negative_factors[:3])} "
            f"lowered the fraud score."
        )

    return summary


# ============================================================
# TEST
# ============================================================

if __name__ == "__main__":

    print("=" * 70)
    print("AI SUMMARY TEST")
    print("=" * 70)

    # --------------------------------------------------------
    # Load model
    # --------------------------------------------------------

    model = joblib.load(
        MODEL_DIR / "xgboost.pkl"
    )

    feature_columns = joblib.load(
        MODEL_DIR / "feature_columns.pkl"
    )

    # --------------------------------------------------------
    # Load SHAP results
    # --------------------------------------------------------

    shap_values = pd.read_csv(
        "reports/shap_values.csv"
    )

    importance = pd.read_csv(
        "reports/shap_feature_importance.csv"
    )

    # --------------------------------------------------------
    # Select sample
    # --------------------------------------------------------

    sample_index = 0

    factors = []

    for feature in importance["Feature"].head(5):

        shap_value = shap_values.loc[
            sample_index,
            feature
        ]

        factors.append({
            "feature": feature,
            "shap_value": float(shap_value)
        })

    # --------------------------------------------------------
    # Load test data
    # --------------------------------------------------------

    test = pd.read_csv(
        PROCESSED_DIR / "fraud_test_processed.csv"
    )

    X_test = test.drop(
        columns=[
            "is_fraud",
            "trans_date_trans_time"
        ]
    )

    X_test = X_test[feature_columns]

    # --------------------------------------------------------
    # Prediction
    # --------------------------------------------------------

    probability = model.predict_proba(
        X_test.iloc[[sample_index]]
    )[0, 1]

    # --------------------------------------------------------
    # Risk
    # --------------------------------------------------------

    risk = get_risk_level(
        probability
    )

    # --------------------------------------------------------
    # Summary
    # --------------------------------------------------------

    summary = generate_summary(
        probability,
        risk,
        factors
    )

    # --------------------------------------------------------
    # Display
    # --------------------------------------------------------

    print("\nFraud Probability:")
    print(
        f"{probability * 100:.2f}%"
    )

    print("\nRisk Level:")
    print(risk)

    print("\nAI Summary:")
    print(summary)

    print("\n" + "=" * 70)
    print("AI SUMMARY TEST COMPLETE")
    print("=" * 70)
