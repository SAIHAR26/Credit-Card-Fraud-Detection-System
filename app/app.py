from datetime import date
from functools import lru_cache
from pathlib import Path
import random
import sys
from typing import Any, Dict, List, Optional

import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, ConfigDict, Field, create_model

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.ai_summary import format_feature_name, generate_summary
from src.fraud_predictor import THRESHOLD, feature_columns, predict_transaction

RAW_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

HISTORY_FEATURES = [
    "previous_amount",
    "previous_transaction_count",
    "historical_avg_amount",
    "time_since_previous_hours",
    "amount_deviation",
    "merchant_distance_km",
]


class Factor(BaseModel):
    feature: str
    feature_label: str
    shap_value: float
    feature_value: float
    impact: str


class PredictionResponse(BaseModel):
    fraud_probability: float
    prediction: int
    prediction_label: str
    risk_level: str
    threshold: float
    top_factors: List[Factor]
    explanation: str
    model_features: Dict[str, float]
    feature_source: str
    actual_label: Optional[int] = None
    actual_label_text: Optional[str] = None


class HealthResponse(BaseModel):
    project: str
    status: str
    model: str
    threshold: float
    feature_count: int


class SampleTransactionResponse(BaseModel):
    user_input: Dict[str, Any]
    transaction: Dict[str, float]
    source_label: str
    actual_label: int
    actual_label_text: str


class UserTransactionInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    amt: float = Field(..., gt=0, description="Transaction amount")
    merchant: str
    category: str
    gender: str
    state: str
    city_pop: int = Field(..., ge=0)
    transaction_date: date
    hour: int = Field(..., ge=0, le=23)
    age: float = Field(..., ge=18, le=100)
    previous_amount: Optional[float] = Field(None, ge=0)
    previous_transaction_count: Optional[int] = Field(None, ge=0)
    historical_avg_amount: Optional[float] = Field(None, ge=0)
    time_since_previous_hours: Optional[float] = Field(None, ge=0)


TransactionInput = create_model(
    "TransactionInput",
    __config__=ConfigDict(extra="forbid"),
    **{
        feature: (
            float,
            Field(
                ...,
                description=f"Engineered model feature: {format_feature_name(feature)}",
            ),
        )
        for feature in feature_columns
    },
)

app = FastAPI(
    title="Credit Card Fraud Detection API",
    description=(
        "Inference API for the trained XGBoost fraud model. The user-facing "
        "endpoint accepts readable transaction fields and converts them into "
        "the 18 engineered model features without changing the trained model."
    ),
    version="1.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "http://127.0.0.1:4173",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@lru_cache(maxsize=1)
def get_raw_reference_data() -> pd.DataFrame:
    paths = [RAW_DIR / "fraudTrain.csv", RAW_DIR / "fraudTest.csv"]
    if not all(path.exists() for path in paths):
        raise FileNotFoundError("Raw fraudTrain.csv and fraudTest.csv are required.")

    frames = [
        pd.read_csv(
            path,
            usecols=["merchant", "category", "gender", "state"],
        )
        for path in paths
    ]
    return pd.concat(frames, axis=0, ignore_index=True)


@lru_cache(maxsize=1)
def get_categorical_mappings() -> Dict[str, Dict[str, int]]:
    raw = get_raw_reference_data()
    mappings = {}

    for column in ["merchant", "category", "state", "gender"]:
        categories = raw[column].astype("category").cat.categories.tolist()
        mappings[column] = {value: index for index, value in enumerate(categories)}

    return mappings


@lru_cache(maxsize=1)
def get_processed_train_medians() -> Dict[str, float]:
    train_path = PROCESSED_DIR / "fraud_train_processed.csv"
    if not train_path.exists():
        raise FileNotFoundError("Processed training dataset was not found.")

    train = pd.read_csv(train_path, usecols=list(feature_columns))
    return {column: float(train[column].median()) for column in feature_columns}


@lru_cache(maxsize=1)
def get_processed_test_data() -> pd.DataFrame:
    test_path = PROCESSED_DIR / "fraud_test_processed.csv"
    if not test_path.exists():
        raise FileNotFoundError("Processed test dataset was not found.")

    return pd.read_csv(test_path)


@lru_cache(maxsize=1)
def get_raw_test_sorted() -> pd.DataFrame:
    raw_path = RAW_DIR / "fraudTest.csv"
    if not raw_path.exists():
        raise FileNotFoundError("Raw fraudTest.csv was not found.")

    raw = pd.read_csv(raw_path)
    raw["trans_date_trans_time"] = pd.to_datetime(raw["trans_date_trans_time"])
    return raw.sort_values("trans_date_trans_time").reset_index(drop=True)


def get_reverse_mappings() -> Dict[str, Dict[int, str]]:
    return {
        column: {encoded: label for label, encoded in mapping.items()}
        for column, mapping in get_categorical_mappings().items()
    }


def encode_value(column: str, value: str) -> int:
    mapping = get_categorical_mappings()[column]

    if value not in mapping:
        raise HTTPException(
            status_code=422,
            detail=f"Unknown {column}: {value}. Use one of the values returned by /input-options.",
        )

    return mapping[value]


def actual_label_text(value: int) -> str:
    return "Fraud" if int(value) == 1 else "Legitimate"


def build_user_input_from_raw(raw_row: pd.Series) -> Dict[str, Any]:
    transaction_time = pd.to_datetime(raw_row["trans_date_trans_time"])
    dob = pd.to_datetime(raw_row["dob"])
    age = (transaction_time - dob).days / 365.25

    return {
        "amt": float(raw_row["amt"]),
        "merchant": raw_row["merchant"],
        "category": raw_row["category"],
        "gender": raw_row["gender"],
        "state": raw_row["state"],
        "city_pop": int(raw_row["city_pop"]),
        "transaction_date": transaction_time.date().isoformat(),
        "hour": int(transaction_time.hour),
        "age": float(max(18, min(100, age))),
    }


def build_manual_model_features(transaction: UserTransactionInput) -> Dict[str, float]:
    medians = get_processed_train_medians()
    transaction_date = pd.Timestamp(transaction.transaction_date)

    model_features = {feature: float(medians[feature]) for feature in feature_columns}
    model_features.update(
        {
            "merchant": float(encode_value("merchant", transaction.merchant)),
            "category": float(encode_value("category", transaction.category)),
            "gender": float(encode_value("gender", transaction.gender)),
            "state": float(encode_value("state", transaction.state)),
            "amt": float(transaction.amt),
            "city_pop": float(transaction.city_pop),
            "hour": float(transaction.hour),
            "day_of_week": float(transaction_date.dayofweek),
            "month": float(transaction_date.month),
            "day": float(transaction_date.day),
            "is_weekend": float(1 if transaction_date.dayofweek >= 5 else 0),
            "age": float(transaction.age),
        }
    )

    history_values = {
        "previous_amount": transaction.previous_amount,
        "previous_transaction_count": transaction.previous_transaction_count,
        "historical_avg_amount": transaction.historical_avg_amount,
        "time_since_previous_hours": transaction.time_since_previous_hours,
    }
    for feature, value in history_values.items():
        if value is not None:
            model_features[feature] = float(value)

    model_features["amount_deviation"] = (
        model_features["amt"] - model_features["historical_avg_amount"]
    )

    return {feature: float(model_features[feature]) for feature in feature_columns}


def format_factor(factor: Dict[str, Any]) -> Dict[str, Any]:
    shap_value = float(factor["shap_value"])

    return {
        "feature": factor["feature"],
        "feature_label": format_feature_name(factor["feature"]),
        "shap_value": shap_value,
        "feature_value": float(factor["feature_value"]),
        "impact": "increased the fraud risk estimate"
        if shap_value > 0
        else "lowered the fraud risk estimate",
    }


def create_prediction_response(
    model_features: Dict[str, float],
    feature_source: str,
    actual_label: Optional[int] = None,
) -> Dict[str, Any]:
    ordered_features = {feature: float(model_features[feature]) for feature in feature_columns}
    result = predict_transaction(ordered_features)
    top_factors = [format_factor(factor) for factor in result["top_factors"]]

    explanation = generate_summary(
        result["fraud_probability"],
        result["risk_level"],
        top_factors,
    )

    return {
        "fraud_probability": result["fraud_probability"],
        "prediction": result["prediction"],
        "prediction_label": "Fraud" if result["prediction"] == 1 else "Legitimate",
        "risk_level": result["risk_level"],
        "threshold": THRESHOLD,
        "top_factors": top_factors,
        "explanation": explanation,
        "model_features": ordered_features,
        "feature_source": feature_source,
        "actual_label": actual_label,
        "actual_label_text": actual_label_text(actual_label)
        if actual_label is not None
        else None,
    }


def get_sample_by_label(label: Optional[int] = None) -> Dict[str, Any]:
    processed = get_processed_test_data()

    if label is None:
        row_index = random.randrange(len(processed))
    else:
        matches = processed.index[processed["is_fraud"].eq(label)]
        if len(matches) == 0:
            raise HTTPException(
                status_code=404,
                detail=f"No test sample found for label {label}.",
            )
        row_index = int(random.choice(matches.tolist()))

    processed_row = processed.iloc[row_index]
    raw_row = get_raw_test_sorted().iloc[row_index]
    model_features = {
        feature: float(processed_row[feature])
        for feature in feature_columns
    }
    label_value = int(processed_row["is_fraud"])

    return {
        "user_input": build_user_input_from_raw(raw_row),
        "transaction": model_features,
        "source_label": "Actual processed test dataset sample",
        "actual_label": label_value,
        "actual_label_text": actual_label_text(label_value),
    }


@app.get("/", response_model=HealthResponse)
def health_check():
    return {
        "project": "Credit Card Fraud Detection System",
        "status": "ready",
        "model": "XGBoost",
        "threshold": THRESHOLD,
        "feature_count": len(feature_columns),
    }


@app.get("/features", response_model=List[str])
def get_features():
    return list(feature_columns)


@app.get("/input-options")
def get_input_options():
    mappings = get_categorical_mappings()

    return {
        "categories": list(mappings["category"].keys()),
        "genders": list(mappings["gender"].keys()),
        "states": list(mappings["state"].keys()),
        "merchants": list(mappings["merchant"].keys()),
        "threshold": THRESHOLD,
        "risk_bands": {
            "LOW": "probability < 0.25",
            "MEDIUM": "0.25 <= probability < 0.60",
            "HIGH": "probability >= 0.60",
        },
    }


@app.get("/sample-transaction", response_model=SampleTransactionResponse)
def get_sample_transaction():
    return get_sample_by_label()


@app.get("/sample-legitimate-transaction", response_model=SampleTransactionResponse)
def get_sample_legitimate_transaction():
    return get_sample_by_label(0)


@app.get("/sample-fraud-transaction", response_model=SampleTransactionResponse)
def get_sample_fraud_transaction():
    return get_sample_by_label(1)


@app.post("/predict", response_model=PredictionResponse)
def predict(transaction: TransactionInput):
    return create_prediction_response(
        transaction.model_dump(),
        "Advanced/debugging mode: caller supplied the complete engineered 18-feature model vector.",
    )


@app.post("/predict-user", response_model=PredictionResponse)
def predict_user(transaction: UserTransactionInput):
    model_features = build_manual_model_features(transaction)
    has_history = any(
        value is not None
        for value in (
            transaction.previous_amount,
            transaction.previous_transaction_count,
            transaction.historical_avg_amount,
            transaction.time_since_previous_hours,
        )
    )

    return create_prediction_response(
        model_features,
        (
            "User input demo/test mode: readable fields were encoded with the training "
            "categorical mappings. "
            + (
                "Supplied prior-transaction history was used; merchant distance and any "
                "history values left blank use training-set medians."
                if has_history
                else "History and merchant-distance features use training-set medians "
                "because no prior history was supplied."
            )
        ),
    )


@app.post("/predict-demo", response_model=PredictionResponse)
def predict_demo(transaction: TransactionInput):
    return create_prediction_response(
        transaction.model_dump(),
        "Dataset demo sample: actual processed test-row features, including historical features, were used.",
    )
