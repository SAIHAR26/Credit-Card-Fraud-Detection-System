import pandas as pd
import numpy as np
from pathlib import Path

PROCESSED_DIR = Path("data/processed")

train = pd.read_csv(PROCESSED_DIR / "fraud_train_processed.csv")
test = pd.read_csv(PROCESSED_DIR / "fraud_test_processed.csv")

print("=" * 70)
print("PREPROCESSING VALIDATION")
print("=" * 70)

# Shapes
print("\n--- SHAPES ---")
print("Train:", train.shape)
print("Test :", test.shape)

# Columns
print("\n--- COLUMN MATCH ---")
train_features = set(train.columns) - {"is_fraud"}
test_features = set(test.columns)

print("Same feature columns:", train_features == test_features)

# Missing values
print("\n--- MISSING VALUES ---")
print("Train:", train.isnull().sum().sum())
print("Test :", test.isnull().sum().sum())

# Infinite values
print("\n--- INFINITE VALUES ---")
print("Train:", np.isinf(train.select_dtypes(include=np.number)).sum().sum())
print("Test :", np.isinf(test.select_dtypes(include=np.number)).sum().sum())

# Target
print("\n--- TARGET ---")
print(train["is_fraud"].value_counts())

# Data types
print("\n--- DATA TYPES ---")
print(train.dtypes)

# Feature ranges
print("\n--- NUMERIC SUMMARY ---")
print(train.describe().T[
    ["min", "max", "mean"]
].to_string())

print("\n" + "=" * 70)
print("VALIDATION COMPLETE")
print("=" * 70)