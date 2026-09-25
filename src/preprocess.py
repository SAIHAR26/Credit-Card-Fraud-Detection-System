import pandas as pd
import numpy as np
from pathlib import Path


# --------------------------------------------------
# PATHS
# --------------------------------------------------

RAW_DIR = Path("data/raw")
PROCESSED_DIR = Path("data/processed")

PROCESSED_DIR.mkdir(parents=True, exist_ok=True)


# --------------------------------------------------
# LOAD DATA
# --------------------------------------------------

print("=" * 70)
print("LOADING DATA")
print("=" * 70)

train = pd.read_csv(RAW_DIR / "fraudTrain.csv")
test = pd.read_csv(RAW_DIR / "fraudTest.csv")

print(f"Train shape: {train.shape}")
print(f"Test shape : {test.shape}")


# --------------------------------------------------
# DATETIME CONVERSION
# --------------------------------------------------

print("\nConverting timestamps...")

train["trans_date_trans_time"] = pd.to_datetime(
    train["trans_date_trans_time"]
)

test["trans_date_trans_time"] = pd.to_datetime(
    test["trans_date_trans_time"]
)


# --------------------------------------------------
# SORT CHRONOLOGICALLY
# --------------------------------------------------

print("Sorting transactions chronologically...")

train = train.sort_values(
    "trans_date_trans_time"
).reset_index(drop=True)

test = test.sort_values(
    "trans_date_trans_time"
).reset_index(drop=True)


# --------------------------------------------------
# TIME FEATURES
# --------------------------------------------------

print("Creating time features...")

for df in [train, test]:

    df["hour"] = df["trans_date_trans_time"].dt.hour

    df["day_of_week"] = (
        df["trans_date_trans_time"].dt.dayofweek
    )

    df["month"] = (
        df["trans_date_trans_time"].dt.month
    )

    df["day"] = (
        df["trans_date_trans_time"].dt.day
    )

    df["is_weekend"] = (
        df["day_of_week"] >= 5
    ).astype(int)


# --------------------------------------------------
# AGE FEATURE
# --------------------------------------------------

print("Creating age feature...")

for df in [train, test]:

    df["dob"] = pd.to_datetime(df["dob"])

    df["age"] = (
        (df["trans_date_trans_time"] - df["dob"])
        .dt.days / 365.25
    )

    df["age"] = df["age"].clip(lower=18, upper=100)


# --------------------------------------------------
# CUSTOMER-MERCHANT DISTANCE
# --------------------------------------------------

print("Creating customer-merchant distance...")

def haversine_distance(lat1, lon1, lat2, lon2):

    R = 6371.0

    lat1 = np.radians(lat1)
    lon1 = np.radians(lon1)
    lat2 = np.radians(lat2)
    lon2 = np.radians(lon2)

    dlat = lat2 - lat1
    dlon = lon2 - lon1

    a = (
        np.sin(dlat / 2) ** 2
        + np.cos(lat1)
        * np.cos(lat2)
        * np.sin(dlon / 2) ** 2
    )

    c = 2 * np.arcsin(np.sqrt(a))

    return R * c


for df in [train, test]:

    df["merchant_distance_km"] = haversine_distance(
        df["lat"],
        df["long"],
        df["merch_lat"],
        df["merch_long"]
    )


# --------------------------------------------------
# LEAKAGE-SAFE CARD BEHAVIOR FEATURES
# --------------------------------------------------

print("Creating card behavioral features...")

def create_card_features(df):

    df = df.sort_values(
        "trans_date_trans_time"
    ).copy()

    # Previous transaction amount
    df["previous_amount"] = (
        df.groupby("cc_num")["amt"]
        .shift(1)
    )

    # Number of previous transactions
    df["previous_transaction_count"] = (
        df.groupby("cc_num")
        .cumcount()
    )

    # Historical amount statistics
    shifted_amount = (
        df.groupby("cc_num")["amt"]
        .shift(1)
    )

    df["historical_avg_amount"] = (
        shifted_amount
        .groupby(df["cc_num"])
        .expanding()
        .mean()
        .reset_index(level=0, drop=True)
    )

    # Time since previous transaction
    previous_time = (
        df.groupby("cc_num")["trans_date_trans_time"]
        .shift(1)
    )

    df["time_since_previous_hours"] = (
        (
            df["trans_date_trans_time"]
            - previous_time
        ).dt.total_seconds() / 3600
    )

    # Difference from historical average
    df["amount_deviation"] = (
        df["amt"] - df["historical_avg_amount"]
    )

    return df


train = create_card_features(train)


# --------------------------------------------------
# TEST CARD FEATURES
# --------------------------------------------------

print("Creating test historical features...")

# For test transactions, we need the historical
# information from the training period first.

card_history = (
    train.groupby("cc_num")
    .agg(
        historical_avg_amount=("amt", "mean"),
        previous_transaction_count=("amt", "count"),
        previous_amount=("amt", "last"),
        last_transaction_time=(
            "trans_date_trans_time",
            "last"
        )
    )
    .reset_index()
)

test = test.merge(
    card_history,
    on="cc_num",
    how="left"
)

test["time_since_previous_hours"] = (
    (
        test["trans_date_trans_time"]
        - test["last_transaction_time"]
    ).dt.total_seconds() / 3600
)

test["amount_deviation"] = (
    test["amt"] - test["historical_avg_amount"]
)


# --------------------------------------------------
# DROP UNNECESSARY COLUMNS
# --------------------------------------------------

print("Dropping unnecessary columns...")

columns_to_drop = [
    "Unnamed: 0",
    "first",
    "last",
    "street",
    "city",
    "zip",
    "job",
    "dob",
    "trans_num",
    "unix_time",
    "lat",
    "long",
    "merch_lat",
    "merch_long",
    "last_transaction_time"
]

train.drop(
    columns=columns_to_drop,
    inplace=True,
    errors="ignore"
)

test.drop(
    columns=columns_to_drop,
    inplace=True,
    errors="ignore"
)


# --------------------------------------------------
# DROP RAW CARD NUMBER
# --------------------------------------------------

train.drop(columns=["cc_num"], inplace=True)
test.drop(columns=["cc_num"], inplace=True)


# --------------------------------------------------
# CATEGORICAL ENCODING
# --------------------------------------------------

print("Encoding categorical features...")

categorical_columns = [
    "merchant",
    "category",
    "state",
    "gender"
]

# Convert categories to category dtype
for column in categorical_columns:

    combined = pd.concat(
        [train[column], test[column]],
        axis=0
    ).astype("category")

    categories = combined.cat.categories

    train[column] = pd.Categorical(
        train[column],
        categories=categories
    ).codes

    test[column] = pd.Categorical(
        test[column],
        categories=categories
    ).codes


# --------------------------------------------------
# HANDLE MISSING VALUES
# --------------------------------------------------

print("Handling missing values...")

numeric_columns = train.select_dtypes(
    include=[np.number]
).columns

for column in numeric_columns:

    if column != "is_fraud":

        median_value = train[column].median()

        train[column] = (
            train[column]
            .fillna(median_value)
        )

        test[column] = (
            test[column]
            .fillna(median_value)
        )


# --------------------------------------------------
# SAVE DATA
# --------------------------------------------------

train_path = PROCESSED_DIR / "fraud_train_processed.csv"
test_path = PROCESSED_DIR / "fraud_test_processed.csv"

train.to_csv(
    train_path,
    index=False
)

test.to_csv(
    test_path,
    index=False
)


# --------------------------------------------------
# FINAL SUMMARY
# --------------------------------------------------

print("\n" + "=" * 70)
print("PREPROCESSING COMPLETE")
print("=" * 70)

print(f"Processed train shape: {train.shape}")
print(f"Processed test shape : {test.shape}")

print("\nTrain columns:")
print(train.columns.tolist())

print("\nMissing values in train:")
print(train.isnull().sum().sum())

print("\nMissing values in test:")
print(test.isnull().sum().sum())

print("\nFraud distribution:")
print(train["is_fraud"].value_counts())

print("\nSaved:")
print(train_path)
print(test_path)

print("=" * 70)