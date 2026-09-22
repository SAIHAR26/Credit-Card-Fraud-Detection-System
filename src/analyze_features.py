import pandas as pd
from pathlib import Path

RAW_DIR = Path("data/raw")

train = pd.read_csv(RAW_DIR / "fraudTrain.csv")
test = pd.read_csv(RAW_DIR / "fraudTest.csv")

print("=" * 70)
print("FEATURE CARDINALITY ANALYSIS")
print("=" * 70)

print("\n--- DATASET SIZES ---")
print(f"Train: {len(train):,}")
print(f"Test : {len(test):,}")

print("\n--- UNIQUE VALUES ---")

columns = [
    "cc_num",
    "merchant",
    "category",
    "city",
    "state",
    "job",
    "zip",
    "gender"
]

for col in columns:
    print(f"{col:12} Train: {train[col].nunique():,} | "
          f"Test: {test[col].nunique():,}")

print("\n--- CARD OVERLAP ---")

train_cards = set(train["cc_num"].unique())
test_cards = set(test["cc_num"].unique())

overlap = train_cards.intersection(test_cards)

print(f"Unique train cards : {len(train_cards):,}")
print(f"Unique test cards  : {len(test_cards):,}")
print(f"Cards in both      : {len(overlap):,}")

print("\n--- TRANSACTIONS PER CARD ---")

card_counts = train["cc_num"].value_counts()

print(f"Average : {card_counts.mean():.2f}")
print(f"Median  : {card_counts.median():.2f}")
print(f"Minimum : {card_counts.min()}")
print(f"Maximum : {card_counts.max()}")

print("\nTop 10 cards by transaction count:")
print(card_counts.head(10).to_string())

print("\n--- CATEGORIES ---")
print(train["category"].value_counts().to_string())

print("\n--- STATES ---")
print(f"Number of states: {train['state'].nunique()}")
print(train["state"].value_counts().head(20).to_string())

print("\n--- GENDER ---")
print(train["gender"].value_counts().to_string())

print("\n--- TEMPORAL RANGE ---")

train_time = pd.to_datetime(train["trans_date_trans_time"])
test_time = pd.to_datetime(test["trans_date_trans_time"])

print(f"Train start: {train_time.min()}")
print(f"Train end  : {train_time.max()}")
print(f"Test start : {test_time.min()}")
print(f"Test end   : {test_time.max()}")

print("\n--- SORTING CHECK ---")

print(
    "Train chronological:",
    train_time.is_monotonic_increasing
)

print(
    "Test chronological:",
    test_time.is_monotonic_increasing
)

print("\n--- JOB CARDINALITY ---")
print(f"Unique jobs: {train['job'].nunique():,}")

print("\n--- CITY CARDINALITY ---")
print(f"Unique cities: {train['city'].nunique():,}")

print("\n--- ZIP CARDINALITY ---")
print(f"Unique ZIP codes: {train['zip'].nunique():,}")

print("\nAnalysis complete.")