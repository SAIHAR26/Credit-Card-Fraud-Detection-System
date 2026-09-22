import pandas as pd
from pathlib import Path

RAW_DIR = Path("data/raw")

files = [
    "fraudTrain.csv",
    "fraudTest.csv",
    "creditcard.csv"
]

for filename in files:
    path = RAW_DIR / filename

    print("\n" + "=" * 80)
    print(f"DATASET: {filename}")
    print("=" * 80)

    df = pd.read_csv(path)

    print(f"\n1. Number of rows: {len(df):,}")
    print(f"2. Number of columns: {len(df.columns)}")

    print("\n3. Column names:")
    for column in df.columns:
        print(f"   - {column}")

    print("\n4. Data types:")
    print(df.dtypes.to_string())

    print("\n5. Missing values:")
    missing = df.isnull().sum()
    missing_pct = (missing / len(df) * 100).round(2)

    missing_table = pd.DataFrame({
        "Missing": missing,
        "Percentage": missing_pct
    })

    missing_table = missing_table[missing_table["Missing"] > 0]

    if missing_table.empty:
        print("   No missing values.")
    else:
        print(missing_table.to_string())

    print(f"\n6. Duplicate rows: {df.duplicated().sum():,}")

    possible_targets = [
        "is_fraud",
        "fraud",
        "Class",
        "class",
        "Fraud"
    ]

    target = next(
        (col for col in possible_targets if col in df.columns),
        None
    )

    print(f"\n7. Target/label column: {target}")

    if target:
        print("\n8. Target class distribution:")

        counts = df[target].value_counts(dropna=False)
        percentages = (
            df[target]
            .value_counts(normalize=True, dropna=False)
            * 100
        ).round(2)

        distribution = pd.DataFrame({
            "Count": counts,
            "Percentage": percentages
        })

        print(distribution.to_string())

    numerical = df.select_dtypes(include="number").columns.tolist()

    print("\n9. Numerical columns:")
    print(numerical)

    categorical = df.select_dtypes(
        include=["object", "category"]
    ).columns.tolist()

    print("\n10. Categorical columns:")
    print(categorical)

    datetime_columns = df.select_dtypes(
        include=["datetime", "datetimetz"]
    ).columns.tolist()

    possible_datetime = [
        col for col in df.columns
        if any(
            word in col.lower()
            for word in ["date", "time", "timestamp"]
        )
    ]

    print("\n11. Date/time columns:")
    print("Detected datetime dtype:", datetime_columns)
    print("Possible date/time columns:", possible_datetime)

    print("\n12. First 5 rows:")
    print(df.head().to_string())

    print()