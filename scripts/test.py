from pathlib import Path

import pandas as pd


def main():
    # Resolve the project root from this script: <project>/scripts/test.py
    project_root = Path(__file__).resolve().parent.parent
    dataset_path = project_root / "data" / "raw" / "fraudTrain.csv"

    if not dataset_path.is_file():
        print(f"\nERROR: Training dataset not found.\nExpected file: {dataset_path}\n")
        return 1

    try:
        df = pd.read_csv(dataset_path)
    except FileNotFoundError:
        print(f"\nERROR: Training dataset not found.\nExpected file: {dataset_path}\n")
        return 1

    print("=" * 80)
    print("                 FRAUD TRAINING DATASET — QUICK OVERVIEW")
    print("=" * 80)
    print(f"\nFile:   {dataset_path.name}")
    print(f"Shape:  {df.shape[0]:,} rows × {df.shape[1]:,} columns")

    print("\nCOLUMNS")
    print("-" * 80)
    for number, column in enumerate(df.columns, start=1):
        print(f"{number:>3}. {column}")

    print("\nFIRST 5 ROWS")
    print("-" * 80)
    with pd.option_context(
        "display.max_columns", None,
        "display.width", None,
        "display.max_colwidth", None,
    ):
        print(df.head(5).to_string(index=False))
    print("\n" + "=" * 80)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
