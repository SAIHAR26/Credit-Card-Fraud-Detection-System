"""Plot transaction counts from the project's train and test CSV files."""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


# Resolve paths relative to the project root, so this works regardless of the
# terminal's current directory when invoked as ``python src/dataset_distribution.py``.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data" / "raw"
OUTPUT = PROJECT_ROOT / "reports" / "dataset_size_class_distribution.png"


def load_class_counts(path: Path, target_column: str = "is_fraud") -> tuple[int, int]:
    """Return legitimate and fraud counts from one CSV, reading only its label."""
    if not path.is_file():
        raise FileNotFoundError(
            f"Dataset not found: {path}\n"
            "Place the dataset in data/raw or update DATA_DIR in this script."
        )

    labels = pd.read_csv(path, usecols=[target_column])[target_column]
    counts = labels.value_counts()
    return int(counts.get(0, 0)), int(counts.get(1, 0))


def main() -> None:
    train_legitimate, train_fraud = load_class_counts(DATA_DIR / "fraudTrain.csv")
    test_legitimate, test_fraud = load_class_counts(DATA_DIR / "fraudTest.csv")

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(11, 5))

    sizes = [train_legitimate + train_fraud, test_legitimate + test_fraud]
    axes[0].bar(["Training", "Testing"], sizes, edgecolor="black")
    axes[0].set_title("Dataset Sizes")
    axes[0].set_ylabel("Number of Transactions")
    for index, value in enumerate(sizes):
        axes[0].text(index, value, f"{value:,}", ha="center", va="bottom")

    labels = ["Legitimate", "Fraud"]
    x = [0, 1]
    width = 0.35
    axes[1].bar(
        [value - width / 2 for value in x],
        [train_legitimate, train_fraud],
        width,
        label="Training",
        edgecolor="black",
    )
    axes[1].bar(
        [value + width / 2 for value in x],
        [test_legitimate, test_fraud],
        width,
        label="Testing",
        edgecolor="black",
    )
    axes[1].set_xticks(x, labels)
    axes[1].set_title("Class Distribution")
    axes[1].set_ylabel("Number of Transactions")
    axes[1].legend()

    fig.suptitle("Dataset Size and Class Distribution")
    fig.tight_layout()
    fig.savefig(OUTPUT, dpi=300, bbox_inches="tight")
    plt.close(fig)

    print(f"Train: {train_legitimate + train_fraud:,} rows "
          f"({train_legitimate:,} legitimate, {train_fraud:,} fraud)")
    print(f"Test:  {test_legitimate + test_fraud:,} rows "
          f"({test_legitimate:,} legitimate, {test_fraud:,} fraud)")
    print(f"Saved chart: {OUTPUT}")


if __name__ == "__main__":
    main()
