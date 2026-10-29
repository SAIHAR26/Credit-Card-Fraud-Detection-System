
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

input_path = Path("reports/shap_feature_importance.csv")
output_path = Path("reports/xgboost_feature_importance.png")

df = pd.read_csv(input_path)
print("Columns found:", df.columns.tolist())

feature_col = next(
    (c for c in df.columns if c.lower() in ["feature", "features", "feature_name"]),
    df.columns[0]
)

importance_candidates = [
    c for c in df.columns
    if any(word in c.lower() for word in ["mean_abs_shap", "importance", "shap"])
    and pd.api.types.is_numeric_dtype(df[c])
]

if not importance_candidates:
    importance_candidates = [
        c for c in df.select_dtypes(include="number").columns
    ]

if not importance_candidates:
    raise ValueError("Could not find a numeric feature-importance column.")

importance_col = importance_candidates[0]

df = df[[feature_col, importance_col]].dropna()
df = df.sort_values(importance_col, ascending=False).head(10)
df = df.sort_values(importance_col)

plt.figure(figsize=(9, 6))
plt.barh(df[feature_col], df[importance_col], edgecolor="black")
plt.title("Top 10 Features by Mean Absolute SHAP Value")
plt.xlabel("Mean Absolute SHAP Value")
plt.ylabel("Feature")
plt.tight_layout()

output_path.parent.mkdir(parents=True, exist_ok=True)
plt.savefig(output_path, dpi=300, bbox_inches="tight")
plt.show()

print(f"Graph saved to: {output_path}")
