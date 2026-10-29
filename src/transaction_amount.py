import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

df = pd.read_csv("data/raw/fraudTrain.csv")
amounts = df["amt"].dropna()

limit = amounts.quantile(0.99)

plt.figure(figsize=(9, 5))
plt.hist(amounts[amounts <= limit], bins=50, edgecolor="black")

plt.title("Distribution of Transaction Amounts (Up to 99th Percentile)")
plt.xlabel("Transaction Amount (amt)")
plt.ylabel("Number of Transactions")
plt.tight_layout()

Path("reports").mkdir(exist_ok=True)
plt.savefig("reports/transaction_amount_distribution.png", dpi=300)
plt.show()