from pathlib import Path
import numpy as np
import pandas as pd

rng = np.random.default_rng(42)
rows = []
start = pd.Timestamp("2024-01-01")
for customer in range(10000, 10800):
    orders = int(rng.integers(1, 14))
    for order in range(orders):
        date = start + pd.Timedelta(days=int(rng.integers(0, 365)))
        for line in range(int(rng.integers(1, 4))):
            rows.append({
                "InvoiceNo": f"{customer}-{order}", "StockCode": f"SKU{rng.integers(1, 101):03d}",
                "Description": "Synthetic retail product", "Quantity": int(rng.integers(1, 8)),
                "InvoiceDate": date, "UnitPrice": round(float(rng.lognormal(2.1, .65)), 2),
                "CustomerID": customer, "Country": "United Kingdom",
            })
out = Path(__file__).resolve().parents[1] / "data" / "sample_transactions.csv"
out.parent.mkdir(exist_ok=True)
pd.DataFrame(rows).to_csv(out, index=False)
print(f"Created {out} with {len(rows):,} rows")

