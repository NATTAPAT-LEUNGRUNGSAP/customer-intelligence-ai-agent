import pandas as pd


def build_customer_features(df: pd.DataFrame, snapshot_date=None) -> pd.DataFrame:
    snapshot = pd.Timestamp(snapshot_date) if snapshot_date else df["InvoiceDate"].max() + pd.Timedelta(days=1)
    base = df.groupby("CustomerID").agg(
        last_purchase=("InvoiceDate", "max"),
        first_purchase=("InvoiceDate", "min"),
        frequency=("InvoiceNo", "nunique"),
        monetary=("Revenue", "sum"),
        units=("Quantity", "sum"),
        unique_products=("StockCode", "nunique"),
    )
    base["recency"] = (snapshot - base["last_purchase"]).dt.days.clip(lower=0)
    base["tenure"] = (snapshot - base["first_purchase"]).dt.days.clip(lower=1)
    base["avg_order_value"] = base["monetary"] / base["frequency"].clip(lower=1)
    base["purchase_rate"] = base["frequency"] / base["tenure"] * 30
    return base.reset_index()

