"""Evidence-based segment product candidates from transaction history."""
from __future__ import annotations

import pandas as pd

NON_PRODUCT_PATTERN = (
    r"POSTAGE|CARRIAGE|BANK CHARGES|AMAZON FEE|DOTCOM POSTAGE|MANUAL|"
    r"ADJUSTMENT|DISCOUNT|CRUK COMMISSION"
)


def rank_segment_products(transactions: pd.DataFrame, segment_customers: pd.DataFrame,
                          top_n: int = 5, ranking: str = "popular") -> pd.DataFrame:
    """Rank products using observed transactions.

    ``popular`` surfaces products with broad segment adoption and revenue.
    ``distinctive`` only surfaces products whose segment penetration exceeds
    the customer-base penetration (lift > 1).
    """
    if ranking not in {"popular", "distinctive"}:
        raise ValueError("ranking must be 'popular' or 'distinctive'")
    required = {"CustomerID", "StockCode", "Description", "InvoiceNo", "Quantity", "Revenue"}
    if not required.issubset(transactions.columns) or segment_customers.empty:
        return pd.DataFrame()
    tx = transactions.copy()
    tx["Description"] = tx["Description"].fillna("").astype(str).str.strip()
    tx = tx[(tx.Description != "") & ~tx.Description.str.upper().str.contains(NON_PRODUCT_PATTERN, regex=True)]
    if tx.empty:
        return pd.DataFrame()

    customer_ids = set(segment_customers.CustomerID.astype(str))
    segment_tx = tx[tx.CustomerID.astype(str).isin(customer_ids)]
    if segment_tx.empty:
        return pd.DataFrame()

    keys = ["StockCode", "Description"]
    segment = segment_tx.groupby(keys).agg(
        buyers=("CustomerID", "nunique"), orders=("InvoiceNo", "nunique"),
        units=("Quantity", "sum"), revenue=("Revenue", "sum"),
    )
    overall = tx.groupby(keys).CustomerID.nunique().rename("overall_buyers")
    ranked = segment.join(overall, how="left").reset_index()
    min_buyers = max(3, round(len(customer_ids) * .005))
    ranked = ranked[ranked.buyers >= min_buyers].copy()
    if ranked.empty:
        return ranked
    ranked["segment_penetration"] = ranked.buyers / len(customer_ids)
    ranked["overall_penetration"] = ranked.overall_buyers / tx.CustomerID.nunique()
    ranked["lift"] = ranked.segment_penetration / ranked.overall_penetration.clip(lower=1e-9)
    ranked["popular_score"] = (
        .45 * ranked.buyers.rank(pct=True)
        + .30 * ranked.revenue.rank(pct=True)
        + .25 * ranked.orders.rank(pct=True)
    )
    ranked["distinctive_score"] = (
        .55 * ranked.lift.rank(pct=True)
        + .25 * ranked.buyers.rank(pct=True)
        + .20 * ranked.revenue.rank(pct=True)
    )
    if ranking == "distinctive":
        # The displayed value is rounded to two decimals, so require the
        # rounded value itself to remain above 1.00x.
        ranked = ranked[ranked.lift.round(2) > 1.0]
        score = "distinctive_score"
    else:
        score = "popular_score"
    return ranked.sort_values([score, "buyers"], ascending=False).head(top_n).reset_index(drop=True)


def product_records(ranked: pd.DataFrame) -> list[dict]:
    if ranked.empty:
        return []
    return [
        {
            "stock_code": str(row.StockCode), "product_name": str(row.Description),
            "segment_buyers": int(row.buyers), "segment_revenue_gbp": round(float(row.revenue), 2),
            "segment_lift": round(float(row.lift), 2),
        }
        for row in ranked.itertuples()
    ]
