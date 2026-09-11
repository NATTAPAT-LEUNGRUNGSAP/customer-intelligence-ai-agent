"""Evidence-based segment product candidates from transaction history."""
from __future__ import annotations

import pandas as pd

NON_PRODUCT_PATTERN = (
    r"POSTAGE|CARRIAGE|BANK CHARGES|AMAZON FEE|DOTCOM POSTAGE|MANUAL|"
    r"ADJUSTMENT|DISCOUNT|CRUK COMMISSION"
)


def _product_transactions(transactions: pd.DataFrame) -> pd.DataFrame:
    """Return sale rows that represent merchandise rather than service lines."""
    required = {"CustomerID", "StockCode", "Description", "InvoiceNo", "Quantity", "Revenue"}
    if not required.issubset(transactions.columns):
        return pd.DataFrame()
    tx = transactions.copy()
    tx["Description"] = tx["Description"].fillna("").astype(str).str.strip()
    return tx[
        (tx.Description != "")
        & ~tx.Description.str.upper().str.contains(NON_PRODUCT_PATTERN, regex=True)
    ]


def rank_overall_products(transactions: pd.DataFrame, top_n: int = 10,
                          ranking: str = "revenue") -> pd.DataFrame:
    """Rank observed products by one explicit, reproducible business metric."""
    allowed = {"revenue", "units", "orders", "customers"}
    if ranking not in allowed:
        raise ValueError(f"ranking must be one of {sorted(allowed)}")
    tx = _product_transactions(transactions)
    if tx.empty:
        return pd.DataFrame()
    ranked = tx.groupby(["StockCode", "Description"]).agg(
        revenue=("Revenue", "sum"),
        units=("Quantity", "sum"),
        orders=("InvoiceNo", "nunique"),
        customers=("CustomerID", "nunique"),
    ).reset_index()
    return ranked.sort_values(
        [ranking, "revenue", "orders"], ascending=False
    ).head(top_n).reset_index(drop=True)


def summarize_product_catalog(transactions: pd.DataFrame) -> dict:
    """Count observed merchandise without inventing a product taxonomy.

    Online Retail has product codes and descriptions but no category field. If
    another source supplies an explicit category column, report it; otherwise
    keep category counts null so an analyst cannot mistake inferred keywords
    for authoritative categories.
    """
    tx = _product_transactions(transactions)
    distinct_products = int(tx["StockCode"].astype(str).nunique()) if not tx.empty else 0
    distinct_names = int(tx["Description"].nunique()) if not tx.empty else 0
    category_lookup = {
        str(column).casefold().replace("_", "").replace(" ", ""): str(column)
        for column in transactions.columns
    }
    category_column = next(
        (category_lookup[name] for name in ("category", "productcategory")
         if name in category_lookup),
        None,
    )
    distinct_categories = None
    if category_column is not None:
        values = transactions[category_column].dropna().astype(str).str.strip()
        distinct_categories = int(values[values != ""].nunique())
    return {
        "distinct_products": distinct_products,
        "distinct_product_names": distinct_names,
        "category_field_available": category_column is not None,
        "category_field": category_column,
        "distinct_categories": distinct_categories,
        "category_note": (
            f"Categories are counted from the explicit '{category_column}' field."
            if category_column is not None
            else "No category field exists in this dataset; categories were not guessed from product names."
        ),
    }


def rank_segment_products(transactions: pd.DataFrame, segment_customers: pd.DataFrame,
                          top_n: int = 5, ranking: str = "popular") -> pd.DataFrame:
    """Rank products using observed transactions.

    ``popular`` surfaces products with broad segment adoption and revenue.
    ``distinctive`` only surfaces products whose segment penetration exceeds
    the customer-base penetration (lift > 1).
    """
    if ranking not in {"popular", "distinctive"}:
        raise ValueError("ranking must be 'popular' or 'distinctive'")
    if segment_customers.empty:
        return pd.DataFrame()
    tx = _product_transactions(transactions)
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


def overall_product_records(ranked: pd.DataFrame) -> list[dict]:
    if ranked.empty:
        return []
    return [
        {
            "rank": index,
            "stock_code": str(row.StockCode),
            "product_name": str(row.Description),
            "revenue_gbp": round(float(row.revenue), 2),
            "units": round(float(row.units), 2),
            "orders": int(row.orders),
            "customers": int(row.customers),
        }
        for index, row in enumerate(ranked.itertuples(), start=1)
    ]
