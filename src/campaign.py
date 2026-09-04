import pandas as pd


def recommend_campaign(segment: pd.DataFrame) -> dict:
    profile = {
        "customers": int(segment.CustomerID.nunique()),
        "revenue": float(segment.monetary.sum()),
        "aov": float(segment.avg_order_value.mean()),
        "recency": float(segment.recency.mean()),
    }
    if profile["recency"] >= 90:
        strategy = ("Win-back", "Test a reminder-led win-back against a modest incentive; exact incentive requires margin approval", "Email + LINE")
    elif segment.frequency.mean() >= 5:
        strategy = ("Retention", "Test recognition or early-access benefits; exact benefit requires offer inventory", "LINE + App")
    else:
        strategy = ("Second purchase", "Test a second-purchase reminder; product and threshold require catalog and margin data", "Email + Retargeting")
    return {"profile": profile, "objective": strategy[0], "offer": strategy[1], "channel": strategy[2]}


def simulate_campaign(customers: int, baseline_conversion: float, expected_conversion: float,
                      average_order_value: float, discount_rate: float = 0) -> dict:
    baseline_revenue = customers * baseline_conversion * average_order_value
    gross_revenue = customers * expected_conversion * average_order_value
    net_revenue = gross_revenue * (1 - discount_rate)
    return {
        "baseline_revenue": baseline_revenue,
        "campaign_revenue_after_discount": net_revenue,
        "incremental_revenue": net_revenue - baseline_revenue,
        "expected_orders": customers * expected_conversion,
    }
