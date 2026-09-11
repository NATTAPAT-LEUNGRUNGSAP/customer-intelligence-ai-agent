from pathlib import Path

import pandas as pd

from src.analyst import parse_analyst_intent
from src.business_tools import CustomerIntelligenceService
from src.contracts import TOOL_CONTRACTS
from src.data import load_transactions_with_report
from src.predictive import (
    ForecastBundle, build_temporal_customer_dataset, forecast_revenue,
    train_revenue_forecaster,
)


ROOT = Path(__file__).resolve().parents[1]


def test_temporal_labels_do_not_leak_future_revenue_into_features():
    rows = []
    start = pd.Timestamp("2024-01-01")
    for customer in range(1, 5):
        days = range(0, 211, 7) if customer < 4 else range(0, 91, 7)
        for day in days:
            rows.append({
                "CustomerID": str(customer), "InvoiceNo": f"{customer}-{day}",
                "InvoiceDate": start + pd.Timedelta(days=day), "StockCode": "A",
                "Quantity": 1, "Revenue": 10.0,
            })
    # This purchase is after the first snapshot and must affect its label, not its monetary feature.
    rows.append({
        "CustomerID": "1", "InvoiceNo": "future-large", "InvoiceDate": start + pd.Timedelta(days=100),
        "StockCode": "B", "Quantity": 1, "Revenue": 10_000.0,
    })
    dataset = build_temporal_customer_dataset(
        pd.DataFrame(rows), horizon_days=30, target="repeat_purchase",
        minimum_history_days=90, snapshot_step_days=30,
    )
    first = dataset[(dataset.CustomerID == "1")
                    & (dataset.snapshot_date == dataset.snapshot_date.min())].iloc[0]
    assert first.label == 1
    assert first.monetary == 130.0


def test_rule_parser_routes_predictive_questions():
    churn, _, _ = parse_analyst_intent(
        "ลูกค้ากลุ่มไหนเสี่ยง churn มากกว่า 60%", provider="Rules only"
    )
    repeat, _, _ = parse_analyst_intent(
        "กลุ่ม loyalty มีโอกาสซื้อซ้ำภายใน 30 วันเท่าไร", provider="Rules only"
    )
    forecast, _, _ = parse_analyst_intent(
        "พยากรณ์รายได้ 4 สัปดาห์ข้างหน้า", provider="Rules only"
    )
    assert churn.action == "churn_prediction" and churn.probability_threshold == .60
    assert repeat.action == "repeat_purchase_prediction" and repeat.segment == "Loyal High Value"
    assert forecast.action == "revenue_forecast" and forecast.forecast_weeks == 4


def test_four_month_request_is_capped_to_supported_weekly_horizon():
    forecast, _, _ = parse_analyst_intent(
        "พยากรณ์รายได้ 4 เดือนข้างหน้า", provider="Rules only"
    )
    assert forecast.action == "revenue_forecast"
    assert forecast.forecast_weeks == 12


def test_predictive_mcp_business_tools_return_valid_aggregate_outputs():
    service = CustomerIntelligenceService(
        database_url="", data_path=ROOT / "data" / "sample_transactions.csv"
    )
    churn = service.execute("predict_churn", {
        "segment": None, "probability_threshold": .60,
    })
    repeat = service.execute("predict_repeat_purchase", {
        "segment": "loyalty", "probability_threshold": .60,
    })
    forecast = service.execute("forecast_revenue", {"weeks": 4})
    explanation = service.execute("explain_predictive_model", {"model": "churn"})

    assert churn["prediction_horizon_days"] == 90
    assert repeat["prediction_horizon_days"] == 30
    assert repeat["resolved_segment"].startswith("Loyal High Value")
    assert len(forecast["forecast"]) == 4
    assert forecast["validation"]["validation_weeks"] >= 4
    validation = forecast["validation"]
    assert validation["selected_method"] in {"ridge", "naive_last", "rolling_mean_4"}
    assert validation["mae"] == min(
        validation["ridge_mae"], validation["baseline_mae"],
        validation["rolling_mean_4_mae"],
    )
    assert explanation["global_drivers"]
    assert "CustomerID" not in str({"churn": churn, "repeat": repeat})


def test_predictive_tools_are_in_canonical_registry():
    assert {
        "predict_churn", "predict_repeat_purchase", "forecast_revenue",
        "explain_predictive_model",
    } <= set(TOOL_CONTRACTS)


def test_recursive_forecast_replaces_infinity_before_next_lag():
    class InfiniteModel:
        def predict(self, frame):
            assert frame.replace([float("inf"), float("-inf")], pd.NA).notna().all().all()
            return [float("inf")]

    history = pd.Series(
        [100.0] * 13,
        index=pd.date_range("2024-01-07", periods=13, freq="W-SUN"),
    )
    bundle = ForecastBundle(
        model=InfiniteModel(), history=history, metrics={}, residual_band=10.0,
        drivers=[], selected_method="ridge",
    )
    points = forecast_revenue(bundle, 4)
    assert len(points) == 4
    assert all(point["predicted_revenue_gbp"] == 200.0 for point in points)


def test_revenue_forecaster_drops_the_trailing_incomplete_week():
    transactions, _ = load_transactions_with_report(
        ROOT / "data" / "sample_transactions.csv"
    )
    bundle = train_revenue_forecaster(transactions)
    assert bundle.metrics["incomplete_week_dropped"] is True
    assert bundle.history.index[-1] < transactions.InvoiceDate.max().normalize()
