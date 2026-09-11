"""Customer Intelligence MCP server.

The server exposes aggregate, read-only analytics. It never accepts raw SQL,
never writes to PostgreSQL, and never returns customer identifiers.
"""
from __future__ import annotations

import argparse
from typing import Annotated, Literal, TypedDict

from pydantic import Field
from mcp.server import MCPServer
from mcp.types import ToolAnnotations

from src.business_tools import CustomerIntelligenceService


class SegmentSummary(TypedDict):
    segment: str
    customers: int
    mean_recency_days: float
    mean_frequency: float
    mean_customer_value_gbp: float
    mean_order_value_gbp: float


class SegmentSummaryResult(TypedDict):
    source: str
    segments: list[SegmentSummary]


class CustomerFilterResult(TypedDict):
    source: str
    resolved_segment: str | None
    matched_customers: int
    customer_share: float
    observed_customer_value_gbp: float
    mean_recency_days: float | None
    mean_frequency: float | None
    high_value_threshold_gbp: float | None


class ProductEvidence(TypedDict):
    stock_code: str
    product_name: str
    segment_buyers: int
    segment_revenue_gbp: float
    segment_lift: float


class SegmentProductsResult(TypedDict):
    source: str
    requested_segment: str
    resolved_segment: str
    ranking: str
    products: list[ProductEvidence]
    evidence_note: str


class OverallProduct(TypedDict):
    rank: int
    stock_code: str
    product_name: str
    revenue_gbp: float
    units: float
    orders: int
    customers: int


class TopProductsResult(TypedDict):
    source: str
    ranking: str
    limit: int
    products: list[OverallProduct]
    evidence_note: str


class ProductCatalogSummaryResult(TypedDict):
    source: str
    distinct_products: int
    distinct_product_names: int
    category_field_available: bool
    category_field: str | None
    distinct_categories: int | None
    category_note: str


class CampaignRecommendationResult(TypedDict):
    source: str
    objective: str
    recommended_segment: str
    customers: int
    observed_segment_revenue_gbp: float
    channel: str
    products: list[ProductEvidence]
    launch_requirements: list[str]


class CampaignSimulationResult(TypedDict):
    baseline_revenue: float
    campaign_revenue_after_discount: float
    incremental_revenue: float
    expected_orders: float
    causal_warning: str


class ClassificationValidation(TypedDict):
    roc_auc: float | None
    average_precision: float | None
    brier_score: float | None
    validation_rows: int
    validation_positive_rate: float
    temporal_split: str
    training_snapshots: int
    validation_snapshots: int


class PredictionSegment(TypedDict):
    segment: str
    customers: int
    mean_probability: float
    flagged_customers: int


class CustomerPredictionResult(TypedDict):
    source: str
    model: str
    prediction_horizon_days: int
    resolved_segment: str | None
    scored_customers: int
    flagged_customers: int
    flagged_share: float
    mean_probability: float
    threshold: float
    probability_name: str
    segment_scores: list[PredictionSegment]
    validation: ClassificationValidation
    data_warning: str


class ForecastPoint(TypedDict):
    week_ending: str
    predicted_revenue_gbp: float
    lower_bound_gbp: float
    upper_bound_gbp: float


class ForecastValidation(TypedDict):
    mae: float
    rmse: float
    mape: float | None
    baseline_mae: float
    ridge_mae: float
    rolling_mean_4_mae: float
    selected_method: str
    normalized_mae: float
    reliability: str
    model_beats_baseline: bool
    validation_weeks: int
    temporal_split: str
    incomplete_week_dropped: bool


class RevenueForecastResult(TypedDict):
    source: str
    model: str
    frequency: str
    history_weeks: int
    forecast: list[ForecastPoint]
    validation: ForecastValidation
    forecast_warning: str


class ModelDriver(TypedDict):
    feature: str
    direction: str
    importance: float


class ModelMetric(TypedDict):
    name: str
    value: float | str | bool | None


class ModelExplanationResult(TypedDict):
    source: str
    model: str
    target: str
    prediction_horizon: str | None
    metrics: list[ModelMetric]
    global_drivers: list[ModelDriver]
    limitations: list[str]


READ_ONLY = ToolAnnotations(read_only_hint=True, open_world_hint=False)
service = CustomerIntelligenceService()
mcp = MCPServer("customer-intelligence")


@mcp.tool(title="Get segment summary", annotations=READ_ONLY)
def get_segment_summary(
    segment: Annotated[
        str | None,
        Field(description="Segment name, or null to return every behavioral segment."),
    ],
) -> SegmentSummaryResult:
    """Return aggregate RFM profiles without exposing customer identifiers."""
    return service.execute("get_segment_summary", {"segment": segment})


@mcp.tool(title="Compare segments", annotations=READ_ONLY)
def compare_segments() -> SegmentSummaryResult:
    """Compare aggregate RFM profiles for every behavioral segment."""
    return service.execute("compare_segments", {})


@mcp.tool(title="Filter customer audience", annotations=READ_ONLY)
def filter_customers(
    segment: Annotated[str | None, Field(description="Behavioral or rule segment, or null.")],
    minimum_recency_days: Annotated[
        int | None,
        Field(ge=0, le=3650, description="Minimum days since last purchase, or null."),
    ],
    high_value: Annotated[
        bool,
        Field(description="When true, require monetary value at or above the customer 70th percentile."),
    ],
) -> CustomerFilterResult:
    """Count and summarize an approved audience; do not return customer IDs."""
    return service.execute("filter_customers", {
        "segment": segment,
        "minimum_recency_days": minimum_recency_days,
        "high_value": high_value,
    })


@mcp.tool(title="Get segment products", annotations=READ_ONLY)
def get_segment_products(
    segment: Annotated[str, Field(min_length=1, max_length=100, description="Segment name or alias.")],
    limit: Annotated[int, Field(ge=1, le=20, description="Maximum products to return.")],
    ranking: Annotated[
        Literal["popular", "distinctive"],
        Field(description="popular ranks adoption/revenue; distinctive ranks segment lift above one."),
    ],
) -> SegmentProductsResult:
    """Rank products using observed transactions for the requested segment."""
    return service.execute("get_segment_products", {
        "segment": segment, "limit": limit, "ranking": ranking,
    })


@mcp.tool(title="Get top products", annotations=READ_ONLY)
def get_top_products(
    limit: Annotated[int, Field(ge=1, le=100, description="Maximum products to return.")],
    ranking: Annotated[
        Literal["revenue", "units", "orders", "customers"],
        Field(description="Exact metric used to rank products."),
    ],
) -> TopProductsResult:
    """Rank observed products overall using one explicit business metric."""
    return service.execute("get_top_products", {"limit": limit, "ranking": ranking})


@mcp.tool(title="Get product catalog summary", annotations=READ_ONLY)
def get_product_catalog_summary() -> ProductCatalogSummaryResult:
    """Count observed products without inventing unavailable categories."""
    return service.execute("get_product_catalog_summary", {})


@mcp.tool(title="Recommend campaign evidence", annotations=READ_ONLY)
def recommend_campaign(
    objective: Literal[
        "Retain valuable customers",
        "Reactivate lapsed customers",
        "Increase average order value",
    ],
) -> CampaignRecommendationResult:
    """Select a target and product evidence with deterministic Python scoring."""
    return service.execute("recommend_campaign", {"objective": objective})


@mcp.tool(title="Simulate campaign scenario", annotations=READ_ONLY)
def simulate_campaign(
    customers: Annotated[int, Field(ge=1, le=10_000_000)],
    baseline_conversion: Annotated[float, Field(ge=0, le=1)],
    expected_conversion: Annotated[float, Field(ge=0, le=1)],
    average_order_value: Annotated[float, Field(gt=0, le=1_000_000)],
    discount_rate: Annotated[float, Field(ge=0, le=1)],
) -> CampaignSimulationResult:
    """Calculate a planning scenario; do not interpret it as causal uplift."""
    return service.execute("simulate_campaign", {
        "customers": customers,
        "baseline_conversion": baseline_conversion,
        "expected_conversion": expected_conversion,
        "average_order_value": average_order_value,
        "discount_rate": discount_rate,
    })


@mcp.tool(title="Predict churn risk", annotations=READ_ONLY)
def predict_churn(
    segment: Annotated[str | None, Field(description="Segment name or null for every customer.")],
    probability_threshold: Annotated[float, Field(ge=0, le=1)],
) -> CustomerPredictionResult:
    """Estimate the probability of no purchase in the following 90 days."""
    return service.execute("predict_churn", {
        "segment": segment, "probability_threshold": probability_threshold,
    })


@mcp.tool(title="Predict repeat purchase", annotations=READ_ONLY)
def predict_repeat_purchase(
    segment: Annotated[str | None, Field(description="Segment name or null for every customer.")],
    probability_threshold: Annotated[float, Field(ge=0, le=1)],
) -> CustomerPredictionResult:
    """Estimate the probability of a repeat purchase in the following 30 days."""
    return service.execute("predict_repeat_purchase", {
        "segment": segment, "probability_threshold": probability_threshold,
    })


@mcp.tool(name="forecast_revenue", title="Forecast weekly revenue", annotations=READ_ONLY)
def forecast_revenue_tool(
    weeks: Annotated[int, Field(ge=1, le=12, description="Number of future weeks.")],
) -> RevenueForecastResult:
    """Forecast weekly revenue with chronological holdout evaluation."""
    return service.execute("forecast_revenue", {"weeks": weeks})


@mcp.tool(title="Explain predictive model", annotations=READ_ONLY)
def explain_predictive_model(
    model: Literal["churn", "repeat_purchase", "revenue_forecast"],
) -> ModelExplanationResult:
    """Return temporal metrics, global model drivers, and known limitations."""
    return service.execute("explain_predictive_model", {"model": model})


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the Customer Intelligence MCP server")
    parser.add_argument("--transport", choices=("stdio", "streamable-http"), default="stdio")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    if args.transport == "stdio":
        mcp.run(transport="stdio")
    else:
        mcp.run(
            transport="streamable-http",
            host=args.host,
            port=args.port,
            json_response=True,
            stateless_http=True,
        )


if __name__ == "__main__":
    main()
