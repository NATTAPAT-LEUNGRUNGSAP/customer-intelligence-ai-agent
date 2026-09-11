"""Canonical JSON Schemas shared by OpenAI adapters, MCP docs, and tests.

The canonical schemas intentionally use the strict subset accepted by OpenAI
function tools: every object is closed and every property is required. Values
that are semantically optional are represented as a union with ``null``.
"""
from __future__ import annotations

from copy import deepcopy
from typing import Iterable


def strict_object(properties: dict, *, required: Iterable[str] | None = None,
                  description: str | None = None) -> dict:
    schema = {
        "type": "object",
        "properties": properties,
        "required": list(required if required is not None else properties),
        "additionalProperties": False,
    }
    if description:
        schema["description"] = description
    return schema


def analyst_intent_schema(actions: Iterable[str], objectives: Iterable[str]) -> dict:
    return strict_object({
        "action": {"type": "string", "enum": list(actions)},
        "objective": {"type": "string", "enum": list(objectives)},
        "recency_days": {"type": ["integer", "null"], "minimum": 0, "maximum": 3650},
        "high_value": {"type": "boolean"},
        "segment": {"type": ["string", "null"], "maxLength": 100},
        "target_customers": {
            "type": ["integer", "null"], "minimum": 1, "maximum": 10_000_000,
        },
        "baseline_conversion": {"type": ["number", "null"], "minimum": 0, "maximum": 1},
        "expected_conversion": {"type": ["number", "null"], "minimum": 0, "maximum": 1},
        "average_order_value": {
            "type": ["number", "null"], "exclusiveMinimum": 0, "maximum": 1_000_000,
        },
        "discount_rate": {"type": ["number", "null"], "minimum": 0, "maximum": 1},
        "probability_threshold": {"type": ["number", "null"], "minimum": 0, "maximum": 1},
        "forecast_weeks": {"type": ["integer", "null"], "minimum": 1, "maximum": 12},
        "product_limit": {"type": ["integer", "null"], "minimum": 1, "maximum": 100},
        "product_ranking": {
            "type": ["string", "null"],
            "enum": ["revenue", "units", "orders", "customers", None],
        },
    })


def narrative_schema() -> dict:
    return strict_object({
        "headline": {"type": "string", "minLength": 1},
        "interpretation": {"type": "string", "minLength": 1},
        "next_step": {"type": "string", "minLength": 1},
    })


def campaign_copy_schema(channels: Iterable[str]) -> dict:
    channel_list = list(channels)
    messages = strict_object(
        {channel: {"type": "string", "minLength": 1} for channel in channel_list},
        required=channel_list,
    )
    return strict_object({"messages": messages})


PRODUCT_ITEM_SCHEMA = strict_object({
    "stock_code": {"type": "string"},
    "product_name": {"type": "string"},
    "segment_buyers": {"type": "integer", "minimum": 0},
    "segment_revenue_gbp": {"type": "number"},
    "segment_lift": {"type": "number", "minimum": 0},
})

OVERALL_PRODUCT_ITEM_SCHEMA = strict_object({
    "rank": {"type": "integer", "minimum": 1},
    "stock_code": {"type": "string"},
    "product_name": {"type": "string"},
    "revenue_gbp": {"type": "number"},
    "units": {"type": "number"},
    "orders": {"type": "integer", "minimum": 0},
    "customers": {"type": "integer", "minimum": 0},
})

SEGMENT_ITEM_SCHEMA = strict_object({
    "segment": {"type": "string"},
    "customers": {"type": "integer", "minimum": 0},
    "mean_recency_days": {"type": "number", "minimum": 0},
    "mean_frequency": {"type": "number", "minimum": 0},
    "mean_customer_value_gbp": {"type": "number", "minimum": 0},
    "mean_order_value_gbp": {"type": "number", "minimum": 0},
})

PREDICTION_SEGMENT_SCHEMA = strict_object({
    "segment": {"type": "string"},
    "customers": {"type": "integer", "minimum": 0},
    "mean_probability": {"type": "number", "minimum": 0, "maximum": 1},
    "flagged_customers": {"type": "integer", "minimum": 0},
})

CLASSIFICATION_VALIDATION_SCHEMA = strict_object({
    "roc_auc": {"type": ["number", "null"], "minimum": 0, "maximum": 1},
    "average_precision": {"type": ["number", "null"], "minimum": 0, "maximum": 1},
    "brier_score": {"type": ["number", "null"], "minimum": 0, "maximum": 1},
    "validation_rows": {"type": "integer", "minimum": 1},
    "validation_positive_rate": {"type": "number", "minimum": 0, "maximum": 1},
    "temporal_split": {"type": "string"},
    "training_snapshots": {"type": "integer", "minimum": 1},
    "validation_snapshots": {"type": "integer", "minimum": 1},
})

PREDICTION_OUTPUT_SCHEMA = strict_object({
    "source": {"type": "string"},
    "model": {"type": "string", "enum": ["churn", "repeat_purchase"]},
    "prediction_horizon_days": {"type": "integer", "minimum": 1},
    "resolved_segment": {"type": ["string", "null"]},
    "scored_customers": {"type": "integer", "minimum": 0},
    "flagged_customers": {"type": "integer", "minimum": 0},
    "flagged_share": {"type": "number", "minimum": 0, "maximum": 1},
    "mean_probability": {"type": "number", "minimum": 0, "maximum": 1},
    "threshold": {"type": "number", "minimum": 0, "maximum": 1},
    "probability_name": {"type": "string"},
    "segment_scores": {"type": "array", "items": PREDICTION_SEGMENT_SCHEMA},
    "validation": CLASSIFICATION_VALIDATION_SCHEMA,
    "data_warning": {"type": "string"},
})

FORECAST_POINT_SCHEMA = strict_object({
    "week_ending": {"type": "string", "format": "date"},
    "predicted_revenue_gbp": {"type": "number", "minimum": 0},
    "lower_bound_gbp": {"type": "number", "minimum": 0},
    "upper_bound_gbp": {"type": "number", "minimum": 0},
})

FORECAST_VALIDATION_SCHEMA = strict_object({
    "mae": {"type": "number", "minimum": 0},
    "rmse": {"type": "number", "minimum": 0},
    "mape": {"type": ["number", "null"], "minimum": 0},
    "baseline_mae": {"type": "number", "minimum": 0},
    "ridge_mae": {"type": "number", "minimum": 0},
    "rolling_mean_4_mae": {"type": "number", "minimum": 0},
    "selected_method": {
        "type": "string", "enum": ["ridge", "naive_last", "rolling_mean_4"],
    },
    "normalized_mae": {"type": "number", "minimum": 0},
    "reliability": {"type": "string", "enum": ["high", "medium", "low"]},
    "model_beats_baseline": {"type": "boolean"},
    "validation_weeks": {"type": "integer", "minimum": 1},
    "temporal_split": {"type": "string"},
    "incomplete_week_dropped": {"type": "boolean"},
})

DRIVER_SCHEMA = strict_object({
    "feature": {"type": "string"},
    "direction": {"type": "string"},
    "importance": {"type": "number", "minimum": 0},
})

METRIC_SCHEMA = strict_object({
    "name": {"type": "string"},
    "value": {"type": ["number", "string", "boolean", "null"]},
})


TOOL_CONTRACTS = {
    "get_segment_summary": {
        "description": "Return aggregate RFM profiles for all segments or one named segment. No customer IDs are returned.",
        "input_schema": strict_object({
            "segment": {
                "type": ["string", "null"],
                "description": "Segment name or null to return every behavioral segment.",
            },
        }),
        "output_schema": strict_object({
            "source": {"type": "string"},
            "segments": {"type": "array", "items": SEGMENT_ITEM_SCHEMA},
        }),
    },
    "compare_segments": {
        "description": "Compare aggregate RFM profiles for every behavioral segment.",
        "input_schema": strict_object({}),
        "output_schema": strict_object({
            "source": {"type": "string"},
            "segments": {"type": "array", "items": SEGMENT_ITEM_SCHEMA},
        }),
    },
    "filter_customers": {
        "description": "Count and summarize customers matching approved aggregate filters without exposing customer identifiers.",
        "input_schema": strict_object({
            "segment": {"type": ["string", "null"]},
            "minimum_recency_days": {"type": ["integer", "null"], "minimum": 0, "maximum": 3650},
            "high_value": {"type": "boolean"},
        }),
        "output_schema": strict_object({
            "source": {"type": "string"},
            "resolved_segment": {"type": ["string", "null"]},
            "matched_customers": {"type": "integer", "minimum": 0},
            "customer_share": {"type": "number", "minimum": 0, "maximum": 1},
            "observed_customer_value_gbp": {"type": "number", "minimum": 0},
            "mean_recency_days": {"type": ["number", "null"], "minimum": 0},
            "mean_frequency": {"type": ["number", "null"], "minimum": 0},
            "high_value_threshold_gbp": {"type": ["number", "null"], "minimum": 0},
        }),
    },
    "get_segment_products": {
        "description": "Rank products from observed transactions for a segment using popularity or segment lift.",
        "input_schema": strict_object({
            "segment": {"type": "string", "minLength": 1, "maxLength": 100},
            "limit": {"type": "integer", "minimum": 1, "maximum": 20},
            "ranking": {"type": "string", "enum": ["popular", "distinctive"]},
        }),
        "output_schema": strict_object({
            "source": {"type": "string"},
            "requested_segment": {"type": "string"},
            "resolved_segment": {"type": "string"},
            "ranking": {"type": "string", "enum": ["popular", "distinctive"]},
            "products": {"type": "array", "items": PRODUCT_ITEM_SCHEMA},
            "evidence_note": {"type": "string"},
        }),
    },
    "get_top_products": {
        "description": "Rank observed products overall by revenue, units, distinct orders, or distinct customers.",
        "input_schema": strict_object({
            "limit": {"type": "integer", "minimum": 1, "maximum": 100},
            "ranking": {"type": "string", "enum": ["revenue", "units", "orders", "customers"]},
        }),
        "output_schema": strict_object({
            "source": {"type": "string"},
            "ranking": {"type": "string", "enum": ["revenue", "units", "orders", "customers"]},
            "limit": {"type": "integer", "minimum": 1, "maximum": 100},
            "products": {"type": "array", "items": OVERALL_PRODUCT_ITEM_SCHEMA},
            "evidence_note": {"type": "string"},
        }),
    },
    "get_product_catalog_summary": {
        "description": "Count distinct observed products and report categories only when an explicit category field exists.",
        "input_schema": strict_object({}),
        "output_schema": strict_object({
            "source": {"type": "string"},
            "distinct_products": {"type": "integer", "minimum": 0},
            "distinct_product_names": {"type": "integer", "minimum": 0},
            "category_field_available": {"type": "boolean"},
            "category_field": {"type": ["string", "null"]},
            "distinct_categories": {"type": ["integer", "null"], "minimum": 0},
            "category_note": {"type": "string"},
        }),
    },
    "recommend_campaign": {
        "description": "Select a target segment and observed product evidence for an approved business objective.",
        "input_schema": strict_object({
            "objective": {
                "type": "string",
                "enum": [
                    "Retain valuable customers",
                    "Reactivate lapsed customers",
                    "Increase average order value",
                ],
            },
        }),
        "output_schema": strict_object({
            "source": {"type": "string"},
            "objective": {"type": "string"},
            "recommended_segment": {"type": "string"},
            "customers": {"type": "integer", "minimum": 0},
            "observed_segment_revenue_gbp": {"type": "number", "minimum": 0},
            "channel": {"type": "string"},
            "products": {"type": "array", "items": PRODUCT_ITEM_SCHEMA},
            "launch_requirements": {"type": "array", "items": {"type": "string"}},
        }),
    },
    "simulate_campaign": {
        "description": "Calculate a deterministic planning scenario. The result is not a causal forecast.",
        "input_schema": strict_object({
            "customers": {"type": "integer", "minimum": 1, "maximum": 10_000_000},
            "baseline_conversion": {"type": "number", "minimum": 0, "maximum": 1},
            "expected_conversion": {"type": "number", "minimum": 0, "maximum": 1},
            "average_order_value": {"type": "number", "exclusiveMinimum": 0, "maximum": 1_000_000},
            "discount_rate": {"type": "number", "minimum": 0, "maximum": 1},
        }),
        "output_schema": strict_object({
            "baseline_revenue": {"type": "number"},
            "campaign_revenue_after_discount": {"type": "number"},
            "incremental_revenue": {"type": "number"},
            "expected_orders": {"type": "number"},
            "causal_warning": {"type": "string"},
        }),
    },
    "predict_churn": {
        "description": "Estimate 90-day no-purchase probability from leakage-safe historical customer snapshots.",
        "input_schema": strict_object({
            "segment": {"type": ["string", "null"]},
            "probability_threshold": {"type": "number", "minimum": 0, "maximum": 1},
        }),
        "output_schema": PREDICTION_OUTPUT_SCHEMA,
    },
    "predict_repeat_purchase": {
        "description": "Estimate 30-day repeat-purchase probability from leakage-safe historical customer snapshots.",
        "input_schema": strict_object({
            "segment": {"type": ["string", "null"]},
            "probability_threshold": {"type": "number", "minimum": 0, "maximum": 1},
        }),
        "output_schema": PREDICTION_OUTPUT_SCHEMA,
    },
    "forecast_revenue": {
        "description": "Forecast one to twelve future weeks of revenue using lagged weekly history and temporal validation.",
        "input_schema": strict_object({
            "weeks": {"type": "integer", "minimum": 1, "maximum": 12},
        }),
        "output_schema": strict_object({
            "source": {"type": "string"},
            "model": {"type": "string"},
            "frequency": {"type": "string", "enum": ["weekly"]},
            "history_weeks": {"type": "integer", "minimum": 1},
            "forecast": {"type": "array", "items": FORECAST_POINT_SCHEMA},
            "validation": FORECAST_VALIDATION_SCHEMA,
            "forecast_warning": {"type": "string"},
        }),
    },
    "explain_predictive_model": {
        "description": "Return validation metrics, global model drivers, and limitations for a predictive model.",
        "input_schema": strict_object({
            "model": {"type": "string", "enum": ["churn", "repeat_purchase", "revenue_forecast"]},
        }),
        "output_schema": strict_object({
            "source": {"type": "string"},
            "model": {"type": "string"},
            "target": {"type": "string"},
            "prediction_horizon": {"type": ["string", "null"]},
            "metrics": {"type": "array", "items": METRIC_SCHEMA},
            "global_drivers": {"type": "array", "items": DRIVER_SCHEMA},
            "limitations": {"type": "array", "items": {"type": "string"}},
        }),
    },
}


def get_tool_contract(name: str) -> dict:
    if name not in TOOL_CONTRACTS:
        raise KeyError(f"Unknown tool contract: {name}")
    return deepcopy(TOOL_CONTRACTS[name])


def openai_function_tool(name: str) -> dict:
    """Adapt one canonical contract to the OpenAI Responses function-tool shape."""
    contract = get_tool_contract(name)
    return {
        "type": "function",
        "name": name,
        "description": contract["description"],
        "parameters": contract["input_schema"],
        "strict": True,
    }


def mcp_tool_descriptor(name: str) -> dict:
    """Return the wire-format fields a client sees during MCP tool discovery."""
    contract = get_tool_contract(name)
    return {
        "name": name,
        "description": contract["description"],
        "inputSchema": contract["input_schema"],
        "outputSchema": contract["output_schema"],
        "annotations": {
            "readOnlyHint": True,
            "destructiveHint": False,
            "idempotentHint": True,
            "openWorldHint": False,
        },
    }
