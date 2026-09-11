"""Read-only business tool layer used by MCP and direct application adapters."""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import pandas as pd

from src.agent import OBJECTIVES, grounded_facts, select_target_segment
from src.campaign import simulate_campaign as calculate_campaign_scenario
from src.contracts import TOOL_CONTRACTS
from src.data import load_transactions_with_report, resolve_default_data
from src.database import load_transactions_from_database
from src.features import build_customer_features
from src.products import (
    overall_product_records, product_records, rank_overall_products,
    rank_segment_products, summarize_product_catalog,
)
from src.predictive import (
    PredictiveEngine, aggregate_customer_predictions, forecast_revenue,
    score_customer_classifier,
)
from src.segmentation import segment_customers


ROOT = Path(__file__).resolve().parents[1]


def _json_number(value: Any, digits: int = 2) -> float:
    return round(float(value), digits)


class CustomerIntelligenceService:
    """Lazy, read-only analytics facade with no model-generated SQL."""

    def __init__(self, database_url: str | None = None, data_path: str | Path | None = None,
                 selected_k: int | None = None):
        self.database_url = database_url if database_url is not None else os.getenv("DATABASE_URL", "")
        configured_path = data_path if data_path is not None else os.getenv("MCP_DATA_PATH", "")
        self.data_path = Path(configured_path) if configured_path else None
        self.selected_k = selected_k
        self._state: dict[str, Any] | None = None
        self._predictive: PredictiveEngine | None = None

    def _load(self) -> dict[str, Any]:
        if self._state is not None:
            return self._state
        if self.database_url.strip():
            transactions, quality = load_transactions_from_database(self.database_url)
            source = "PostgreSQL · transactions"
        else:
            path = self.data_path or resolve_default_data(ROOT)
            transactions, quality = load_transactions_with_report(path)
            source = f"CSV · {path.name}"
        features = build_customer_features(transactions)
        segmentation = segment_customers(features, selected_k=self.selected_k)
        self._state = {
            "source": source,
            "transactions": transactions,
            "quality": quality,
            "segmentation": segmentation,
            "customers": segmentation.customers,
        }
        self._predictive = PredictiveEngine(transactions)
        return self._state

    def refresh(self) -> None:
        self._state = None
        self._predictive = None

    @staticmethod
    def _segment_alias(value: str) -> str:
        folded = value.casefold().strip()
        aliases = {
            "loyalty": "loyal", "vip": "loyal high value",
            "ลูกค้าประจำ": "loyal", "ลูกค้ามูลค่าสูง": "loyal high value",
            "หลับ": "hibernating", "เสี่ยง": "at risk",
        }
        return aliases.get(folded, folded)

    def _resolve_segment(self, requested: str) -> tuple[str, str]:
        customers = self._load()["customers"]
        needle = self._segment_alias(requested)
        for column in ("cluster_persona", "rule_segment"):
            names = customers[column].dropna().astype(str).unique().tolist()
            exact = next((name for name in names if name.casefold() == needle), None)
            if exact:
                return column, exact
            partial = next((name for name in names
                            if needle in name.casefold() or name.casefold() in needle), None)
            if partial:
                return column, partial
        available = sorted(customers.cluster_persona.dropna().astype(str).unique())
        raise ValueError(f"Unknown segment '{requested}'. Available behavioral segments: {available}")

    def get_segment_summary(self, segment: str | None) -> dict:
        state = self._load()
        customers = state["customers"]
        if segment is not None:
            column, resolved = self._resolve_segment(segment)
            customers = customers[customers[column] == resolved]
            group_column = column
        else:
            group_column = "cluster_persona"
        table = customers.groupby(group_column).agg(
            customers=("CustomerID", "nunique"),
            mean_recency_days=("recency", "mean"),
            mean_frequency=("frequency", "mean"),
            mean_customer_value_gbp=("monetary", "mean"),
            mean_order_value_gbp=("avg_order_value", "mean"),
        ).reset_index().rename(columns={group_column: "segment"})
        records = []
        for row in table.sort_values("customers", ascending=False).itertuples():
            records.append({
                "segment": str(row.segment),
                "customers": int(row.customers),
                "mean_recency_days": _json_number(row.mean_recency_days),
                "mean_frequency": _json_number(row.mean_frequency),
                "mean_customer_value_gbp": _json_number(row.mean_customer_value_gbp),
                "mean_order_value_gbp": _json_number(row.mean_order_value_gbp),
            })
        return {"source": state["source"], "segments": records}

    def compare_segments(self) -> dict:
        return self.get_segment_summary(None)

    def filter_customers(self, segment: str | None, minimum_recency_days: int | None,
                         high_value: bool) -> dict:
        state = self._load()
        all_customers = state["customers"]
        subset = all_customers.copy()
        resolved = None
        if segment is not None:
            column, resolved = self._resolve_segment(segment)
            subset = subset[subset[column] == resolved]
        if minimum_recency_days is not None:
            subset = subset[subset.recency >= minimum_recency_days]
        threshold = None
        if high_value:
            threshold = float(all_customers.monetary.quantile(.70))
            subset = subset[subset.monetary >= threshold]
        matched = int(subset.CustomerID.nunique())
        total = int(all_customers.CustomerID.nunique())
        return {
            "source": state["source"],
            "resolved_segment": resolved,
            "matched_customers": matched,
            "customer_share": _json_number(matched / total if total else 0, 4),
            "observed_customer_value_gbp": _json_number(subset.monetary.sum()),
            "mean_recency_days": _json_number(subset.recency.mean(), 1) if matched else None,
            "mean_frequency": _json_number(subset.frequency.mean(), 1) if matched else None,
            "high_value_threshold_gbp": _json_number(threshold) if threshold is not None else None,
        }

    def get_segment_products(self, segment: str, limit: int, ranking: str) -> dict:
        state = self._load()
        column, resolved = self._resolve_segment(segment)
        subset = state["customers"][state["customers"][column] == resolved]
        ranked = rank_segment_products(state["transactions"], subset, top_n=limit, ranking=ranking)
        return {
            "source": state["source"],
            "requested_segment": segment,
            "resolved_segment": resolved,
            "ranking": ranking,
            "products": product_records(ranked),
            "evidence_note": "Ranked from observed segment transactions; this is not individual preference evidence.",
        }

    def get_top_products(self, limit: int, ranking: str) -> dict:
        state = self._load()
        ranked = rank_overall_products(
            state["transactions"], top_n=limit, ranking=ranking
        )
        return {
            "source": state["source"],
            "ranking": ranking,
            "limit": limit,
            "products": overall_product_records(ranked),
            "evidence_note": (
                "Ranked from cleaned observed transactions. 'Best-selling' defaults to revenue; "
                "choose units, orders, or customers for a different definition."
            ),
        }

    def get_product_catalog_summary(self) -> dict:
        state = self._load()
        return {
            "source": state["source"],
            **summarize_product_catalog(state["transactions"]),
        }

    def recommend_campaign(self, objective: str) -> dict:
        if objective not in OBJECTIVES:
            raise ValueError(f"Unsupported objective: {objective}")
        state = self._load()
        chosen, subset, _ = select_target_segment(state["customers"], objective)
        facts = grounded_facts(chosen, subset, state["transactions"])
        return {
            "source": state["source"],
            "objective": objective,
            "recommended_segment": chosen,
            "customers": int(facts["customers"]),
            "observed_segment_revenue_gbp": _json_number(facts["observed_segment_revenue_gbp"]),
            "channel": str(facts["rule_based_channel"]),
            "products": list(facts.get("campaign_product_candidates", [])),
            "launch_requirements": [
                "Confirm inventory and margin",
                "Approve the offer before launch",
                "Use a randomized control group to measure incremental uplift",
            ],
        }

    def predict_customers(self, model_name: str, segment: str | None,
                          probability_threshold: float) -> dict:
        state = self._load()
        if self._predictive is None:
            self._predictive = PredictiveEngine(state["transactions"])
        bundle = self._predictive.classifier(model_name)
        scored = score_customer_classifier(bundle, state["transactions"], state["customers"])
        resolved = None
        if segment is not None:
            column, resolved = self._resolve_segment(segment)
            scored = scored[scored[column] == resolved]
        probability_name = (
            "probability of no purchase in the next 90 days"
            if model_name == "churn"
            else "probability of a repeat purchase in the next 30 days"
        )
        aggregate = aggregate_customer_predictions(
            scored, probability_threshold, probability_name
        )
        return {
            "source": state["source"],
            "model": model_name,
            "prediction_horizon_days": int(bundle.horizon_days),
            "resolved_segment": resolved,
            **aggregate,
            "validation": bundle.metrics,
            "data_warning": (
                "Probabilities are model estimates from historical behavior, not certainties or causal effects. "
                "Revalidate on newer production data before operational use."
            ),
        }

    def forecast_revenue(self, weeks: int) -> dict:
        state = self._load()
        if self._predictive is None:
            self._predictive = PredictiveEngine(state["transactions"])
        bundle = self._predictive.revenue_forecaster()
        return {
            "source": state["source"],
            "model": bundle.selected_method,
            "frequency": "weekly",
            "history_weeks": int(len(bundle.history)),
            "forecast": forecast_revenue(bundle, weeks),
            "validation": bundle.metrics,
            "forecast_warning": (
                "Forecast intervals use recent holdout residuals and are planning ranges, not guarantees. "
                f"The selected method is {bundle.selected_method}, chosen by lowest temporal holdout MAE. "
                "Non-finite or explosive recursive values fall back to, or are capped by, recent observed revenue. "
                "If model_beats_baseline is false, prefer the naive baseline until the model is improved."
            ),
        }

    def explain_predictive_model(self, model: str) -> dict:
        state = self._load()
        if self._predictive is None:
            self._predictive = PredictiveEngine(state["transactions"])
        if model == "revenue_forecast":
            bundle = self._predictive.revenue_forecaster()
            target = "weekly observed revenue"
            horizon = None
            limitations = [
                "Only about one year of the Online Retail history is available.",
                "Prediction bands are empirical residual ranges, not formal confidence intervals.",
                "Promotions, prices, holidays, inventory, and external events are not modeled.",
            ]
        else:
            bundle = self._predictive.classifier(model)
            target = (
                "no purchase in the following 90 days" if model == "churn"
                else "at least one purchase in the following 30 days"
            )
            horizon = f"{bundle.horizon_days} days"
            limitations = [
                "Churn is a behavioral proxy because no explicit churn event exists in the dataset.",
                "Probabilities may drift when customer behavior or data collection changes.",
                "Global coefficients do not explain an individual customer prediction.",
            ]
        metrics = [{"name": key, "value": value} for key, value in bundle.metrics.items()]
        return {
            "source": state["source"],
            "model": model,
            "target": target,
            "prediction_horizon": horizon,
            "metrics": metrics,
            "global_drivers": bundle.drivers[:8],
            "limitations": limitations,
        }

    @staticmethod
    def simulate_campaign(customers: int, baseline_conversion: float,
                          expected_conversion: float, average_order_value: float,
                          discount_rate: float) -> dict:
        result = calculate_campaign_scenario(
            customers, baseline_conversion, expected_conversion,
            average_order_value, discount_rate,
        )
        return {
            **{key: _json_number(value) for key, value in result.items()},
            "causal_warning": "Planning scenario only; validate uplift with a randomized control group.",
        }

    def execute(self, name: str, arguments: dict) -> dict:
        """Validate both sides of a tool call against the canonical contract."""
        if name not in TOOL_CONTRACTS:
            raise ValueError(f"Unknown tool: {name}")
        try:
            from jsonschema import validate
        except ImportError as exc:
            raise RuntimeError("Install requirements.txt to validate tool contracts.") from exc
        contract = TOOL_CONTRACTS[name]
        validate(instance=arguments, schema=contract["input_schema"])
        handlers = {
            "get_segment_summary": lambda: self.get_segment_summary(**arguments),
            "compare_segments": self.compare_segments,
            "filter_customers": lambda: self.filter_customers(**arguments),
            "get_segment_products": lambda: self.get_segment_products(**arguments),
            "get_top_products": lambda: self.get_top_products(**arguments),
            "get_product_catalog_summary": self.get_product_catalog_summary,
            "recommend_campaign": lambda: self.recommend_campaign(**arguments),
            "simulate_campaign": lambda: self.simulate_campaign(**arguments),
            "predict_churn": lambda: self.predict_customers("churn", **arguments),
            "predict_repeat_purchase": lambda: self.predict_customers("repeat_purchase", **arguments),
            "forecast_revenue": lambda: self.forecast_revenue(**arguments),
            "explain_predictive_model": lambda: self.explain_predictive_model(**arguments),
        }
        result = handlers[name]()
        validate(instance=result, schema=contract["output_schema"])
        return result
