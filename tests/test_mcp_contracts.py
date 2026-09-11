import asyncio
from types import SimpleNamespace

import pandas as pd
import pytest
from jsonschema import Draft202012Validator, ValidationError

from src.business_tools import CustomerIntelligenceService
from src.agent import _call_copy_model
from src.contracts import (
    TOOL_CONTRACTS,
    analyst_intent_schema,
    campaign_copy_schema,
    mcp_tool_descriptor,
    narrative_schema,
    openai_function_tool,
)


def _assert_openai_strict(schema: dict) -> None:
    if schema.get("type") == "object":
        assert schema.get("additionalProperties") is False
        assert set(schema.get("required", [])) == set(schema.get("properties", {}))
        for child in schema.get("properties", {}).values():
            _assert_openai_strict(child)
    if schema.get("type") == "array":
        _assert_openai_strict(schema["items"])


def test_every_canonical_contract_is_valid_and_openai_strict():
    for contract in TOOL_CONTRACTS.values():
        Draft202012Validator.check_schema(contract["input_schema"])
        Draft202012Validator.check_schema(contract["output_schema"])
        _assert_openai_strict(contract["input_schema"])
        _assert_openai_strict(contract["output_schema"])


def test_mcp_and_openai_adapters_share_one_input_schema():
    for name, contract in TOOL_CONTRACTS.items():
        assert mcp_tool_descriptor(name)["inputSchema"] == contract["input_schema"]
        tool = openai_function_tool(name)
        assert tool["parameters"] == contract["input_schema"]
        assert tool["strict"] is True


def test_model_output_schemas_are_strict():
    for schema in (
        analyst_intent_schema(["segment_overview"], ["Retain valuable customers"]),
        narrative_schema(),
        campaign_copy_schema(["Email", "LINE"]),
    ):
        Draft202012Validator.check_schema(schema)
        _assert_openai_strict(schema)


def test_service_validates_input_before_execution():
    service = CustomerIntelligenceService()
    with pytest.raises(ValidationError):
        service.execute("simulate_campaign", {
            "customers": 1000,
            "baseline_conversion": 4,
            "expected_conversion": .07,
            "average_order_value": 50,
            "discount_rate": .1,
        })


def test_service_segment_products_returns_grounded_structured_output():
    customers = pd.DataFrame([
        {"CustomerID": "1", "cluster_persona": "Loyal High Value (C0)", "rule_segment": "Champions",
         "recency": 2, "frequency": 3, "monetary": 100, "avg_order_value": 50},
        {"CustomerID": "2", "cluster_persona": "Loyal High Value (C0)", "rule_segment": "Champions",
         "recency": 3, "frequency": 2, "monetary": 80, "avg_order_value": 40},
        {"CustomerID": "3", "cluster_persona": "Loyal High Value (C0)", "rule_segment": "Champions",
         "recency": 4, "frequency": 2, "monetary": 70, "avg_order_value": 35},
    ])
    transactions = pd.DataFrame([
        {"CustomerID": customer, "StockCode": "A", "Description": "RED MUG",
         "InvoiceNo": customer, "Quantity": 1, "Revenue": 20}
        for customer in ("1", "2", "3")
    ])
    service = CustomerIntelligenceService(database_url="")
    service._state = {"source": "test", "customers": customers, "transactions": transactions}
    result = service.execute("get_segment_products", {
        "segment": "loyalty", "limit": 5, "ranking": "popular",
    })
    assert result["resolved_segment"] == "Loyal High Value (C0)"
    assert result["products"][0]["product_name"] == "RED MUG"
    assert "CustomerID" not in str(result)


def test_service_top_products_returns_ranked_aggregate_output():
    service = CustomerIntelligenceService(
        database_url="", data_path="data/sample_transactions.csv"
    )
    result = service.execute("get_top_products", {"limit": 10, "ranking": "revenue"})
    assert len(result["products"]) == 10
    assert result["products"][0]["rank"] == 1
    assert result["products"][0]["revenue_gbp"] >= result["products"][-1]["revenue_gbp"]
    assert "CustomerID" not in str(result)


def test_service_product_catalog_summary_reports_missing_taxonomy_truthfully():
    service = CustomerIntelligenceService(
        database_url="", data_path="data/sample_transactions.csv"
    )
    result = service.execute("get_product_catalog_summary", {})
    assert result["distinct_products"] > 0
    assert result["category_field_available"] is False
    assert result["category_field"] is None
    assert result["distinct_categories"] is None


def test_mcp_server_lists_expected_tools_when_sdk_is_installed():
    from mcp import Client
    from mcp_server import mcp

    async def list_names():
        async with Client(mcp) as client:
            result = await client.list_tools()
            return {tool.name for tool in result.tools}

    names = asyncio.run(list_names())
    assert set(TOOL_CONTRACTS) <= names


def test_openai_copy_call_sends_strict_structured_output(monkeypatch):
    import openai

    captured = {}

    class FakeResponses:
        def create(self, **kwargs):
            captured.update(kwargs)
            return SimpleNamespace(output_text='{"headline":"ok"}')

    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setattr(openai, "OpenAI", lambda: SimpleNamespace(responses=FakeResponses()))
    schema = narrative_schema()
    assert _call_copy_model("prompt", "OpenAI", "test-model", schema) == '{"headline":"ok"}'
    assert captured["text"]["format"] == {
        "type": "json_schema",
        "name": "validated_customer_intelligence_output",
        "schema": schema,
        "strict": True,
    }
