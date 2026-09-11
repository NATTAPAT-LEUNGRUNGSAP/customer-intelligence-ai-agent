import pandas as pd
import src.agent as agent_module
import src.analyst as analyst_module
from src.data import DataQualityReport, clean_transactions_dataframe, load_transactions, load_transactions_with_report
from src.features import build_customer_features
from src.campaign import simulate_campaign
from src.segmentation import segment_customers
from src.agent import (CampaignPlan, audit_generated_campaign, build_campaign_constraints,
                       build_safe_campaign_fallback, build_safe_campaign_plan,
                       generate_campaign, generate_campaign_plan, grounded_facts,
                       render_campaign_plan, render_observed_facts,
                       select_target_segment, validate_campaign_plan)
from src.products import rank_overall_products, rank_segment_products, summarize_product_catalog
from src.database import DATABASE_INSERT_CHUNK_SIZE, SCHEMA_SQL
from src.analyst import parse_analyst_intent, run_analyst_query


def test_features_aggregate_transactions(tmp_path):
    path = tmp_path / "t.csv"
    pd.DataFrame([
        {"InvoiceNo":"1", "StockCode":"A", "Quantity":2, "InvoiceDate":"2024-01-01", "UnitPrice":10, "CustomerID":1},
        {"InvoiceNo":"2", "StockCode":"B", "Quantity":1, "InvoiceDate":"2024-01-10", "UnitPrice":20, "CustomerID":1},
    ]).to_csv(path, index=False)
    f = build_customer_features(load_transactions(path), "2024-01-11").iloc[0]
    assert f.frequency == 2 and f.monetary == 40 and f.recency == 1


def test_simulator_accounts_for_discount():
    result = simulate_campaign(1000, .04, .07, 100, .10)
    assert result["incremental_revenue"] == 2300


def test_quality_report_counts_each_removal_stage(tmp_path):
    path = tmp_path / "quality.csv"
    pd.DataFrame([
        {"InvoiceNo":"1", "StockCode":"A", "Quantity":2, "InvoiceDate":"2024-01-01", "UnitPrice":10, "CustomerID":1},
        {"InvoiceNo":"C2", "StockCode":"B", "Quantity":1, "InvoiceDate":"2024-01-02", "UnitPrice":20, "CustomerID":1},
        {"InvoiceNo":"3", "StockCode":"C", "Quantity":1, "InvoiceDate":"bad", "UnitPrice":20, "CustomerID":2},
    ]).to_csv(path, index=False)
    clean, report = load_transactions_with_report(path)
    assert len(clean) == 1
    assert report.cancelled_rows_removed == 1
    assert report.invalid_required_fields_removed == 1


def test_mixed_invoice_date_formats_are_retained(tmp_path):
    path = tmp_path / "mixed_dates.csv"
    pd.DataFrame([
        {"InvoiceNo":"1", "StockCode":"A", "Quantity":1, "InvoiceDate":"12/1/2010 8:26", "UnitPrice":10, "CustomerID":1},
        {"InvoiceNo":"2", "StockCode":"B", "Quantity":1, "InvoiceDate":"01-04-2011 10:00", "UnitPrice":20, "CustomerID":2},
    ]).to_csv(path, index=False)
    clean, report = load_transactions_with_report(path)
    assert len(clean) == 2
    assert report.invalid_invoice_date_removed == 0


def test_dataframe_cleaner_supports_non_csv_sources():
    raw = pd.DataFrame([
        {"InvoiceNo":"1", "StockCode":"A", "Quantity":2, "InvoiceDate":"2024-01-01",
         "UnitPrice":10, "CustomerID":"1", "Description":"Mug"},
    ])
    clean, report = clean_transactions_dataframe(raw)
    assert len(clean) == 1 and clean.iloc[0].Revenue == 20
    assert report.customers == 1


def test_cleaner_rejects_nonfinite_numeric_values():
    raw = pd.DataFrame([
        {"InvoiceNo":"1", "StockCode":"A", "Quantity":1, "InvoiceDate":"2024-01-01",
         "UnitPrice":10, "CustomerID":1, "Description":"Valid"},
        {"InvoiceNo":"2", "StockCode":"B", "Quantity":float("inf"), "InvoiceDate":"2024-01-02",
         "UnitPrice":20, "CustomerID":2, "Description":"Infinite quantity"},
        {"InvoiceNo":"3", "StockCode":"C", "Quantity":1e308, "InvoiceDate":"2024-01-03",
         "UnitPrice":1e308, "CustomerID":3, "Description":"Overflow revenue"},
    ])
    clean, report = clean_transactions_dataframe(raw)
    assert len(clean) == 1
    assert report.invalid_numeric_removed == 2


def test_sub_cent_unit_price_survives_cleaning_and_database_precision():
    raw = pd.DataFrame([
        {"InvoiceNo":"1", "StockCode":"A", "Quantity":1, "InvoiceDate":"2024-01-01",
         "UnitPrice":0.001, "CustomerID":"1", "Description":"Sample"},
    ])
    clean, _ = clean_transactions_dataframe(raw)
    assert len(clean) == 1 and clean.iloc[0].UnitPrice == 0.001
    assert "NUMERIC(14,4)" in SCHEMA_SQL
    assert DATABASE_INSERT_CHUNK_SIZE == 1_000
    assert "data_import_audit" in SCHEMA_SQL


def test_data_quality_report_can_be_restored_from_postgres_json():
    values = {name: index for index, name in enumerate(DataQualityReport.__dataclass_fields__, start=1)}
    restored = DataQualityReport.from_dict({**values, "future_field": 999})
    assert restored.to_dict() == values


def test_manual_cluster_count_is_respected():
    rows = []
    for i in range(20):
        rows.append({"CustomerID": str(i), "recency": i + 1, "frequency": (i % 5) + 1,
                     "monetary": (i + 1) * 100, "avg_order_value": (i + 1) * 20,
                     "unique_products": (i % 7) + 1, "units": 1, "tenure": 100,
                     "purchase_rate": 1, "first_purchase": pd.Timestamp("2024-01-01"),
                     "last_purchase": pd.Timestamp("2024-02-01")})
    result = segment_customers(pd.DataFrame(rows), selected_k=4)
    assert result.selected_k == 4
    assert result.customers.cluster.nunique() == 4


def test_agent_targeting_and_rules_brief_are_grounded():
    customers = pd.DataFrame([
        {"CustomerID":"1", "cluster_persona":"Loyal", "recency":10, "frequency":10, "monetary":1000, "avg_order_value":100},
        {"CustomerID":"2", "cluster_persona":"Lapsed", "recency":200, "frequency":2, "monetary":600, "avg_order_value":300},
    ])
    target, subset, ranking = select_target_segment(customers, "Reactivate lapsed customers")
    assert target == "Lapsed"
    facts = grounded_facts(target, subset)
    brief = generate_campaign("Draft a campaign", "Reactivate lapsed customers", "English", facts)
    facts_block = render_observed_facts(facts, "English")
    assert "1" in facts_block and "£600.00" in facts_block
    assert "£600.00" not in brief


def test_guardrail_flags_invented_money_percent_and_personalization():
    facts = {
        "mean_customer_value_gbp": 600.0,
        "mean_order_value_gbp": 300.0,
        "observed_segment_revenue_gbp": 600.0,
        "product_level_preferences_available": False,
    }
    warnings = audit_generated_campaign(
        "Give 15% off, forecast £999, and recommend สินค้าที่คุณชอบ", facts
    )
    assert len(warnings) == 3


def test_guardrail_allows_user_supplied_percentage_and_money():
    facts = {
        "mean_customer_value_gbp": 600.0,
        "mean_order_value_gbp": 300.0,
        "observed_segment_revenue_gbp": 1200.0,
        "product_level_preferences_available": False,
    }
    assert audit_generated_campaign("Test 10% with budget £600.00", facts, "Test 10% with budget £600.00") == []


def test_segment_product_ranking_uses_observed_transactions():
    tx = pd.DataFrame([
        {"CustomerID":"1", "StockCode":"A", "Description":"Red Mug", "InvoiceNo":"1", "Quantity":2, "Revenue":20},
        {"CustomerID":"2", "StockCode":"A", "Description":"Red Mug", "InvoiceNo":"2", "Quantity":1, "Revenue":10},
        {"CustomerID":"3", "StockCode":"A", "Description":"Red Mug", "InvoiceNo":"3", "Quantity":1, "Revenue":10},
        {"CustomerID":"4", "StockCode":"A", "Description":"Red Mug", "InvoiceNo":"4", "Quantity":1, "Revenue":10},
        {"CustomerID":"5", "StockCode":"A", "Description":"Red Mug", "InvoiceNo":"5", "Quantity":1, "Revenue":10},
        {"CustomerID":"6", "StockCode":"A", "Description":"Red Mug", "InvoiceNo":"6", "Quantity":1, "Revenue":10},
        {"CustomerID":"1", "StockCode":"B", "Description":"Blue Bowl", "InvoiceNo":"1", "Quantity":1, "Revenue":12},
        {"CustomerID":"2", "StockCode":"B", "Description":"Blue Bowl", "InvoiceNo":"2", "Quantity":1, "Revenue":12},
        {"CustomerID":"3", "StockCode":"B", "Description":"Blue Bowl", "InvoiceNo":"3", "Quantity":1, "Revenue":12},
        {"CustomerID":"1", "StockCode":"POST", "Description":"POSTAGE", "InvoiceNo":"1", "Quantity":1, "Revenue":5},
    ])
    subset = pd.DataFrame({"CustomerID":["1", "2", "3"]})
    ranked = rank_segment_products(tx, subset, top_n=3)
    assert {"Red Mug", "Blue Bowl"}.issubset(set(ranked.Description))
    assert "POSTAGE" not in ranked.Description.tolist()
    distinctive = rank_segment_products(tx, subset, top_n=3, ranking="distinctive")
    assert distinctive.iloc[0].Description == "Blue Bowl"
    assert (distinctive.lift > 1).all()


def test_overall_product_ranking_uses_requested_metric_and_excludes_fees():
    tx = pd.DataFrame([
        {"CustomerID":"1", "StockCode":"A", "Description":"High Revenue", "InvoiceNo":"1", "Quantity":1, "Revenue":100},
        {"CustomerID":"2", "StockCode":"B", "Description":"Many Units", "InvoiceNo":"2", "Quantity":20, "Revenue":20},
        {"CustomerID":"1", "StockCode":"POST", "Description":"POSTAGE", "InvoiceNo":"3", "Quantity":100, "Revenue":500},
    ])
    revenue = rank_overall_products(tx, top_n=10, ranking="revenue")
    units = rank_overall_products(tx, top_n=10, ranking="units")
    assert revenue.iloc[0].Description == "High Revenue"
    assert units.iloc[0].Description == "Many Units"
    assert "POSTAGE" not in set(revenue.Description)


def test_product_catalog_summary_does_not_invent_missing_categories():
    tx = pd.DataFrame([
        {"CustomerID":"1", "StockCode":"A", "Description":"Red Mug", "InvoiceNo":"1", "Quantity":1, "Revenue":10},
        {"CustomerID":"2", "StockCode":"B", "Description":"Blue Bowl", "InvoiceNo":"2", "Quantity":1, "Revenue":12},
        {"CustomerID":"1", "StockCode":"POST", "Description":"POSTAGE", "InvoiceNo":"3", "Quantity":1, "Revenue":5},
    ])
    summary = summarize_product_catalog(tx)
    assert summary["distinct_products"] == 2
    assert summary["distinct_product_names"] == 2
    assert summary["category_field_available"] is False
    assert summary["distinct_categories"] is None


def test_product_catalog_summary_uses_explicit_category_field_when_available():
    tx = pd.DataFrame([
        {"CustomerID":"1", "StockCode":"A", "Description":"Red Mug", "InvoiceNo":"1", "Quantity":1, "Revenue":10, "Category":"Kitchen"},
        {"CustomerID":"2", "StockCode":"B", "Description":"Blue Bowl", "InvoiceNo":"2", "Quantity":1, "Revenue":12, "Category":"Kitchen"},
        {"CustomerID":"3", "StockCode":"C", "Description":"Toy", "InvoiceNo":"3", "Quantity":1, "Revenue":15, "Category":"Gifts"},
    ])
    summary = summarize_product_catalog(tx)
    assert summary["category_field"] == "Category"
    assert summary["distinct_categories"] == 2


def test_analyst_routes_top_products_with_limit_and_metric():
    revenue, _, _ = parse_analyst_intent("10 สินค้าที่ขายดีที่สุด", provider="Rules only")
    units, _, _ = parse_analyst_intent(
        "5 สินค้าที่ขายได้จำนวนชิ้นมากที่สุด", provider="Rules only"
    )
    assert revenue.action == "top_products"
    assert revenue.product_limit == 10 and revenue.product_ranking == "revenue"
    assert units.action == "top_products"
    assert units.product_limit == 5 and units.product_ranking == "units"


def test_analyst_distinguishes_catalog_count_from_top_products():
    count, _, _ = parse_analyst_intent("สินค้ามีกี่ประเภท", provider="Rules only")
    total, _, _ = parse_analyst_intent("มีสินค้าทั้งหมดกี่รายการ", provider="Rules only")
    top, _, _ = parse_analyst_intent("สินค้า 10 อันดับแรกตามรายได้", provider="Rules only")
    assert count.action == "product_catalog_summary"
    assert total.action == "product_catalog_summary"
    assert top.action == "top_products"


def test_guardrail_flags_unresolved_template_placeholders():
    facts = {"product_level_preferences_available": False}
    warnings = audit_generated_campaign("ส่งถึง [ชื่อ] ผ่าน [ช่องทาง Line] with {{offer}}", facts)
    assert any("placeholder" in warning.lower() for warning in warnings)


def test_guardrail_requires_exact_observed_campaign_product():
    facts = {
        "product_level_preferences_available": False,
        "campaign_product_candidates": [{"product_name": "REGENCY CAKESTAND 3 TIER"}],
    }
    missing = audit_generated_campaign("ส่งข้อความชวนลูกค้ากลับมา", facts)
    assert any("exact observed campaign product" in warning for warning in missing)
    assert audit_generated_campaign("แนะนำ REGENCY CAKESTAND 3 TIER ซึ่งขายดีในระดับ Segment", facts) == []


def test_guardrail_flags_thai_preference_promotion_and_availability_claims():
    facts = {"product_level_preferences_available": False}
    warnings = audit_generated_campaign(
        "สินค้าที่คุณชื่นชอบเลือกซื้อใหม่ได้แล้ว และเรามีโปรโมชั่นพิเศษสำหรับคุณ", facts
    )
    assert any("personalization" in warning.lower() for warning in warnings)
    assert any("availability" in warning.lower() for warning in warnings)
    assert any("promotion" in warning.lower() for warning in warnings)


def test_guardrail_flags_unsupported_channel_attributes_and_product_lists():
    facts = {
        "product_level_preferences_available": False,
        "product_catalog_and_margin_available": False,
        "rule_based_channel": "Email + LINE",
        "campaign_product_candidates": [
            {"product_name": "RED MUG"}, {"product_name": "BLUE BOWL"},
        ],
    }
    warnings = audit_generated_campaign(
        "Email: RED MUG และ BLUE BOWL เป็นสินค้าคลาสสิก เหมาะสำหรับตกแต่งบ้าน เลือกซื้อที่เว็บไซต์", facts
    )
    assert any("unsupported communication channel" in warning.lower() for warning in warnings)
    assert any("product attributes" in warning.lower() for warning in warnings)
    assert any("more than one observed product" in warning.lower() for warning in warnings)


def test_python_safe_fallback_passes_the_same_guardrail():
    facts = {
        "product_level_preferences_available": False,
        "product_catalog_and_margin_available": False,
        "rule_based_channel": "Email + LINE",
        "campaign_product_candidates": [
            {"product_name": "RED MUG"}, {"product_name": "BLUE BOWL"},
        ],
    }
    fallback = build_safe_campaign_fallback(facts, "Thai", "เขียนข้อความ Email และ LINE")
    assert audit_generated_campaign(fallback, facts, "เขียนข้อความ Email และ LINE") == []


def test_guardrail_blocks_fake_individual_purchase_history():
    facts = {
        "product_level_preferences_available": False,
        "campaign_product_candidates": [{"product_name": "RETROSPOT SMALL TUBE MATCHES"}],
    }
    warnings = audit_generated_campaign(
        "ท่านเคยเลือกซื้อ RETROSPOT SMALL TUBE MATCHES",
        facts,
    )
    assert any("personalization" in warning.lower() for warning in warnings)


def test_requested_control_group_percentage_is_required_and_fallback_preserves_it():
    facts = {
        "product_level_preferences_available": False,
        "rule_based_channel": "Email + LINE",
        "campaign_product_candidates": [{"product_name": "RED MUG"}],
    }
    question = "ทำ A/B test โดยใช้ control group 10%"
    missing = audit_generated_campaign(
        "Email: ขอแนะนำ RED MUG; A/B test with a control group.", facts, question
    )
    assert any("omitted" in warning.lower() and "10%" in warning for warning in missing)
    fallback = build_safe_campaign_fallback(facts, "Thai", question)
    assert "10%" in fallback
    assert audit_generated_campaign(fallback, facts, question) == []


def _structured_campaign_facts():
    return {
        "product_level_preferences_available": False,
        "product_catalog_and_margin_available": False,
        "rule_based_channel": "Email + LINE",
        "campaign_product_candidates": [
            {"product_name": "JUMBO BAG PAISLEY PARK"},
            {"product_name": "VINTAGE DOILY JUMBO BAG RED"},
        ],
    }


def test_campaign_constraints_lock_explicit_product_and_exclude_forbidden_channel():
    facts = _structured_campaign_facts()
    question = (
        "ใช้ VINTAGE DOILY JUMBO BAG RED เพียงสินค้าเดียวผ่าน Email และ LINE "
        "ห้ามใช้ App และใช้ control group 10%"
    )
    constraints = build_campaign_constraints(question, "Retain valuable customers", "Thai", facts)
    assert constraints.product == "VINTAGE DOILY JUMBO BAG RED"
    assert constraints.channels == ("Email", "LINE")
    assert constraints.control_group_percentage == "10%"


def test_safe_plan_uses_one_locked_product_across_every_channel():
    facts = _structured_campaign_facts()
    question = "ใช้ VINTAGE DOILY JUMBO BAG RED เพียงสินค้าเดียว control group 10%"
    plan = build_safe_campaign_plan(question, "Retain valuable customers", "Thai", facts)
    assert set(plan.messages) == {"Email", "LINE"}
    assert all("VINTAGE DOILY JUMBO BAG RED" in message for message in plan.messages.values())
    assert all("JUMBO BAG PAISLEY PARK" not in message for message in plan.messages.values())
    assert plan.constraints.control_group_percentage == "10%"
    assert validate_campaign_plan(plan, facts, question) == []
    rendered = render_campaign_plan(plan)
    assert "Control group: 10%" in rendered
    assert "Treatment group" not in rendered


def test_structured_generation_rejects_bad_json_then_accepts_repair():
    facts = _structured_campaign_facts()
    question = "ใช้ VINTAGE DOILY JUMBO BAG RED เพียงสินค้าเดียว control group 10%"
    outputs = iter([
        '{"messages":{"Email":"สวัสดี [Customer Name]","LINE":"ลองสินค้าใหม่"}}',
        '{"messages":{"Email":"ขอแนะนำ VINTAGE DOILY JUMBO BAG RED สามารถตอบกลับทาง Email นี้ได้",'
        '"LINE":"ขอแนะนำ VINTAGE DOILY JUMBO BAG RED สามารถตอบกลับทาง LINE นี้ได้"}}',
    ])
    original = agent_module._call_copy_model
    agent_module._call_copy_model = lambda *args, **kwargs: next(outputs)
    try:
        result = generate_campaign_plan(
            question, "Retain valuable customers", "Thai", facts, "Ollama", "test-model"
        )
    finally:
        agent_module._call_copy_model = original
    assert result.attempts == 2
    assert not result.used_fallback
    assert len(result.rejected_outputs) == 1
    assert validate_campaign_plan(result.plan, facts, question) == []


def test_structured_generation_falls_back_after_two_invalid_attempts():
    facts = _structured_campaign_facts()
    question = "ใช้ VINTAGE DOILY JUMBO BAG RED เพียงสินค้าเดียว control group 10%"
    original = agent_module._call_copy_model
    agent_module._call_copy_model = lambda *args, **kwargs: "not json"
    try:
        result = generate_campaign_plan(
            question, "Retain valuable customers", "Thai", facts, "Ollama", "test-model"
        )
    finally:
        agent_module._call_copy_model = original
    assert result.used_fallback and result.attempts == 2
    assert len(result.rejected_outputs) == 2
    assert all("VINTAGE DOILY JUMBO BAG RED" in text for text in result.plan.messages.values())
    assert validate_campaign_plan(result.plan, facts, question) == []


def test_guardrail_blocks_latest_qwen_claim_patterns():
    facts = {
        "product_level_preferences_available": False,
        "product_catalog_and_margin_available": False,
        "rule_based_channel": "Email + LINE",
        "campaign_product_candidates": [{"product_name": "RETROSPOT SMALL TUBE MATCHES"}],
    }
    draft = """RECOMMENDATION: Utilize RETROSPOT SMALL TUBE MATCHES as it has a segment lift of 1.04.
Email: พบกับสินค้าที่ไม่เคยลอง RETROSPOT SMALL TUBE MATCHES ที่มีลักษณะเฉพาะและน่าสนใจ ราคาไม่แพงเกินไป
LINE: แนะนำสินค้าใหม่ RETROSPOT SMALL TUBE MATCHES เป็นของขวัญชิ้นพิเศษสำหรับคุณ
KPIS AND A/B TEST: A/B Test the subject line and monitor open rate."""
    warnings = audit_generated_campaign(draft, facts, "เขียนข้อความ Email และ LINE")
    assert any("new products" in warning.lower() for warning in warnings)
    assert any("gift or incentive" in warning.lower() for warning in warnings)
    assert any("product attributes" in warning.lower() for warning in warnings)
    assert any("price positioning" in warning.lower() for warning in warnings)
    assert any("product-lift metric" in warning.lower() for warning in warnings)
    assert any("control group" in warning.lower() for warning in warnings)


def test_claim_guardrail_flags_membership_and_new_stock_claims():
    facts = {"mean_customer_value_gbp":1, "mean_order_value_gbp":1,
             "observed_segment_revenue_gbp":1, "customers":1,
             "mean_recency_days":1, "mean_frequency":1,
             "product_level_preferences_available":False}
    warnings = audit_generated_campaign("คุณเป็นสมาชิกพิเศษของเรา และเรามีสินค้าใหม่พร้อมส่ง", facts)
    assert any("Membership" in w for w in warnings)
    assert any("New products" in w for w in warnings)
    assert any("Availability" in w for w in warnings)


def _analyst_fixture():
    customers = pd.DataFrame([
        {"CustomerID":"1", "cluster_persona":"Loyal High Value (C0)", "rule_segment":"Champions",
         "recency":10, "frequency":8, "monetary":1000, "avg_order_value":125},
        {"CustomerID":"2", "cluster_persona":"Loyal High Value (C0)", "rule_segment":"Champions",
         "recency":20, "frequency":6, "monetary":800, "avg_order_value":133},
        {"CustomerID":"3", "cluster_persona":"Hibernating Low Value (C1)", "rule_segment":"Hibernating",
         "recency":120, "frequency":1, "monetary":100, "avg_order_value":100},
        {"CustomerID":"4", "cluster_persona":"Occasional Premium (C2)", "rule_segment":"At Risk High Value",
         "recency":100, "frequency":2, "monetary":1200, "avg_order_value":600},
    ])
    transactions = pd.DataFrame([
        {"CustomerID":"1", "StockCode":"A", "Description":"RED MUG", "InvoiceNo":"1", "Quantity":1, "Revenue":20},
        {"CustomerID":"2", "StockCode":"A", "Description":"RED MUG", "InvoiceNo":"2", "Quantity":1, "Revenue":20},
        {"CustomerID":"3", "StockCode":"A", "Description":"RED MUG", "InvoiceNo":"3", "Quantity":1, "Revenue":20},
        {"CustomerID":"4", "StockCode":"A", "Description":"RED MUG", "InvoiceNo":"4", "Quantity":1, "Revenue":20},
    ])
    diagnostics = pd.DataFrame([
        {"k":3, "silhouette":.25, "davies_bouldin":1.2, "stability":.99,
         "smallest_cluster_share":.20, "selection_score":.33}
    ])
    return customers, transactions, diagnostics


def test_analyst_rule_intent_extracts_reactivation_filter():
    intent, source, warnings = parse_analyst_intent(
        "หาลูกค้ามูลค่าสูงที่ไม่ได้ซื้อมากกว่า 90 วัน", "Thai", "Rules only"
    )
    assert intent.action == "filter_customers"
    assert intent.objective == "Reactivate lapsed customers"
    assert intent.recency_days == 90 and intent.high_value
    assert source == "Python rules" and warnings == []


def test_analyst_rule_intent_extracts_simulation_assumptions():
    intent, _, _ = parse_analyst_intent(
        "ถ้ามีลูกค้า 1000 คน baseline conversion 4% เพิ่มเป็น 7% AOV 50 และส่วนลด 10%",
        "Thai", "Rules only",
    )
    assert intent.action == "campaign_simulation"
    assert intent.target_customers == 1000
    assert intent.baseline_conversion == .04
    assert intent.expected_conversion == .07
    assert intent.average_order_value == 50
    assert intent.discount_rate == .10


def test_analyst_routes_filter_and_returns_python_evidence():
    customers, transactions, diagnostics = _analyst_fixture()
    result = run_analyst_query(
        "หาลูกค้ามูลค่าสูงที่ไม่ได้ซื้อมากกว่า 90 วัน",
        customers, transactions, 3, 3, diagnostics, "Thai", "Rules only",
    )
    assert result.intent.action == "filter_customers"
    assert result.evidence["matched_customers"] == 1
    assert result.table.CustomerID.tolist() == ["4"]
    assert "Validated customer filter" in result.tool_trace


def test_analyst_routes_simulator_without_llm_arithmetic():
    customers, transactions, diagnostics = _analyst_fixture()
    result = run_analyst_query(
        "ถ้ามีลูกค้า 1000 คน baseline conversion 4% เพิ่มเป็น 7% AOV 50 และส่วนลด 10%",
        customers, transactions, 3, 3, diagnostics, "Thai", "Rules only",
    )
    assert result.evidence["scenario_results"]["incremental_revenue"] == 1150
    assert result.tool_trace[-2] == "Deterministic campaign simulator"


def test_analyst_rejects_invalid_llm_intent_and_falls_back_to_rules():
    original = analyst_module._call_copy_model
    analyst_module._call_copy_model = lambda *args, **kwargs: '{"action":"delete_database"}'
    try:
        intent, source, warnings = parse_analyst_intent(
            "เปรียบเทียบแต่ละ segment", "Thai", "Ollama", "test-model"
        )
    finally:
        analyst_module._call_copy_model = original
    assert intent.action == "compare_segments"
    assert source == "Python rules fallback"
    assert warnings and "rejected" in warnings[0]


def test_analyst_semantic_guardrail_keeps_explicit_campaign_action():
    original = analyst_module._call_copy_model
    analyst_module._call_copy_model = lambda *args, **kwargs: (
        '{"action":"churn_prediction","objective":"Reactivate lapsed customers",'
        '"recency_days":null,"high_value":false,"segment":null,"target_customers":null,'
        '"baseline_conversion":null,"expected_conversion":null,"average_order_value":null,'
        '"discount_rate":null,"probability_threshold":null,"forecast_weeks":null,'
        '"product_limit":null,"product_ranking":null}'
    )
    try:
        intent, source, warnings = parse_analyst_intent(
            "แนะนำกลุ่มเป้าหมายสำหรับแคมเปญดึงลูกค้ากลับมา",
            "Thai", "Ollama", "test-model",
        )
    finally:
        analyst_module._call_copy_model = original
    assert intent.action == "campaign_recommendation"
    assert source == "Python semantic guardrail"
    assert warnings and "explicit request" in warnings[0]


def test_analyst_rejects_numeric_llm_narrative_and_uses_safe_interpretation():
    customers, transactions, diagnostics = _analyst_fixture()
    outputs = iter([
        '{"action":"segment_overview","objective":"Retain valuable customers",'
        '"recency_days":null,"high_value":false,"segment":null,"target_customers":null,'
        '"baseline_conversion":null,"expected_conversion":null,"average_order_value":null,'
        '"discount_rate":null}',
        '{"headline":"พบ 3 กลุ่ม","interpretation":"ยอดเพิ่ม 20%",'
        '"next_step":"ส่งทันที"}',
    ])
    original = analyst_module._call_copy_model
    analyst_module._call_copy_model = lambda *args, **kwargs: next(outputs)
    try:
        result = run_analyst_query(
            "สรุปกลุ่มลูกค้า", customers, transactions, 3, 3, diagnostics,
            "Thai", "Ollama", "test-model",
        )
    finally:
        analyst_module._call_copy_model = original
    assert result.intent_source == "Ollama structured intent"
    assert result.narrative_source == "Python interpretation fallback"
    assert any("unsupported numeric text" in warning for warning in result.warnings)
