import pandas as pd
from src.data import load_transactions, load_transactions_with_report
from src.features import build_customer_features
from src.campaign import simulate_campaign
from src.segmentation import segment_customers
from src.agent import (audit_generated_campaign, generate_campaign, grounded_facts,
                       render_observed_facts, select_target_segment)
from src.products import rank_segment_products


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


def test_guardrail_flags_unresolved_template_placeholders():
    facts = {"product_level_preferences_available": False}
    warnings = audit_generated_campaign("Send to [customers] with {{offer}}", facts)
    assert any("placeholder" in warning.lower() for warning in warnings)


def test_claim_guardrail_flags_membership_and_new_stock_claims():
    facts = {"mean_customer_value_gbp":1, "mean_order_value_gbp":1,
             "observed_segment_revenue_gbp":1, "customers":1,
             "mean_recency_days":1, "mean_frequency":1,
             "product_level_preferences_available":False}
    warnings = audit_generated_campaign("คุณเป็นสมาชิกพิเศษของเรา และเรามีสินค้าใหม่พร้อมส่ง", facts)
    assert any("Membership" in w for w in warnings)
    assert any("New products" in w for w in warnings)
    assert any("Availability" in w for w in warnings)
