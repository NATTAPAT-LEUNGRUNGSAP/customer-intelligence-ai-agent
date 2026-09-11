"""Grounded natural-language analyst with validated intent and tool routing."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
import json
import re

import pandas as pd

from src.agent import OBJECTIVES, _call_copy_model, grounded_facts, select_target_segment
from src.campaign import simulate_campaign
from src.contracts import analyst_intent_schema, narrative_schema
from src.products import (
    overall_product_records, product_records, rank_overall_products,
    rank_segment_products, summarize_product_catalog,
)
from src.predictive import (
    PredictiveEngine, aggregate_customer_predictions, forecast_revenue,
    score_customer_classifier,
)


ANALYST_ACTIONS = (
    "segment_overview",
    "compare_segments",
    "filter_customers",
    "segment_products",
    "product_catalog_summary",
    "top_products",
    "campaign_recommendation",
    "campaign_simulation",
    "churn_prediction",
    "repeat_purchase_prediction",
    "revenue_forecast",
)


@dataclass(frozen=True)
class AnalystIntent:
    """Small, validated decision surface produced from a natural-language question."""

    action: str
    objective: str
    language: str
    recency_days: int | None = None
    high_value: bool = False
    segment: str | None = None
    target_customers: int | None = None
    baseline_conversion: float | None = None
    expected_conversion: float | None = None
    average_order_value: float | None = None
    discount_rate: float | None = None
    probability_threshold: float | None = None
    forecast_weeks: int | None = None
    product_limit: int | None = None
    product_ranking: str | None = None


@dataclass
class AnalystResult:
    intent: AnalystIntent
    intent_source: str
    narrative_source: str
    answer: str
    evidence: dict
    tool_trace: list[str]
    table: pd.DataFrame = field(default_factory=pd.DataFrame)
    warnings: list[str] = field(default_factory=list)


def _json_object(raw: str) -> dict:
    candidate = raw.strip()
    if candidate.startswith("```"):
        candidate = re.sub(r"^```(?:json)?\s*|\s*```$", "", candidate, flags=re.IGNORECASE)
    start, end = candidate.find("{"), candidate.rfind("}")
    if start < 0 or end < start:
        raise ValueError("The model did not return a JSON object.")
    payload = json.loads(candidate[start:end + 1])
    if not isinstance(payload, dict):
        raise ValueError("The model response must be a JSON object.")
    return payload


def _percent_near(text: str, labels: tuple[str, ...]) -> float | None:
    joined = "|".join(labels)
    patterns = (
        rf"(?:{joined}).{{0,35}}?(\d+(?:\.\d+)?)\s*%",
        rf"(\d+(?:\.\d+)?)\s*%.{{0,35}}?(?:{joined})",
    )
    for pattern in patterns:
        match = re.search(pattern, text, flags=re.IGNORECASE)
        if match:
            return float(match.group(1)) / 100
    return None


def _rule_intent(question: str, language: str) -> AnalystIntent:
    """Deterministic fallback that covers common Thai and English analyst requests."""
    text = question.casefold()
    if re.search(r"จำลอง|ถ้า|simulation|what[- ]?if|baseline|conversion|incremental", text):
        action = "campaign_simulation"
    elif re.search(r"churn|เลิกซื้อ|เสี่ยงหาย|ไม่กลับมาซื้อ", text):
        action = "churn_prediction"
    elif re.search(r"forecast|พยากรณ์|คาดการณ์รายได้|รายได้(?:สัปดาห์|เดือน|ปี)?หน้า", text):
        action = "revenue_forecast"
    elif re.search(r"repeat purchase|ซื้อซ้ำ|กลับมาซื้อ|โอกาสซื้อ", text):
        action = "repeat_purchase_prediction"
    elif re.search(r"สินค้า|product", text) and re.search(r"กลุ่ม|segment|loyal|vip|hibernat|premium", text):
        action = "segment_products"
    elif (
        re.search(r"สินค้า|products?|items?|catalog", text)
        and re.search(r"กี่|จำนวน|ทั้งหมด|ประเภท|หมวดหมู่|how many|count|types?|categor", text)
        and not re.search(r"มากที่สุด|ขายดี|อันดับ|top|best|highest|most", text)
    ):
        action = "product_catalog_summary"
    elif re.search(r"สินค้า|products?|items?", text):
        action = "top_products"
    elif re.search(r"เปรียบเทียบ|แตกต่าง|เทียบ|compare|difference|versus|\bvs\b", text):
        action = "compare_segments"
    elif re.search(r"แคมเปญ|campaign|สินค้าแนะนำ|recommend|retain|reactivat|upsell", text):
        action = "campaign_recommendation"
    elif re.search(r"หา(?:กลุ่ม|ลูกค้า)|ลูกค้าที่|filter|find customers?|customers? who", text):
        action = "filter_customers"
    else:
        action = "segment_overview"

    if re.search(r"หาย|ไม่ได้ซื้อ|ดึงกลับ|กลับมา|lapsed|reactivat|at.?risk|churn", text):
        objective = "Reactivate lapsed customers"
    elif re.search(r"ยอดต่อออเดอร์|มูลค่าต่อออเดอร์|เพิ่มยอด|aov|average order|upsell|bundle", text):
        objective = "Increase average order value"
    else:
        objective = "Retain valuable customers"

    recency = None
    day_match = re.search(r"(?:มากกว่า|เกิน|อย่างน้อย|over|more than|at least|>)?\s*(\d{1,4})\s*(?:วัน|days?)", text)
    if day_match:
        recency = int(day_match.group(1))
    high_value = bool(re.search(r"มูลค่าสูง|ใช้จ่ายสูง|ยอดสูง|high.?value|spend(?:ing)? a lot|high spend", text))
    customer_match = re.search(r"(\d[\d,]*)\s*(?:คน|customers?)", text)
    aov_match = re.search(r"(?:aov|average order value|มูลค่าต่อออเดอร์).{0,20}?(?:£|gbp)?\s*(\d+(?:\.\d+)?)", text)
    probability_threshold = None
    if action in {"churn_prediction", "repeat_purchase_prediction"}:
        threshold_match = re.search(r"(?:เกิน|มากกว่า|threshold|probability).{0,15}?(\d+(?:\.\d+)?)\s*%", text)
        if threshold_match:
            probability_threshold = float(threshold_match.group(1)) / 100
    forecast_weeks = None
    if action == "revenue_forecast":
        week_match = re.search(r"(\d{1,2})\s*(?:สัปดาห์|weeks?)", text)
        month_match = re.search(r"(\d{1,2})\s*(?:เดือน|months?)", text)
        if week_match:
            forecast_weeks = min(12, int(week_match.group(1)))
        elif month_match:
            forecast_weeks = min(12, int(month_match.group(1)) * 4)
        elif re.search(r"เดือนหน้า|next month", text):
            forecast_weeks = 4
    product_limit = None
    product_ranking = None
    if action == "top_products":
        limit_match = re.search(r"(\d{1,3})\s*(?:สินค้า|products?|อันดับ|รายการ|items?)", text)
        product_limit = min(100, int(limit_match.group(1))) if limit_match else 10
        if re.search(r"จำนวนชิ้น|ชิ้น|quantity|units?", text):
            product_ranking = "units"
        elif re.search(r"ออเดอร์|คำสั่งซื้อ|invoices?|orders?", text):
            product_ranking = "orders"
        elif re.search(r"ผู้ซื้อ|จำนวนลูกค้า|buyers?|customers?", text):
            product_ranking = "customers"
        else:
            product_ranking = "revenue"
    segment = None
    segment_aliases = {
        "loyalty": "Loyal High Value", "loyal": "Loyal High Value", "vip": "Loyal High Value",
        "hibernating": "Hibernating Low Value", "premium": "Occasional Premium",
        "ลูกค้าประจำ": "Loyal High Value",
    }
    if action in {"segment_products", "churn_prediction", "repeat_purchase_prediction"}:
        for alias, canonical in segment_aliases.items():
            if alias in text:
                segment = canonical
                break
    return AnalystIntent(
        action=action,
        objective=objective,
        language=language,
        recency_days=recency,
        high_value=high_value,
        segment=segment,
        target_customers=int(customer_match.group(1).replace(",", "")) if customer_match else None,
        baseline_conversion=_percent_near(text, ("baseline", "ฐานเดิม", "เดิม")),
        expected_conversion=_percent_near(text, ("expected", "คาดหวัง", "เพิ่มเป็น", "เป็น")),
        average_order_value=float(aov_match.group(1)) if aov_match else None,
        discount_rate=_percent_near(text, ("discount", "ส่วนลด", "coupon", "คูปอง")),
        probability_threshold=probability_threshold,
        forecast_weeks=forecast_weeks,
        product_limit=product_limit,
        product_ranking=product_ranking,
    )


def _intent_schema() -> dict:
    return analyst_intent_schema(ANALYST_ACTIONS, OBJECTIVES)


def _bounded_number(value, low: float, high: float, integer: bool = False):
    if value is None:
        return None
    number = float(value)
    if not low <= number <= high:
        raise ValueError(f"Numeric intent value must be between {low} and {high}.")
    return int(number) if integer else number


def _validated_intent(payload: dict, language: str) -> AnalystIntent:
    action = payload.get("action")
    objective = payload.get("objective")
    if action not in ANALYST_ACTIONS:
        raise ValueError("Unsupported analyst action.")
    if objective not in OBJECTIVES:
        raise ValueError("Unsupported business objective.")
    segment = payload.get("segment")
    if segment is not None and (not isinstance(segment, str) or len(segment) > 100):
        raise ValueError("Invalid segment name.")
    if not isinstance(payload.get("high_value"), bool):
        raise ValueError("high_value must be true or false.")
    product_ranking = payload.get("product_ranking")
    if product_ranking not in {None, "revenue", "units", "orders", "customers"}:
        raise ValueError("Unsupported product ranking metric.")
    return AnalystIntent(
        action=action,
        objective=objective,
        language=language,
        recency_days=_bounded_number(payload.get("recency_days"), 0, 3650, True),
        high_value=payload["high_value"],
        segment=segment,
        target_customers=_bounded_number(payload.get("target_customers"), 1, 10_000_000, True),
        baseline_conversion=_bounded_number(payload.get("baseline_conversion"), 0, 1),
        expected_conversion=_bounded_number(payload.get("expected_conversion"), 0, 1),
        average_order_value=_bounded_number(payload.get("average_order_value"), 0.01, 1_000_000),
        discount_rate=_bounded_number(payload.get("discount_rate"), 0, 1),
        probability_threshold=_bounded_number(payload.get("probability_threshold"), 0, 1),
        forecast_weeks=_bounded_number(payload.get("forecast_weeks"), 1, 12, True),
        product_limit=_bounded_number(payload.get("product_limit"), 1, 100, True),
        product_ranking=product_ranking,
    )


def parse_analyst_intent(question: str, language: str = "Thai",
                         provider: str = "Rules only", model: str | None = None
                         ) -> tuple[AnalystIntent, str, list[str]]:
    """Parse intent with an LLM when requested and fail closed to deterministic rules."""
    fallback = _rule_intent(question, language)
    if provider == "Rules only":
        return fallback, "Python rules", []
    schema = _intent_schema()
    prompt = f"""Classify this customer-analytics question into the supplied JSON schema.
Choose exactly one allowed action and one allowed objective. Extract only values explicitly stated by the user.
Percentages must be decimals between zero and one. Use null for missing values. Do not invent a segment or number.
Question: {question}
Return JSON only."""
    try:
        raw = _call_copy_model(prompt, provider, model, schema)
        parsed = _validated_intent(_json_object(raw), language)
        # Explicit business phrases are deterministic safety anchors. The LLM
        # may fill structured details, but it cannot reroute a clear request to
        # an unrelated analytical model.
        if fallback.action != "segment_overview" and parsed.action != fallback.action:
            return fallback, "Python semantic guardrail", [
                f"Intent model proposed '{parsed.action}', but the explicit request maps to "
                f"'{fallback.action}'. Python used the explicit action."
            ]
        return parsed, f"{provider} structured intent", []
    except Exception as exc:
        return fallback, "Python rules fallback", [f"Intent model rejected: {exc}"]


def _profile_table(customers: pd.DataFrame) -> pd.DataFrame:
    return customers.groupby("cluster_persona").agg(
        customers=("CustomerID", "nunique"),
        recency=("recency", "mean"),
        frequency=("frequency", "mean"),
        monetary=("monetary", "mean"),
        avg_order_value=("avg_order_value", "mean"),
    ).reset_index().sort_values("customers", ascending=False)


def _records(table: pd.DataFrame) -> list[dict]:
    records = []
    for row in table.to_dict("records"):
        records.append({key: (round(float(value), 2) if isinstance(value, float) else value)
                        for key, value in row.items()})
    return records


def _resolve_segment(customers: pd.DataFrame, requested: str | None) -> str | None:
    if not requested:
        return None
    requested_folded = requested.casefold()
    requested_folded = {"loyalty": "loyal", "vip": "loyal high value"}.get(
        requested_folded, requested_folded
    )
    names = customers.cluster_persona.dropna().astype(str).unique().tolist()
    return next((name for name in names
                 if requested_folded in name.casefold() or name.casefold() in requested_folded), None)


def _execute_intent(intent: AnalystIntent, customers: pd.DataFrame,
                    transactions: pd.DataFrame, selected_k: int,
                    automatic_k: int, diagnostics: pd.DataFrame
                    ) -> tuple[dict, pd.DataFrame, list[str]]:
    profiles = _profile_table(customers)
    trace = ["Customer feature table", "Behavioral segment profiles"]
    if intent.action in {"segment_overview", "compare_segments"}:
        selected_row = diagnostics[diagnostics.k == selected_k]
        evidence = {
            "action": intent.action,
            "selected_k": int(selected_k),
            "automatic_k": int(automatic_k),
            "selected_k_silhouette": round(float(selected_row.iloc[0].silhouette), 3),
            "segments": _records(profiles),
        }
        trace.append("Model diagnostics")
        return evidence, profiles, trace

    if intent.action == "filter_customers":
        subset = customers.copy()
        filters = []
        resolved = _resolve_segment(customers, intent.segment)
        if intent.segment and not resolved:
            raise ValueError(f"Unknown segment: {intent.segment}")
        if resolved:
            subset = subset[subset.cluster_persona == resolved]
            filters.append(f"segment = {resolved}")
        if intent.recency_days is not None:
            subset = subset[subset.recency >= intent.recency_days]
            filters.append(f"recency >= {intent.recency_days} days")
        value_threshold = None
        if intent.high_value:
            value_threshold = float(customers.monetary.quantile(.70))
            subset = subset[subset.monetary >= value_threshold]
            filters.append("monetary >= customer 70th percentile")
        evidence = {
            "action": intent.action,
            "filters": filters or ["no additional filters"],
            "matched_customers": int(subset.CustomerID.nunique()),
            "customer_share": round(float(subset.CustomerID.nunique() / customers.CustomerID.nunique()), 4),
            "observed_customer_value_gbp": round(float(subset.monetary.sum()), 2),
            "mean_recency_days": round(float(subset.recency.mean()), 1) if len(subset) else None,
            "mean_frequency": round(float(subset.frequency.mean()), 1) if len(subset) else None,
            "high_value_threshold_gbp": round(value_threshold, 2) if value_threshold is not None else None,
        }
        columns = ["CustomerID", "cluster_persona", "rule_segment", "recency",
                   "frequency", "monetary", "avg_order_value"]
        trace.append("Validated customer filter")
        return evidence, subset[columns].sort_values("monetary", ascending=False).head(100), trace

    if intent.action == "segment_products":
        resolved = _resolve_segment(customers, intent.segment)
        if not resolved:
            raise ValueError(
                "Specify a known behavioral segment, for example Loyal High Value or Hibernating Low Value."
            )
        subset = customers[customers.cluster_persona == resolved]
        ranked = rank_segment_products(transactions, subset, top_n=10, ranking="popular")
        evidence = {
            "action": intent.action,
            "requested_segment": intent.segment,
            "resolved_segment": resolved,
            "products": product_records(ranked),
            "evidence_note": "Observed at segment level; not proof of an individual's preference.",
        }
        trace.append("Segment product ranking")
        return evidence, ranked, trace

    if intent.action == "top_products":
        limit = intent.product_limit or 10
        ranking = intent.product_ranking or "revenue"
        if ranking not in {"revenue", "units", "orders", "customers"}:
            raise ValueError("Unsupported product ranking metric.")
        ranked = rank_overall_products(
            transactions, top_n=limit, ranking=ranking
        )
        evidence = {
            "action": intent.action,
            "ranking": ranking,
            "limit": limit,
            "products": overall_product_records(ranked),
            "evidence_note": (
                "Calculated from cleaned observed transactions. "
                "Best-selling defaults to total revenue."
            ),
        }
        trace.extend(["Clean product transaction filter", f"Product ranking by {ranking}"])
        return evidence, ranked, trace

    if intent.action == "product_catalog_summary":
        summary = summarize_product_catalog(transactions)
        evidence = {
            "action": intent.action,
            **summary,
        }
        table = pd.DataFrame([{
            "measure": "Distinct observed products",
            "value": summary["distinct_products"],
        }, {
            "measure": "Distinct product names",
            "value": summary["distinct_product_names"],
        }, {
            "measure": "Distinct categories",
            "value": summary["distinct_categories"],
        }])
        trace.extend(["Clean product transaction filter", "Product catalog summary"])
        return evidence, table, trace

    if intent.action == "campaign_recommendation":
        chosen, subset, ranking = select_target_segment(customers, intent.objective)
        facts = grounded_facts(chosen, subset, transactions)
        evidence = {
            "action": intent.action,
            "objective": intent.objective,
            "recommended_segment": chosen,
            "segment_facts": facts,
        }
        trace.extend(["Objective scoring", "Segment product ranking", "Grounded campaign facts"])
        return evidence, ranking, trace

    if intent.action in {"churn_prediction", "repeat_purchase_prediction"}:
        model_name = "churn" if intent.action == "churn_prediction" else "repeat_purchase"
        engine = PredictiveEngine(transactions)
        bundle = engine.classifier(model_name)
        scored = score_customer_classifier(bundle, transactions, customers)
        resolved = _resolve_segment(customers, intent.segment)
        if intent.segment and not resolved:
            raise ValueError(f"Unknown behavioral segment: {intent.segment}")
        if resolved:
            scored = scored[scored.cluster_persona == resolved]
        threshold = intent.probability_threshold if intent.probability_threshold is not None else .60
        probability_name = (
            "probability of no purchase in the next 90 days"
            if model_name == "churn"
            else "probability of a repeat purchase in the next 30 days"
        )
        aggregate = aggregate_customer_predictions(scored, threshold, probability_name)
        evidence = {
            "action": intent.action,
            "model": model_name,
            "prediction_horizon_days": bundle.horizon_days,
            "resolved_segment": resolved,
            **aggregate,
            "validation": bundle.metrics,
            "warning": "Model probability, not certainty or causal effect.",
        }
        table = pd.DataFrame(aggregate["segment_scores"])
        trace.extend(["Historical rolling snapshots", "Temporal holdout validation", f"{model_name} probability model"])
        return evidence, table, trace

    if intent.action == "revenue_forecast":
        weeks = intent.forecast_weeks or 4
        engine = PredictiveEngine(transactions)
        bundle = engine.revenue_forecaster()
        points = forecast_revenue(bundle, weeks)
        evidence = {
            "action": intent.action,
            "model": bundle.selected_method,
            "forecast_weeks": weeks,
            "history_weeks": len(bundle.history),
            "forecast": points,
            "validation": bundle.metrics,
            "warning": (
                f"Planning forecast using {bundle.selected_method}, selected by temporal holdout MAE; not a guarantee. "
                "Non-finite or explosive recursive values use a recent-history safety fallback or cap."
            ),
        }
        trace.extend(["Weekly revenue aggregation", "Temporal holdout validation", "Recursive revenue forecast"])
        return evidence, pd.DataFrame(points), trace

    n = intent.target_customers or 1000
    base = intent.baseline_conversion if intent.baseline_conversion is not None else .04
    expected = intent.expected_conversion if intent.expected_conversion is not None else .07
    aov = intent.average_order_value or 50.0
    discount = intent.discount_rate if intent.discount_rate is not None else .10
    simulated = simulate_campaign(n, base, expected, aov, discount)
    evidence = {
        "action": intent.action,
        "assumptions": {
            "target_customers": n,
            "baseline_conversion": base,
            "expected_conversion": expected,
            "average_order_value_gbp": aov,
            "discount_rate": discount,
        },
        "scenario_results": simulated,
        "causal_warning": "Scenario estimate only; validate incremental uplift with a randomized holdout.",
    }
    trace.append("Deterministic campaign simulator")
    return evidence, pd.DataFrame([simulated]), trace


def _fallback_narrative(intent: AnalystIntent) -> str:
    thai = intent.language == "Thai"
    messages = {
        "segment_overview": (
            "### ภาพรวมกลุ่มลูกค้า\nใช้โปรไฟล์ด้านล่างเพื่อดูขนาดและพฤติกรรมของแต่ละกลุ่ม ควรตีความ Cluster เป็นกลุ่มพฤติกรรมโดยประมาณ ไม่ใช่ประเภทลูกค้าที่ถูกต้องตายตัว",
            "### Segment overview\nUse the profiles below to compare audience size and behavior. Treat clusters as useful behavioral approximations, not objectively true customer types.",
        ),
        "compare_segments": (
            "### การเปรียบเทียบ Segment\nพิจารณา Recency, Frequency, Monetary และมูลค่าต่อออเดอร์ร่วมกันก่อนเลือกกลุ่มเป้าหมาย และตรวจว่ากลุ่มมีขนาดเพียงพอสำหรับการทดลอง",
            "### Segment comparison\nCompare recency, frequency, monetary value, and order value together before targeting, then confirm the audience is large enough for an experiment.",
        ),
        "filter_customers": (
            "### กลุ่มลูกค้าที่ตรงเงื่อนไข\nผลลัพธ์เกิดจากตัวกรองที่ตรวจสอบได้ใน Python ควรใช้รายชื่อตัวอย่างเพื่อสำรวจ และกำหนดสิทธิ์เข้าถึงก่อนนำข้อมูลลูกค้าออกไปใช้งาน",
            "### Matching customer audience\nPython applied the visible filters deterministically. Use the sample for exploration and apply access controls before operational use.",
        ),
        "segment_products": (
            "### สินค้าของ Segment\nรายการด้านล่างจัดอันดับจากธุรกรรมจริงของลูกค้าในกลุ่ม เป็นหลักฐานระดับ Segment ไม่ใช่การยืนยันว่าลูกค้าแต่ละคนชอบหรือเคยซื้อสินค้านั้น",
            "### Segment products\nThe list below is ranked from observed transactions in the segment. It is segment-level evidence, not proof of an individual customer's preference or purchase history.",
        ),
        "top_products": (
            "### สินค้าขายดี\nรายการด้านล่างคำนวณจากธุรกรรมที่ผ่านการทำความสะอาด โดยแสดงเกณฑ์การจัดอันดับอย่างชัดเจน คำว่า ‘ขายดีที่สุด’ จะหมายถึงรายได้รวม หากไม่ได้ระบุเกณฑ์อื่น",
            "### Top products\nThe list below is calculated from cleaned observed transactions using the displayed ranking metric. ‘Best-selling’ means total revenue unless another metric is requested.",
        ),
        "product_catalog_summary": (
            "### สรุปรายการสินค้า\nระบบนับรหัสและชื่อสินค้าที่ไม่ซ้ำจากธุรกรรมที่ผ่านการทำความสะอาด หากข้อมูลไม่มีคอลัมน์หมวดหมู่ ระบบจะไม่เดาหมวดหมู่จากชื่อสินค้า",
            "### Product catalog summary\nThe system counts distinct product codes and names from cleaned transactions. If the data has no category field, it does not infer categories from product names.",
        ),
        "campaign_recommendation": (
            "### คำแนะนำแคมเปญ\nกลุ่มเป้าหมายและสินค้าอ้างอิงมาจากคะแนนและธุรกรรมจริง ขั้นต่อไปคือตรวจสต็อก มาร์จิน และข้อเสนอที่ได้รับอนุมัติก่อนสร้างข้อความและทดลองกับกลุ่มควบคุม",
            "### Campaign recommendation\nThe target and product evidence come from observed data. Confirm inventory, margin, and an approved offer before generating copy and testing against a control group.",
        ),
        "campaign_simulation": (
            "### การตีความ Scenario\nผลด้านล่างคำนวณจากสมมติฐานที่ระบุ จึงใช้เพื่อวางแผนเท่านั้น ไม่ใช่หลักฐานว่าแคมเปญทำให้รายได้เพิ่ม ต้องยืนยันด้วยการสุ่มกลุ่มควบคุม",
            "### Scenario interpretation\nThe result below follows the stated assumptions. It supports planning but does not prove causal uplift; validate it with a randomized control group.",
        ),
        "churn_prediction": (
            "### ความเสี่ยงที่ลูกค้าจะไม่กลับมาซื้อ\nPython สร้างโมเดลจาก Snapshot ในอดีตและประเมินโอกาสไม่ซื้อภายในเก้าสิบวัน ผลเป็นความน่าจะเป็น ไม่ใช่คำยืนยันว่าลูกค้าจะเลิกซื้อแน่นอน",
            "### Customer inactivity risk\nPython trained on historical snapshots and estimates no-purchase risk over ninety days. The result is a probability, not a certainty.",
        ),
        "repeat_purchase_prediction": (
            "### โอกาสซื้อซ้ำ\nPython ประเมินโอกาสกลับมาซื้อภายในสามสิบวันจากพฤติกรรมก่อนหน้า ควรติดตามคุณภาพโมเดลและทดสอบแคมเปญด้วยกลุ่มควบคุม",
            "### Repeat-purchase probability\nPython estimates a purchase within thirty days from prior behavior. Monitor model quality and validate campaign impact with a control group.",
        ),
        "revenue_forecast": (
            "### พยากรณ์รายได้\nPython พยากรณ์รายได้รายสัปดาห์จากค่า Lag แนวโน้ม และฤดูกาล พร้อมประเมินย้อนหลังตามลำดับเวลา ช่วงคาดการณ์ไม่ใช่การรับประกัน",
            "### Revenue forecast\nPython forecasts weekly revenue from lags, trend, and seasonality with chronological validation. Prediction ranges are not guarantees.",
        ),
    }
    return messages[intent.action][0 if thai else 1]


def _narrative_schema() -> dict:
    return narrative_schema()


def _analyst_narrative(intent: AnalystIntent, evidence: dict, provider: str,
                       model: str | None) -> tuple[str, str, list[str]]:
    fallback = _fallback_narrative(intent)
    if provider == "Rules only":
        return fallback, "Python interpretation", []
    prompt = f"""Write a short business interpretation in {intent.language} using the supplied evidence.
Return JSON with exactly headline, interpretation, and next_step. Do not include digits, currency symbols,
percentages, customer identifiers, invented causes, guaranteed outcomes, or facts absent from the evidence.
The application displays every number separately using Python. Treat clustering as an approximation and
simulation as non-causal. Evidence: {json.dumps(evidence, ensure_ascii=False, default=str)}
Return JSON only."""
    try:
        payload = _json_object(_call_copy_model(prompt, provider, model, _narrative_schema()))
        if set(payload) != {"headline", "interpretation", "next_step"}:
            raise ValueError("Narrative keys do not match the approved schema.")
        if not all(isinstance(payload[key], str) and payload[key].strip() for key in payload):
            raise ValueError("Narrative fields must be non-empty text.")
        combined = " ".join(payload.values())
        if re.search(r"\d|[%£$€]", combined):
            raise ValueError("Narrative repeated a protected numeric claim.")
        if re.search(r"guarantee|แน่นอน|รับประกัน|จะเพิ่ม", combined, flags=re.IGNORECASE):
            raise ValueError("Narrative made a guaranteed-outcome claim.")
        answer = f"### {payload['headline']}\n\n{payload['interpretation']}\n\n**Next step:** {payload['next_step']}"
        return answer, f"{provider} validated interpretation", []
    except Exception as exc:
        if "protected numeric claim" in str(exc):
            warning = (
                "AI explanation was replaced by the safe Python explanation because "
                "it introduced numeric text outside the protected evidence display."
            )
        else:
            warning = f"AI explanation was replaced by the safe Python explanation: {exc}"
        return fallback, "Python interpretation fallback", [warning]


def run_analyst_query(question: str, customers: pd.DataFrame, transactions: pd.DataFrame,
                      selected_k: int, automatic_k: int, diagnostics: pd.DataFrame,
                      language: str = "Thai", provider: str = "Rules only",
                      model: str | None = None) -> AnalystResult:
    """Parse, route, execute, and interpret a customer-analytics question."""
    if not question.strip():
        raise ValueError("Enter an analyst question.")
    intent, intent_source, warnings = parse_analyst_intent(question, language, provider, model)
    if intent.action == "revenue_forecast":
        month_match = re.search(r"(\d{1,2})\s*(?:เดือน|months?)", question.casefold())
        if month_match and int(month_match.group(1)) * 4 > 12:
            warnings.append(
                "คำขอเกินขอบเขตข้อมูลปัจจุบัน ระบบจึงจำกัดไว้ที่ 12 สัปดาห์"
                if language == "Thai"
                else "The requested horizon exceeds the current data limit and was capped at 12 weeks."
            )
    evidence, table, trace = _execute_intent(
        intent, customers, transactions, selected_k, automatic_k, diagnostics
    )
    answer, narrative_source, narrative_warnings = _analyst_narrative(
        intent, evidence, provider, model
    )
    parser_tool = "LLM structured intent parser" if provider != "Rules only" else "Python intent parser"
    return AnalystResult(
        intent=intent,
        intent_source=intent_source,
        narrative_source=narrative_source,
        answer=answer,
        evidence=evidence,
        tool_trace=[parser_tool, *trace, "Validated business interpretation"],
        table=table,
        warnings=[*warnings, *narrative_warnings],
    )


def intent_as_dict(intent: AnalystIntent) -> dict:
    return asdict(intent)
