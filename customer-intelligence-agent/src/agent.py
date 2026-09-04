"""Grounded campaign agent: deterministic targeting, optional LLM copywriting."""
from __future__ import annotations

import json
import os
import re
from urllib import request

import numpy as np
import pandas as pd

from src.campaign import recommend_campaign
from src.products import product_records, rank_segment_products

OBJECTIVES = {
    "Retain valuable customers": "retention",
    "Reactivate lapsed customers": "reactivation",
    "Increase average order value": "upsell",
}

SEGMENT_DEFINITIONS = {
    "Loyal High Value": "Customers with high purchase frequency and high observed customer value who purchased relatively recently.",
    "Occasional Premium": "Customers who purchase infrequently but have relatively high average order value; this does not mean abnormal customers.",
    "Hibernating Low Value": "Customers with long purchase recency, low frequency, and relatively low observed customer value.",
    "Recent / New Buyers": "Customers who purchased recently and have limited purchase history.",
    "Lapsed Valuable": "Customers with meaningful historical value whose most recent purchase was a long time ago.",
    "Value Regulars": "Customers with recurring mid-range purchasing behavior.",
}

SEGMENT_DEFINITIONS_TH = {
    "Loyal High Value": "ลูกค้ามูลค่าสูงที่ซื้อบ่อยและซื้อค่อนข้างล่าสุด",
    "Occasional Premium": "ลูกค้าที่ซื้อไม่บ่อย แต่มูลค่าต่อออเดอร์ค่อนข้างสูง ไม่ได้หมายถึงลูกค้าที่ผิดปกติ",
    "Hibernating Low Value": "ลูกค้าที่หายไปนาน ซื้อไม่บ่อย และมีมูลค่าที่สังเกตได้ค่อนข้างต่ำ",
    "Recent / New Buyers": "ลูกค้าที่เพิ่งซื้อและยังมีประวัติการซื้อไม่มาก",
    "Lapsed Valuable": "ลูกค้าที่เคยมีมูลค่า แต่ไม่ได้ซื้อมาเป็นเวลานาน",
    "Value Regulars": "ลูกค้าที่ซื้อซ้ำและมีมูลค่าระดับกลาง",
}


def _minmax(series: pd.Series) -> pd.Series:
    width = series.max() - series.min()
    return (series - series.min()) / width if width else pd.Series(0.5, index=series.index)


def select_target_segment(customers: pd.DataFrame, objective: str) -> tuple[str, pd.DataFrame, pd.DataFrame]:
    """Select an audience with transparent scoring; the LLM never chooses the numbers."""
    profiles = customers.groupby("cluster_persona").agg(
        customers=("CustomerID", "nunique"),
        recency=("recency", "mean"), frequency=("frequency", "mean"),
        monetary=("monetary", "mean"), avg_order_value=("avg_order_value", "mean"),
    )
    r = _minmax(profiles.recency)
    f = _minmax(profiles.frequency)
    m = _minmax(profiles.monetary)
    m_log = _minmax(pd.Series(np.log1p(profiles.monetary), index=profiles.index))
    aov = _minmax(profiles.avg_order_value)
    key = OBJECTIVES.get(objective, "retention")
    if key == "reactivation":
        # Balance dormancy with recoverable historic value; avoid spending the
        # campaign budget on the oldest but consistently lowest-value audience.
        profiles["target_score"] = .65 * r + .55 * m_log - .20 * f
    elif key == "upsell":
        profiles["target_score"] = .80 * aov + .10 * m + .10 * (1 - r)
    else:
        profiles["target_score"] = .45 * m + .35 * f + .20 * (1 - r)
    target = str(profiles.target_score.idxmax())
    subset = customers[customers.cluster_persona == target].copy()
    return target, subset, profiles.sort_values("target_score", ascending=False).reset_index()


def grounded_facts(segment_name: str, subset: pd.DataFrame,
                   transactions: pd.DataFrame | None = None) -> dict:
    rec = recommend_campaign(subset)
    base_name = segment_name.rsplit(" (C", 1)[0]
    if transactions is not None:
        popular_products = product_records(rank_segment_products(transactions, subset, ranking="popular"))
        distinctive_products = product_records(rank_segment_products(transactions, subset, ranking="distinctive"))
    else:
        popular_products, distinctive_products = [], []
    return {
        "segment": segment_name,
        "segment_definition": SEGMENT_DEFINITIONS.get(base_name, "A behavioral cluster defined by the supplied metrics."),
        "segment_definition_th": SEGMENT_DEFINITIONS_TH.get(base_name, "กลุ่มลูกค้าที่แบ่งจากพฤติกรรมตามตัวชี้วัดที่แสดง"),
        "customers": rec["profile"]["customers"],
        "mean_recency_days": round(rec["profile"]["recency"], 1),
        "mean_frequency": round(float(subset.frequency.mean()), 1),
        "mean_customer_value_gbp": round(float(subset.monetary.mean()), 2),
        "mean_order_value_gbp": round(rec["profile"]["aov"], 2),
        "observed_segment_revenue_gbp": round(rec["profile"]["revenue"], 2),
        "rule_based_offer": rec["offer"],
        "rule_based_channel": rec["channel"],
        "product_level_preferences_available": False,
        "segment_product_candidates_available": bool(popular_products or distinctive_products),
        "popular_product_candidates": popular_products,
        "distinctive_product_candidates": distinctive_products,
        "product_catalog_and_margin_available": False,
        "historical_campaign_outcomes_available": False,
    }


def render_observed_facts(facts: dict, language: str) -> str:
    """Render metrics deterministically so the LLM cannot rename or reinterpret them."""
    if language == "Thai":
        block = f"""### ข้อเท็จจริงจาก Python

- **Segment:** `{facts['segment']}`
- **ความหมาย:** {facts['segment_definition_th']}
- **จำนวนลูกค้า:** {facts['customers']:,} คน
- **ซื้อครั้งล่าสุดเฉลี่ย:** {facts['mean_recency_days']} วัน
- **ความถี่การซื้อเฉลี่ย:** {facts['mean_frequency']} ครั้งต่อลูกค้า
- **มูลค่าลูกค้าเฉลี่ย:** £{facts['mean_customer_value_gbp']:,.2f}
- **มูลค่าต่อออเดอร์เฉลี่ย:** £{facts['mean_order_value_gbp']:,.2f}
- **รายได้ที่สังเกตได้ของกลุ่ม:** £{facts['observed_segment_revenue_gbp']:,.2f}
"""
        if facts.get("popular_product_candidates"):
            block += "\n**สินค้าขายดีในอดีตของ Segment (ไม่ใช่ความชอบรายบุคคล):**\n"
            for p in facts["popular_product_candidates"]:
                block += f"\n- `{p['stock_code']}` {p['product_name']} — ผู้ซื้อ {p['segment_buyers']:,} คน, รายได้ £{p['segment_revenue_gbp']:,.2f}, lift {p['segment_lift']:.2f}x"
        block += "\n\n**สินค้าที่ Segment นิยมมากกว่าฐานลูกค้ารวม (Lift > 1):**\n"
        if facts.get("distinctive_product_candidates"):
            for p in facts["distinctive_product_candidates"]:
                block += f"\n- `{p['stock_code']}` {p['product_name']} — ผู้ซื้อ {p['segment_buyers']:,} คน, รายได้ £{p['segment_revenue_gbp']:,.2f}, lift {p['segment_lift']:.2f}x"
        else:
            block += "\n- ไม่พบสินค้าที่ผ่านเกณฑ์ Lift > 1 และจำนวนผู้ซื้อขั้นต่ำ"
        return block
    block = f"""### Python-calculated facts

- **Segment:** `{facts['segment']}`
- **Definition:** {facts['segment_definition']}
- **Customers:** {facts['customers']:,}
- **Mean recency:** {facts['mean_recency_days']} days
- **Mean purchase frequency:** {facts['mean_frequency']} purchases per customer
- **Mean customer value:** £{facts['mean_customer_value_gbp']:,.2f}
- **Mean order value:** £{facts['mean_order_value_gbp']:,.2f}
- **Observed segment revenue:** £{facts['observed_segment_revenue_gbp']:,.2f}
"""
    if facts.get("popular_product_candidates"):
        block += "\n**Historically popular segment products (not individual preferences):**\n"
        for p in facts["popular_product_candidates"]:
            block += f"\n- `{p['stock_code']}` {p['product_name']} — {p['segment_buyers']:,} buyers, £{p['segment_revenue_gbp']:,.2f} revenue, {p['segment_lift']:.2f}x lift"
    block += "\n\n**Products more prevalent in this segment than the full customer base (Lift > 1):**\n"
    if facts.get("distinctive_product_candidates"):
        for p in facts["distinctive_product_candidates"]:
            block += f"\n- `{p['stock_code']}` {p['product_name']} — {p['segment_buyers']:,} buyers, £{p['segment_revenue_gbp']:,.2f} revenue, {p['segment_lift']:.2f}x lift"
    else:
        block += "\n- No products met both the Lift > 1 and minimum-buyer thresholds."
    return block


def _prompt(question: str, objective: str, language: str, facts: dict) -> str:
    return f"""You are a marketing campaign strategist. Create a concise campaign brief in {language}.
Use the facts only to reason. A trusted Python component displays all facts separately. Do not repeat, paraphrase,
translate, rename, or explain the segment, metrics, customer counts, currency values, or observed behavior. Never
write any number, currency amount, percentage, product name, discount, margin, conversion rate, or uplift unless it
appears verbatim in USER REQUEST. Individual product preferences are unavailable. You may mention only exact product
names from popular_product_candidates or distinctive_product_candidates and only as historically popular or
distinctive at segment level; never call them a customer's favorite. Do not claim membership, VIP status, new
arrivals, availability, stock, exclusivity, or current promotions. Never output unresolved template placeholders.
If data is missing, put it under DATA NEEDED and say it requires approval.
Use exactly these sections: RECOMMENDATION, MESSAGE DRAFTS, KPIS AND A/B TEST, DATA NEEDED. Do not claim causal impact.

BUSINESS OBJECTIVE: {objective}
USER REQUEST: {question}
FACTS: {json.dumps(facts, ensure_ascii=False)}
"""


def _rules_brief(objective: str, language: str, facts: dict) -> str:
    if language == "Thai":
        return f"""### ข้อเสนอแนะ

**แนวทางข้อเสนอ:** {facts['rule_based_offer']}  
**ช่องทาง:** {facts['rule_based_channel']}  
**KPI:** Conversion rate, revenue per targeted customer, repeat purchase rate และ unsubscribe rate  
**การทดสอบ:** แบ่ง Holdout group ก่อนส่งจริง และเปรียบเทียบ incremental conversion/revenue  
**ข้อจำกัด:** ยังไม่มีข้อมูลต้นทุนสินค้า มาร์จิน และผล Campaign เดิม จึงยังไม่ควรกำหนดส่วนลดหรือคาดการณ์ uplift
"""
    return f"""### Recommendation

**Offer concept:** {facts['rule_based_offer']}  
**Channel:** {facts['rule_based_channel']}  
**KPIs:** conversion, revenue per targeted customer, repeat purchase, unsubscribe rate  
**Test:** retain a randomized holdout group and measure incremental conversion and revenue  
**Limitation:** product cost, margin, and prior campaign outcomes are unavailable; discount and uplift require approval/testing.
"""


def generate_campaign(question: str, objective: str, language: str, facts: dict,
                      provider: str = "Rules only", model: str | None = None) -> str:
    prompt = _prompt(question, objective, language, facts)
    if provider == "Rules only":
        return _rules_brief(objective, language, facts)
    if provider == "OpenAI":
        if not os.getenv("OPENAI_API_KEY"):
            raise RuntimeError("OPENAI_API_KEY is not set. Use Rules only or configure the environment variable.")
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise RuntimeError("Install dependencies from requirements.txt to use OpenAI.") from exc
        response = OpenAI().responses.create(model=model or os.getenv("OPENAI_MODEL", "gpt-4o-mini"), input=prompt)
        return response.output_text
    if provider == "Ollama":
        payload = json.dumps({
            "model": model or os.getenv("OLLAMA_MODEL", "qwen2.5:7b"),
            "messages": [{"role": "user", "content": prompt}], "stream": False,
        }).encode()
        endpoint = os.getenv("OLLAMA_URL", "http://localhost:11434/api/chat")
        try:
            with request.urlopen(request.Request(endpoint, data=payload, headers={"Content-Type": "application/json"}), timeout=120) as res:
                return json.loads(res.read())["message"]["content"]
        except Exception as exc:
            raise RuntimeError("Could not reach Ollama. Confirm Ollama is running and the model is installed.") from exc
    raise ValueError(f"Unknown provider: {provider}")


def audit_generated_campaign(text: str, facts: dict, question: str = "") -> list[str]:
    """Flag unsupported high-risk claims after generation; warnings do not alter the draft."""
    warnings = []
    placeholders = re.findall(r"\[[A-Za-z_][A-Za-z0-9_]*\]|\{\{[^{}]+\}\}", text)
    if placeholders:
        warnings.append("Unresolved template placeholder detected: " + ", ".join(dict.fromkeys(placeholders)))
    allowed_money = [float(x.replace(",", "")) for x in re.findall(r"£\s*([0-9][0-9,]*(?:\.[0-9]+)?)", question)]
    for raw in re.findall(r"£\s*([0-9][0-9,]*(?:\.[0-9]+)?)", text):
        value = float(raw.replace(",", ""))
        if not any(abs(value - allowed) <= max(1.0, abs(allowed) * .005) for allowed in allowed_money):
            warnings.append(f"LLM-generated or repeated currency amount detected: £{raw}")

    allowed_percentages = {p.replace(" ", "") for p in re.findall(r"\d+(?:\.\d+)?\s*%", question)}
    for pct in re.findall(r"\d+(?:\.\d+)?\s*%", text):
        if pct.replace(" ", "") not in allowed_percentages:
            warnings.append(f"Unapproved percentage detected: {pct.strip()}")

    for key in ("customers", "mean_recency_days", "mean_frequency"):
        if key not in facts:
            continue
        value = str(facts[key])
        if re.search(rf"(?<![\d.]){re.escape(value)}(?![\d.])", text) and value not in question:
            warnings.append(f"LLM repeated a protected fact ({key}={value}); use the Python facts section instead.")

    if not facts.get("product_level_preferences_available"):
        unsupported_phrases = [
            "personalized best seller", "favorite product", "products you love",
            "สินค้าที่คุณชอบ", "สินค้าที่คุณโปรดปราน", "รายชื่อสินค้า",
        ]
        if any(phrase.lower() in text.lower() for phrase in unsupported_phrases):
            warnings.append("Product personalization was suggested without product-preference data.")
    unsupported_claim_patterns = {
        r"คุณเป็นสมาชิก": "Membership status was claimed without membership data.",
        r"สมาชิกพิเศษของเรา": "Membership status was claimed without membership data.",
        r"คุณคือ\s*(?:vip|วีไอพี)": "VIP status was claimed without loyalty-tier data.",
        r"(?:มี|พบกับ)\s*(?:สินค้า|ของ)ใหม่": "New products were claimed without product launch data.",
        r"พร้อมส่ง|มีสินค้าในสต็อก": "Availability was claimed without inventory data.",
        r"you are (?:a |an )?(?:member|vip)": "Membership or VIP status was claimed without supporting data.",
        r"new arrivals|available now|in stock": "Newness or availability was claimed without inventory data.",
    }
    for pattern, message in unsupported_claim_patterns.items():
        if re.search(pattern, text, flags=re.IGNORECASE):
            warnings.append(message)
    return list(dict.fromkeys(warnings))
