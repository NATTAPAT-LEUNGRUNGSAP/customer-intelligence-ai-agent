"""Grounded campaign agent: deterministic targeting, optional LLM copywriting."""
from __future__ import annotations

from dataclasses import dataclass, field
import json
import os
import re
from urllib import request

import numpy as np
import pandas as pd

from src.campaign import recommend_campaign
from src.contracts import campaign_copy_schema
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

CHANNEL_PATTERNS = {
    "Email": r"\bemail\b|อีเมล",
    "LINE": r"\bline\b|ไลน์",
    "App": r"\bapp\b|แอป(?:พลิเคชัน)?",
    "Website": r"\bweb(?:site)?\b|เว็บไซต์",
    "SMS": r"\bsms\b|ข้อความสั้น",
    "Phone": r"\bphone\b|\bcall center\b|โทรศัพท์|คอลเซ็นเตอร์",
    "Store": r"\bstore\b|หน้าร้าน",
    "Retargeting": r"\bretargeting\b|รีทาร์เก็ต",
}


@dataclass(frozen=True)
class CampaignConstraints:
    """Python-owned campaign decisions that an LLM cannot modify."""

    objective: str
    language: str
    product: str | None
    channels: tuple[str, ...]
    control_group_percentage: str | None


@dataclass
class CampaignPlan:
    """Validated campaign artifact rendered only after structural checks pass."""

    constraints: CampaignConstraints
    messages: dict[str, str]
    source: str
    kpis: tuple[str, ...] = (
        "Open rate",
        "Click-through rate",
        "Conversion rate",
        "Revenue per targeted customer",
    )
    data_needed: tuple[str, ...] = (
        "Product availability",
        "Product margin",
        "Approved offer",
        "Approved response destination",
    )


@dataclass
class CampaignGenerationResult:
    """Approved plan plus an audit trail of any rejected LLM attempts."""

    plan: CampaignPlan
    used_fallback: bool
    attempts: int
    rejected_outputs: list[str] = field(default_factory=list)
    rejected_errors: list[str] = field(default_factory=list)


def _minmax(series: pd.Series) -> pd.Series:
    series = pd.to_numeric(series, errors="coerce").replace([np.inf, -np.inf], np.nan)
    if series.notna().any():
        series = series.fillna(series.median())
    else:
        return pd.Series(0.5, index=series.index)
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
    # Prefer products that over-index in the segment. If none qualify, use
    # popular products as campaign creative evidence instead.
    campaign_products = (distinctive_products or popular_products)[:3]
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
        "campaign_product_candidates": campaign_products,
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


def _allowed_channels(facts: dict, question: str = "") -> list[str]:
    """Return channels explicitly supported by rules or the user's request."""
    evidence = f"{facts.get('rule_based_channel', '')} {question}"
    channels = [name for name, pattern in CHANNEL_PATTERNS.items()
                if re.search(pattern, evidence, flags=re.IGNORECASE)]
    for name, pattern in CHANNEL_PATTERNS.items():
        forbidden = (
            rf"(?:ห้าม(?:ใช้|ส่ง|กล่าวถึง)?|ไม่ใช้).{{0,20}}(?:{pattern})"
            rf"|(?:do not use|don't use|exclude|without).{{0,20}}(?:{pattern})"
        )
        if re.search(forbidden, question, flags=re.IGNORECASE) and name in channels:
            channels.remove(name)
    return channels


def _requested_control_group_percentages(question: str) -> list[str]:
    """Extract percentages explicitly tied to a requested holdout/control group."""
    group = r"(?:control\s*group|holdout(?:\s*group)?|กลุ่มควบคุม|กลุ่ม\s*holdout)"
    percentage = r"(\d+(?:\.\d+)?\s*%)"
    matches = []
    for pattern in (rf"{group}.{{0,40}}?{percentage}", rf"{percentage}.{{0,40}}?{group}"):
        matches.extend(re.findall(pattern, question, flags=re.IGNORECASE))
    return list(dict.fromkeys(value.replace(" ", "") for value in matches))


def build_campaign_constraints(question: str, objective: str, language: str,
                               facts: dict) -> CampaignConstraints:
    """Compile free-text intent into a small Python-owned decision surface."""
    products = [str(p.get("product_name", "")).strip()
                for p in facts.get("campaign_product_candidates", []) if p.get("product_name")]
    explicitly_named = [name for name in products if name.casefold() in question.casefold()]
    product = explicitly_named[0] if explicitly_named else (products[0] if products else None)
    channels = tuple((_allowed_channels(facts, question) or ["Email"])[:2])
    percentages = _requested_control_group_percentages(question)
    return CampaignConstraints(
        objective=objective,
        language=language,
        product=product,
        channels=channels,
        control_group_percentage=percentages[0] if percentages else None,
    )


def _copy_schema(constraints: CampaignConstraints) -> dict:
    return campaign_copy_schema(constraints.channels)


def _prompt(question: str, constraints: CampaignConstraints,
            previous_errors: list[str] | None = None) -> str:
    expected = {"messages": {channel: "customer-facing copy" for channel in constraints.channels}}
    repair = ""
    if previous_errors:
        repair = "\nPREVIOUS OUTPUT WAS REJECTED FOR: " + json.dumps(previous_errors, ensure_ascii=False)
    return f"""Write only customer-facing campaign copy in {constraints.language}.
Return one JSON object matching this exact shape: {json.dumps(expected, ensure_ascii=False)}
Do not add markdown, commentary, recommendation, KPIs, A/B configuration, data-needed fields, or extra JSON keys.
Every message must contain the exact product name {json.dumps(constraints.product, ensure_ascii=False)}.
The allowed message keys are exactly {json.dumps(list(constraints.channels), ensure_ascii=False)}.
Never use a customer name or placeholder. Never say the recipient previously bought, liked, or preferred the product.
Never claim a gift, discount, promotion, membership, VIP status, newness, availability, stock, exclusivity, product
attribute, price positioning, causal effect, or personal preference. Do not include any number, percentage, money,
link, URL, channel instruction, control group, treatment group, or A/B language. Use a generic invitation only.
Business objective: {constraints.objective}
User context (instructions only; do not repeat it): {question}{repair}
"""


def _parse_copy_json(raw: str) -> dict[str, str]:
    candidate = raw.strip()
    if candidate.startswith("```"):
        candidate = re.sub(r"^```(?:json)?\s*|\s*```$", "", candidate, flags=re.IGNORECASE)
    start, end = candidate.find("{"), candidate.rfind("}")
    if start < 0 or end < start:
        raise ValueError("The model did not return a JSON object.")
    payload = json.loads(candidate[start:end + 1])
    if not isinstance(payload, dict) or not isinstance(payload.get("messages"), dict):
        raise ValueError("JSON must contain a messages object.")
    return payload["messages"]


def _plan_from_messages(constraints: CampaignConstraints, messages: dict[str, str],
                        source: str) -> CampaignPlan:
    return CampaignPlan(constraints=constraints, messages=messages, source=source)


def build_safe_campaign_plan(question: str, objective: str, language: str,
                             facts: dict) -> CampaignPlan:
    """Create deterministic copy from the same locked constraints used for LLM output."""
    constraints = build_campaign_constraints(question, objective, language, facts)
    messages = {}
    for channel in constraints.channels:
        if language == "Thai":
            product_text = f"ขอแนะนำ {constraints.product} " if constraints.product else "ขอเชิญกลับมาเลือกชมสินค้า "
            messages[channel] = f"ยินดีต้อนรับกลับมา {product_text}หากสนใจสามารถตอบกลับผ่าน {channel} นี้ได้"
        else:
            product_text = f"We would like to introduce {constraints.product}. " if constraints.product else "We invite you to browse again. "
            messages[channel] = f"Welcome back. {product_text}Reply through this {channel} channel for more information."
    return _plan_from_messages(constraints, messages, "Python safe plan")


def validate_campaign_plan(plan: CampaignPlan, facts: dict, question: str = "") -> list[str]:
    """Validate structure, locked decisions, and residual natural-language claims."""
    errors = []
    expected_channels = set(plan.constraints.channels)
    actual_channels = set(plan.messages)
    if actual_channels != expected_channels:
        errors.append(
            "Message channels do not match the Python-locked channels: "
            f"expected {sorted(expected_channels)}, received {sorted(actual_channels)}."
        )
    candidate_products = [str(p.get("product_name", "")).strip()
                          for p in facts.get("campaign_product_candidates", []) if p.get("product_name")]
    for channel in plan.constraints.channels:
        message = plan.messages.get(channel)
        if not isinstance(message, str) or not message.strip():
            errors.append(f"{channel} message is missing or empty.")
            continue
        if plan.constraints.product and plan.constraints.product.casefold() not in message.casefold():
            errors.append(f"{channel} message omitted the Python-locked product.")
        other_products = [name for name in candidate_products
                          if name != plan.constraints.product and name.casefold() in message.casefold()]
        if other_products:
            errors.append(f"{channel} message added another product: {', '.join(other_products)}.")
        if re.search(r"\bA/B\b|control\s*group|treatment\s*group|holdout|กลุ่มควบคุม|กลุ่มทดลอง",
                     message, flags=re.IGNORECASE):
            errors.append(f"{channel} message contains experiment configuration owned by Python.")
    rendered_messages = "\n".join(f"{channel}: {message}" for channel, message in plan.messages.items())
    errors.extend(audit_generated_campaign(
        rendered_messages, facts, question, require_control_percentage=False
    ))
    return list(dict.fromkeys(errors))


def render_campaign_plan(plan: CampaignPlan) -> str:
    """Render locked fields and approved copy; the LLM never authors the surrounding plan."""
    c = plan.constraints
    product = c.product or ("ยังไม่มีสินค้าอ้างอิง" if c.language == "Thai" else "No evidenced product")
    messages = "\n".join(f"- **{channel}:** {plan.messages[channel]}" for channel in c.channels)
    control = c.control_group_percentage or ("กำหนดก่อนส่งจริง" if c.language == "Thai" else "Set before launch")
    if c.language == "Thai":
        return f"""### แผนแคมเปญที่ผ่านการตรวจสอบ

**ค่าที่ Python ล็อกไว้**
- สินค้า: `{product}`
- ช่องทาง: {', '.join(c.channels)}
- Control group: {control}

**ข้อความพร้อมใช้**
{messages}

**KPI:** {', '.join(plan.kpis)}

**ข้อมูลที่ต้องมีก่อนส่งจริง:** สถานะสินค้า, มาร์จิน, ข้อเสนอที่อนุมัติ และปลายทางตอบกลับที่อนุมัติ
"""
    return f"""### Validated campaign plan

**Python-locked configuration**
- Product: `{product}`
- Channels: {', '.join(c.channels)}
- Control group: {control}

**Ready-to-use messages**
{messages}

**KPIs:** {', '.join(plan.kpis)}

**Required before launch:** product availability, margin, approved offer, and approved response destination
"""


def build_safe_campaign_fallback(facts: dict, language: str, question: str = "",
                                 objective: str = "Campaign objective") -> str:
    """Backward-compatible rendered deterministic plan."""
    return render_campaign_plan(build_safe_campaign_plan(question, objective, language, facts))


def _call_copy_model(prompt: str, provider: str, model: str | None,
                     schema: dict) -> str:
    if provider == "OpenAI":
        if not os.getenv("OPENAI_API_KEY"):
            raise RuntimeError("OPENAI_API_KEY is not set. Use Rules only or configure the environment variable.")
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise RuntimeError("Install dependencies from requirements.txt to use OpenAI.") from exc
        response = OpenAI().responses.create(
            model=model or os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
            input=prompt,
            text={
                "format": {
                    "type": "json_schema",
                    "name": "validated_customer_intelligence_output",
                    "schema": schema,
                    "strict": True,
                }
            },
        )
        return response.output_text
    if provider == "Ollama":
        payload = json.dumps({
            "model": model or os.getenv("OLLAMA_MODEL", "qwen2.5:7b"),
            "messages": [
                {"role": "system", "content": "Return only JSON that follows the supplied schema."},
                {"role": "user", "content": prompt},
            ],
            "format": schema,
            "options": {"temperature": 0.1},
            "stream": False,
        }).encode()
        endpoint = os.getenv("OLLAMA_URL", "http://localhost:11434/api/chat")
        try:
            with request.urlopen(request.Request(endpoint, data=payload, headers={"Content-Type": "application/json"}), timeout=120) as res:
                return json.loads(res.read())["message"]["content"]
        except Exception as exc:
            raise RuntimeError("Could not reach Ollama. Confirm Ollama is running and the model is installed.") from exc
    raise ValueError(f"Unknown provider: {provider}")


def generate_campaign_plan(question: str, objective: str, language: str, facts: dict,
                           provider: str = "Rules only", model: str | None = None) -> CampaignGenerationResult:
    """Generate message-only JSON, retry once, then return a deterministic approved plan."""
    constraints = build_campaign_constraints(question, objective, language, facts)
    if provider == "Rules only":
        plan = build_safe_campaign_plan(question, objective, language, facts)
        return CampaignGenerationResult(plan=plan, used_fallback=False, attempts=0)

    rejected_outputs, rejected_errors = [], []
    previous_errors = None
    for attempt in range(1, 3):
        prompt = _prompt(question, constraints, previous_errors)
        raw = _call_copy_model(prompt, provider, model, _copy_schema(constraints))
        try:
            messages = _parse_copy_json(raw)
            plan = _plan_from_messages(constraints, messages, f"{provider} structured copy")
            errors = validate_campaign_plan(plan, facts, question)
        except (ValueError, json.JSONDecodeError) as exc:
            errors = [str(exc)]
        if not errors:
            return CampaignGenerationResult(
                plan=plan,
                used_fallback=False,
                attempts=attempt,
                rejected_outputs=rejected_outputs,
                rejected_errors=rejected_errors,
            )
        rejected_outputs.append(raw)
        rejected_errors.extend(errors)
        previous_errors = errors

    fallback = build_safe_campaign_plan(question, objective, language, facts)
    fallback_errors = validate_campaign_plan(fallback, facts, question)
    if fallback_errors:
        raise RuntimeError("Python safe plan failed validation: " + "; ".join(fallback_errors))
    return CampaignGenerationResult(
        plan=fallback,
        used_fallback=True,
        attempts=2,
        rejected_outputs=rejected_outputs,
        rejected_errors=list(dict.fromkeys(rejected_errors)),
    )


def generate_campaign(question: str, objective: str, language: str, facts: dict,
                      provider: str = "Rules only", model: str | None = None) -> str:
    """Backward-compatible entry point returning only the approved rendered plan."""
    result = generate_campaign_plan(question, objective, language, facts, provider, model)
    return render_campaign_plan(result.plan)


def audit_generated_campaign(text: str, facts: dict, question: str = "",
                             require_control_percentage: bool = True) -> list[str]:
    """Flag unsupported high-risk claims after generation; warnings do not alter the draft."""
    warnings = []
    placeholders = re.findall(r"\[[^\[\]\r\n]{1,120}\]|\{\{[^{}\r\n]+\}\}", text)
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
            "favourite product", "previously loved", "products you liked",
            "สินค้าที่คุณชอบ", "สินค้าที่คุณโปรดปราน", "ผลิตภัณฑ์ที่คุณชอบ",
            "ผลิตภัณฑ์ที่คุณโปรดปราน", "เคยชื่นชอบ", "เคยโปรดปราน", "รายชื่อสินค้า",
        ]
        preference_patterns = [
            r"(?:สินค้า|ผลิตภัณฑ์).{0,30}คุณ.{0,15}(?:ชอบ|ชื่นชอบ|โปรดปราน|ถูกใจ)",
            r"products?.{0,20}you.{0,12}(?:like|liked|love|loved|prefer|preferred)",
            r"(?:คุณ|ท่าน|ลูกค้ารายนี้).{0,20}เคย.{0,20}(?:ซื้อ|สั่งซื้อ|เลือกซื้อ)",
            r"\byou\b.{0,20}(?:previously\s+|have\s+)?(?:bought|purchased|ordered)\b",
        ]
        if (any(phrase.lower() in text.lower() for phrase in unsupported_phrases)
                or any(re.search(pattern, text, flags=re.IGNORECASE) for pattern in preference_patterns)):
            warnings.append("Product personalization was suggested without product-preference data.")
    unsupported_claim_patterns = {
        r"คุณเป็นสมาชิก": "Membership status was claimed without membership data.",
        r"สมาชิกพิเศษของเรา": "Membership status was claimed without membership data.",
        r"คุณคือ\s*(?:vip|วีไอพี)": "VIP status was claimed without loyalty-tier data.",
        r"(?:แนะนำ|พบกับ|มี|ลอง|เลือก|เปิดตัว).{0,12}(?:สินค้า|ผลิตภัณฑ์|ของ)(?:รุ่น)?ใหม่": "New products were claimed without product launch data.",
        r"พร้อมส่ง|มีสินค้าในสต็อก": "Availability was claimed without inventory data.",
        r"เลือกซื้อ(?:ใหม่|อีกครั้ง)?(?:ได้)?แล้ว|กลับมาวางขาย|พร้อมให้(?:เลือก)?ซื้อ": "Availability was claimed without inventory data.",
        r"(?:เรา|ขณะนี้|ตอนนี้).{0,20}(?:มี|จัด)(?:โปรโมชั่น|โปรโมชัน|ข้อเสนอ|สิทธิ)พิเศษ": "An active promotion was claimed without campaign approval data.",
        r"(?:โปรโมชั่น|โปรโมชัน|ข้อเสนอ|สิทธิ)พิเศษสำหรับ": "An active promotion was claimed without campaign approval data.",
        r"(?:ของขวัญ|คูปอง|สิทธิ).{0,20}(?:พิเศษ|สำหรับคุณ|มอบให้)": "A gift or incentive was claimed without campaign approval data.",
        r"you are (?:a |an )?(?:member|vip)": "Membership or VIP status was claimed without supporting data.",
        r"new arrivals|new (?:product|item)|available now|in stock": "Newness or availability was claimed without inventory data.",
        r"shop (?:it|them) again|back (?:on sale|in store)": "Availability was claimed without inventory data.",
        r"(?:our|a) special (?:promotion|offer)(?: is| for| now)": "An active promotion was claimed without campaign approval data.",
    }
    for pattern, message in unsupported_claim_patterns.items():
        if re.search(pattern, text, flags=re.IGNORECASE):
            warnings.append(message)

    required_products = [
        str(p.get("product_name", "")).strip()
        for p in facts.get("campaign_product_candidates", [])
        if p.get("product_name")
    ]
    if required_products and not any(name.casefold() in text.casefold() for name in required_products):
        warnings.append("No exact observed campaign product was used; regenerate using a Python-supplied product name.")

    allowed_channels = set(_allowed_channels(facts, question))
    for channel, pattern in CHANNEL_PATTERNS.items():
        if re.search(pattern, text, flags=re.IGNORECASE) and channel not in allowed_channels:
            warnings.append(f"Unsupported communication channel detected: {channel}.")

    attribute_scan = text
    for name in required_products:
        attribute_scan = re.sub(re.escape(name), "", attribute_scan, flags=re.IGNORECASE)
    unsupported_attribute_patterns = [
        r"คลาสสิก|พรีเมียม|คุณภาพสูง|สวยงาม|หรูหรา|ลักษณะ(?:เฉพาะ|น่าสนใจ)|น่าสนใจ|เหมาะสำหรับ|เหมาะกับ|ใช้สำหรับ|ตกแต่งบ้าน",
        r"\bclassic\b|\bpremium\b|\bunique\b|\binteresting\b|high[- ]quality|luxurious|perfect for|ideal for|suitable for",
    ]
    if (not facts.get("product_catalog_and_margin_available")
            and any(re.search(pattern, attribute_scan, flags=re.IGNORECASE)
                    for pattern in unsupported_attribute_patterns)):
        warnings.append("Unsupported product attributes or use cases were added without catalog evidence.")

    unsupported_price_patterns = [
        r"ราคา.{0,12}(?:ไม่แพง|ราคาถูก|คุ้มค่า|เหมาะสม|จับต้องได้)",
        r"\b(?:affordable|inexpensive|good value|value for money)\b",
    ]
    if (not facts.get("product_catalog_and_margin_available")
            and any(re.search(pattern, attribute_scan, flags=re.IGNORECASE)
                    for pattern in unsupported_price_patterns)):
        warnings.append("Unsupported product price positioning was added without catalog or benchmark evidence.")

    if re.search(r"(?:segment\s+)?lift\s*(?:of|=|:)?\s*\d", text, flags=re.IGNORECASE):
        warnings.append("LLM repeated a protected product-lift metric; use the Python facts section instead.")

    if (re.search(r"\bA/B\b", text, flags=re.IGNORECASE)
            and not re.search(r"holdout|control group|กลุ่มควบคุม", text, flags=re.IGNORECASE)):
        warnings.append("A/B plan does not specify a holdout or control group.")

    text_percentages = {p.replace(" ", "") for p in re.findall(r"\d+(?:\.\d+)?\s*%", text)}
    if require_control_percentage:
        for requested in _requested_control_group_percentages(question):
            if requested not in text_percentages:
                warnings.append(f"User-requested control-group percentage was omitted: {requested}.")

    for line in text.splitlines():
        if not re.search(r"\bemail\b|\bline\b|อีเมล|ไลน์", line, flags=re.IGNORECASE):
            continue
        names_in_line = {name for name in required_products if name.casefold() in line.casefold()}
        if len(names_in_line) > 1:
            warnings.append("More than one observed product was used in a single Email/LINE message.")
    return list(dict.fromkeys(warnings))
