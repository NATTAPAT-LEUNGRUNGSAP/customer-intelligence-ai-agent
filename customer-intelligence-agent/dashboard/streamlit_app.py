from pathlib import Path
import sys
import plotly.express as px
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.data import load_transactions_with_report, resolve_default_data
from src.features import build_customer_features
from src.segmentation import segment_customers
from src.campaign import simulate_campaign
from src.agent import (OBJECTIVES, audit_generated_campaign, generate_campaign,
                       grounded_facts, render_observed_facts, select_target_segment)

st.set_page_config(page_title="Customer Intelligence Agent", layout="wide")
st.title("Customer Intelligence AI Agent")
st.caption("Auditable segmentation and grounded campaign decisions")

uploaded = st.sidebar.file_uploader("Upload transaction CSV", type="csv")
source = uploaded if uploaded else resolve_default_data(ROOT)
try:
    tx, quality = load_transactions_with_report(source)
    features = build_customer_features(tx)
    cluster_choice = st.sidebar.selectbox(
        "Behavioral cluster count",
        ["Auto", 3, 4, 5, 6],
        help="Auto uses statistical diagnostics. Manual selection lets you compare business usefulness.",
    )
    manual_k = None if cluster_choice == "Auto" else int(cluster_choice)
    result = segment_customers(features, selected_k=manual_k)
except Exception as exc:
    st.error(str(exc)); st.stop()

overview, explorer, agent, simulator = st.tabs(["Overview", "Segmentation", "Campaign Agent", "Simulator"])
with overview:
    a, b, c = st.columns(3)
    a.metric("Customers", f"{features.CustomerID.nunique():,}")
    b.metric("Orders", f"{tx.InvoiceNo.nunique():,}")
    c.metric("Revenue", f"£{tx.Revenue.sum():,.0f}")
    st.subheader("Data quality audit")
    q1, q2, q3, q4 = st.columns(4)
    q1.metric("Raw rows", f"{quality.raw_rows:,}")
    q2.metric("Clean rows", f"{quality.clean_rows:,}")
    q3.metric("Missing customer ID", f"{quality.missing_customer_id_removed:,}")
    q4.metric("Cancelled removed", f"{quality.cancelled_rows_removed:,}")
    if quality.invalid_invoice_date_removed:
        st.warning(f"Could not parse {quality.invalid_invoice_date_removed:,} invoice dates. Check the source date format.")
    with st.expander("View complete cleaning report"):
        st.json(quality.to_dict())
        st.caption("Counts follow the cleaning order, so every removed row is counted once.")
    chart_data = features.assign(log_monetary=lambda d: __import__('numpy').log1p(d.monetary))
    st.plotly_chart(px.histogram(chart_data, x="log_monetary", nbins=45,
                                title="Customer lifetime spend (log scale)",
                                labels={"log_monetary": "log(1 + monetary)"}), use_container_width=True)
with explorer:
    if manual_k is None:
        st.write(f"Auto selected **K={result.selected_k}** using separation, stability, and minimum cluster size.")
    else:
        st.write(f"Showing manual **K={result.selected_k}**. Statistical auto-selection recommends **K={result.automatic_k}**.")
    st.dataframe(result.diagnostics.style.format({"silhouette":"{:.3f}", "davies_bouldin":"{:.3f}",
                                                  "stability":"{:.3f}", "smallest_cluster_share":"{:.1%}",
                                                  "selection_score":"{:.3f}"}), use_container_width=True)
    st.plotly_chart(px.line(result.diagnostics, x="k", y="silhouette", markers=True,
                            title="Silhouette by candidate K"), use_container_width=True)
    st.plotly_chart(px.scatter(result.customers, x="recency", y="monetary", size="frequency",
                               color="cluster_persona", hover_data=["CustomerID", "rule_segment"],
                               log_y=True, title="Behavioral clusters (monetary on log scale)"), use_container_width=True)
    st.dataframe(result.customers.groupby(["cluster_persona", "rule_segment"]).size().rename("customers").reset_index(), use_container_width=True)
    profiles = result.customers.groupby("cluster_persona").agg(
        customers=("CustomerID", "nunique"), recency=("recency", "mean"),
        frequency=("frequency", "mean"), monetary=("monetary", "mean"),
        avg_order_value=("avg_order_value", "mean")).reset_index()
    st.subheader("Cluster profiles")
    st.dataframe(profiles.style.format({"recency":"{:.1f}", "frequency":"{:.1f}",
                                        "monetary":"£{:,.0f}", "avg_order_value":"£{:,.0f}"}),
                 use_container_width=True)
    st.caption("Choose Auto or another K in the sidebar. Prefer a solution that is statistically stable and creates distinct, actionable audiences.")
with agent:
    st.subheader("Grounded Campaign Agent")
    objective = st.selectbox("Business objective", list(OBJECTIVES))
    question = st.text_area("What should the agent help with?", value="Create an actionable campaign for this objective.")
    col1, col2, col3 = st.columns(3)
    provider = col1.selectbox("Generation provider", ["Rules only", "OpenAI", "Ollama"])
    language = col2.selectbox("Output language", ["Thai", "English"])
    default_model = "gpt-4o-mini" if provider == "OpenAI" else "qwen2.5:7b"
    model = col3.text_input("Model", value=default_model, disabled=provider == "Rules only")
    chosen, subset, ranking = select_target_segment(result.customers, objective)
    facts = grounded_facts(chosen, subset, tx)
    st.write(f"**Recommended target:** {chosen}")
    st.caption("Target selection is calculated in Python. The language model only writes the grounded brief.")
    with st.expander("View targeting evidence and scores"):
        st.dataframe(ranking.style.format({"customers":"{:,.0f}", "recency":"{:.1f}", "frequency":"{:.1f}",
                                           "monetary":"£{:,.0f}", "avg_order_value":"£{:,.0f}",
                                           "target_score":"{:.3f}"}), use_container_width=True)
        st.json(facts)
    if st.button("Generate / regenerate campaign brief", type="primary"):
        try:
            with st.spinner("Building a grounded campaign brief..."):
                draft = generate_campaign(question, objective, language, facts, provider, model)
            facts_block = render_observed_facts(facts, language)
            st.markdown(facts_block)
            st.markdown("### AI recommendation" if language == "English" else "### คำแนะนำจาก AI")
            st.markdown(draft)
            warnings = audit_generated_campaign(draft, facts, question)
            if warnings:
                st.warning("Guardrail review found claims that need human approval:\n\n- " + "\n- ".join(warnings))
            else:
                st.success("Guardrail review passed: no unsupported currency, percentage, or product-personalization claims detected.")
                full_brief = facts_block + "\n\n" + draft
                st.download_button("Download brief", full_brief, file_name="campaign_brief.md", mime="text/markdown")
        except Exception as exc:
            st.error(str(exc))
with simulator:
    n = st.number_input("Target customers", 1, value=1000)
    base = st.slider("Baseline conversion", 0.0, 0.5, 0.04, 0.01)
    expected = st.slider("Expected conversion", 0.0, 0.5, 0.07, 0.01)
    aov = st.number_input("Average order value", 1.0, value=50.0)
    discount = st.slider("Discount rate", 0.0, 0.5, 0.10, 0.01)
    sim = simulate_campaign(n, base, expected, aov, discount)
    st.metric("Simulated incremental revenue", f"£{sim['incremental_revenue']:,.0f}")
    st.caption("Scenario estimate—not observed causal uplift. Validate with an A/B test.")
