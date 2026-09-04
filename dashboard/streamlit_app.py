from io import BytesIO
import os
from pathlib import Path
import sys
import numpy as np
import plotly.express as px
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.data import load_transactions_with_report, resolve_default_data
from src.database import load_transactions_from_database
from src.features import build_customer_features
from src.segmentation import segment_customers
from src.campaign import simulate_campaign
from src.agent import (OBJECTIVES, build_campaign_constraints, generate_campaign_plan,
                       grounded_facts, render_campaign_plan, render_observed_facts,
                       select_target_segment)

try:
    from dotenv import load_dotenv
    load_dotenv(ROOT / ".env")
except ImportError:
    pass


@st.cache_data(show_spinner=False)
def load_csv_path(path: str):
    return load_transactions_with_report(path)


@st.cache_data(show_spinner=False)
def load_csv_bytes(contents: bytes):
    return load_transactions_with_report(BytesIO(contents))


@st.cache_data(ttl=300, show_spinner=False)
def load_database(database_url: str):
    return load_transactions_from_database(database_url)


@st.cache_data(show_spinner=False)
def run_segmentation(transactions, selected_k):
    customer_features = build_customer_features(transactions)
    segmentation = segment_customers(customer_features, selected_k=selected_k)
    return customer_features, segmentation


st.set_page_config(page_title="Customer Intelligence Agent", page_icon="📊", layout="wide")
st.markdown("""
<style>
    .block-container {padding-top: 2rem; padding-bottom: 3rem; max-width: 1450px;}
    [data-testid="stMetric"] {background: #f7f9fc; border: 1px solid #e5e9f0; padding: 1rem; border-radius: .75rem;}
    [data-testid="stSidebar"] {border-right: 1px solid #e5e9f0;}
    .stTabs [data-baseweb="tab-list"] {gap: .5rem;}
</style>
""", unsafe_allow_html=True)
st.title("Customer Intelligence AI Agent")
st.caption("Raw transactions → auditable cleaning → hybrid segmentation → grounded campaign decisions")

st.sidebar.header("Data connection")
source_mode = st.sidebar.radio("Source", ["CSV", "PostgreSQL"], horizontal=True)
using_demo_data = False
try:
    with st.spinner("Loading and validating transactions..."):
        if source_mode == "PostgreSQL":
            database_url = st.sidebar.text_input(
                "DATABASE_URL", value=os.getenv("DATABASE_URL", ""), type="password",
                help="Example: postgresql+psycopg://user:password@localhost:5432/database",
            )
            if not database_url:
                st.info("Enter DATABASE_URL in the sidebar to use PostgreSQL.")
                st.stop()
            tx, quality = load_database(database_url)
            source_name = "PostgreSQL · transactions"
        else:
            uploaded = st.sidebar.file_uploader("Upload transaction CSV", type="csv")
            if uploaded:
                tx, quality = load_csv_bytes(uploaded.getvalue())
                source_name = f"Uploaded CSV · {uploaded.name}"
            else:
                source = resolve_default_data(ROOT)
                tx, quality = load_csv_path(str(source))
                source_name = f"Local CSV · {source.name}"
                using_demo_data = source.name == "sample_transactions.csv"

    st.sidebar.success(source_name)
    if using_demo_data:
        st.warning(
            "Demo data active: results use synthetic customers and products. "
            "Upload the Kaggle CSV or save it as data/online_retail.csv for portfolio results."
        )
    cluster_choice = st.sidebar.selectbox(
        "Behavioral cluster count",
        ["Auto", 3, 4, 5, 6],
        help="Auto uses statistical diagnostics. Manual selection lets you compare business usefulness.",
    )
    manual_k = None if cluster_choice == "Auto" else int(cluster_choice)
    with st.spinner("Building customer features and segments..."):
        features, result = run_segmentation(tx, manual_k)
except Exception as exc:
    st.error(str(exc)); st.stop()

retention_rate = quality.clean_rows / quality.raw_rows if quality.raw_rows else 0
st.caption(
    f"Source: **{source_name}** · Retained **{retention_rate:.1%}** of raw rows · "
    f"Model run: **K={result.selected_k}**"
)

overview, explorer, agent, simulator = st.tabs(["Overview", "Segmentation", "Campaign Agent", "Simulator"])
with overview:
    a, b, c, d = st.columns(4)
    a.metric("Customers", f"{features.CustomerID.nunique():,}")
    b.metric("Orders", f"{tx.InvoiceNo.nunique():,}")
    c.metric("Revenue", f"£{tx.Revenue.sum():,.0f}")
    order_revenue = tx.groupby("InvoiceNo").Revenue.sum()
    d.metric("Average order value", f"£{order_revenue.mean():,.0f}")
    st.subheader("Data quality audit")
    q1, q2, q3, q4, q5 = st.columns(5)
    q1.metric("Raw rows", f"{quality.raw_rows:,}")
    q2.metric("Clean rows", f"{quality.clean_rows:,}")
    q3.metric("Rows removed", f"{quality.raw_rows - quality.clean_rows:,}")
    q4.metric("Missing customer ID", f"{quality.missing_customer_id_removed:,}")
    q5.metric("Cancelled removed", f"{quality.cancelled_rows_removed:,}")
    if quality.invalid_invoice_date_removed:
        st.warning(f"Could not parse {quality.invalid_invoice_date_removed:,} invoice dates. Check the source date format.")
    with st.expander("View complete cleaning report"):
        st.json(quality.to_dict())
        st.caption("Counts follow the cleaning order, so every removed row is counted once.")
    chart_data = features.assign(log_monetary=lambda d: np.log1p(d.monetary))
    left, right = st.columns(2)
    left.plotly_chart(px.histogram(chart_data, x="log_monetary", nbins=45,
                                  title="Customer lifetime spend (log scale)",
                                  labels={"log_monetary": "log(1 + monetary)"}),
                      use_container_width=True)
    composition = result.customers.groupby("cluster_persona").size().rename("customers").reset_index()
    right.plotly_chart(px.bar(composition.sort_values("customers"), x="customers", y="cluster_persona",
                              orientation="h", title="Customer mix by behavioral segment",
                              labels={"cluster_persona": "Segment"}), use_container_width=True)
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
    constraints = build_campaign_constraints(question, objective, language, facts)
    st.write(f"**Recommended target:** {chosen}")
    st.caption("Targeting and campaign configuration are locked by Python. The language model can write message copy only.")
    p1, p2, p3 = st.columns(3)
    p1.metric("Locked product", constraints.product or "No evidenced product")
    p2.metric("Locked channels", " + ".join(constraints.channels))
    p3.metric("Locked control group", constraints.control_group_percentage or "Set before launch")
    with st.expander("View targeting evidence and scores"):
        st.dataframe(ranking.style.format({"customers":"{:,.0f}", "recency":"{:.1f}", "frequency":"{:.1f}",
                                           "monetary":"£{:,.0f}", "avg_order_value":"£{:,.0f}",
                                           "target_score":"{:.3f}"}), use_container_width=True)
        st.json(facts)
    if st.button("Generate / regenerate campaign brief", type="primary"):
        try:
            with st.spinner("Building a grounded campaign brief..."):
                generation = generate_campaign_plan(question, objective, language, facts, provider, model)
            facts_block = render_observed_facts(facts, language)
            approved_plan = render_campaign_plan(generation.plan)
            st.markdown(facts_block)
            if generation.rejected_outputs:
                if generation.used_fallback:
                    st.error(
                        "Both structured AI attempts failed validation. "
                        "Python replaced them with a plan built from the same locked configuration."
                    )
                else:
                    st.warning("The first structured AI attempt was rejected and automatically repaired.")
                with st.expander("View rejected AI attempts and validation errors"):
                    st.write(generation.rejected_errors)
                    for index, rejected in enumerate(generation.rejected_outputs, start=1):
                        st.code(rejected, language="json")
                        st.caption(f"Rejected attempt {index}")
            st.markdown(approved_plan)
            if generation.used_fallback:
                st.info("Deterministic Python copy is active; no rejected AI text is downloadable.")
                file_name = "campaign_plan_safe.md"
                button_label = "Download validated safe plan"
            else:
                source = "Python" if provider == "Rules only" else provider
                st.success(f"Validated structured plan ready. Message copy source: {source}.")
                file_name = "campaign_plan_validated.md"
                button_label = "Download validated plan"
            full_brief = facts_block + "\n\n" + approved_plan
            st.download_button(button_label, full_brief, file_name=file_name, mime="text/markdown")
        except Exception as exc:
            st.error(str(exc))
with simulator:
    n = st.number_input("Target customers", 1, value=1000)
    base = st.slider("Baseline conversion", 0.0, 0.5, 0.04, 0.01)
    expected = st.slider("Expected conversion", 0.0, 0.5, 0.07, 0.01)
    aov = st.number_input("Average order value", 1.0, value=50.0)
    discount = st.slider("Discount rate", 0.0, 0.5, 0.10, 0.01)
    sim = simulate_campaign(n, base, expected, aov, discount)
    s1, s2, s3, s4 = st.columns(4)
    s1.metric("Baseline revenue", f"£{sim['baseline_revenue']:,.0f}")
    s2.metric("Campaign revenue", f"£{sim['campaign_revenue_after_discount']:,.0f}")
    s3.metric("Expected orders", f"{sim['expected_orders']:,.0f}")
    s4.metric("Incremental revenue", f"£{sim['incremental_revenue']:,.0f}")
    st.caption("Scenario estimate—not observed causal uplift. Validate with an A/B test.")
