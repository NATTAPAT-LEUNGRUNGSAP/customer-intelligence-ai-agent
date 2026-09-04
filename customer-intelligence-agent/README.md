# Customer Intelligence AI Agent

Portfolio project that turns raw retail transactions into customer segments and actionable campaigns.

## What it demonstrates

- SQL-ready transaction model and data validation
- RFM feature engineering
- Hybrid segmentation: business rules + K-Means
- Auditable row-by-row cleaning summary
- Mixed-format date parsing with separate missing-ID and invalid-date counts
- Outlier-resistant clustering using percentile caps, log transform, and RobustScaler
- Cluster selection using separation, stability, and minimum audience size
- Auto or manual K comparison for statistical quality versus business usefulness
- Cluster personas and deterministic campaign recommendations
- Interactive Streamlit dashboard
- Grounded Campaign Agent with deterministic audience selection
- Rules-only, OpenAI, and local Ollama generation modes
- Post-generation guardrail audit for invented money, percentages, and unsupported personalization
- Deterministic Python facts block; LLM output is limited to recommendations and copy
- Segment-level product ranking using buyer penetration, revenue, orders, and lift
- Claim guardrails for membership, VIP, product newness, and inventory availability

The LLM is optional by design. All metrics, segments, and calculations come from Python. An LLM can later rewrite the grounded campaign brief without inventing numbers.

## Quick start

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python scripts/generate_sample_data.py
streamlit run dashboard/streamlit_app.py
```

Then open `http://localhost:8501`.

The Campaign Agent works immediately in `Rules only` mode. For OpenAI, set
`OPENAI_API_KEY` in your environment (never commit it). For local generation,
start Ollama and install the model named in `.env.example`.

## Use the Kaggle Online Retail dataset

Download the Online Retail transaction CSV and save it as `data/online_retail.csv`. The loader accepts the standard columns:

`InvoiceNo, StockCode, Description, Quantity, InvoiceDate, UnitPrice, CustomerID, Country`

Then run the dashboard. If that file is absent, it falls back to `data/sample_transactions.csv`.

## Architecture

```text
Raw CSV / PostgreSQL
        -> validation and cleaning
        -> customer-level RFM features
        -> rule-based lifecycle segment + K-Means behavior cluster
        -> cluster evaluation and persona
        -> campaign recommendation and what-if simulator
        -> Streamlit dashboard
```

## Why hybrid segmentation?

Rules make lifecycle labels stable and explainable (Champions, At Risk, New). K-Means discovers behavioral groups without assuming thresholds. Keeping both lets a marketer act on a familiar lifecycle label while still learning patterns from data.

## Repository map

- `src/data.py`: loading, cleaning, and schema validation
- `src/features.py`: RFM and supporting customer features
- `src/segmentation.py`: rules, K-Means, model selection, personas
- `src/campaign.py`: grounded campaign recommendations and simulator
- `dashboard/streamlit_app.py`: four-tab interactive application
- `tests/`: core unit tests
- `sql/schema.sql`: PostgreSQL starter schema

## Roadmap

1. Current: reproducible segmentation and campaign dashboard
2. Add PostgreSQL + FastAPI and persist model runs
3. Add product/catalog and campaign history tables
4. Add optional RAG only for unstructured product descriptions and brand guidelines
5. Measure campaign uplift with A/B tests; never present simulated revenue as observed impact

## Portfolio talking point

> SQL handles structured facts, ML discovers customer behavior, business rules preserve explainability, and the agent converts validated outputs into an auditable action plan.
