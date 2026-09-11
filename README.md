# Customer Intelligence AI Agent

Hosted demo with local Ollama: [setup guide](docs/LOCAL_LLM_WEB.md).
Includes an authenticated, limited gateway for scheduled demonstrations; PC uptime
and a running HTTPS tunnel are required. Public LLM access is not enabled by default.

Team Docker setup: see [TEAM_SETUP.md](docs/TEAM_SETUP.md). Use
`docker compose -f compose.team.yml up --build -d` for an isolated demo database
with automatic first-run seeding, Streamlit, and MCP. Local Ollama is optional.

An auditable customer-intelligence portfolio project that converts raw retail transactions into customer segments, product evidence, predictive scores, revenue forecasts, and guarded campaign briefs.

The system deliberately separates responsibilities: Python and SQL calculate facts, machine learning discovers behavior, business rules preserve explainability, and the LLM writes copy that must pass deterministic guardrails.

## Business problem

Marketing teams need to answer four connected questions: which customers behave similarly, which audience fits a business objective, which observed product can support the campaign, and how to create copy without allowing an LLM to invent customer facts. This application makes every numerical decision reproducible and keeps generated language inside a narrow, validated boundary.

**Stack:** Python · pandas · scikit-learn · PostgreSQL · SQLAlchemy · Streamlit · Plotly · MCP · Ollama/OpenAI · Docker · GitHub Actions

## Demonstrated result

The current PostgreSQL run processes the Online Retail dataset from raw transactions to a validated campaign plan:

| Result | Value |
|---|---:|
| Raw transaction rows | 541,909 |
| Clean transaction rows | 392,692 |
| Customers | 4,338 |
| Orders | 18,532 |
| Observed revenue | £8,887,209 |
| Selected behavioral clusters | 3 |
| Selected-K silhouette | 0.259 |
| Selected-K stability | 0.994 |

![Dashboard overview and persistent data-quality audit](docs/images/overview-data-quality.png)

The model selected `K=3` because it had the best combined separation/stability score while keeping the smallest audience at 27.7% of customers. The silhouette score is modest, so the project describes the groups as useful behavioral approximations rather than objectively true customer types.

![Automatic cluster-count diagnostics](docs/images/model-selection.png)

The campaign layer then locks decisions in Python before asking the LLM for message copy. In the demonstrated AOV scenario, the selected product, Email/LINE channels, and 10% control group remain identical in the final plan.

![Python-locked campaign configuration](docs/images/campaign-locked-config.png)

![Validated campaign plan](docs/images/validated-campaign-plan.png)

See [Model Card](docs/MODEL_CARD.md), [Acceptance Tests](docs/ACCEPTANCE_TESTS.md), and [Interview Guide](docs/INTERVIEW_GUIDE.md) for evaluation details and portfolio talking points.

## Dashboard walkthrough

### 1. PostgreSQL overview and data-quality audit

The dashboard can read the transaction table directly from PostgreSQL. The landing view reports the active data source, retained-row percentage, selected model run, customer count, order count, revenue, average order value, and the most important cleaning outcomes.

![PostgreSQL dashboard overview](docs/images/postgresql-dashboard-overview.png)

The complete cleaning report keeps every removal reason auditable and counts each rejected row once according to the cleaning order.

![Complete data-cleaning audit](docs/images/complete-cleaning-audit.png)

### 2. Customer and segment exploration

The overview compares the lifetime-spend distribution with the size of each behavioral audience.

![Customer spend distribution and segment mix](docs/images/customer-distribution-and-segment-mix.png)

Automatic model selection compares candidate values of `K` using separation, stability, Davies–Bouldin score, and minimum audience size rather than choosing the number of clusters manually without evidence.

![Candidate cluster evaluation table](docs/images/candidate-cluster-evaluation.png)

![Silhouette score by candidate K](docs/images/silhouette-by-candidate-k.png)

The behavioral scatter plot makes the three discovered audiences visible across recency and monetary value.

![Behavioral cluster scatter plot](docs/images/behavioral-cluster-scatter.png)

The hybrid cross-tab shows how ML-discovered behavior intersects with explainable business-rule segments.

![Hybrid ML and rule segment matrix](docs/images/hybrid-segment-matrix.png)

Cluster profiles summarize the audience size and typical RFM behavior used to assign human-readable personas.

![Behavioral cluster profiles](docs/images/cluster-profiles.png)

### 3. Campaign scenario simulator

The simulator estimates baseline revenue, campaign revenue, expected orders, and incremental revenue from explicit assumptions. Its output is a planning scenario—not causal evidence—and the interface directs users to validate uplift with a randomized A/B test.

![Campaign what-if simulator](docs/images/campaign-simulator.png)

## What the application delivers

- CSV upload, local CSV, or PostgreSQL transaction source
- Persistent import audit with original/clean row counts plus duplicate, missing, cancellation, and invalid-value counts
- Non-finite and overflow numeric rejection before values reach scikit-learn
- Customer-level RFM and behavioral feature engineering
- Hybrid segmentation: explainable lifecycle rules plus K-Means clustering
- Automatic K selection using silhouette, stability, Davies–Bouldin, and minimum audience size
- Segment profiles, cluster visualizations, and manual K comparison
- Grounded AI Analyst for Thai/English questions with validated structured intent
- Semantic guardrail that prevents an LLM from rerouting an explicit request to an unrelated tool
- Leakage-aware churn-risk prediction for no purchase in the next 90 days
- Repeat-purchase prediction for purchase in the next 30 days
- Weekly revenue forecasting for 1–12 weeks with temporal selection across Ridge, last-week, and four-week-mean methods
- Forecast reliability gate that hides aggregate totals when normalized validation error is high
- Temporal holdout metrics and global model-driver explanations
- A real read-only MCP server with discoverable tools and structured results
- Canonical JSON Schemas adapted to MCP and OpenAI strict function tools
- Segment-product questions such as `กลุ่ม loyalty มีสินค้าอะไรบ้าง`
- Product catalog counts that distinguish products from unavailable category taxonomy
- Overall top-product questions with explicit revenue, units, orders, or customer ranking
- Auditable tool routing across segment overview, comparison, customer filters, campaign recommendation, and simulation
- Optional LLM business interpretation with protected numerical evidence rendered by Python
- Popular products and distinctive products with segment lift greater than 1
- Deterministic campaign targeting for retention, reactivation, and average-order-value objectives
- Rules-only, OpenAI, and local Ollama message generation
- Python-owned `CampaignPlan` that locks product, channels, control group, KPIs, and launch requirements
- JSON message-only LLM output with schema validation and one automatic repair attempt
- Guardrails for unsupported claims inside message copy
- Deterministic Python plan from the same locked constraints when both LLM attempts fail
- What-if campaign simulator with an explicit A/B-testing disclaimer

## Architecture

```mermaid
flowchart TD
    A[CSV or PostgreSQL] --> B[Validation and cleaning]
    B --> C[Customer feature engineering]
    C --> D[Business-rule segments]
    C --> E[K-Means clusters]
    B --> P[Predictive models]
    D --> F[Auditable customer profile]
    E --> F
    Q[Natural-language question] --> R[Validated intent and tool router]
    M[MCP client or Inspector] --> N[Customer Intelligence MCP server]
    N --> C
    N --> G
    R --> F
    R --> P
    F --> G[Target and product evidence]
    G --> H[Python locks CampaignPlan]
    H --> I[LLM writes message-only JSON]
    I --> J{Schema and claim checks}
    J -->|Pass| K[Rendered validated plan]
    J -->|Fail twice| L[Python copy from same plan]
```

## Repository map

| Path | Responsibility |
|---|---|
| `dashboard/streamlit_app.py` | Interactive dashboard and user workflow |
| `src/data.py` | Shared validation and cleaning for every source |
| `src/database.py` | PostgreSQL read/write adapter |
| `src/features.py` | RFM and behavioral features |
| `src/segmentation.py` | Rules, clustering, model selection, and personas |
| `src/predictive.py` | Temporal churn/repeat models and weekly revenue forecast |
| `src/products.py` | Overall and segment-level product rankings from observed transactions |
| `src/analyst.py` | Structured intent, approved tool routing, evidence, and validated interpretation |
| `src/agent.py` | Targeting, LLM generation, guardrails, and safe fallback |
| `src/contracts.py` | Canonical strict JSON Schemas and MCP/OpenAI adapters |
| `src/business_tools.py` | Read-only analytics shared by MCP and application adapters |
| `mcp_server.py` | Discoverable MCP tools over stdio or Streamable HTTP |
| `src/campaign.py` | Rule-based strategy and campaign simulator |
| `scripts/load_csv_to_postgres.py` | Validated CSV-to-PostgreSQL loader |
| `sql/schema.sql` | Database schema and indexes |
| `tests/test_core.py` | Core data, ML, product, and agent tests |
| `tests/test_predictive.py` | Leakage, predictive routing, contracts, and model tests |
| `docs/MODEL_CARD.md` | Model assumptions, evaluation, limitations, and appropriate use |
| `docs/ACCEPTANCE_TESTS.md` | Evidence-backed release checklist |
| `docs/INTERVIEW_GUIDE.md` | Short project pitch and likely technical questions |

## Quick start on Windows

From PowerShell inside the project folder:

```powershell
py -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m streamlit run dashboard\streamlit_app.py
```

Open `http://localhost:8501`. If `data/online_retail.csv` is absent, the app clearly labels the bundled synthetic dataset as Demo data.

## Use the Kaggle Online Retail dataset

Save the transaction CSV as:

```text
data/online_retail.csv
```

Required columns:

```text
InvoiceNo, StockCode, Description, Quantity, InvoiceDate, UnitPrice, CustomerID, Country
```

You can also upload the file from the sidebar without copying it into the repository. Do not commit customer data, API keys, or `.env`.

## Local Ollama

Install and start Ollama, then pull the default model:

```powershell
ollama pull qwen2.5:7b
```

Select `Ollama` in the Campaign Agent tab. All customer metrics and campaign decisions remain Python-calculated; Ollama returns message-only JSON.

## PostgreSQL workflow

Copy `.env.example` to `.env`, then start PostgreSQL:

```powershell
docker compose up -d db
```

Load the validated dataset:

```powershell
.\.venv\Scripts\python.exe scripts\load_csv_to_postgres.py data\online_retail.csv
```

If the table already contains rows and you intentionally want to replace them:

```powershell
.\.venv\Scripts\python.exe scripts\load_csv_to_postgres.py data\online_retail.csv --replace
```

Run the `--replace` command once after upgrading from an earlier project version so the original CSV cleaning report is recorded for the Dashboard.

Start Streamlit, select `PostgreSQL` in the sidebar, and use:

```text
postgresql+psycopg://portfolio:portfolio@localhost:5432/customer_intelligence
```

The loader refuses to append to a non-empty table unless `--replace` is explicitly supplied, preventing accidental duplicate imports. It also stores the original cleaning report in `data_import_audit`, so PostgreSQL mode retains the CSV's pre-cleaning row counts and removal reasons.

## Run everything with Docker

```powershell
docker compose up --build
```

The dashboard will be available at `http://localhost:8501`. It starts with Demo CSV data; load PostgreSQL separately when you want to demonstrate the database path.

The MCP endpoint is available locally at `http://localhost:8000/mcp`. It is intentionally unauthenticated for local portfolio demonstration only; add HTTPS authentication before any public deployment.

## Demonstrate the MCP server

Install the dependencies, then launch the official MCP Inspector from PowerShell:

```powershell
.\.venv\Scripts\mcp.exe dev mcp_server.py
```

Open the URL printed by the command, choose **Tools**, and call `get_segment_products` with:

```json
{
  "segment": "loyalty",
  "limit": 5,
  "ranking": "popular"
}
```

To run a local Streamable HTTP endpoint instead:

```powershell
.\.venv\Scripts\python.exe mcp_server.py --transport streamable-http
```

To show that MCP and OpenAI are generated from the same canonical input contract:

```powershell
.\.venv\Scripts\python.exe scripts\show_api_contracts.py get_segment_products
```

Replace the tool name with `predict_churn`, `predict_repeat_purchase`,
`forecast_revenue`, or `explain_predictive_model` to inspect a predictive
contract. The first predictive request trains and caches the models in the
running process; production deployment should train, version, and promote model
artifacts offline.

See [MCP Demonstration Guide](docs/MCP_GUIDE.md) for the presentation flow, schemas, security boundary, and remote OpenAI connection example.

## Run tests

```powershell
.\.venv\Scripts\python.exe -m pytest -q
```

The repository currently contains core, adversarial, schema, and in-memory MCP integration tests. GitHub Actions runs the same test suite on every push and pull request.

## AI Analyst examples

Open the `AI Analyst` tab and ask in Thai or English, for example:

```text
หาลูกค้าที่มีมูลค่าสูงและไม่ได้ซื้อมากกว่า 90 วัน
สินค้ามีกี่ประเภท
10 สินค้าที่ขายดีที่สุด
10 สินค้าที่ขายได้จำนวนชิ้นมากที่สุด
เปรียบเทียบแต่ละ behavioral segment ให้หน่อย
แนะนำกลุ่มเป้าหมายสำหรับแคมเปญดึงลูกค้ากลับมา
ทำนายลูกค้าที่เสี่ยง churn มากกว่า 60%
กลุ่ม loyalty มีโอกาสซื้อซ้ำภายใน 30 วันเท่าไร
พยากรณ์รายได้ 4 สัปดาห์ข้างหน้า
ถ้ามีลูกค้า 1000 คน baseline conversion 4% เพิ่มเป็น 7% AOV 50 และส่วนลด 10% จะเป็นอย่างไร
```

With OpenAI or Ollama selected, the LLM returns a schema-constrained intent and a qualitative business interpretation. Python validates the intent, runs only an allow-listed analytical function or predictive model, calculates every metric, and exposes an evidence block plus tool trace. The LLM does not calculate risk probabilities or forecasts. Invalid model output falls back to deterministic intent rules or a safe Python interpretation. The analyst never executes model-written SQL and never writes to the database.

## Why hybrid segmentation?

Rules answer stable business questions such as “new,” “champion,” and “at risk.” K-Means discovers behavioral patterns without assuming those thresholds. Keeping both avoids pretending that an unsupervised cluster is automatically a business persona.

## Why SQL instead of a vector database?

Transactions, prices, quantities, and customer IDs are structured facts, so SQL provides exact filtering and aggregation. A vector database becomes useful only when the project adds unstructured product descriptions, brand guidelines, campaign documents, or customer feedback.

## Structured campaign behavior

Python compiles the user's request and observed evidence into a locked `CampaignPlan`. Product, channels, control-group percentage, KPIs, and launch requirements are never generated by the LLM. The model returns only a JSON `messages` object whose keys must exactly equal the approved channels, and every message must use the same Python-selected product.

Each message is checked for placeholders and unsupported individual purchase, preference, inventory, promotion, product-attribute, and price claims. A rejected response receives one repair attempt. If that also fails, Python creates safe copy from the exact same locked plan; rejected model text is inspectable but never downloadable as an approved result.

## Portfolio talking point

> SQL handles structured facts, ML discovers customer behavior, and Python owns campaign decisions. The LLM is restricted to message-only JSON; schema validation, claim checks, one repair attempt, and a same-plan fallback prevent free-form text from changing products, channels, or experiment settings.

## Next production steps

1. Persist model-run metadata and segment assignments.
2. Add product margin, inventory, and approved-offer tables.
3. Track campaign delivery and outcome events.
4. Evaluate incremental uplift with randomized holdouts.
5. Add RAG only for approved unstructured marketing knowledge.
6. Persist, calibrate, monitor, and version predictive-model artifacts.
