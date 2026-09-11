# Customer Intelligence MCP Demonstration Guide

## What the MCP server is

`mcp_server.py` is the protocol boundary between an MCP host and the project's
read-only Python analytics. PostgreSQL is the data store, SQLAlchemy is the
database adapter, and OpenAI/Ollama are model providers; none of those is the
MCP server itself.

The server exposes twelve tools:

| Tool | Evidence returned |
|---|---|
| `get_segment_summary` | Aggregate RFM profiles |
| `compare_segments` | All behavioral segment profiles |
| `filter_customers` | Aggregate audience size and value, without IDs |
| `get_segment_products` | Observed product popularity or segment lift |
| `get_top_products` | Overall product ranking by revenue, units, orders, or customers |
| `get_product_catalog_summary` | Distinct product counts and category availability without inferred taxonomy |
| `recommend_campaign` | Deterministic target and product evidence |
| `simulate_campaign` | Deterministic planning scenario with a causal warning |
| `predict_churn` | Aggregate 90-day risk scores and temporal validation metrics |
| `predict_repeat_purchase` | Aggregate 30-day repeat scores by optional segment |
| `forecast_revenue` | Weekly forecast, candidate selection, reliability, uncertainty band, and temporal validation |
| `explain_predictive_model` | Method, label definition, metrics, and global drivers |

All tools are read-only. They reject raw SQL, perform no database writes, and
do not return `CustomerID`.

## Local demonstration with MCP Inspector

From PowerShell in the repository root:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\mcp.exe dev mcp_server.py
```

The Inspector requires Node.js/npm because it launches through `npx`.

1. Open the URL printed in PowerShell.
2. Select **Tools**.
3. Confirm that eleven tools appear.
4. Select `get_top_products`.
5. Call it with:

```json
{
  "limit": 10,
  "ranking": "revenue"
}
```

Show these four pieces of evidence to the reviewer:

1. The client discovered the tool instead of hard-coding a database query.
2. The form was generated from `inputSchema`.
3. The response appears in `structuredContent` and conforms to `outputSchema`.
4. The result identifies its source and contains aggregate product evidence,
   not customer identifiers.

## Data source

The server loads data lazily on the first call:

1. If `DATABASE_URL` is set, it reads the `transactions` table from PostgreSQL.
2. Otherwise it uses `MCP_DATA_PATH` when configured.
3. Otherwise it uses `data/online_retail.csv`, falling back to bundled demo data.

For the real PostgreSQL demonstration:

```powershell
$env:DATABASE_URL="postgresql+psycopg://portfolio:portfolio@localhost:5432/customer_intelligence"
.\.venv\Scripts\mcp.exe dev mcp_server.py
```

## Streamable HTTP

Start the local HTTP endpoint:

```powershell
.\.venv\Scripts\python.exe mcp_server.py --transport streamable-http
```

The endpoint is:

```text
http://127.0.0.1:8000/mcp
```

Docker can start PostgreSQL, Streamlit, and MCP together:

```powershell
docker compose up --build
```

This local endpoint has no authentication and must not be exposed publicly.
A public deployment needs HTTPS, authentication/authorization, origin and host
validation, restricted tools, audit logs, and approval for any future write tool.

## OpenAI Responses API connection

OpenAI needs a remotely reachable HTTPS MCP endpoint; it cannot access a
`localhost` endpoint on the presenter's computer.

```python
from openai import OpenAI

client = OpenAI()
response = client.responses.create(
    model="YOUR_SUPPORTED_MODEL",
    input="กลุ่ม Loyal High Value มีสินค้าอะไรบ้าง",
    tools=[{
        "type": "mcp",
        "server_label": "customer_intelligence",
        "server_url": "https://YOUR-DOMAIN.example/mcp",
        "allowed_tools": [
            "get_segment_summary",
            "get_segment_products",
            "simulate_campaign",
        ],
        "require_approval": "never",
    }],
)
print(response.output_text)
```

Keep `require_approval="always"` for any future tool that sends a campaign,
changes customer data, or performs another external side effect.

## Schema normalization

`src/contracts.py` is the canonical registry. Its schemas follow the strict
portable subset used by this project:

- every root is an object;
- every property is listed in `required`;
- optional values use a nullable type;
- every object sets `additionalProperties: false`;
- numeric ranges and enums are explicit;
- each tool defines both input and output schemas.

`openai_function_tool(name)` maps `input_schema` to OpenAI `parameters` and
sets `strict: true`. `mcp_tool_descriptor(name)` maps the same contract to MCP
`inputSchema` and `outputSchema`. The MCP Python SDK also derives its discovery
schema from the matching type hints and validates typed structured output.

Print both representations side by side:

```powershell
.\.venv\Scripts\python.exe scripts\show_api_contracts.py get_segment_products
```

Predictive example:

```powershell
.\.venv\Scripts\python.exe scripts\show_api_contracts.py predict_churn
```

MCP `structuredContent` and OpenAI Structured Outputs solve different problems:
the first contracts server-produced tool results; the second constrains
model-produced JSON.

## Short answer for a presentation

> Our MCP server exposes twelve read-only customer-intelligence tools. An MCP host
> discovers them through their schemas, chooses a tool, and sends validated
> arguments. Python—not the LLM—queries the prepared transaction data, performs
> segmentation, prediction, or forecasting, and returns schema-validated structured
> evidence. The LLM then explains that evidence. A canonical JSON Schema
> registry is adapted to both MCP and OpenAI strict function tools.
