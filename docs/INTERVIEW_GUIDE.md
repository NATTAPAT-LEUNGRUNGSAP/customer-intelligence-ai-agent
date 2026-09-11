# Interview Guide

## 60-second project pitch

I built an end-to-end customer intelligence application using PostgreSQL, Python, scikit-learn, Streamlit, MCP, and an optional local or hosted LLM. It cleans transaction data with a persistent audit trail, combines K-Means with explainable business-rule segments, predicts 90-day churn risk and 30-day repeat purchase, and forecasts weekly revenue.

The key design decision is that the LLM does not make numerical or targeting decisions. In the AI Analyst, it may classify a question into a validated intent and explain Python-owned evidence; an allow-list prevents it from executing arbitrary SQL or database writes. In the Campaign Agent, Python selects the audience and product, locks the channels and experiment configuration in a `CampaignPlan`, and asks the model only for JSON message copy. Invalid model output falls back to deterministic Python behavior. This prevents fluent text from silently changing business decisions.

## What makes the analyst agentic?

The user asks a business question rather than selecting a fixed report. The model can convert that request into a structured intent, while Python validates it and routes it to an approved descriptive, predictive, campaign, or simulation tool. The answer exposes its intent, evidence, and tool trace. This is bounded tool orchestration rather than unrestricted model autonomy.

For explicit requests, Python also compares the model-selected action with a
deterministic semantic rule. A mismatch fails closed to the explicit action, so
a fluent but incorrect model classification cannot silently invoke an unrelated
predictive tool.

For example, “ten best-selling products” routes to `get_top_products`. Python
defines the otherwise ambiguous word “best-selling” as total revenue by default;
the user can instead request units, distinct orders, or distinct customers.
In contrast, “how many product categories are there?” routes to
`get_product_catalog_summary`. The Online Retail source has product codes and
descriptions but no category field, so the tool reports that categories are
unavailable instead of manufacturing a taxonomy from product-name keywords.

## Does the LLM predict churn or revenue?

No. The LLM interprets the request, selects an allowed tool, and explains the
returned evidence. Scikit-learn creates probabilities and forecasts from
historical data. Keeping language orchestration separate from statistical
calculation makes the result testable and prevents the model from inventing a
number.

For revenue forecasting, the system holds out the latest complete weeks and
compares Ridge with last-week and four-week-mean baselines. It deploys the
lowest-MAE candidate and labels reliability from normalized holdout error. A
low-reliability run does not display an aggregate forecast total.

## How did you prevent predictive leakage?

Customer features are calculated only from transactions at or before each
historical snapshot. Labels look forward 30 days for repeat purchase or 90 days
for churn. Validation uses the latest eligible snapshots, never a random row
split that mixes the same time periods.

## Why use both rules and clustering?

Rules provide stable definitions such as New Customers, Champions, and At Risk High Value. K-Means discovers combinations of behavior that fixed thresholds may miss. The application displays both so an analyst can compare statistical patterns with business interpretation.

## Why was K=3 selected?

K=3 had the highest combined selection score. Its silhouette score was 0.259, stability was 0.994, and its smallest cluster still contained 27.7% of customers. I would not claim perfect separation; I would describe the result as stable and operationally usable for this dataset.

## Why not let the LLM choose the campaign?

LLMs are useful for language generation but unreliable for exact IDs, amounts, channels, and experimental settings. Those decisions are compiled and validated by Python. The LLM receives only a narrow message-writing task.

## Why PostgreSQL instead of a vector database?

Customer IDs, orders, quantities, prices, and dates are structured facts. SQL provides exact filters, constraints, and reproducible aggregation. A vector database would become appropriate only after adding unstructured approved content such as brand guidelines, campaign documents, or customer feedback.

## What does the control group do?

It reserves a random subset of eligible customers who do not receive the campaign. Comparing treatment and control outcomes estimates incremental effect rather than counting customers who would have purchased anyway. The current application locks the requested percentage but does not yet execute campaign assignment or causal analysis.

## What would you build next in production?

1. Product margin, inventory, and approved-offer tables
2. Persisted model versions and customer assignments
3. Campaign delivery, exposure, conversion, and opt-out events
4. Randomized assignment and statistical uplift reporting
5. Drift monitoring and scheduled retraining
6. RAG only for approved unstructured marketing knowledge
7. Probability calibration, fairness checks, forecast covariates, and model registry

## Claims to avoid

Do not say the clusters are objectively correct customer types, the recommended product maximizes revenue, or the simulator predicts causal uplift. Say the solution produces stable behavioral hypotheses and auditable campaign plans that must be validated experimentally.
