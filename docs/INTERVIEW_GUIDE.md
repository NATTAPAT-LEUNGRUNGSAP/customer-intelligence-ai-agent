# Interview Guide

## 60-second project pitch

I built an end-to-end customer intelligence application using PostgreSQL, Python, scikit-learn, Streamlit, and an optional local or hosted LLM. It cleans transaction data with a persistent audit trail, engineers customer-level behavioral features, compares K-Means solutions, and combines ML clusters with explainable business-rule segments.

The key design decision is that the LLM does not make numerical or targeting decisions. Python selects the audience and product, locks the channels and experiment configuration in a `CampaignPlan`, and asks the model only for JSON message copy. The output is schema-checked and claim-checked, retried once, and replaced by deterministic Python copy if it still fails. This prevents fluent text from silently changing business decisions.

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

## Claims to avoid

Do not say the clusters are objectively correct customer types, the recommended product maximizes revenue, or the simulator predicts causal uplift. Say the solution produces stable behavioral hypotheses and auditable campaign plans that must be validated experimentally.
