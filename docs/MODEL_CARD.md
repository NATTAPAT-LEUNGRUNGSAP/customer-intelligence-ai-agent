# Model Card: Customer Behavioral Segmentation

## Intended use

This model groups retail customers with similar observed transaction behavior so an analyst can explore audiences and design testable campaigns. It supports portfolio demonstrations and decision support. It does not prove that a campaign will cause incremental sales.

## Dataset and cleaning

The demonstrated run uses the Online Retail transaction dataset. The PostgreSQL import audit records 541,909 raw rows and 392,692 retained rows. Rows are removed deterministically for duplicate records, missing customer IDs, invalid required values, cancellations, and non-positive quantity or price.

The database stores the latest original cleaning report in `data_import_audit`; therefore PostgreSQL mode does not incorrectly present already-cleaned rows as 100% retained.

## Customer-level features

| Feature | Meaning |
|---|---|
| `recency` | Days since the customer's latest purchase |
| `frequency` | Number of distinct invoices |
| `monetary` | Total observed transaction revenue |
| `avg_order_value` | Monetary value divided by distinct invoices |
| `unique_products` | Number of distinct stock codes purchased |

The model caps inputs at the 99th percentile, applies `log1p`, and then uses `RobustScaler`. Displayed customer facts remain uncapped.

## Algorithm and model selection

K-Means is fitted for candidate values `K=3..6` using `n_init=20` and a fixed random seed. Automatic selection combines:

- Silhouette score for separation
- Davies–Bouldin score as a secondary diagnostic
- Adjusted Rand Index across two seeds for stability
- Minimum cluster share to avoid unusably small audiences

Observed diagnostics:

| K | Silhouette | Davies–Bouldin | Stability | Smallest cluster | Selection score |
|---:|---:|---:|---:|---:|---:|
| 3 | 0.259 | 1.241 | 0.994 | 27.7% | 0.339 |
| 4 | 0.249 | 1.265 | 1.000 | 22.8% | 0.329 |
| 5 | 0.234 | 1.211 | 0.992 | 15.6% | 0.314 |
| 6 | 0.225 | 1.296 | 0.991 | 11.8% | 0.305 |

`K=3` is selected automatically. Manual K selection remains available for business comparison.

## Resulting profiles

| Behavioral persona | Customers | Mean recency | Mean frequency | Mean value | Mean order value |
|---|---:|---:|---:|---:|---:|
| Loyal High Value | 1,381 | 32.2 days | 9.4 | £5,025 | £489 |
| Occasional Premium | 1,755 | 129.3 days | 2.0 | £958 | £550 |
| Hibernating Low Value | 1,202 | 158.5 days | 1.7 | £222 | £143 |

Persona names are deterministic summaries of relative cluster profiles; K-Means itself produces only numeric labels.

## Campaign safeguards

Python owns the selected segment, product, channels, control-group percentage, KPIs, and launch requirements. The LLM is restricted to a JSON `messages` object. It cannot alter locked configuration fields.

Each message is validated for exact channel keys, the locked product, additional products, placeholders, unsupported individual purchase or preference claims, promotions, inventory, attributes, and price positioning. One rejected output is automatically repaired; a second failure activates Python copy built from the same `CampaignPlan`.

## Limitations

- A silhouette score of 0.259 indicates overlapping behavior, not perfectly separated natural groups.
- No product cost, margin, live inventory, approved promotion, or campaign-response history is available.
- Product ranking measures segment popularity/distinctiveness; it is not a causal recommendation model.
- Customer-level product preference is intentionally unavailable, so messages must not claim an individual previously bought or liked a product.
- The simulator is scenario analysis, not an uplift estimator.
- Cluster profiles may drift as customer behavior or the observation window changes.

## Responsible interpretation

Use segments as hypotheses for analysis and controlled experiments. Do not use them for credit, pricing discrimination, eligibility, or other high-impact decisions. Validate campaign impact with randomized holdouts and monitor opt-out and complaint rates.
