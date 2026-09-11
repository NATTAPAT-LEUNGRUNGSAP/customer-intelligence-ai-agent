# Model Card: Customer Segmentation and Predictive Analytics

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

## AI Analyst safeguards

The optional LLM analyst is limited to two language tasks: selecting a schema-constrained intent and writing a qualitative interpretation. Python validates the intent and dispatches an allow-listed descriptive, predictive, campaign, or simulation tool.

The model cannot submit SQL, write to PostgreSQL, or supply executable code. Python calculates and renders all customer counts, monetary values, probabilities, forecasts, diagnostics, filters, rankings, and simulator outputs. The qualitative narrative rejects unsupported numeric claims and guaranteed-outcome language; rejected intent or narrative output falls back to deterministic Python behavior. The UI exposes the validated intent, evidence, tool trace, and fallback warnings.

When a question contains an explicit supported action, such as requesting a
campaign, forecast, churn prediction, or product ranking, a semantic guardrail
prevents the LLM from routing it to a different analytical tool. Non-finite and
overflow numeric inputs are rejected during cleaning before they can reach a
scikit-learn estimator.

## Predictive customer models

The classification models use rolling historical snapshots. Features contain
only transactions on or before each snapshot date; labels are constructed from
the future label window. The final eligible snapshot ends before the dataset's
maximum date by the prediction horizon, which reduces right-censoring and
prevents future transactions from leaking into features.

| Model | Operational label | Horizon | Algorithm |
|---|---|---:|---|
| Churn risk | No purchase after the snapshot | 90 days | Scaled logistic regression with balanced class weights |
| Repeat purchase | At least one purchase after the snapshot | 30 days | Scaled logistic regression with balanced class weights |

Inputs are recency, frequency, monetary value, average order value, unique
products, units, customer tenure, and observed purchase rate. Missing values are
median-imputed, non-negative values receive `log1p`, and features are
standardized. Validation is chronological: the most recent eligible snapshots
are held out instead of randomly mixing earlier and later customer states.

The application reports ROC-AUC, average precision, Brier score, validation
size, positive rate, and the exact training/validation snapshot boundary at
runtime. Probabilities are aggregate decision-support scores, not certainties.
Global drivers are model coefficients and do not explain an individual causal
reason.

## Revenue forecast

Observed positive weekly revenue is resampled to weeks ending Sunday, and a
trailing incomplete week is excluded. Three candidates are evaluated on the
same chronological holdout: log-linear Ridge with lag/trend/calendar features,
the previous observed week, and the rolling four-week mean. The method with the
lowest holdout MAE produces the operational forecast.

The application reports the candidate MAEs, selected-method MAE, RMSE, MAPE when
defined, normalized MAE, reliability level, and the exact temporal split.
Forecasts cover 1–12 weeks and include an empirical uncertainty band based on
the selected method's holdout residuals. Aggregate forecast revenue is hidden
in the dashboard when normalized MAE exceeds forty percent of mean validation
revenue.

Requests longer than the supported twelve-week horizon are capped and disclosed
to the user rather than silently extrapolated farther than the data supports.

This is a short-horizon time-series baseline. It does not know future promotions,
holidays, inventory, prices, or external market conditions.

Ridge recursive forecasts pass through a numerical safety layer. A non-finite value
falls back to the latest observed weekly revenue, and explosive extrapolation is
capped relative to recent observed history before it can become the next lag.

## Limitations

- A silhouette score of 0.259 indicates overlapping behavior, not perfectly separated natural groups.
- No product cost, margin, live inventory, approved promotion, or campaign-response history is available.
- Product ranking measures segment popularity/distinctiveness; it is not a causal recommendation model.
- Customer-level product preference is intentionally unavailable, so messages must not claim an individual previously bought or liked a product.
- The simulator is scenario analysis, not an uplift estimator.
- Churn is a behavioral proxy (no future purchase within 90 days), not a confirmed account closure.
- Repeat-purchase labels observe only the dataset's recorded transactions; purchases elsewhere are unknown.
- Predictive performance can change across time and businesses; the displayed temporal validation must be reviewed before use.
- Probability calibration, subgroup fairness, drift monitoring, and scheduled retraining are not productionized.
- Revenue uncertainty bands are empirical diagnostics, not guaranteed statistical coverage.
- Cluster profiles may drift as customer behavior or the observation window changes.
- Natural-language intent classification can misunderstand ambiguous requests; users must inspect the displayed structured intent and filters.
- The analyst's generated interpretation is decision support, not an autonomous approval or campaign-delivery mechanism.

## Responsible interpretation

Use segments as hypotheses for analysis and controlled experiments. Do not use them for credit, pricing discrimination, eligibility, or other high-impact decisions. Validate campaign impact with randomized holdouts and monitor opt-out and complaint rates.
