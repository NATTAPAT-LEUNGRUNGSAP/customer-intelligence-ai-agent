# Acceptance Tests

## Release status

Core portfolio flow: **PASS**

| Area | Acceptance criterion | Evidence | Status |
|---|---|---|---|
| PostgreSQL | Dashboard loads the validated transaction table | 392,692 clean rows displayed | PASS |
| Import audit | Original row count survives database loading | 541,909 raw and 149,217 removed | PASS |
| Feature engineering | Customer and order totals are reproducible | 4,338 customers and 18,532 orders | PASS |
| Model selection | Automatic K uses recorded diagnostics | K=3, silhouette 0.259, stability 0.994 | PASS |
| Audience usability | No selected cluster is extremely small | Smallest cluster share 27.7% | PASS |
| Targeting | Business objective maps to a Python-ranked segment | Target and score table shown in UI | PASS |
| Product grounding | Product comes from observed segment transactions | Product evidence table and stock code displayed | PASS |
| Structured generation | LLM returns message-only JSON | Schema validated before rendering | PASS |
| Locked configuration | Product, channels, and control group cannot be changed by LLM | Locked configuration shown before generation | PASS |
| Repair | Invalid first LLM result receives one retry | Covered by adversarial test | PASS |
| Safe plan | Two invalid results activate same-configuration Python copy | Covered by adversarial test | PASS |
| Simulator | Discount-aware scenario arithmetic is correct | Covered by unit test | PASS |

## Screenshot evidence

### Data-quality audit

![Data-quality audit](images/overview-data-quality.png)

### Model selection

![K selection diagnostics](images/model-selection.png)

### Segment product evidence

![Observed segment and product evidence](images/segment-product-evidence.png)

### Locked and validated campaign

![Locked campaign configuration](images/campaign-locked-config.png)

![Validated campaign plan](images/validated-campaign-plan.png)

## Automated tests

Run locally:

```powershell
.\.venv\Scripts\python.exe -m pytest -q
```

Expected result:

```text
25 passed
```

The tests cover cleaning, mixed date formats, sub-cent prices, feature aggregation, segmentation, product lift, targeting, structured campaign constraints, invalid JSON repair, deterministic fallback, unsupported claims, and simulator arithmetic.

## Manual release check

Before recording a demo:

1. Start PostgreSQL and Streamlit.
2. Select PostgreSQL and confirm the raw/clean counts above.
3. Leave K on Auto and confirm K=3.
4. Run each business objective once.
5. Confirm the displayed product appears in every generated channel message.
6. Confirm the displayed control-group value appears once in the locked plan and no treatment-group value is generated.
7. Download the validated plan and confirm rejected model text is absent.
