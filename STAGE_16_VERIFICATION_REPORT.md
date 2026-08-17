# Stage 16 — Application Verification Report

**Project:** Customer Churn Intelligence  
**Test date:** 15 August 2026  
**Application owner:** Emmanuel Onuoha  
**Status:** Verified for local presentation and deployment preparation

## Verification Summary

| Test area | Result | Evidence |
|---|---|---|
| Calculations against source data | Passed | Raw and processed customer count, IDs, churn count, active count, monthly charges, and revenue reconcile. |
| Filters and dashboard interactions | Passed | Filter logic matches direct dataframe queries; live Contract filter changes the displayed population. |
| Model inputs and predictions | Passed | Production pipeline loads, probabilities remain bounded, predictor form submits, and eight local explanation factors render. |
| Missing-value behavior | Passed | Clean application data has no unresolved nulls; production imputers score numeric/categorical nulls and unseen categories safely. |
| Application performance | Passed | Warm Executive rerender: 1.75 seconds; tested interactions approximately 1.2–1.6 seconds. |
| Responsive visual rendering | Passed | Edge captures at 1440×1000, 1024×900, and 390×844; no horizontal page overflow. |
| Errors and edge cases | Passed | Empty filters, unknown filter fields, invalid campaign size/rates/costs, and unseen model categories are covered. |
| Clean-environment reproducibility | Passed | Fresh Python 3.12 virtual environment installed pinned dependencies, loaded 7,043 rows, reproduced probability 0.661674, and returned HTTP 200 from Streamlit. |

## Automated Test Result

`python -m pytest tests -q --durations=15`

- **24 passed**
- **0 failed**
- One expected scikit-learn warning confirms that an unseen categorical value is encoded safely with `handle_unknown="ignore"`.

## Source-Data Reconciliation

- Raw rows: 7,043
- Unique raw and processed customer IDs: 7,043
- Churned customers: 1,869
- Active customers: 5,174
- Churned-customer monthly charges: $139,130.85
- Raw and processed average monthly charge: exact match
- Raw and processed total revenue: exact match within floating-point tolerance
- High-risk active campaign population: 947 unique customers

## Model and Missing Values

The production pipeline contains median numeric imputation, most-frequent categorical imputation, standardized numeric inputs, and one-hot encoding with unknown-category handling. The tests replace monthly charge and contract with missing values and supply an unseen internet type; the model still returns a finite probability between zero and one.

The interactive predictor uses constrained widgets, preventing blank or out-of-range values during normal use. Post-outcome and leakage-prone fields remain excluded from prediction.

## Performance

The first automated Executive render includes Python/module/model initialization and took 7.10 seconds in the test runner. Cached warm rerender took 1.75 seconds. Filter, predictor, and simulator interaction tests completed in approximately 1.2–1.6 seconds. Data and model loading use Streamlit caching.

## Responsive Rendering

Real Edge browser checks were performed after waiting for the page title and all five KPI elements:

- Desktop: 1440×1000 — passed, no horizontal overflow
- Tablet: 1024×900 — passed, no horizontal overflow
- Mobile: 390×844 — passed, no horizontal overflow; sidebar starts collapsed
- Dark desktop: 1440×1000 — passed, no horizontal overflow; themed charts and controls rendered correctly

Evidence is stored under `artifacts/testing/screenshots/`. Responsive fixes include automatic sidebar state, readable multi-row KPI layout, sidebar-specific contrast, shorter chart titles, and a mobile CSS breakpoint.

## Error and Edge-Case Controls

- Missing project artifacts produce a clear startup error and stop execution.
- Empty customer-filter results produce an explanatory empty state.
- Unknown programmatic filters raise an explicit `KeyError`.
- Campaign sizes outside the eligible population raise an explicit `ValueError`.
- Rates outside 0–100%, negative costs, and invalid horizons are rejected.
- Geographic minimum-volume filtering has an empty-state response.
- The simulator clearly identifies estimates as planning assumptions rather than realized causal impact.

## Reproducibility and Deployment Readiness

- Production packages are pinned in `requirements.txt`.
- Python 3.12 is declared in `runtime.txt`.
- Test-only packages are pinned in `requirements-dev.txt`.
- Streamlit theme and headless configuration are stored in `.streamlit/config.toml`.
- A clean Python 3.12 environment successfully installed the pinned production requirements.
- Clean-process Streamlit health endpoint: HTTP 200, body `ok`.
- The temporary verification environment was removed after the successful test.

Public hosting is not part of this verification result. Deployment to Streamlit Community Cloud or another platform remains a separate external publishing action.

---

**Developed and Designed by Emmanuel Onuoha**
