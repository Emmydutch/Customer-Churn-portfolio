# Telecom Customer Churn Intelligence

![Python 3.12](https://img.shields.io/badge/Python-3.12-3776AB?logo=python&logoColor=white)
![Streamlit 1.62.0](https://img.shields.io/badge/Streamlit-1.62.0-FF4B4B?logo=streamlit&logoColor=white)
[![Project tests](https://github.com/Emmydutch/customer-churn-portfolio/actions/workflows/tests.yml/badge.svg)](https://github.com/Emmydutch/customer-churn-portfolio/actions/workflows/tests.yml)
![Tests](https://img.shields.io/badge/tests-34%20passed-2A9D8F)
![License](https://img.shields.io/badge/license-MIT-17345B)
[![Live Streamlit app](https://img.shields.io/badge/Live%20dashboard-Open%20app-FF4B4B?logo=streamlit&logoColor=white)](https://customer-churn-portfolio-z3hvzwkxs3yu8zvenx9yuk.streamlit.app/)

An end-to-end telecom retention portfolio project combining business analysis, leakage-safe predictive modeling, explainable customer risk, campaign economics, and a presentation-ready Streamlit application.

<p align="center">
  <img src="assets/emmanuel-onuoha.jpg" alt="Emmanuel Onuoha" width="150">
</p>

<p align="center"><strong>Developed and Designed by Emmanuel Onuoha</strong></p>

<p align="center">
  <img src="artifacts/testing/deployed_screenshots/desktop-1440x1000.png" alt="Deployed Customer Churn Intelligence Streamlit dashboard" width="100%">
</p>

## At a Glance

| Portfolio outcome | Result |
|---|---:|
| Customers analyzed | 7,043 |
| Observed churn rate | 26.54% |
| High-risk active customers | 947 |
| Selected threshold | 0.32 |
| Holdout ROC-AUC | 89.85% |
| Holdout recall | 83.16% |
| Base scenario net benefit | $18,775 |
| Automated tests | 34 passed |

> The retention result is a transparent planning scenario, not realized causal impact. Campaign lift must be validated with a controlled pilot.

## Repository Guide

- **Open the live dashboard:** [Customer Churn Intelligence](https://customer-churn-portfolio-z3hvzwkxs3yu8zvenx9yuk.streamlit.app/).
- **Run the application locally:** follow [Run locally](#run-locally).
- **Review the full analysis:** open [`customer churn.ipynb`](customer%20churn.ipynb).
- **Inspect reusable code:** browse [`src/`](src/).
- **Review verification evidence:** read [`STAGE_16_VERIFICATION_REPORT.md`](STAGE_16_VERIFICATION_REPORT.md).
- **Read the concise portfolio story:** see [`PORTFOLIO_SUMMARY.md`](PORTFOLIO_SUMMARY.md).
- **Confirm data provenance:** see [`DATASET_NOTICE.md`](DATASET_NOTICE.md).

## Project Overview

This portfolio project delivers an evidence-based customer churn intelligence solution for a telecommunications company. It identifies where churn is concentrated, investigates why customers leave, estimates pre-churn customer risk without outcome leakage, and translates the findings into practical retention actions.

The completed solution combines a reproducible analytical notebook with an interactive Streamlit application for executive and operational decision-making.

> **Central business question:** Which customers are most likely to churn, why are they leaving, and what actions could improve retention?

## Business Objectives

- Measure churn consistently and quantify its commercial significance.
- Profile churned, retained, and newly joined customers separately.
- Identify demographic, contractual, service, billing, engagement, and geographic churn patterns.
- Determine the leading recorded categories and reasons for customer departure.
- Prioritize high-risk and high-value customer segments.
- Build an interpretable churn model using information available before departure.
- Recommend segment-specific retention actions.
- Present the results through an interactive Streamlit dashboard.

## Intended Users

The final dashboard is intended for executive leadership, customer retention and CRM teams, marketing, customer service, product and network teams, finance, and data analysts.

## Dataset

The analysis uses `TelcoCustomerChurn.csv`, the combined 50-column form of the **Telecom Customer Churn** dataset. The dataset originates from **IBM Cognos Analytics** and is distributed through the [Maven Analytics Data Playground](https://mavenanalytics.io/data-playground/telecom-customer-churn), which identifies it as **Public Domain**. It describes a fictional California telecommunications company rather than real subscribers.

| Attribute | Value |
|---|---:|
| Customer records | 7,043 |
| Variables | 50 |
| Unique customer IDs | 7,043 |
| Geographic coverage | California, United States |
| Reported period | Q3 |
| Churned customers | 1,869 |
| Active customers | 5,174 |

Because Maven identifies the dataset as Public Domain, publishing the fictional customer IDs and detailed geographic fields is permitted. The application nevertheless uses them conservatively: identifiers are excluded from modeling, geography is analyzed in aggregate, and no row-level customer download is exposed. See [`DATASET_NOTICE.md`](DATASET_NOTICE.md) for attribution, redistribution, and real-data safeguards.

## Data-Quality Summary

- No duplicated rows or customer IDs were found.
- No customer identifier is missing.
- All binary fields contain valid `Yes` and `No` values.
- Age, tenure, charge, count, coordinate, and score fields passed range checks.
- Customer status agrees with the churn label for every record.
- Revenue totals reconcile with their supplied charge components.
- Four columns contain structural rather than random blanks:
  - `Offer`: no recorded offer
  - `InternetType`: not applicable without internet service
  - `ChurnCategory`: available only for churned customers
  - `ChurnReason`: available only for churned customers

No rows currently require deletion because of duplication, invalidity, or missing identifiers.

## KPI Framework

| KPI | Definition | Baseline |
|---|---|---:|
| Total Customers | Distinct customer IDs | 7,043 |
| Churned Customers | `ChurnLabel = Yes` | 1,869 |
| Retained / Stayed Customers | `CustomerStatus = Stayed` | 4,720 |
| New Customers | `CustomerStatus = Joined` | 454 |
| Active Customers | Stayed + Joined | 5,174 |
| Overall Churn Rate | Churned ÷ Total | 26.54% |
| Portfolio Non-Churn Rate | Active ÷ Total | 73.46% |
| Established-Customer Retention | Stayed ÷ (Stayed + Churned) | 71.63% |
| Average Monthly Charge | Mean customer monthly charge | $64.76 |
| Average Tenure | Mean customer tenure | 32.39 months |
| Churned-Customer Accumulated Revenue | Historical total revenue from churners | $3,684,459.82 |
| Churned-Customer Monthly Revenue Exposure | Sum of monthly charges associated with churners | $139,130.85 |
| Average Customer Lifetime Value | Mean supplied CLTV | $4,400.30 |
| Churned-Customer CLTV | Sum of supplied CLTV for churners | $7,755,256 |
| High-Risk Active Customers | Active customers at or above the validated 0.32 threshold | 947 |

Historical revenue from churned customers is not presented as realized revenue loss. The high-risk KPI will be populated only after the project's independent model and intervention threshold have been validated; the supplied `ChurnScore` will not be substituted for that process.

## Leakage and Responsible-Use Controls

- `ChurnLabel` is the modeling target.
- `CustomerStatus` directly reveals the outcome and will not be a predictor.
- `ChurnCategory` and `ChurnReason` are post-churn diagnostic fields only.
- The supplied `ChurnScore` will not be used as an input to the new model.
- `SatisfactionScore` and `CLTV` require timing and provenance review before predictive use.
- Observed associations will not be presented as proof of causation.

## Data Preparation

The reusable pipeline in `src/data_preparation.py` validates the raw schema and business rules before creating `data/processed/telco_customer_churn_clean.csv`. It:

- Preserves all 7,043 customer records and all 50 source variables.
- Standardizes column names to `snake_case`.
- Converts structural blanks to `No Offer`, `No Internet`, or `Not Applicable`.
- Stores customer IDs and ZIP codes as identifiers rather than measurements.
- Adds `churn_flag` and `active_flag` for transparent aggregation.
- Reconciles customer status, service relationships, and total revenue.
- Documents fields excluded from predictive modeling because of leakage, constant values, identifiers, or uncertain provenance.

The original CSV is never overwritten. The prepared export contains 7,043 rows, 52 columns, no duplicated customer IDs, and no remaining null values.

## Feature Engineering

The reusable pipeline in `src/feature_engineering.py` adds 12 business-readable features:

- Tenure and age groups
- Monthly-charge bands
- Customer-value segments
- Total service and protection/support-service counts
- Referral groups
- Contract-risk groups
- A transparent descriptive risk score and segment
- Average revenue per tenure month
- Customer-engagement profiles

The resulting `data/processed/telco_customer_churn_features.csv` contains 7,043 rows and 64 columns. The descriptive risk segment uses five visible conditions and is not presented as a model probability or as the final high-risk KPI.

## Exploratory Analysis Highlights

- Overall churn is 26.54%, representing 1,869 customers.
- Month-to-month customers account for 1,655 churn events and have a 45.84% churn rate.
- Churn is highest during months 1–6 at 53.33% and falls to 9.51% during months 49–72.
- Fiber-optic customers have a 40.72% churn rate.
- Customers without online security or premium technical support churn at approximately twice the rate of subscribers.
- Senior citizens churn at 41.68%, while customers aged 70+ churn at 41.65%.
- San Diego records 64.91% churn among 285 customers and requires operational investigation.
- Competitor-related reasons explain 45.00% of recorded churn.
- Churned customers represent $139,130.85 in monthly-charge exposure and $3.68 million in accumulated historical revenue.

These are descriptive associations. They identify where deeper analysis and intervention testing should focus but do not prove causation.

## Priority Customer Segments

Seven overlapping actionable segments were ranked using observed churn, active-customer reach, monthly-charge exposure, and a transparent opportunity heuristic.

| Rank | Segment | Observed churn | Active customers |
|---:|---|---:|---:|
| 1 | Month-to-month without premium support | 48.54% | 1,536 |
| 2 | Fiber optic on month-to-month contract | 58.82% | 775 |
| 3 | High-value with high descriptive risk | 43.29% | 811 |
| 4 | Early-tenure month-to-month | 57.06% | 587 |
| 5 | Early-tenure fiber month-to-month | 73.30% | 224 |
| 6 | Senior with monthly charge of $90+ | 39.43% | 278 |
| 7 | San Diego customers | 64.91% | 100 |

Segments overlap and their customer counts or commercial values must not be added together. The priority ranking is descriptive, not a prediction of individual churn or campaign return.

## Churn Drivers and Statistical Validation

Categorical relationships were tested with chi-square tests and bias-corrected Cramér's V. Numerical distributions were compared with Mann–Whitney tests and rank-biserial effects. Benjamini–Hochberg correction controls the false-discovery rate across multiple tests.

The strongest credible bivariate relationships are:

- Contract: Cramér's V = 0.453, moderate effect
- Internet type: Cramér's V = 0.304, moderate effect
- Tenure: rank-biserial = −0.482, moderate effect
- Monthly charge: rank-biserial = +0.242, small effect
- Payment method, online security, premium support, offers, and referrals: smaller supporting effects

Outcome-derived fields show near-perfect associations and are excluded from prediction. `CustomerStatus`, `ChurnCategory`, `ChurnReason`, and `ChurnScore` are not legitimate model inputs. `SatisfactionScore` and CLTV remain conditional because their collection timing or derivation is undocumented.

All 64 analytical fields have a documented governance status in `data/processed/feature_governance.csv`. Statistical significance is not treated as proof of business importance or causation.

## Validated Analytical Findings

Confidence intervals were added for the most important claims. Overall churn is 26.54% with a Wilson 95% confidence interval of 25.52%–27.58%.

Selected validated comparisons include:

- Months 1–6 vs months 49–72: +43.82 percentage points, 95% CI +40.96 to +46.61
- Month-to-month vs two-year: +43.30 points, 95% CI +41.48 to +45.04
- San Diego vs other cities: +39.99 points, 95% CI +34.19 to +45.40
- Fiber optic vs DSL: +22.14 points, 95% CI +19.53 to +24.66
- No premium support vs premium support: +16.02 points, 95% CI +13.96 to +18.00

Bootstrap intervals also support higher average monthly charges and lower median tenure among churned customers. The evidence register in `validated_analytical_findings.csv` records each claim, interval, validation status, and interpretation limit. All findings remain observational and should not be interpreted as causal campaign effects.

## Predictive Modeling

A leakage-safe scikit-learn pipeline compares a dummy baseline, logistic regression, decision tree, random forest, and gradient boosting using five-fold stratified cross-validation. Preprocessing is fitted inside every fold, and 20% of customers are retained as an untouched stratified test set.

Logistic regression was selected because its cross-validated PR-AUC was within the project's 0.02 practical-equivalence tolerance of the best model while remaining easier to audit.

| Holdout metric | Result |
|---|---:|
| ROC-AUC | 0.898 |
| PR-AUC | 0.745 |
| Brier score | 0.112 |
| Accuracy | 83.39% |
| Balanced accuracy | 77.68% |
| Precision | 70.00% |
| Recall | 65.51% |
| F1 | 67.68% |

The model identified 245 of 374 holdout churners, missed 129, and incorrectly flagged 105 non-churners. The 0.50 classification threshold remains a technical default and will be assessed against retention capacity and business costs before deployment.

## Threshold, Calibration, and Fairness Evaluation

Threshold 0.32 was selected from out-of-fold training predictions as the highest-precision operating point retaining at least 80% recall. On the untouched holdout set it achieved:

| Metric | Result |
|---|---:|
| Precision | 61.58% |
| Recall | 83.16% |
| F1 | 70.76% |
| F2 | 77.71% |
| Balanced accuracy | 82.21% |
| False negatives | 63 |
| True positives | 311 |

Calibration is strong overall: expected calibration error is 2.07 percentage points, calibration intercept is −0.001, and calibration slope is 1.063.

Gender, age, and senior status remain excluded from prediction and are used only for subgroup auditing. The audit found a 6.67-point recall gap by gender, 13.00 points by senior status, and 26.03 points across age groups. These differences require monitoring and do not by themselves prove fairness or discrimination.

After refitting the locked pipeline on all labeled records, 947 of 5,174 active customers meet the high-risk threshold. They represent $68,729.50 in monthly charges and $3,931,619 in supplied CLTV.

## Model Explanation

The selected logistic model is explained through:

- Holdout permutation importance using PR-AUC loss
- Signed logistic coefficients and conditional odds multipliers
- Marginal prediction profiles
- Exact mean-centered customer-level log-odds contributions

The strongest global ranking inputs are monthly charge, referrals, contract, marital status, tenure, internet type, phone service, dependents, premium support, payment method, and online security.

Exact compact explanations were generated for all 5,174 active customers. Example active customer `4927-WWOOZ` receives a 95.56% modeled churn probability, driven primarily by monthly charge, no referrals, month-to-month contract, and early tenure.

SHAP is not installed in the project environment. The selected model is linear, so exact coefficient-based additive decomposition is used without approximation. Explanations describe model behavior and conditional associations—not causal effects.

## Retention Strategy and Business Impact

The validated 0.32 operating threshold identifies 947 active customers for retention consideration. Collectively, these customers represent $68,729.50 in monthly charges and $3,931,619 in supplied customer lifetime value. Their model-weighted expected churn volume is 502.25 customers, corresponding to $37,316 in risk-weighted monthly revenue exposure.

Each eligible customer is assigned one mutually exclusive primary intervention while overlapping risk flags are retained for operational context. The intervention portfolio covers first-90-day onboarding, early fiber assurance, fiber quality and contract support, technical-support bundles, senior-customer assistance, competitor-response offers, geographic investigation, and high-value outreach.

| Scenario | Expected customers saved | Campaign cost | Retained gross margin | Estimated net benefit | ROI |
|---|---:|---:|---:|---:|---:|
| Conservative | 26.37 | $16,272 | $8,228 | -$8,044 | -49.43% |
| Base | 65.92 | $22,366 | $41,141 | $18,775 | 83.95% |
| Optimistic | 138.75 | $28,422 | $129,888 | $101,466 | 357.00% |

These results are transparent planning scenarios rather than realized causal impact. The base case assumes 75% campaign reach, 50% offer acceptance, a 35% incremental save rate among accepting customers, a 12-month value horizon, and a 70% gross margin. Actual incremental lift, offer acceptance, and customer value preservation must be measured through randomized holdout tests before wider rollout.

Stage 13 adds `src/retention_strategy.py` and exports the campaign population, scenario results, intervention-level results, sensitivity matrix, intervention catalog, and strategy report to `artifacts/retention/`.

## Current Project Structure

```text
Customer Churn/
|-- .streamlit/                 # Streamlit theme and server configuration
|-- artifacts/
|   |-- evaluation/             # Threshold, calibration, fairness, and risk outputs
|   |-- explanations/           # Global and customer-level model explanations
|   |-- figures/                # Portfolio-ready exported analytical charts
|   |-- modeling/               # Model comparison, predictions, and serialized pipeline
|   |-- retention/              # Campaign population and scenario economics
|   `-- testing/                # Responsive screenshots and viewport evidence
|-- assets/                     # Emmanuel Onuoha profile image
|-- data/processed/             # Clean, engineered, and validated analytical exports
|-- src/                        # Reusable preparation, analysis, modeling, and retention modules
|-- tests/                      # Automated data, model, Streamlit, and browser checks
|-- app.py                      # Interactive Streamlit application
|-- customer churn.ipynb        # End-to-end analytical notebook
|-- DATASET_NOTICE.md           # Source, licence, and redistribution safeguards
|-- LICENSE                     # MIT licence for original project work
|-- README.md                   # Portfolio documentation and run instructions
|-- requirements.txt            # Pinned production dependencies
|-- requirements-dev.txt        # Pinned testing dependencies
|-- runtime.txt                 # Deployment Python version
`-- TelcoCustomerChurn.csv      # Public-domain fictional source data
```

This structure is the completed local and publicly deployed project layout.

## Project Progress

- [x] Stage 1 — Business problem, objectives, scope, stakeholders, and success criteria
- [x] Stage 2 — Dataset understanding, data dictionary, and data-quality audit
- [x] Stage 3 — Business KPI framework
- [x] Stage 4 — Data cleaning and preparation
- [x] Stage 5 — Feature engineering
- [x] Stage 6 — Exploratory data analysis
- [x] Stage 7 — Customer segmentation and cohort analysis
- [x] Stage 8 — Churn-driver assessment and statistical validation
- [x] Stage 9 — Confidence intervals and validated analytical findings
- [x] Stage 10 — Predictive modeling and model comparison
- [x] Stage 11 — Threshold selection, calibration, and fairness evaluation
- [x] Stage 12 — Global and individual model explanation
- [x] Stage 13 — Retention recommendations and business-impact scenario analysis
- [x] Stage 14 — Interactive Streamlit dashboard
- [x] Stage 15 — Professional application design and local validation
- [x] Stage 16 — Stability, interaction, performance, responsive, and reproducibility verification
- [x] Stage 17 — Recruiter-friendly GitHub repository preparation
- [x] Stage 18 — Public deployment, live verification, and presentation
- [x] Public deployment
- [x] Post-deployment enhancement — Prescriptive Decision Centre

## Current Limitations

- The dataset is a customer-level snapshot rather than monthly event history.
- All records belong to Q3, preventing time-trend analysis.
- All customers are located in California, limiting geographic generalization.
- Churn dates, campaign history, support interactions, and acquisition costs are unavailable.
- Retention returns can therefore be modeled only as transparent scenarios, not claimed as realized impact.

## Notebook

The detailed analytical workflow, checks, definitions, and interpretations are maintained in [`customer churn.ipynb`](customer%20churn.ipynb).

## Application Status

The presentation-ready Streamlit application is implemented in `app.py`. It includes eight interactive views:

- Executive Overview
- Customer Analysis
- Churn Drivers
- Geographic Analysis
- Customer Risk Predictor
- Decision Centre
- Retention Simulator
- Methodology

### Prescriptive Decision Centre

The Decision Centre converts churn probabilities into a transparent, budget-constrained retention plan. It calculates probability-weighted expected saves, retained gross margin, intervention cost, expected net benefit, and ROI for each eligible active customer. Management can change campaign capacity, expected budget, business objective, reach, acceptance, incremental save rate, value horizon, margin, and cost assumptions.

The default policy selects 500 customers under a $15,000 expected budget, with approximately 40.7 expected incremental saves, $11,561 expected campaign cost, and $17,473 expected net benefit. These are planning estimates rather than realized causal results. Customer identifiers are excluded from the public decision table and download; the exported plan is aggregated by intervention.

The interface includes Emmanuel Onuoha's photograph and authorship branding, a user-controlled light/dark theme, cached data and model loading, responsive layouts, consistent formatting, explanatory notes, empty-state handling, model-level and customer-level explanations, responsible-use guidance, and transparent campaign assumptions.

### Run locally

From the project directory, install the declared dependencies and start Streamlit:

```powershell
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

Streamlit will display the local address, normally `http://localhost:8501`.

The deployment environment is validated with Streamlit 1.62.0. All eight navigation views, the submitted customer-prediction workflow, and the Decision Centre capacity control render without application exceptions.

## Testing and Verification

Stage 16 is documented in [`STAGE_16_VERIFICATION_REPORT.md`](STAGE_16_VERIFICATION_REPORT.md). The expanded automated suite contains 34 passing tests covering source-data reconciliation, filters, predictions, missing and unseen inputs, campaign calculations, prescriptive decision economics, budget and capacity constraints, sensitivity, invalid assumptions, all eight application pages, live interactions, theme switching, and performance.

Run the tests locally:

```powershell
python -m pip install -r requirements-dev.txt
python -m pytest tests -q --durations=15
```

Responsive Edge-browser evidence is stored in `artifacts/testing/screenshots/` for local validation and `artifacts/testing/deployed_screenshots/` for the public deployment. Desktop 1440×1000, tablet 1024×900, mobile 390×844, and dark-mode priority pages passed without horizontal page overflow. The repeatable browser checks are `tests/visual_viewport_check.py` and `tests/live_deployment_check.py`.

The production dependencies are pinned in `requirements.txt`, and `runtime.txt` declares Python 3.12. A newly created isolated Python 3.12 environment installed those requirements, loaded the application data and model, reproduced the expected smoke-test probability, and served a healthy Streamlit endpoint before the temporary environment was removed.

## Repository Licence and Citation

The original project code, analysis, documentation, tests, and visual design are released under the [MIT License](LICENSE). The fictional source dataset is separately identified as Public Domain in [`DATASET_NOTICE.md`](DATASET_NOTICE.md). Citation metadata is provided in [`CITATION.cff`](CITATION.cff).

## Live Application

**Public dashboard:** [Open Customer Churn Intelligence](https://customer-churn-portfolio-z3hvzwkxs3yu8zvenx9yuk.streamlit.app/)

The application is deployed on Streamlit Community Cloud with Python 3.12 and Streamlit 1.62.0. Live browser checks covered all seven navigation routes, prediction submission, retention controls, desktop/tablet/mobile rendering, and light/dark presentation. Fresh cloud sessions took approximately 21–51 seconds during verification; warm navigation was substantially faster. Community Cloud may sleep after inactivity, so the first visit can take longer than subsequent interactions.

- Deployment instructions: [`DEPLOYMENT.md`](DEPLOYMENT.md)
- Short presentation script: [`PRESENTATION_SCRIPT.md`](PRESENTATION_SCRIPT.md)
- Deployment validation report: [`STAGE_18_DEPLOYMENT_REPORT.md`](STAGE_18_DEPLOYMENT_REPORT.md)

---

**Developed and Designed by Emmanuel Onuoha**
