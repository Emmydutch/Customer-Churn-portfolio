# Visual Statistics Appendix

## Telecom Customer Churn Intelligence

**Emmanuel Onuoha — 21 August 2026**

This appendix presents selected statistical and decision-support visuals from the independent technical report. The dataset is fictional. Associations are not causal, and retention economics are scenario estimates.

![Figure A1. Selected validated churn-risk differences with 95% confidence intervals.](figures/01_validated_risk_differences.png)

Early tenure and month-to-month contracts show the largest absolute descriptive risk differences. Confidence intervals quantify sampling uncertainty but do not remove confounding.

![Figure A2. Prioritised risk segments.](figures/05_priority_segments.png)

The largest segments combine substantial churn exposure with meaningful active-customer value. Segments overlap and should not be summed.

![Figure A3. Retrospective churn categories and recorded reasons.](figures/04_why_customers_churned.png)

Competitor-related reasons dominate retrospective records. These post-churn fields were excluded from predictive modelling.

![Figure A4. Cross-validated precision–recall trade-off.](figures/02_cross_validated_precision_recall.png)

Logistic regression was selected for its balance of discrimination, recall, calibration, and interpretability rather than the highest ROC-AUC alone.

![Figure A5. Holdout confusion matrix at threshold 0.32.](figures/03_confusion_matrix.png)

The selected operating point captured 311 of 374 holdout churners while flagging 505 of 1,409 customers.

![Figure A6. Scenario-based retention economics.](figures/06_retention_scenarios.png)

The range from negative conservative net benefit to strongly positive optimistic benefit demonstrates sensitivity to campaign assumptions.

![Figure A7. Implemented prescriptive Decision Centre.](figures/07_decision_centre.png)

The interactive implementation exposes campaign assumptions, capacity, budget, and decision objective to the user.

---

**Developed and Designed by Emmanuel Onuoha**
