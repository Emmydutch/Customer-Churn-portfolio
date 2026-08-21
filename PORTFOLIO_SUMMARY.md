# Customer Churn Intelligence

An end-to-end telecom retention project that turns a public-domain fictional customer dataset into an executive decision-support application.

## Portfolio Snapshot

- Audited and prepared 7,043 customer records with reproducible validation controls.
- Measured 26.54% overall churn and identified 947 high-risk active customers.
- Built and compared interpretable classification pipelines without outcome leakage.
- Selected a logistic model with 89.85% holdout ROC-AUC and 83.16% recall at the business threshold.
- Added global and customer-level model explanations.
- Developed retention scenarios linking customer risk, campaign cost, expected saves, and net benefit.
- Added a prescriptive Decision Centre that ranks positive expected-value interventions under editable budget and campaign-capacity constraints.
- Delivered an eight-page Streamlit application with light/dark themes, responsive layouts, interactive filters, a customer predictor, geographic analysis, a Decision Centre, and a retention simulator.
- Verified the project with 34 automated tests, live browser interactions, responsive viewport checks, and a clean Python 3.12 deployment environment.
- Prepared and fully executed a self-contained Kaggle edition with a prescriptive retention Decision Centre and explicit source/licence attribution.
- Produced an original Academia-ready independent technical report and visual-statistics appendix in PDF and editable DOCX formats.

## Business Outcome

The project helps executives answer three practical questions:

1. Which customers and segments show the highest churn exposure?
2. Which pre-churn characteristics does the model rely on when prioritizing risk?
3. Under explicit campaign assumptions, how many customers and how much gross margin could potentially be retained?

## Technology

Python, pandas, scikit-learn, Plotly, Streamlit, Jupyter, pytest, and Selenium.

## Presentation

- **Local application:** `streamlit run app.py`
- **GitHub repository:** https://github.com/Emmydutch/Customer-Churn-portfolio
- **Live demo:** https://customer-churn-portfolio-z3hvzwkxs3yu8zvenx9yuk.streamlit.app/
- **Kaggle profile:** https://www.kaggle.com/onuohaemmanuel
- **Detailed documentation:** `README.md`
- **Verification reports:** `STAGE_16_VERIFICATION_REPORT.md` and `STAGE_18_DEPLOYMENT_REPORT.md`

The analysis, narrative, code, application, tests, and visual design are original portfolio work. Dataset attribution and licensing are documented separately in `DATASET_NOTICE.md`.

---

**Developed and Designed by Emmanuel Onuoha**
