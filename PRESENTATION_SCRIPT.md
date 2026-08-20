# Short Project Presentation Script

## 90-second version

Hello, I’m Emmanuel Onuoha, and this is my Telecom Customer Churn Intelligence project.

The business problem was to determine which customers are most likely to leave, understand the patterns associated with churn, and translate those findings into practical retention decisions.

I analyzed 7,043 fictional telecom customer records. The observed churn rate was 26.54%, with the strongest exposure concentrated among early-tenure, month-to-month, fiber customers—particularly those without premium technical support.

I built a leakage-controlled modeling pipeline and compared multiple classifiers. I selected an interpretable logistic regression model that achieved 89.85% ROC-AUC and 83.16% recall on the holdout data at a 0.32 operating threshold. The model identifies 947 active high-risk customers and provides both portfolio-level and individual prediction explanations.

I then connected customer risk to business action through a retention simulator. Under the documented base assumptions, the campaign could save approximately 66 customers and generate an estimated net benefit of about $18,775. These figures are planning scenarios and should be validated through a controlled campaign pilot.

The final Streamlit application contains an executive overview, customer analysis, churn drivers, geographic analysis, an explainable customer-risk predictor, a prescriptive Decision Centre, a retention simulator, and methodology documentation. The Decision Centre translates risk into budget-constrained next-best actions while keeping save-rate assumptions explicit. The application supports light and dark themes and was verified with 34 automated tests, responsive browser checks, and a clean deployment environment.

This project demonstrates my ability to move from a business question through reproducible analysis and responsible modeling to an executive-ready data product.

## Suggested demonstration flow

1. Start with the Executive Overview and headline KPIs.
2. Show the Customer Analysis filters and priority segments.
3. Explain the difference between recorded churn reasons and pre-churn model drivers.
4. Score one customer in the Risk Predictor and discuss the explanation.
5. Allocate a constrained campaign in the Decision Centre and explain why this is scenario-based rather than causal optimization.
6. Adjust broader campaign assumptions in the Retention Simulator.
7. Close with methodology, limitations, and responsible-use controls.

---

**Developed and Designed by Emmanuel Onuoha**
