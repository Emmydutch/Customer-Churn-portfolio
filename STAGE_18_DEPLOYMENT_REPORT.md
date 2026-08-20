# Stage 18 — Deployment and Presentation Report

**Project:** Customer Churn Intelligence

**Application owner:** Emmanuel Onuoha

**Validation date:** 20 August 2026

**Status:** Publicly deployed and browser-verified

## Release Validation

The complete application was installed and tested in a clean Python 3.12 environment using the pinned production and development dependencies.

- Streamlit version: 1.62.0
- Automated tests: 34 passed, 0 failed
- Local application startup: passed
- Streamlit health endpoint: HTTP 200 with response `ok`
- Application entrypoint: `app.py`
- Required secrets: none
- Runtime data and model artifacts: stored in the repository

One deployment-portability issue was corrected during validation: the Streamlit application test now resolves the entrypoint from the repository root instead of relying on the test runner's current directory.

## Deployment Package

The release includes:

- the local eight-page Streamlit application, including the prescriptive Decision Centre;
- source and processed datasets with attribution documentation;
- reproducible preparation, analysis, modeling, explanation, and simulation modules;
- the executed portfolio notebook;
- pinned dependencies and Python runtime configuration;
- Streamlit light and dark theme configuration;
- automated tests and responsive-screen evidence;
- GitHub Actions continuous-integration checks;
- recruiter-facing README, portfolio summary, deployment guide, screenshots, licence, citation metadata, and presentation script.

## Community Cloud Settings

| Setting | Value |
|---|---|
| Repository | `Emmydutch/customer-churn-portfolio` |
| Branch | `main` |
| Main file | `app.py` |
| Python | `3.12` |
| Secrets | None |

**Public URL:** https://customer-churn-portfolio-z3hvzwkxs3yu8zvenx9yuk.streamlit.app/

## Public Deployment Verification

- All eight navigation routes opened in a live Edge session after the Decision Centre deployment.
- The customer-risk form accepted a live submission and rendered its prediction result.
- The Retention Simulator rendered its controls and scenario outputs; calculation interactions remain covered by the automated suite.
- Desktop 1440×1000, tablet 1024×900, and mobile 390×844 had no horizontal overflow.
- Executive Overview, Customer Risk Predictor, Geographic Analysis, and Retention Simulator rendered correctly in dark mode.
- The Decision Centre passed local dark-mode and responsive visual verification without horizontal overflow.
- The deployed Decision Centre rendered successfully in the live navigation audit; exact budget/capacity recalculation is covered by the deterministic AppTest suite.
- Fresh cloud sessions took approximately 21–51 seconds during the verification window; a warm desktop load completed in approximately 10 seconds.
- Community Cloud wake and session startup introduced intermittent latency, but no application exception was observed in successful sessions.

## Presentation Assets

- `PORTFOLIO_SUMMARY.md` provides the concise recruiter-facing project narrative.
- `PRESENTATION_SCRIPT.md` provides a 90-second walkthrough and recommended demonstration order.
- `artifacts/testing/screenshots/` contains desktop, tablet, mobile, and dark-theme evidence.
- `artifacts/testing/deployed_screenshots/` contains equivalent evidence captured from the public application.
- `DEPLOYMENT.md` contains local and cloud deployment instructions and troubleshooting guidance.

## Release Decision

The application is published and suitable for portfolio presentation. The remaining optional release action is creation of the GitHub `v1.0.0` tag and release entry.

---

**Developed and Designed by Emmanuel Onuoha**
