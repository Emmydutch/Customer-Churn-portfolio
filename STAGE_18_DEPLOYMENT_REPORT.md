# Stage 18 — Deployment and Presentation Report

**Project:** Customer Churn Intelligence

**Application owner:** Emmanuel Onuoha

**Validation date:** 20 August 2026

**Status:** Ready for GitHub and Streamlit Community Cloud deployment

## Release Validation

The complete application was installed and tested in a clean Python 3.12 environment using the pinned production and development dependencies.

- Streamlit version: 1.62.0
- Automated tests: 24 passed, 0 failed
- Local application startup: passed
- Streamlit health endpoint: HTTP 200 with response `ok`
- Application entrypoint: `app.py`
- Required secrets: none
- Runtime data and model artifacts: stored in the repository

One deployment-portability issue was corrected during validation: the Streamlit application test now resolves the entrypoint from the repository root instead of relying on the test runner's current directory.

## Deployment Package

The release includes:

- the local seven-page Streamlit application;
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

The final public URL must be added to `README.md` and `PORTFOLIO_SUMMARY.md` after the repository owner completes the GitHub-authorized Streamlit Community Cloud deployment.

## Presentation Assets

- `PORTFOLIO_SUMMARY.md` provides the concise recruiter-facing project narrative.
- `PRESENTATION_SCRIPT.md` provides a 90-second walkthrough and recommended demonstration order.
- `artifacts/testing/screenshots/` contains desktop, tablet, mobile, and dark-theme evidence.
- `DEPLOYMENT.md` contains local and cloud deployment instructions and troubleshooting guidance.

## Release Decision

The application is technically ready for publication. Public deployment remains an account-authorized release action because Streamlit Community Cloud must be connected to the repository owner's GitHub account.

---

**Developed and Designed by Emmanuel Onuoha**
