# Deployment Guide

## Validated local deployment

The application has been validated in a clean Python 3.12 environment with Streamlit 1.62.0.

```powershell
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

Expected local address: `http://localhost:8501`

Health endpoint: `http://localhost:8501/_stcore/health`

Public deployment: https://customer-churn-portfolio-z3hvzwkxs3yu8zvenx9yuk.streamlit.app/

## Streamlit Community Cloud

Use the following deployment settings:

| Setting | Value |
|---|---|
| Repository | `Emmydutch/customer-churn-portfolio` |
| Branch | `main` |
| Main file path | `app.py` |
| Python version | `3.12` |
| Secrets | None required |

Deployment steps:

1. Sign in to [Streamlit Community Cloud](https://share.streamlit.io/) with the GitHub account that can access the repository.
2. Connect or authorize GitHub if the repository is not visible.
3. Select **Create app**.
4. Choose the repository, `main` branch, and `app.py` entrypoint shown above.
5. In advanced settings, select Python 3.12. No secrets are required.
6. Select **Deploy** and wait for dependency installation and application startup.
7. Open every application page, test the light/dark switch, submit a customer prediction, and change retention assumptions.
8. Copy the public URL into `README.md` and `PORTFOLIO_SUMMARY.md`.

## Deployment verification

- The production dependencies are pinned in `requirements.txt`.
- The project declares Python 3.12 in `runtime.txt`.
- `.streamlit/config.toml` contains the application theme and headless server configuration.
- The source dataset and all model artifacts required at runtime are committed to the repository.
- No secrets or external databases are required.
- All 24 automated tests pass under the clean deployment environment.

## Troubleshooting

- **Dependency build failure:** confirm the deployment uses Python 3.12 and the current `requirements.txt`.
- **Model-loading failure:** confirm `artifacts/evaluation/production_churn_model.joblib` exists in the deployed commit.
- **Missing data:** confirm `data/processed/` and the artifact folders are present in the deployed commit.
- **Map tiles unavailable:** the geographic page depends on an external public basemap; other app functionality remains local to the repository.
- **App sleeps when unused:** Community Cloud may place inactive applications to sleep; opening the URL wakes the app.

## Alternative platforms

Render and Hugging Face Spaces are compatible alternatives, but Community Cloud is preferred because the application is a native Streamlit project and the repository contains all required deployment files.

---

**Developed and Designed by Emmanuel Onuoha**
