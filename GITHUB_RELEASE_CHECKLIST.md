# GitHub Release Checklist

## Repository content

- [x] Public-domain dataset included with source and licence notice
- [x] Reproducible preparation, analysis, modeling, explanation, and retention modules
- [x] Executed analytical notebook
- [x] Streamlit application with light and dark themes
- [x] Pinned production and development dependencies
- [x] Streamlit configuration and Python runtime declaration
- [x] Automated test suite and responsive browser evidence
- [x] Screenshots, project limitations, responsible-use guidance, and run instructions
- [x] MIT project licence and citation metadata
- [x] Cache, virtual-environment, browser-profile, and secret exclusions

## Final local verification

```powershell
python -m pip install -r requirements-dev.txt
python -m pytest tests -q
python -m streamlit run app.py
```

Confirm that:

- all 24 automated tests pass;
- every application page opens in light and dark themes;
- the predictor returns a probability and explanation;
- the retention simulator responds to changed assumptions;
- `git status` contains only intended release changes;
- no `.streamlit/secrets.toml`, virtual environment, cache, or browser profile is tracked.

## GitHub release

- [ ] Commit the Stage 17 repository-preparation changes
- [ ] Push `main` to `origin`
- [ ] Confirm the GitHub Actions workflow passes
- [ ] Add repository description, topics, and social preview
- [ ] Deploy the app and add the public URL to `README.md` and `PORTFOLIO_SUMMARY.md`
- [ ] Create release `v1.0.0`

Community Cloud deployment values are documented in `DEPLOYMENT.md`.

Suggested repository topics:

`customer-churn`, `telecom`, `machine-learning`, `streamlit`, `explainable-ai`, `retention-analytics`, `python`, `portfolio-project`

---

**Developed and Designed by Emmanuel Onuoha**
