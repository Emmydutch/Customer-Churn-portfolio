# Academia Publication Package

This directory contains the independent technical report, visual-statistics appendix, Academia.edu metadata, generated figures, and reproducible publication builder for Emmanuel Onuoha's Customer Churn Intelligence project.

## Build

From the repository root:

```powershell
python scripts/build_academia_package.py
```

Generated outputs are written to `academia/output/`:

- `Emmanuel_Onuoha_Telecom_Churn_Technical_Report.pdf`
- `Emmanuel_Onuoha_Telecom_Churn_Technical_Report.docx`
- `Emmanuel_Onuoha_Telecom_Churn_Visual_Appendix.pdf`
- `Emmanuel_Onuoha_Telecom_Churn_Visual_Appendix.docx`

## Publication safeguards

- Present the work as an independent technical report, not a peer-reviewed article.
- Preserve the dataset attribution and licensing statement.
- Treat associations and model explanations as non-causal.
- Treat retention economics as planning scenarios until validated by a controlled experiment.
- Check the generated PDF preview and hyperlinks before publishing.

See `ACADEMIA_METADATA.md` for the upload title, abstract, tags, biography, citation, and profile post.

Academia profile: https://independent.academia.edu/emmanuelonuoha24
