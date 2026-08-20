# Kaggle Publication Package

This directory contains the Kaggle edition of **Telecom Customer Churn: From Prediction to Prescriptive Retention Decisions** by Emmanuel Onuoha.

## Build

From the repository root:

```powershell
python scripts/build_kaggle_package.py
```

The build creates:

- `telecom-churn-prescriptive-retention.ipynb` — portable Kaggle notebook;
- `kernel-metadata.json` — public notebook metadata for `onuohaemmanuel`;
- `dataset/` — companion input package with the source CSV, attribution, licence, and reusable modules.

## Publication order

1. Create the companion dataset first:

   ```powershell
   kaggle datasets create -p kaggle/dataset --public
   ```

2. Confirm that this dataset exists:

   ```text
   onuohaemmanuel/telecom-customer-churn-portfolio-data
   ```

3. Push and execute the notebook:

   ```powershell
   kaggle kernels push -p kaggle
   ```

4. On Kaggle, inspect the execution log, confirm that every validation cell passes, add an attractive cover image and relevant tags, and publish the saved version.

If a local network blocks the CLI's Google Storage upload, use `Kaggle-upload-package.zip` on Kaggle's **Create Dataset** page. Keep the dataset slug as `telecom-customer-churn-portfolio-data`; after it is public, import `telecom-churn-prescriptive-retention.ipynb` as a new notebook and attach that dataset.

## Presentation guidance

- Keep the notebook public only after **Save & Run All** succeeds.
- Use the attached companion dataset rather than uploading files interactively.
- Preserve the original dataset attribution and mixed licensing notice.
- Add the live Streamlit and GitHub links to the Kaggle notebook description.
- Suggested tags: `customer churn`, `telecom`, `classification`, `explainable AI`, `prescriptive analytics`, and `business analytics`.

---

**Developed and Designed by Emmanuel Onuoha**
