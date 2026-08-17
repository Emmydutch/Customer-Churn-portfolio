# Dataset Attribution and Redistribution Notice

## Dataset identity

- **Dataset:** Telecom Customer Churn
- **Original source/author:** IBM Cognos Analytics
- **Distributor:** Maven Analytics Data Playground
- **URL:** https://mavenanalytics.io/data-playground/telecom-customer-churn
- **Licence:** Public Domain, as stated by Maven Analytics
- **Dataset description:** Customer churn records for a fictional telecommunications company serving 7,043 customers in California.

The supplied `TelcoCustomerChurn.csv` is the combined 50-column form of this dataset. The customer identifiers, locations, service information, churn outcomes, and financial fields describe fictional customers rather than real subscribers.

## Publication decision

The source page labels the dataset Public Domain. On that basis, redistribution of the source data—including fictional customer IDs and geographic fields—is permitted for this portfolio. Attribution is retained voluntarily for transparency and provenance.

Even though the records are fictional, the application follows data-minimisation principles:

- Executive and analytical pages show aggregate results.
- The deployed application does not expose a downloadable row-level customer list.
- Customer IDs are treated as identifiers, never as model features.
- Latitude, longitude, city, and ZIP code are used for aggregate geographic analysis and are excluded from model prediction.

If this project is ever adapted to real customer data, the raw dataset, customer identifiers, coordinates, and row-level risk scores must not be placed in a public repository without the data owner's explicit authorization and an appropriate privacy review.

## Project licence

The original analysis, application code, documentation, tests, and visual design created by Emmanuel Onuoha are released under the repository's MIT License. The dataset retains its separate Public Domain designation.

---

**Developed and Designed by Emmanuel Onuoha**
