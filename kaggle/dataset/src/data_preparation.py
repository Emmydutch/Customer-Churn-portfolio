"""Data preparation pipeline for the telecom customer churn project.

The raw file is never modified. Cleaning decisions in this module are designed to
be deterministic, auditable, and reusable by both the notebook and Streamlit app.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


COLUMN_NAMES = {
    "CustomerID": "customer_id",
    "Gender": "gender",
    "Age": "age",
    "Under30": "under_30",
    "SeniorCitizen": "senior_citizen",
    "Married": "married",
    "Dependents": "dependents",
    "NumberofDependents": "number_of_dependents",
    "Country": "country",
    "State": "state",
    "City": "city",
    "ZipCode": "zip_code",
    "Latitude": "latitude",
    "Longitude": "longitude",
    "Population": "population",
    "Quarter": "quarter",
    "ReferredaFriend": "referred_a_friend",
    "Number_of_Referrals": "number_of_referrals",
    "TenureinMonths": "tenure_in_months",
    "Offer": "offer",
    "PhoneService": "phone_service",
    "AvgMonthlyLongDistanceCharges": "avg_monthly_long_distance_charges",
    "MultipleLines": "multiple_lines",
    "InternetService": "internet_service",
    "InternetType": "internet_type",
    "AvgMonthlyGBDownload": "avg_monthly_gb_download",
    "OnlineSecurity": "online_security",
    "OnlineBackup": "online_backup",
    "DeviceProtectionPlan": "device_protection_plan",
    "PremiumTechSupport": "premium_tech_support",
    "StreamingTV": "streaming_tv",
    "StreamingMovies": "streaming_movies",
    "StreamingMusic": "streaming_music",
    "UnlimitedData": "unlimited_data",
    "Contract": "contract",
    "PaperlessBilling": "paperless_billing",
    "PaymentMethod": "payment_method",
    "MonthlyCharge": "monthly_charge",
    "TotalCharges": "total_charges",
    "TotalRefunds": "total_refunds",
    "TotalExtraDataCharges": "total_extra_data_charges",
    "TotalLongDistanceCharges": "total_long_distance_charges",
    "TotalRevenue": "total_revenue",
    "SatisfactionScore": "satisfaction_score",
    "CustomerStatus": "customer_status",
    "ChurnLabel": "churn_label",
    "ChurnScore": "churn_score",
    "CLTV": "cltv",
    "ChurnCategory": "churn_category",
    "ChurnReason": "churn_reason",
}

YES_NO_COLUMNS = [
    "under_30",
    "senior_citizen",
    "married",
    "dependents",
    "referred_a_friend",
    "phone_service",
    "multiple_lines",
    "internet_service",
    "online_security",
    "online_backup",
    "device_protection_plan",
    "premium_tech_support",
    "streaming_tv",
    "streaming_movies",
    "streaming_music",
    "unlimited_data",
    "paperless_billing",
    "churn_label",
]

CATEGORICAL_COLUMNS = [
    "gender",
    *YES_NO_COLUMNS,
    "country",
    "state",
    "city",
    "quarter",
    "offer",
    "internet_type",
    "contract",
    "payment_method",
    "customer_status",
    "churn_category",
    "churn_reason",
]

MODEL_EXCLUDED_COLUMNS = [
    "customer_id",       # identifier
    "country",           # constant
    "state",             # constant
    "quarter",           # constant
    "customer_status",   # directly reveals the outcome
    "churn_label",       # target, stored separately
    "churn_flag",        # numeric restatement of the target
    "active_flag",       # inverse numeric restatement of the target
    "churn_score",       # pre-existing model output
    "churn_category",    # post-churn information
    "churn_reason",      # post-churn information
    "satisfaction_score",# timing/provenance not confirmed
    "cltv",              # calculation timing/provenance not confirmed
]


def validate_raw_data(df: pd.DataFrame) -> None:
    """Raise a clear error when the supplied raw data violates known rules."""

    missing_columns = sorted(set(COLUMN_NAMES) - set(df.columns))
    unexpected_columns = sorted(set(df.columns) - set(COLUMN_NAMES))
    if missing_columns or unexpected_columns:
        raise ValueError(
            f"Schema mismatch. Missing={missing_columns}; unexpected={unexpected_columns}"
        )

    if df.empty:
        raise ValueError("The raw dataset contains no customer records.")
    if df["CustomerID"].isna().any():
        raise ValueError("CustomerID contains missing values.")
    if df["CustomerID"].duplicated().any():
        raise ValueError("CustomerID must be unique.")
    if df.duplicated().any():
        raise ValueError("The raw dataset contains duplicate rows.")

    raw_yes_no_columns = [
        source for source, prepared in COLUMN_NAMES.items()
        if prepared in YES_NO_COLUMNS
    ]
    invalid_binary = {
        column: sorted(set(df[column].dropna()) - {"Yes", "No"})
        for column in raw_yes_no_columns
    }
    invalid_binary = {key: values for key, values in invalid_binary.items() if values}
    if invalid_binary:
        raise ValueError(f"Invalid Yes/No values found: {invalid_binary}")

    checks = {
        "Under30 disagrees with Age": (df["Age"].lt(30) != df["Under30"].eq("Yes")),
        "SeniorCitizen disagrees with Age": (
            df["Age"].ge(65) != df["SeniorCitizen"].eq("Yes")
        ),
        "Dependents disagrees with dependent count": (
            df["NumberofDependents"].gt(0) != df["Dependents"].eq("Yes")
        ),
        "Referral flag disagrees with referral count": (
            df["Number_of_Referrals"].gt(0) != df["ReferredaFriend"].eq("Yes")
        ),
        "CustomerStatus disagrees with ChurnLabel": (
            df["CustomerStatus"].eq("Churned") != df["ChurnLabel"].eq("Yes")
        ),
        "No phone service but non-zero long-distance usage": (
            df["PhoneService"].eq("No")
            & (df["AvgMonthlyLongDistanceCharges"].ne(0)
               | df["TotalLongDistanceCharges"].ne(0))
        ),
        "No internet but internet attributes present": (
            df["InternetService"].eq("No")
            & (df["InternetType"].notna() | df["AvgMonthlyGBDownload"].ne(0))
        ),
        "Internet customer missing InternetType": (
            df["InternetService"].eq("Yes") & df["InternetType"].isna()
        ),
        "Non-churner has post-churn details": (
            df["ChurnLabel"].eq("No")
            & (df["ChurnCategory"].notna() | df["ChurnReason"].notna())
        ),
        "Churner is missing post-churn details": (
            df["ChurnLabel"].eq("Yes")
            & (df["ChurnCategory"].isna() | df["ChurnReason"].isna())
        ),
    }
    failed = {name: int(mask.sum()) for name, mask in checks.items() if mask.any()}
    if failed:
        raise ValueError(f"Logical consistency checks failed: {failed}")

    non_negative_columns = [
        "Age",
        "NumberofDependents",
        "Population",
        "Number_of_Referrals",
        "TenureinMonths",
        "AvgMonthlyLongDistanceCharges",
        "AvgMonthlyGBDownload",
        "MonthlyCharge",
        "TotalCharges",
        "TotalRefunds",
        "TotalExtraDataCharges",
        "TotalLongDistanceCharges",
        "TotalRevenue",
        "SatisfactionScore",
        "ChurnScore",
        "CLTV",
    ]
    negative_counts = {
        column: int(df[column].lt(0).sum()) for column in non_negative_columns
        if df[column].lt(0).any()
    }
    if negative_counts:
        raise ValueError(f"Unexpected negative values found: {negative_counts}")

    if not df["Latitude"].between(-90, 90).all():
        raise ValueError("Latitude contains values outside [-90, 90].")
    if not df["Longitude"].between(-180, 180).all():
        raise ValueError("Longitude contains values outside [-180, 180].")

    expected_revenue = (
        df["TotalCharges"]
        + df["TotalLongDistanceCharges"]
        + df["TotalExtraDataCharges"]
        - df["TotalRefunds"]
    )
    if not np.allclose(df["TotalRevenue"], expected_revenue, atol=0.01):
        raise ValueError("TotalRevenue does not reconcile with its charge components.")


def prepare_customer_data(raw_df: pd.DataFrame) -> pd.DataFrame:
    """Validate and clean a raw customer dataframe without mutating the input."""

    validate_raw_data(raw_df)
    cleaned = raw_df.copy(deep=True).rename(columns=COLUMN_NAMES)

    # Convert structural blanks to meaningful, visible business categories.
    cleaned["offer"] = cleaned["offer"].fillna("No Offer")
    cleaned["internet_type"] = cleaned["internet_type"].fillna("No Internet")
    cleaned["churn_category"] = cleaned["churn_category"].fillna("Not Applicable")
    cleaned["churn_reason"] = cleaned["churn_reason"].fillna("Not Applicable")

    # Identifiers that look numeric must remain strings to preserve semantics.
    cleaned["customer_id"] = cleaned["customer_id"].astype("string")
    cleaned["zip_code"] = cleaned["zip_code"].astype("string").str.zfill(5)

    # Category dtype reduces in-memory size and documents analytical intent.
    for column in CATEGORICAL_COLUMNS:
        cleaned[column] = cleaned[column].astype("category")

    cleaned["churn_flag"] = cleaned["churn_label"].eq("Yes").astype("int8")
    cleaned["active_flag"] = cleaned["churn_label"].eq("No").astype("int8")

    if cleaned.isna().any().any():
        nulls = cleaned.isna().sum().loc[lambda values: values.gt(0)].to_dict()
        raise ValueError(f"Unexpected nulls remain after preparation: {nulls}")
    if len(cleaned) != len(raw_df):
        raise AssertionError("Preparation changed the customer row count.")
    if cleaned["customer_id"].nunique() != len(cleaned):
        raise AssertionError("Prepared customer_id values are not unique.")
    if not cleaned["churn_flag"].eq(cleaned["customer_status"].eq("Churned")).all():
        raise AssertionError("Prepared churn flags do not reconcile with customer status.")

    return cleaned


def export_clean_data(cleaned_df: pd.DataFrame, output_path: str | Path) -> Path:
    """Export a prepared dataframe to CSV and return the resolved output path."""

    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    cleaned_df.to_csv(destination, index=False)
    return destination.resolve()


def build_clean_dataset(
    raw_path: str | Path = "TelcoCustomerChurn.csv",
    output_path: str | Path = "data/processed/telco_customer_churn_clean.csv",
) -> tuple[pd.DataFrame, Path]:
    """Load, validate, prepare, and export the customer dataset."""

    raw_df = pd.read_csv(raw_path)
    cleaned_df = prepare_customer_data(raw_df)
    destination = export_clean_data(cleaned_df, output_path)
    return cleaned_df, destination


if __name__ == "__main__":
    prepared, saved_to = build_clean_dataset()
    print(f"Prepared {len(prepared):,} customers and {prepared.shape[1]} columns.")
    print(f"Saved clean dataset to: {saved_to}")
