"""Business feature engineering for the telecom customer churn project."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


PROTECTION_SUPPORT_SERVICES = [
    "online_security",
    "online_backup",
    "device_protection_plan",
    "premium_tech_support",
]

COUNTED_SERVICES = [
    "phone_service",
    "internet_service",
    *PROTECTION_SUPPORT_SERVICES,
    "streaming_tv",
    "streaming_movies",
    "streaming_music",
    "unlimited_data",
]

FEATURE_COLUMNS = [
    "tenure_group",
    "age_group",
    "monthly_charge_band",
    "customer_value_segment",
    "service_count",
    "protection_support_service_count",
    "referral_group",
    "contract_risk_group",
    "descriptive_risk_points",
    "descriptive_risk_segment",
    "avg_revenue_per_tenure_month",
    "customer_engagement_profile",
]


def _require_columns(df: pd.DataFrame, columns: list[str]) -> None:
    missing = sorted(set(columns) - set(df.columns))
    if missing:
        raise ValueError(f"Feature engineering requires missing columns: {missing}")


def add_business_features(clean_df: pd.DataFrame) -> pd.DataFrame:
    """Return a copy of clean customer data with interpretable business features."""

    required = [
        "customer_id",
        "age",
        "tenure_in_months",
        "monthly_charge",
        "cltv",
        "number_of_referrals",
        "contract",
        "internet_type",
        "premium_tech_support",
        "total_revenue",
        "churn_label",
        "active_flag",
        *COUNTED_SERVICES,
    ]
    _require_columns(clean_df, required)

    featured = clean_df.copy(deep=True)

    featured["tenure_group"] = pd.cut(
        featured["tenure_in_months"],
        bins=[0, 6, 12, 24, 48, np.inf],
        labels=["1–6 months", "7–12 months", "13–24 months", "25–48 months", "49–72 months"],
        include_lowest=True,
    )
    featured["age_group"] = pd.cut(
        featured["age"],
        bins=[-np.inf, 29, 44, 59, 69, np.inf],
        labels=["19–29", "30–44", "45–59", "60–69", "70+"],
    )
    featured["monthly_charge_band"] = pd.cut(
        featured["monthly_charge"],
        bins=[-np.inf, 35, 70, 90, np.inf],
        labels=["Under $35", "$35–$69.99", "$70–$89.99", "$90+"],
        right=False,
    )

    # Fixed thresholds are the observed full-portfolio CLTV quartiles from Stage 2.
    # Keeping them fixed ensures that dashboard filters do not redefine segments.
    featured["customer_value_segment"] = pd.cut(
        featured["cltv"],
        bins=[-np.inf, 3_469, 4_527, 5_380.5, np.inf],
        labels=["Emerging Value", "Core Value", "High Value", "Premium Value"],
        right=False,
    )

    featured["service_count"] = (
        featured[COUNTED_SERVICES].eq("Yes").sum(axis=1).astype("int8")
    )
    featured["protection_support_service_count"] = (
        featured[PROTECTION_SUPPORT_SERVICES].eq("Yes").sum(axis=1).astype("int8")
    )

    featured["referral_group"] = pd.cut(
        featured["number_of_referrals"],
        bins=[-1, 0, 2, 5, np.inf],
        labels=["No Referrals", "1–2 Referrals", "3–5 Referrals", "6+ Referrals"],
    )

    featured["contract_risk_group"] = featured["contract"].map(
        {
            "Month-to-Month": "Higher Contract Risk",
            "One Year": "Moderate Contract Risk",
            "Two Year": "Lower Contract Risk",
        }
    ).astype(
        pd.CategoricalDtype(
            ["Lower Contract Risk", "Moderate Contract Risk", "Higher Contract Risk"],
            ordered=True,
        )
    )

    # This rule-based score is descriptive and is not a model prediction.
    risk_indicators = pd.DataFrame(
        {
            "month_to_month": featured["contract"].eq("Month-to-Month"),
            "early_tenure": featured["tenure_in_months"].le(12),
            "fiber_optic": featured["internet_type"].eq("Fiber Optic"),
            "without_premium_support": featured["premium_tech_support"].eq("No"),
            "monthly_charge_90_plus": featured["monthly_charge"].ge(90),
        },
        index=featured.index,
    )
    featured["descriptive_risk_points"] = risk_indicators.sum(axis=1).astype("int8")
    featured["descriptive_risk_segment"] = pd.cut(
        featured["descriptive_risk_points"],
        bins=[-1, 1, 2, 3, 5],
        labels=["Lower", "Elevated", "High", "Very High"],
    )

    featured["avg_revenue_per_tenure_month"] = (
        featured["total_revenue"] / featured["tenure_in_months"]
    ).round(2)

    engagement_conditions = [
        featured["number_of_referrals"].gt(0),
        featured["protection_support_service_count"].ge(2),
        featured["service_count"].ge(5),
    ]
    featured["customer_engagement_profile"] = pd.Categorical(
        np.select(
            engagement_conditions,
            ["Advocate", "Protection Engaged", "Multi-Service"],
            default="Basic Relationship",
        ),
        categories=["Basic Relationship", "Multi-Service", "Protection Engaged", "Advocate"],
        ordered=True,
    )

    validate_engineered_features(clean_df, featured)
    return featured


def validate_engineered_features(
    clean_df: pd.DataFrame, featured_df: pd.DataFrame
) -> None:
    """Verify that engineered features are complete and source data is preserved."""

    if len(featured_df) != len(clean_df):
        raise AssertionError("Feature engineering changed the customer row count.")
    if not featured_df["customer_id"].equals(clean_df["customer_id"]):
        raise AssertionError("Feature engineering changed customer identifiers or order.")
    if featured_df[FEATURE_COLUMNS].isna().any().any():
        nulls = featured_df[FEATURE_COLUMNS].isna().sum()
        raise AssertionError(
            f"Engineered features contain nulls: {nulls.loc[nulls.gt(0)].to_dict()}"
        )
    if not featured_df["service_count"].between(0, len(COUNTED_SERVICES)).all():
        raise AssertionError("service_count is outside its valid range.")
    if not featured_df["protection_support_service_count"].between(
        0, len(PROTECTION_SUPPORT_SERVICES)
    ).all():
        raise AssertionError("protection_support_service_count is outside its valid range.")
    if not featured_df["descriptive_risk_points"].between(0, 5).all():
        raise AssertionError("descriptive_risk_points is outside [0, 5].")
    if not featured_df["avg_revenue_per_tenure_month"].gt(0).all():
        raise AssertionError("Average revenue per tenure month must be positive.")


def build_feature_dataset(
    clean_path: str | Path = "data/processed/telco_customer_churn_clean.csv",
    output_path: str | Path = "data/processed/telco_customer_churn_features.csv",
) -> tuple[pd.DataFrame, Path]:
    """Load the clean CSV, add business features, and export the result."""

    clean_df = pd.read_csv(
        clean_path,
        dtype={"customer_id": "string", "zip_code": "string"},
    )
    featured_df = add_business_features(clean_df)
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    featured_df.to_csv(destination, index=False)
    return featured_df, destination.resolve()


if __name__ == "__main__":
    featured, saved_to = build_feature_dataset()
    print(f"Engineered {len(FEATURE_COLUMNS)} features for {len(featured):,} customers.")
    print(f"Feature-enhanced shape: {featured.shape}")
    print(f"Saved feature dataset to: {saved_to}")
