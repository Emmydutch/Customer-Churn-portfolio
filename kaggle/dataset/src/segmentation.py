"""Customer risk segments and lifecycle cohort summaries."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import numpy as np
import pandas as pd


SegmentRule = Callable[[pd.DataFrame], pd.Series]


ACTIONABLE_SEGMENTS: dict[str, dict[str, str | SegmentRule]] = {
    "month_to_month_no_support": {
        "segment": "Month-to-month without premium support",
        "definition": "Month-to-month contract and no premium technical support",
        "recommended_focus": "Proactive support trial and value-led contract migration",
        "rule": lambda df: df["contract"].eq("Month-to-Month")
        & df["premium_tech_support"].eq("No"),
    },
    "fiber_month_to_month": {
        "segment": "Fiber optic on month-to-month contract",
        "definition": "Fiber-optic internet and month-to-month contract",
        "recommended_focus": "Fiber experience review and targeted commitment offer",
        "rule": lambda df: df["internet_type"].eq("Fiber Optic")
        & df["contract"].eq("Month-to-Month"),
    },
    "high_value_high_risk": {
        "segment": "High-value with high descriptive risk",
        "definition": "High/Premium Value and High/Very High descriptive risk",
        "recommended_focus": "Priority human outreach with value-protection treatment",
        "rule": lambda df: df["customer_value_segment"].isin(
            ["High Value", "Premium Value"]
        )
        & df["descriptive_risk_segment"].isin(["High", "Very High"]),
    },
    "early_tenure_month_to_month": {
        "segment": "Early-tenure month-to-month",
        "definition": "Tenure of 1–6 months and month-to-month contract",
        "recommended_focus": "First-six-month onboarding and service assurance",
        "rule": lambda df: df["tenure_in_months"].le(6)
        & df["contract"].eq("Month-to-Month"),
    },
    "early_fiber_month_to_month": {
        "segment": "Early-tenure fiber month-to-month",
        "definition": "Tenure of 1–12 months, fiber optic, and month-to-month contract",
        "recommended_focus": "Early fiber check-in, issue resolution, and plan review",
        "rule": lambda df: df["tenure_in_months"].le(12)
        & df["internet_type"].eq("Fiber Optic")
        & df["contract"].eq("Month-to-Month"),
    },
    "senior_high_charge": {
        "segment": "Senior with $90+ monthly charge",
        "definition": "Senior citizen and monthly charge of at least $90",
        "recommended_focus": "Accessible bill review and service-fit consultation",
        "rule": lambda df: df["senior_citizen"].eq("Yes")
        & df["monthly_charge"].ge(90),
    },
    "san_diego": {
        "segment": "San Diego customers",
        "definition": "Customer city is San Diego",
        "recommended_focus": "Local network, service, acquisition, and competitor investigation",
        "rule": lambda df: df["city"].eq("San Diego"),
    },
}


def build_segment_priority_table(df: pd.DataFrame) -> pd.DataFrame:
    """Summarize overlapping actionable segments and rank commercial opportunity."""

    rows: list[dict[str, object]] = []
    for segment_id, spec in ACTIONABLE_SEGMENTS.items():
        rule = spec["rule"]
        if not callable(rule):
            raise TypeError(f"Segment rule is not callable: {segment_id}")
        mask = rule(df)
        active_mask = mask & df["active_flag"].eq(1)
        customers = int(mask.sum())
        churned = int(df.loc[mask, "churn_flag"].sum())
        active_customers = int(active_mask.sum())
        churn_rate = churned / customers if customers else np.nan
        active_monthly_exposure = float(df.loc[active_mask, "monthly_charge"].sum())
        active_cltv = float(df.loc[active_mask, "cltv"].sum())

        rows.append(
            {
                "segment_id": segment_id,
                "segment": spec["segment"],
                "definition": spec["definition"],
                "customers": customers,
                "churned_customers": churned,
                "observed_churn_rate": churn_rate,
                "active_customers": active_customers,
                "active_monthly_charge_exposure": active_monthly_exposure,
                "active_cltv": active_cltv,
                "opportunity_index": churn_rate * active_monthly_exposure,
                "recommended_focus": spec["recommended_focus"],
            }
        )

    priority = pd.DataFrame(rows).sort_values(
        ["opportunity_index", "observed_churn_rate"], ascending=False
    )
    priority.insert(0, "priority_rank", range(1, len(priority) + 1))
    return priority.reset_index(drop=True)


def build_tenure_contract_cohorts(df: pd.DataFrame) -> pd.DataFrame:
    """Return long-form lifecycle cohorts by tenure group and contract."""

    cohort = (
        df.groupby(["tenure_group", "contract"], observed=True)
        .agg(
            customers=("customer_id", "size"),
            churned_customers=("churn_flag", "sum"),
            active_customers=("active_flag", "sum"),
            monthly_charge_exposure=("monthly_charge", "sum"),
        )
        .reset_index()
    )
    cohort["observed_churn_rate"] = (
        cohort["churned_customers"] / cohort["customers"]
    )
    return cohort


def build_tenure_internet_cohorts(df: pd.DataFrame) -> pd.DataFrame:
    """Return long-form lifecycle cohorts by tenure group and internet type."""

    cohort = (
        df.groupby(["tenure_group", "internet_type"], observed=True)
        .agg(
            customers=("customer_id", "size"),
            churned_customers=("churn_flag", "sum"),
            active_customers=("active_flag", "sum"),
        )
        .reset_index()
    )
    cohort["observed_churn_rate"] = (
        cohort["churned_customers"] / cohort["customers"]
    )
    return cohort


def build_geographic_hotspots(
    df: pd.DataFrame,
    geography: str,
    minimum_customers: int,
) -> pd.DataFrame:
    """Rank geographic areas after applying an explicit minimum sample size."""

    if geography not in {"city", "zip_code"}:
        raise ValueError("geography must be 'city' or 'zip_code'.")
    summary = (
        df.groupby(geography, observed=True)
        .agg(
            customers=("customer_id", "size"),
            churned_customers=("churn_flag", "sum"),
            active_customers=("active_flag", "sum"),
            active_monthly_charge_exposure=(
                "monthly_charge",
                lambda values: values[df.loc[values.index, "active_flag"].eq(1)].sum(),
            ),
        )
        .reset_index()
    )
    summary["observed_churn_rate"] = (
        summary["churned_customers"] / summary["customers"]
    )
    return summary.loc[summary["customers"].ge(minimum_customers)].sort_values(
        ["observed_churn_rate", "customers"], ascending=False
    ).reset_index(drop=True)


def export_segmentation_outputs(
    feature_path: str | Path = "data/processed/telco_customer_churn_features.csv",
    output_directory: str | Path = "data/processed",
) -> dict[str, Path]:
    """Build and export all Stage 7 segmentation tables."""

    df = pd.read_csv(
        feature_path,
        dtype={"customer_id": "string", "zip_code": "string"},
    )
    output_dir = Path(output_directory)
    output_dir.mkdir(parents=True, exist_ok=True)

    outputs = {
        "prioritized_segments": output_dir / "prioritized_risk_segments.csv",
        "tenure_contract_cohorts": output_dir / "tenure_contract_cohorts.csv",
        "tenure_internet_cohorts": output_dir / "tenure_internet_cohorts.csv",
        "city_hotspots": output_dir / "city_churn_hotspots.csv",
        "zip_hotspots": output_dir / "zip_churn_hotspots.csv",
    }
    build_segment_priority_table(df).to_csv(outputs["prioritized_segments"], index=False)
    build_tenure_contract_cohorts(df).to_csv(outputs["tenure_contract_cohorts"], index=False)
    build_tenure_internet_cohorts(df).to_csv(outputs["tenure_internet_cohorts"], index=False)
    build_geographic_hotspots(df, "city", 50).to_csv(outputs["city_hotspots"], index=False)
    build_geographic_hotspots(df, "zip_code", 20).to_csv(outputs["zip_hotspots"], index=False)
    return {name: path.resolve() for name, path in outputs.items()}


if __name__ == "__main__":
    exported = export_segmentation_outputs()
    for name, path in exported.items():
        print(f"{name}: {path}")
