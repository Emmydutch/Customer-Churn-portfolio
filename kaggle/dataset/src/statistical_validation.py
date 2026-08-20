"""Statistical association tests and leakage governance for churn features."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency, mannwhitneyu
from statsmodels.stats.multitest import multipletests


CATEGORICAL_ASSOCIATION_COLUMNS = [
    "gender",
    "under_30",
    "senior_citizen",
    "married",
    "dependents",
    "referred_a_friend",
    "offer",
    "phone_service",
    "multiple_lines",
    "internet_service",
    "internet_type",
    "online_security",
    "online_backup",
    "device_protection_plan",
    "premium_tech_support",
    "streaming_tv",
    "streaming_movies",
    "streaming_music",
    "unlimited_data",
    "contract",
    "paperless_billing",
    "payment_method",
    "customer_status",
    "churn_category",
    "churn_reason",
]

NUMERICAL_ASSOCIATION_COLUMNS = [
    "age",
    "number_of_dependents",
    "number_of_referrals",
    "tenure_in_months",
    "avg_monthly_long_distance_charges",
    "avg_monthly_gb_download",
    "monthly_charge",
    "total_charges",
    "total_refunds",
    "total_extra_data_charges",
    "total_long_distance_charges",
    "total_revenue",
    "satisfaction_score",
    "churn_score",
    "cltv",
]


def _effect_label(value: float, thresholds: tuple[float, float, float]) -> str:
    magnitude = abs(value)
    if magnitude < thresholds[0]:
        return "Negligible"
    if magnitude < thresholds[1]:
        return "Small"
    if magnitude < thresholds[2]:
        return "Moderate"
    return "Strong"


def categorical_churn_associations(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate chi-square tests and bias-corrected Cramer's V."""

    rows: list[dict[str, object]] = []
    for feature in CATEGORICAL_ASSOCIATION_COLUMNS:
        table = pd.crosstab(df[feature], df["churn_flag"], dropna=False)
        chi2, p_value, _, expected = chi2_contingency(table)
        n = table.to_numpy().sum()
        rows_count, columns_count = table.shape
        phi_squared = chi2 / n
        corrected_phi = max(
            0.0,
            phi_squared
            - ((columns_count - 1) * (rows_count - 1)) / max(n - 1, 1),
        )
        corrected_rows = rows_count - ((rows_count - 1) ** 2) / max(n - 1, 1)
        corrected_columns = columns_count - ((columns_count - 1) ** 2) / max(n - 1, 1)
        denominator = min(corrected_columns - 1, corrected_rows - 1)
        cramers_v = np.sqrt(corrected_phi / denominator) if denominator > 0 else 0.0

        rows.append(
            {
                "feature": feature,
                "levels": int(table.shape[0]),
                "chi_square": chi2,
                "p_value": p_value,
                "cramers_v": cramers_v,
                "effect_strength": _effect_label(cramers_v, (0.10, 0.30, 0.50)),
                "minimum_expected_count": float(expected.min()),
                "cells_expected_below_5": int((expected < 5).sum()),
                "assumption_flag": "Review sparse cells" if (expected < 5).any() else "Pass",
            }
        )

    result = pd.DataFrame(rows)
    result["adjusted_p_value"] = multipletests(
        result["p_value"], method="fdr_bh"
    )[1]
    result["significant_after_fdr"] = result["adjusted_p_value"].lt(0.05)
    return result.sort_values("cramers_v", ascending=False).reset_index(drop=True)


def numerical_churn_associations(df: pd.DataFrame) -> pd.DataFrame:
    """Compare churn groups with Mann-Whitney tests and rank-biserial effects."""

    rows: list[dict[str, object]] = []
    for feature in NUMERICAL_ASSOCIATION_COLUMNS:
        churned = df.loc[df["churn_flag"].eq(1), feature].dropna()
        not_churned = df.loc[df["churn_flag"].eq(0), feature].dropna()
        test = mannwhitneyu(churned, not_churned, alternative="two-sided")
        rank_biserial = (2 * test.statistic) / (len(churned) * len(not_churned)) - 1
        rows.append(
            {
                "feature": feature,
                "churned_median": float(churned.median()),
                "not_churned_median": float(not_churned.median()),
                "median_difference": float(churned.median() - not_churned.median()),
                "mann_whitney_u": float(test.statistic),
                "p_value": float(test.pvalue),
                "rank_biserial": float(rank_biserial),
                "effect_strength": _effect_label(rank_biserial, (0.10, 0.30, 0.50)),
                "direction": "Higher among churned" if rank_biserial > 0 else "Lower among churned",
            }
        )

    result = pd.DataFrame(rows)
    result["adjusted_p_value"] = multipletests(
        result["p_value"], method="fdr_bh"
    )[1]
    result["significant_after_fdr"] = result["adjusted_p_value"].lt(0.05)
    return result.sort_values(
        "rank_biserial", key=lambda values: values.abs(), ascending=False
    ).reset_index(drop=True)


def build_feature_governance(df: pd.DataFrame) -> pd.DataFrame:
    """Classify every feature by modeling availability and leakage risk."""

    governance: dict[str, tuple[str, str]] = {}

    def assign(columns: list[str], status: str, rationale: str) -> None:
        for column in columns:
            governance[column] = (status, rationale)

    assign(["churn_label", "churn_flag"], "Target", "Defines or numerically restates the churn outcome")
    assign(
        ["customer_status", "active_flag"],
        "Exclude — direct leakage",
        "Directly reveals whether the customer churned",
    )
    assign(
        ["churn_category", "churn_reason"],
        "Exclude — post-outcome",
        "Recorded only to explain a completed churn event",
    )
    assign(
        ["churn_score"],
        "Exclude — external model output",
        "Existing score has unknown training data and would create circularity",
    )
    assign(
        ["satisfaction_score"],
        "Conditional — timing unknown",
        "Perfect/near-perfect outcome separation suggests possible post-churn collection",
    )
    assign(
        ["cltv", "customer_value_segment"],
        "Conditional — provenance unknown",
        "Calculation method and availability date are not documented",
    )
    assign(["customer_id"], "Exclude — identifier", "Unique reference has no intended behavioral meaning")
    assign(
        ["country", "state", "quarter"],
        "Exclude — constant",
        "Contains one value and provides no within-dataset variation",
    )
    assign(
        ["city", "zip_code", "latitude", "longitude", "population"],
        "Descriptive — geography",
        "Useful for hotspot analysis; baseline modeling risks sparse location proxies",
    )
    assign(
        ["gender", "age", "under_30", "senior_citizen"],
        "Conditional — fairness review",
        "Potentially sensitive demographic information requiring fairness assessment",
    )
    assign(
        [
            "tenure_group",
            "age_group",
            "monthly_charge_band",
            "referral_group",
            "contract_risk_group",
            "descriptive_risk_points",
            "descriptive_risk_segment",
            "customer_engagement_profile",
            "avg_revenue_per_tenure_month",
        ],
        "Descriptive — engineered reporting feature",
        "Derived from other fields; exclude from baseline to avoid duplication or circular rules",
    )
    assign(
        ["service_count", "protection_support_service_count"],
        "Conditional — engineered aggregate",
        "Potentially useful but duplicates component service indicators; compare in alternatives",
    )

    for column in df.columns:
        governance.setdefault(
            column,
            (
                "Eligible pre-churn candidate",
                "Plausibly available before churn; subject to preprocessing and collinearity review",
            ),
        )

    result = pd.DataFrame(
        [
            {"feature": feature, "governance_status": status, "rationale": rationale}
            for feature, (status, rationale) in governance.items()
            if feature in df.columns
        ]
    )
    if set(result["feature"]) != set(df.columns):
        raise AssertionError("Feature governance does not cover every dataset column.")
    return result.sort_values(["governance_status", "feature"]).reset_index(drop=True)


def numerical_redundancy_report(
    df: pd.DataFrame, threshold: float = 0.70
) -> pd.DataFrame:
    """Return highly associated numerical feature pairs using Spearman correlation."""

    numerical = [column for column in NUMERICAL_ASSOCIATION_COLUMNS if column in df]
    correlations = df[numerical].corr(method="spearman")
    rows = []
    for index, first in enumerate(numerical):
        for second in numerical[index + 1 :]:
            correlation = float(correlations.loc[first, second])
            if abs(correlation) >= threshold:
                rows.append(
                    {
                        "feature_1": first,
                        "feature_2": second,
                        "spearman_correlation": correlation,
                        "absolute_correlation": abs(correlation),
                        "review_reason": "Potential redundancy or shared accumulation with tenure",
                    }
                )
    return pd.DataFrame(rows).sort_values(
        "absolute_correlation", ascending=False
    ).reset_index(drop=True)


def export_validation_outputs(
    feature_path: str | Path = "data/processed/telco_customer_churn_features.csv",
    output_directory: str | Path = "data/processed",
) -> dict[str, Path]:
    """Run and export statistical and governance assessments."""

    df = pd.read_csv(
        feature_path,
        dtype={"customer_id": "string", "zip_code": "string"},
    )
    output_dir = Path(output_directory)
    output_dir.mkdir(parents=True, exist_ok=True)
    outputs = {
        "categorical_associations": output_dir / "categorical_churn_associations.csv",
        "numerical_associations": output_dir / "numerical_churn_associations.csv",
        "numerical_redundancy": output_dir / "numerical_redundancy_report.csv",
        "feature_governance": output_dir / "feature_governance.csv",
    }
    categorical_churn_associations(df).to_csv(outputs["categorical_associations"], index=False)
    numerical_churn_associations(df).to_csv(outputs["numerical_associations"], index=False)
    numerical_redundancy_report(df).to_csv(outputs["numerical_redundancy"], index=False)
    build_feature_governance(df).to_csv(outputs["feature_governance"], index=False)
    return {name: path.resolve() for name, path in outputs.items()}


if __name__ == "__main__":
    exported = export_validation_outputs()
    for name, path in exported.items():
        print(f"{name}: {path}")
