"""Confidence intervals and validated analytical findings for customer churn."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import numpy as np
import pandas as pd
from statsmodels.stats.proportion import (
    confint_proportions_2indep,
    proportion_confint,
)


MaskRule = Callable[[pd.DataFrame], pd.Series]


COMPARISONS: dict[str, dict[str, str | MaskRule]] = {
    "month_to_month_vs_two_year": {
        "claim": "Month-to-month customers have greater churn exposure than two-year customers",
        "exposed_label": "Month-to-Month",
        "reference_label": "Two Year",
        "exposed": lambda df: df["contract"].eq("Month-to-Month"),
        "reference": lambda df: df["contract"].eq("Two Year"),
    },
    "early_vs_long_tenure": {
        "claim": "Customers in months 1–6 have greater churn exposure than customers in months 49–72",
        "exposed_label": "1–6 months",
        "reference_label": "49–72 months",
        "exposed": lambda df: df["tenure_in_months"].between(1, 6),
        "reference": lambda df: df["tenure_in_months"].between(49, 72),
    },
    "fiber_vs_dsl": {
        "claim": "Fiber-optic customers have greater churn exposure than DSL customers",
        "exposed_label": "Fiber Optic",
        "reference_label": "DSL",
        "exposed": lambda df: df["internet_type"].eq("Fiber Optic"),
        "reference": lambda df: df["internet_type"].eq("DSL"),
    },
    "no_support_vs_support": {
        "claim": "Customers without premium technical support have greater churn exposure",
        "exposed_label": "No Premium Support",
        "reference_label": "Premium Support",
        "exposed": lambda df: df["premium_tech_support"].eq("No"),
        "reference": lambda df: df["premium_tech_support"].eq("Yes"),
    },
    "no_security_vs_security": {
        "claim": "Customers without online security have greater churn exposure",
        "exposed_label": "No Online Security",
        "reference_label": "Online Security",
        "exposed": lambda df: df["online_security"].eq("No"),
        "reference": lambda df: df["online_security"].eq("Yes"),
    },
    "senior_vs_non_senior": {
        "claim": "Senior customers have greater churn exposure than non-senior customers",
        "exposed_label": "Senior",
        "reference_label": "Non-Senior",
        "exposed": lambda df: df["senior_citizen"].eq("Yes"),
        "reference": lambda df: df["senior_citizen"].eq("No"),
    },
    "no_dependents_vs_dependents": {
        "claim": "Customers without dependents have greater churn exposure",
        "exposed_label": "No Dependents",
        "reference_label": "Dependents",
        "exposed": lambda df: df["dependents"].eq("No"),
        "reference": lambda df: df["dependents"].eq("Yes"),
    },
    "bank_vs_credit_card": {
        "claim": "Bank-withdrawal customers have greater churn exposure than credit-card customers",
        "exposed_label": "Bank Withdrawal",
        "reference_label": "Credit Card",
        "exposed": lambda df: df["payment_method"].eq("Bank Withdrawal"),
        "reference": lambda df: df["payment_method"].eq("Credit Card"),
    },
    "san_diego_vs_other_cities": {
        "claim": "San Diego customers have greater churn exposure than customers elsewhere",
        "exposed_label": "San Diego",
        "reference_label": "Other Cities",
        "exposed": lambda df: df["city"].eq("San Diego"),
        "reference": lambda df: df["city"].ne("San Diego"),
    },
}


def overall_churn_interval(df: pd.DataFrame, alpha: float = 0.05) -> pd.Series:
    """Calculate the overall churn rate and Wilson confidence interval."""

    churned = int(df["churn_flag"].sum())
    customers = len(df)
    lower, upper = proportion_confint(churned, customers, alpha=alpha, method="wilson")
    return pd.Series(
        {
            "customers": customers,
            "churned_customers": churned,
            "churn_rate": churned / customers,
            "ci_lower": float(lower),
            "ci_upper": float(upper),
            "confidence_level": 1 - alpha,
            "method": "Wilson score interval",
        }
    )


def build_segment_comparisons(
    df: pd.DataFrame, alpha: float = 0.05
) -> pd.DataFrame:
    """Calculate rate, risk-difference, and risk-ratio intervals for key claims."""

    rows: list[dict[str, object]] = []
    for comparison_id, spec in COMPARISONS.items():
        exposed_rule = spec["exposed"]
        reference_rule = spec["reference"]
        if not callable(exposed_rule) or not callable(reference_rule):
            raise TypeError(f"Comparison masks must be callable: {comparison_id}")
        exposed_mask = exposed_rule(df)
        reference_mask = reference_rule(df)
        if (exposed_mask & reference_mask).any():
            raise ValueError(f"Comparison groups overlap: {comparison_id}")

        exposed_n = int(exposed_mask.sum())
        reference_n = int(reference_mask.sum())
        exposed_churned = int(df.loc[exposed_mask, "churn_flag"].sum())
        reference_churned = int(df.loc[reference_mask, "churn_flag"].sum())
        exposed_rate = exposed_churned / exposed_n
        reference_rate = reference_churned / reference_n
        exposed_ci = proportion_confint(
            exposed_churned, exposed_n, alpha=alpha, method="wilson"
        )
        reference_ci = proportion_confint(
            reference_churned, reference_n, alpha=alpha, method="wilson"
        )
        difference_ci = confint_proportions_2indep(
            exposed_churned,
            exposed_n,
            reference_churned,
            reference_n,
            method="newcomb",
            compare="diff",
            alpha=alpha,
        )
        ratio_ci = confint_proportions_2indep(
            exposed_churned,
            exposed_n,
            reference_churned,
            reference_n,
            method="log-adjusted",
            compare="ratio",
            alpha=alpha,
        )

        rows.append(
            {
                "comparison_id": comparison_id,
                "claim": spec["claim"],
                "exposed_group": spec["exposed_label"],
                "reference_group": spec["reference_label"],
                "exposed_customers": exposed_n,
                "reference_customers": reference_n,
                "exposed_churn_rate": exposed_rate,
                "exposed_rate_ci_lower": float(exposed_ci[0]),
                "exposed_rate_ci_upper": float(exposed_ci[1]),
                "reference_churn_rate": reference_rate,
                "reference_rate_ci_lower": float(reference_ci[0]),
                "reference_rate_ci_upper": float(reference_ci[1]),
                "risk_difference": exposed_rate - reference_rate,
                "risk_difference_ci_lower": float(difference_ci[0]),
                "risk_difference_ci_upper": float(difference_ci[1]),
                "risk_ratio": exposed_rate / reference_rate,
                "risk_ratio_ci_lower": float(ratio_ci[0]),
                "risk_ratio_ci_upper": float(ratio_ci[1]),
                "validated_difference": bool(
                    difference_ci[0] > 0 and ratio_ci[0] > 1
                ),
            }
        )

    return pd.DataFrame(rows).sort_values(
        "risk_difference", ascending=False
    ).reset_index(drop=True)


def _bootstrap_difference(
    first: np.ndarray,
    second: np.ndarray,
    statistic: Callable[[np.ndarray], float],
    iterations: int,
    seed: int,
    alpha: float,
) -> tuple[float, float, float]:
    rng = np.random.default_rng(seed)
    estimates = np.empty(iterations)
    for index in range(iterations):
        first_sample = rng.choice(first, size=len(first), replace=True)
        second_sample = rng.choice(second, size=len(second), replace=True)
        estimates[index] = statistic(first_sample) - statistic(second_sample)
    estimate = statistic(first) - statistic(second)
    lower, upper = np.quantile(estimates, [alpha / 2, 1 - alpha / 2])
    return float(estimate), float(lower), float(upper)


def build_numeric_confidence_intervals(
    df: pd.DataFrame,
    iterations: int = 3_000,
    seed: int = 42,
    alpha: float = 0.05,
) -> pd.DataFrame:
    """Bootstrap key numerical differences between churned and non-churned groups."""

    churned = df["churn_flag"].eq(1)
    specifications = [
        ("monthly_charge", "mean", np.mean),
        ("tenure_in_months", "median", np.median),
        ("total_revenue", "median", np.median),
        ("cltv", "median", np.median),
    ]
    rows = []
    for offset, (feature, statistic_name, statistic) in enumerate(specifications):
        first = df.loc[churned, feature].to_numpy(dtype=float)
        second = df.loc[~churned, feature].to_numpy(dtype=float)
        estimate, lower, upper = _bootstrap_difference(
            first,
            second,
            statistic,
            iterations=iterations,
            seed=seed + offset,
            alpha=alpha,
        )
        rows.append(
            {
                "feature": feature,
                "statistic": statistic_name,
                "difference_churned_minus_not_churned": estimate,
                "ci_lower": lower,
                "ci_upper": upper,
                "confidence_level": 1 - alpha,
                "bootstrap_iterations": iterations,
                "interval_excludes_zero": bool(lower > 0 or upper < 0),
            }
        )
    return pd.DataFrame(rows)


def build_validated_findings(
    df: pd.DataFrame,
    comparisons: pd.DataFrame,
    numeric_intervals: pd.DataFrame,
) -> pd.DataFrame:
    """Create a concise evidence register for portfolio reporting."""

    overall = overall_churn_interval(df)
    findings = [
        {
            "finding": "Overall portfolio churn",
            "evidence": (
                f"{overall['churn_rate']:.2%} "
                f"(95% CI {overall['ci_lower']:.2%}–{overall['ci_upper']:.2%})"
            ),
            "validation_status": "Validated descriptive estimate",
            "interpretation_limit": "Snapshot estimate; not a time-series churn rate",
        }
    ]
    for row in comparisons.itertuples(index=False):
        findings.append(
            {
                "finding": row.claim,
                "evidence": (
                    f"Risk difference {row.risk_difference:.2%} "
                    f"(95% CI {row.risk_difference_ci_lower:.2%}–"
                    f"{row.risk_difference_ci_upper:.2%}); "
                    f"risk ratio {row.risk_ratio:.2f} "
                    f"(95% CI {row.risk_ratio_ci_lower:.2f}–{row.risk_ratio_ci_upper:.2f})"
                ),
                "validation_status": (
                    "Validated association" if row.validated_difference else "Inconclusive"
                ),
                "interpretation_limit": "Bivariate observational comparison; not causal",
            }
        )
    for row in numeric_intervals.itertuples(index=False):
        findings.append(
            {
                "finding": f"Churn-group difference in {row.feature} ({row.statistic})",
                "evidence": (
                    f"Difference {row.difference_churned_minus_not_churned:,.2f} "
                    f"(bootstrap 95% CI {row.ci_lower:,.2f}–{row.ci_upper:,.2f})"
                ),
                "validation_status": (
                    "Validated difference" if row.interval_excludes_zero else "Inconclusive"
                ),
                "interpretation_limit": (
                    "Distributional comparison; CLTV remains provenance-conditional"
                    if row.feature == "cltv"
                    else "Distributional comparison; not causal"
                ),
            }
        )
    return pd.DataFrame(findings)


def export_analytical_validation(
    feature_path: str | Path = "data/processed/telco_customer_churn_features.csv",
    output_directory: str | Path = "data/processed",
) -> dict[str, Path]:
    """Calculate and export Stage 9 confidence-interval evidence."""

    df = pd.read_csv(
        feature_path,
        dtype={"customer_id": "string", "zip_code": "string"},
    )
    comparisons = build_segment_comparisons(df)
    numeric_intervals = build_numeric_confidence_intervals(df)
    findings = build_validated_findings(df, comparisons, numeric_intervals)
    output_dir = Path(output_directory)
    output_dir.mkdir(parents=True, exist_ok=True)
    outputs = {
        "segment_comparisons": output_dir / "validated_segment_comparisons.csv",
        "numeric_intervals": output_dir / "validated_numeric_differences.csv",
        "validated_findings": output_dir / "validated_analytical_findings.csv",
    }
    comparisons.to_csv(outputs["segment_comparisons"], index=False)
    numeric_intervals.to_csv(outputs["numeric_intervals"], index=False)
    findings.to_csv(outputs["validated_findings"], index=False)
    return {name: path.resolve() for name, path in outputs.items()}


if __name__ == "__main__":
    exported = export_analytical_validation()
    for name, path in exported.items():
        print(f"{name}: {path}")
