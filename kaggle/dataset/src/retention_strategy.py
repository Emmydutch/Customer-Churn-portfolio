"""Retention intervention assignment and transparent business-impact simulation."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


INTERVENTION_CATALOG: dict[str, dict[str, object]] = {
    "High-value outreach": {
        "priority": 1,
        "action": "Dedicated retention specialist and personalized value-protection review",
        "rationale": "Protect active High/Premium Value customers with High/Very High descriptive risk",
        "base_offer_cost": 75.0,
        "success_measure": "Save rate, retained monthly charges, net benefit, and 90/180-day retention",
    },
    "Early fiber assurance": {
        "priority": 2,
        "action": "First-90-day fiber service-health check, rapid issue resolution, and plan review",
        "rationale": "Address concentrated early-tenure fiber and month-to-month exposure",
        "base_offer_cost": 40.0,
        "success_measure": "Issue resolution, repeat contact, 90-day retention, and contract conversion",
    },
    "Early onboarding": {
        "priority": 3,
        "action": "Structured 30/60/90-day onboarding support and service education",
        "rationale": "Reduce first-six-month month-to-month churn exposure",
        "base_offer_cost": 25.0,
        "success_measure": "Onboarding completion, early support resolution, and 90/180-day retention",
    },
    "Senior assistance": {
        "priority": 4,
        "action": "Accessible bill review, assisted support, and service-fit consultation",
        "rationale": "Support senior customers with monthly charges of at least $90",
        "base_offer_cost": 20.0,
        "success_measure": "Resolution, plan-fit improvement, satisfaction, and retention",
    },
    "Fiber quality and contract": {
        "priority": 5,
        "action": "Fiber diagnostics followed by a value-led contract-upgrade incentive",
        "rationale": "Treat service concerns before requesting longer commitment",
        "base_offer_cost": 50.0,
        "success_measure": "Reliability, issue resolution, contract conversion, and churn versus control",
    },
    "Support bundle and contract": {
        "priority": 6,
        "action": "Time-limited premium-support bundle with a longer-contract value option",
        "rationale": "Address broad month-to-month exposure without premium support",
        "base_offer_cost": 45.0,
        "success_measure": "Support activation, contract conversion, utilization, and retention",
    },
    "San Diego investigation": {
        "priority": 7,
        "action": "Local network, acquisition, service, and competitor root-cause investigation",
        "rationale": "Respond to the geographic hotspot before applying blanket discounts",
        "base_offer_cost": 15.0,
        "success_measure": "Reliability, complaints, resolved causes, and local churn trend",
    },
    "Competitor-response retention": {
        "priority": 8,
        "action": "Personalized device, data, speed, or value response based on customer needs",
        "rationale": "Provide a governed fallback for other model-identified high-risk customers",
        "base_offer_cost": 55.0,
        "success_measure": "Offer acceptance, save rate, incremental margin, and retention versus control",
    },
}

SCENARIOS: dict[str, dict[str, float]] = {
    "Conservative": {
        "reach_rate": 0.60,
        "acceptance_rate": 0.35,
        "incremental_save_rate": 0.25,
        "retention_horizon_months": 6,
        "gross_margin_rate": 0.70,
        "contact_cost_per_target": 4.0,
        "offer_cost_multiplier": 1.20,
    },
    "Base": {
        "reach_rate": 0.75,
        "acceptance_rate": 0.50,
        "incremental_save_rate": 0.35,
        "retention_horizon_months": 12,
        "gross_margin_rate": 0.70,
        "contact_cost_per_target": 4.0,
        "offer_cost_multiplier": 1.00,
    },
    "Optimistic": {
        "reach_rate": 0.85,
        "acceptance_rate": 0.65,
        "incremental_save_rate": 0.50,
        "retention_horizon_months": 18,
        "gross_margin_rate": 0.70,
        "contact_cost_per_target": 4.0,
        "offer_cost_multiplier": 0.90,
    },
}


def load_campaign_population(
    feature_path: str | Path,
    score_path: str | Path,
) -> pd.DataFrame:
    """Return active customers meeting the selected model threshold."""

    features = pd.read_csv(
        feature_path,
        dtype={"customer_id": "string", "zip_code": "string"},
    )
    scores = pd.read_csv(score_path, dtype={"customer_id": "string"})[
        [
            "customer_id",
            "predicted_churn_probability",
            "high_risk_at_selected_threshold",
        ]
    ]
    merged = features.merge(scores, on="customer_id", how="inner", validate="one_to_one")
    eligible = merged.loc[
        merged["active_flag"].eq(1)
        & merged["high_risk_at_selected_threshold"].astype(bool)
    ].copy()
    if len(eligible) != 947:
        raise AssertionError(f"Expected 947 eligible active customers; found {len(eligible)}")
    return eligible.reset_index(drop=True)


def assign_primary_intervention(eligible: pd.DataFrame) -> pd.DataFrame:
    """Assign one primary action while retaining overlapping segment flags."""

    assigned = eligible.copy()
    assigned["flag_high_value_high_risk"] = (
        assigned["customer_value_segment"].isin(["High Value", "Premium Value"])
        & assigned["descriptive_risk_segment"].isin(["High", "Very High"])
    )
    assigned["flag_early_fiber"] = (
        assigned["tenure_in_months"].le(12)
        & assigned["internet_type"].eq("Fiber Optic")
        & assigned["contract"].eq("Month-to-Month")
    )
    assigned["flag_early_month_to_month"] = (
        assigned["tenure_in_months"].le(6)
        & assigned["contract"].eq("Month-to-Month")
    )
    assigned["flag_senior_high_charge"] = (
        assigned["senior_citizen"].eq("Yes")
        & assigned["monthly_charge"].ge(90)
    )
    assigned["flag_fiber_month_to_month"] = (
        assigned["internet_type"].eq("Fiber Optic")
        & assigned["contract"].eq("Month-to-Month")
    )
    assigned["flag_month_to_month_no_support"] = (
        assigned["contract"].eq("Month-to-Month")
        & assigned["premium_tech_support"].eq("No")
    )
    assigned["flag_san_diego"] = assigned["city"].eq("San Diego")

    conditions = [
        assigned["flag_high_value_high_risk"],
        assigned["flag_early_fiber"],
        assigned["flag_early_month_to_month"],
        assigned["flag_senior_high_charge"],
        assigned["flag_fiber_month_to_month"],
        assigned["flag_month_to_month_no_support"],
        assigned["flag_san_diego"],
    ]
    choices = [
        "High-value outreach",
        "Early fiber assurance",
        "Early onboarding",
        "Senior assistance",
        "Fiber quality and contract",
        "Support bundle and contract",
        "San Diego investigation",
    ]
    assigned["primary_intervention"] = np.select(
        conditions, choices, default="Competitor-response retention"
    )
    assigned["intervention_priority"] = assigned["primary_intervention"].map(
        lambda value: INTERVENTION_CATALOG[value]["priority"]
    )
    assigned["recommended_action"] = assigned["primary_intervention"].map(
        lambda value: INTERVENTION_CATALOG[value]["action"]
    )
    assigned["base_offer_cost"] = assigned["primary_intervention"].map(
        lambda value: INTERVENTION_CATALOG[value]["base_offer_cost"]
    )
    return assigned


def simulate_campaign(
    campaign: pd.DataFrame,
    *,
    scenario_name: str,
    reach_rate: float,
    acceptance_rate: float,
    incremental_save_rate: float,
    retention_horizon_months: float,
    gross_margin_rate: float,
    contact_cost_per_target: float,
    offer_cost_multiplier: float,
) -> dict[str, float | str]:
    """Estimate campaign cost and benefit under editable assumptions."""

    rate_values = {
        "reach_rate": reach_rate,
        "acceptance_rate": acceptance_rate,
        "incremental_save_rate": incremental_save_rate,
        "gross_margin_rate": gross_margin_rate,
    }
    if any(value < 0 or value > 1 for value in rate_values.values()):
        raise ValueError(f"Rates must be in [0, 1]: {rate_values}")
    if retention_horizon_months <= 0 or contact_cost_per_target < 0 or offer_cost_multiplier < 0:
        raise ValueError("Horizon must be positive and cost assumptions non-negative.")

    expected_churners = float(campaign["predicted_churn_probability"].sum())
    expected_reached = len(campaign) * reach_rate
    expected_acceptors = expected_reached * acceptance_rate
    expected_saved = (
        expected_churners * reach_rate * acceptance_rate * incremental_save_rate
    )
    contact_cost = len(campaign) * contact_cost_per_target
    expected_offer_cost = float(
        (
            campaign["base_offer_cost"]
            * reach_rate
            * acceptance_rate
            * offer_cost_multiplier
        ).sum()
    )
    expected_retained_gross_margin = float(
        (
            campaign["predicted_churn_probability"]
            * campaign["monthly_charge"]
            * reach_rate
            * acceptance_rate
            * incremental_save_rate
            * retention_horizon_months
            * gross_margin_rate
        ).sum()
    )
    total_campaign_cost = contact_cost + expected_offer_cost
    estimated_net_benefit = expected_retained_gross_margin - total_campaign_cost
    roi = estimated_net_benefit / total_campaign_cost if total_campaign_cost else np.nan
    cost_per_expected_save = total_campaign_cost / expected_saved if expected_saved else np.nan

    return {
        "scenario": scenario_name,
        "eligible_customers": int(len(campaign)),
        "model_weighted_expected_churners": expected_churners,
        "expected_reached_customers": expected_reached,
        "expected_offer_acceptors": expected_acceptors,
        "expected_customers_saved": expected_saved,
        "contact_cost": contact_cost,
        "expected_offer_cost": expected_offer_cost,
        "total_campaign_cost": total_campaign_cost,
        "expected_retained_gross_margin": expected_retained_gross_margin,
        "estimated_net_benefit": estimated_net_benefit,
        "estimated_roi": roi,
        "cost_per_expected_save": cost_per_expected_save,
        "reach_rate": reach_rate,
        "acceptance_rate": acceptance_rate,
        "incremental_save_rate": incremental_save_rate,
        "retention_horizon_months": retention_horizon_months,
        "gross_margin_rate": gross_margin_rate,
        "contact_cost_per_target": contact_cost_per_target,
        "offer_cost_multiplier": offer_cost_multiplier,
    }


def simulate_by_intervention(
    campaign: pd.DataFrame,
    scenario_name: str,
    assumptions: dict[str, float],
) -> pd.DataFrame:
    """Apply one scenario separately to each mutually exclusive primary action."""

    rows = []
    for intervention, subset in campaign.groupby("primary_intervention", observed=True):
        result = simulate_campaign(
            subset,
            scenario_name=scenario_name,
            **assumptions,
        )
        result["primary_intervention"] = intervention
        result["recommended_action"] = INTERVENTION_CATALOG[intervention]["action"]
        result["success_measure"] = INTERVENTION_CATALOG[intervention]["success_measure"]
        result["monthly_charge_exposure"] = float(subset["monthly_charge"].sum())
        result["cltv"] = float(subset["cltv"].sum())
        rows.append(result)
    return pd.DataFrame(rows).sort_values(
        "estimated_net_benefit", ascending=False
    ).reset_index(drop=True)


def build_sensitivity_matrix(campaign: pd.DataFrame) -> pd.DataFrame:
    """Vary acceptance and incremental save rates around base assumptions."""

    rows = []
    for acceptance_rate in [0.30, 0.40, 0.50, 0.60, 0.70]:
        for save_rate in [0.15, 0.25, 0.35, 0.45, 0.55]:
            assumptions = SCENARIOS["Base"].copy()
            assumptions["acceptance_rate"] = acceptance_rate
            assumptions["incremental_save_rate"] = save_rate
            result = simulate_campaign(
                campaign,
                scenario_name="Sensitivity",
                **assumptions,
            )
            rows.append(result)
    return pd.DataFrame(rows)


def run_retention_strategy(
    feature_path: str | Path = "data/processed/telco_customer_churn_features.csv",
    score_path: str | Path = "artifacts/evaluation/portfolio_risk_scores.csv",
    output_directory: str | Path = "artifacts/retention",
) -> dict[str, object]:
    """Create campaign assignments, scenarios, and segment recommendations."""

    eligible = load_campaign_population(feature_path, score_path)
    campaign = assign_primary_intervention(eligible)
    scenario_results = pd.DataFrame(
        [
            simulate_campaign(campaign, scenario_name=name, **assumptions)
            for name, assumptions in SCENARIOS.items()
        ]
    )
    base_by_intervention = simulate_by_intervention(
        campaign, "Base", SCENARIOS["Base"]
    )
    sensitivity = build_sensitivity_matrix(campaign)
    intervention_catalog = pd.DataFrame(
        [
            {"primary_intervention": name, **details}
            for name, details in INTERVENTION_CATALOG.items()
        ]
    ).sort_values("priority")

    output_dir = Path(output_directory)
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "campaign_population": output_dir / "retention_campaign_population.csv",
        "scenario_results": output_dir / "retention_scenario_results.csv",
        "base_by_intervention": output_dir / "base_scenario_by_intervention.csv",
        "sensitivity": output_dir / "retention_sensitivity_matrix.csv",
        "intervention_catalog": output_dir / "intervention_catalog.csv",
        "report": output_dir / "retention_strategy_report.json",
    }
    campaign.to_csv(paths["campaign_population"], index=False)
    scenario_results.to_csv(paths["scenario_results"], index=False)
    base_by_intervention.to_csv(paths["base_by_intervention"], index=False)
    sensitivity.to_csv(paths["sensitivity"], index=False)
    intervention_catalog.to_csv(paths["intervention_catalog"], index=False)

    report = {
        "eligible_customers": int(len(campaign)),
        "model_weighted_expected_churners": float(
            campaign["predicted_churn_probability"].sum()
        ),
        "total_monthly_charge_exposure": float(campaign["monthly_charge"].sum()),
        "risk_weighted_monthly_charge_exposure": float(
            (
                campaign["predicted_churn_probability"]
                * campaign["monthly_charge"]
            ).sum()
        ),
        "total_cltv": float(campaign["cltv"].sum()),
        "primary_intervention_counts": {
            key: int(value)
            for key, value in campaign["primary_intervention"].value_counts().items()
        },
        "scenario_results": scenario_results.to_dict(orient="records"),
        "interpretation_warning": (
            "All financial results are assumption-driven scenarios, not realized savings. "
            "Segments require controlled treatment/comparison evaluation before scale-up."
        ),
    }
    paths["report"].write_text(json.dumps(report, indent=2), encoding="utf-8")

    return {
        "campaign": campaign,
        "scenario_results": scenario_results,
        "base_by_intervention": base_by_intervention,
        "sensitivity": sensitivity,
        "intervention_catalog": intervention_catalog,
        "report": report,
        "paths": {name: path.resolve() for name, path in paths.items()},
    }


if __name__ == "__main__":
    result = run_retention_strategy()
    print(json.dumps(result["report"], indent=2))
    for name, path in result["paths"].items():
        print(f"{name}: {path}")
