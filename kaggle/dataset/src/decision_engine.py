"""Transparent prescriptive retention decision policy.

The engine converts pre-churn probabilities into scenario-based customer economics,
then applies a deterministic budget and capacity constrained ranking. It does not
estimate causal treatment effects; incremental save rates remain explicit inputs.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


REQUIRED_COLUMNS = {
    "customer_id",
    "predicted_churn_probability",
    "monthly_charge",
    "base_offer_cost",
    "primary_intervention",
    "recommended_action",
    "customer_value_segment",
    "descriptive_risk_segment",
}

OBJECTIVES = {
    "Maximize expected net benefit": "expected_net_benefit",
    "Maximize customers saved per dollar": "save_efficiency",
    "Prioritize highest churn risk": "predicted_churn_probability",
}


def _validate_assumptions(
    *,
    reach_rate: float,
    acceptance_rate: float,
    incremental_save_rate: float,
    retention_horizon_months: float,
    gross_margin_rate: float,
    contact_cost_per_target: float,
    offer_cost_multiplier: float,
) -> None:
    rates = {
        "reach_rate": reach_rate,
        "acceptance_rate": acceptance_rate,
        "incremental_save_rate": incremental_save_rate,
        "gross_margin_rate": gross_margin_rate,
    }
    if any(value < 0 or value > 1 for value in rates.values()):
        raise ValueError(f"Rates must be between 0 and 1: {rates}")
    if retention_horizon_months <= 0:
        raise ValueError("Retention horizon must be positive.")
    if contact_cost_per_target < 0 or offer_cost_multiplier < 0:
        raise ValueError("Cost assumptions must be non-negative.")


def score_retention_decisions(
    campaign: pd.DataFrame,
    *,
    reach_rate: float,
    acceptance_rate: float,
    incremental_save_rate: float,
    retention_horizon_months: float,
    gross_margin_rate: float,
    contact_cost_per_target: float,
    offer_cost_multiplier: float,
) -> pd.DataFrame:
    """Calculate expected customer-level economics under explicit assumptions."""

    missing = REQUIRED_COLUMNS.difference(campaign.columns)
    if missing:
        raise KeyError(f"Decision input is missing required columns: {sorted(missing)}")
    _validate_assumptions(
        reach_rate=reach_rate,
        acceptance_rate=acceptance_rate,
        incremental_save_rate=incremental_save_rate,
        retention_horizon_months=retention_horizon_months,
        gross_margin_rate=gross_margin_rate,
        contact_cost_per_target=contact_cost_per_target,
        offer_cost_multiplier=offer_cost_multiplier,
    )

    scored = campaign.copy()
    treatment_probability = reach_rate * acceptance_rate
    scored["expected_incremental_saves"] = (
        scored["predicted_churn_probability"]
        * treatment_probability
        * incremental_save_rate
    )
    scored["expected_retained_gross_margin"] = (
        scored["expected_incremental_saves"]
        * scored["monthly_charge"]
        * retention_horizon_months
        * gross_margin_rate
    )
    scored["expected_contact_cost"] = float(contact_cost_per_target)
    scored["expected_offer_cost"] = (
        scored["base_offer_cost"] * treatment_probability * offer_cost_multiplier
    )
    scored["expected_campaign_cost"] = (
        scored["expected_contact_cost"] + scored["expected_offer_cost"]
    )
    scored["expected_net_benefit"] = (
        scored["expected_retained_gross_margin"] - scored["expected_campaign_cost"]
    )
    scored["expected_roi"] = np.where(
        scored["expected_campaign_cost"].gt(0),
        scored["expected_net_benefit"] / scored["expected_campaign_cost"],
        np.nan,
    )
    scored["save_efficiency"] = np.where(
        scored["expected_campaign_cost"].gt(0),
        scored["expected_incremental_saves"] / scored["expected_campaign_cost"],
        np.nan,
    )
    scored["economically_eligible"] = scored["expected_net_benefit"].gt(0)
    return scored


def select_decision_portfolio(
    scored: pd.DataFrame,
    *,
    budget: float,
    capacity: int,
    objective: str = "Maximize expected net benefit",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Select a transparent ranked portfolio within expected budget and capacity.

    This is a deterministic constrained ranking policy, not a causal treatment
    optimizer. Only customers with positive scenario-based net benefit are selected.
    """

    if budget < 0:
        raise ValueError("Budget must be non-negative.")
    if capacity < 0:
        raise ValueError("Capacity must be non-negative.")
    if objective not in OBJECTIVES:
        raise ValueError(f"Unknown decision objective: {objective}")

    rank_column = OBJECTIVES[objective]
    ranked = scored.sort_values(
        [rank_column, "expected_net_benefit", "customer_id"],
        ascending=[False, False, True],
        kind="mergesort",
    ).copy()
    ranked["selected_for_campaign"] = False
    ranked["exclusion_reason"] = "Not selected within current constraints"

    selected_indices: list[int] = []
    spend = 0.0
    for index, row in ranked.iterrows():
        if not bool(row["economically_eligible"]):
            ranked.at[index, "exclusion_reason"] = "Non-positive expected net benefit"
            continue
        if len(selected_indices) >= capacity:
            ranked.at[index, "exclusion_reason"] = "Campaign capacity reached"
            continue
        candidate_cost = float(row["expected_campaign_cost"])
        if spend + candidate_cost > budget + 1e-9:
            ranked.at[index, "exclusion_reason"] = "Expected budget constraint"
            continue
        selected_indices.append(index)
        spend += candidate_cost
        ranked.at[index, "selected_for_campaign"] = True
        ranked.at[index, "exclusion_reason"] = "Selected"

    selected = ranked.loc[selected_indices].copy()
    selected["decision_rank"] = np.arange(1, len(selected) + 1)
    if selected.empty:
        selected["decision_category"] = pd.Series(dtype="string")
    else:
        priority_cutoff = selected["expected_net_benefit"].quantile(0.75)
        selected["decision_category"] = np.select(
            [
                selected["expected_net_benefit"].ge(priority_cutoff),
                selected["base_offer_cost"].le(30),
            ],
            ["Priority intervention", "Low-cost intervention"],
            default="Targeted intervention",
        )
    return selected.reset_index(drop=True), ranked.reset_index(drop=True)


def summarize_decision_portfolio(selected: pd.DataFrame, ranked: pd.DataFrame) -> dict[str, float | int]:
    """Return executive decision metrics for a selected portfolio."""

    cost = float(selected["expected_campaign_cost"].sum()) if len(selected) else 0.0
    net = float(selected["expected_net_benefit"].sum()) if len(selected) else 0.0
    return {
        "selected_customers": int(len(selected)),
        "expected_campaign_cost": cost,
        "expected_customers_saved": float(selected["expected_incremental_saves"].sum()) if len(selected) else 0.0,
        "expected_retained_gross_margin": float(selected["expected_retained_gross_margin"].sum()) if len(selected) else 0.0,
        "expected_net_benefit": net,
        "expected_roi": net / cost if cost else np.nan,
        "economically_ineligible_customers": int((~ranked["economically_eligible"]).sum()),
        "budget_excluded_customers": int(ranked["exclusion_reason"].eq("Expected budget constraint").sum()),
        "capacity_excluded_customers": int(ranked["exclusion_reason"].eq("Campaign capacity reached").sum()),
    }


def build_decision_sensitivity(
    campaign: pd.DataFrame,
    *,
    save_rates: list[float],
    budget: float,
    capacity: int,
    objective: str,
    common_assumptions: dict[str, float],
) -> pd.DataFrame:
    """Rebuild the decision portfolio across uncertain incremental save rates."""

    rows = []
    for save_rate in save_rates:
        scored = score_retention_decisions(
            campaign,
            incremental_save_rate=save_rate,
            **common_assumptions,
        )
        selected, ranked = select_decision_portfolio(
            scored, budget=budget, capacity=capacity, objective=objective
        )
        rows.append({
            "incremental_save_rate": save_rate,
            **summarize_decision_portfolio(selected, ranked),
        })
    return pd.DataFrame(rows)
