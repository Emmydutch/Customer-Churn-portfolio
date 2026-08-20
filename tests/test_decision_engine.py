"""Tests for the transparent prescriptive retention decision policy."""

from __future__ import annotations

import math
from pathlib import Path

import pandas as pd
import pytest

from src.decision_engine import (
    build_decision_sensitivity,
    score_retention_decisions,
    select_decision_portfolio,
    summarize_decision_portfolio,
)


ROOT = Path(__file__).resolve().parents[1]
BASE_ASSUMPTIONS = {
    "reach_rate": 0.75,
    "acceptance_rate": 0.50,
    "incremental_save_rate": 0.35,
    "retention_horizon_months": 12,
    "gross_margin_rate": 0.70,
    "contact_cost_per_target": 4.0,
    "offer_cost_multiplier": 1.0,
}


@pytest.fixture(scope="module")
def campaign() -> pd.DataFrame:
    return pd.read_csv(ROOT / "artifacts" / "retention" / "retention_campaign_population.csv")


def test_customer_economics_reconcile(campaign):
    scored = score_retention_decisions(campaign, **BASE_ASSUMPTIONS)
    row = scored.iloc[0]
    expected_saves = row.predicted_churn_probability * 0.75 * 0.50 * 0.35
    expected_margin = expected_saves * row.monthly_charge * 12 * 0.70
    expected_cost = 4.0 + row.base_offer_cost * 0.75 * 0.50
    assert math.isclose(row.expected_incremental_saves, expected_saves)
    assert math.isclose(row.expected_retained_gross_margin, expected_margin)
    assert math.isclose(row.expected_campaign_cost, expected_cost)
    assert math.isclose(row.expected_net_benefit, expected_margin - expected_cost)


def test_selection_respects_budget_capacity_and_positive_value(campaign):
    scored = score_retention_decisions(campaign, **BASE_ASSUMPTIONS)
    selected, ranked = select_decision_portfolio(
        scored,
        budget=5_000,
        capacity=100,
        objective="Maximize expected net benefit",
    )
    assert 0 < len(selected) <= 100
    assert selected.expected_campaign_cost.sum() <= 5_000 + 1e-9
    assert selected.expected_net_benefit.gt(0).all()
    assert selected.decision_rank.tolist() == list(range(1, len(selected) + 1))
    assert ranked.selected_for_campaign.sum() == len(selected)


def test_selection_is_deterministic(campaign):
    scored = score_retention_decisions(campaign, **BASE_ASSUMPTIONS)
    first, _ = select_decision_portfolio(
        scored, budget=7_500, capacity=200, objective="Maximize expected net benefit"
    )
    second, _ = select_decision_portfolio(
        scored, budget=7_500, capacity=200, objective="Maximize expected net benefit"
    )
    assert first.customer_id.tolist() == second.customer_id.tolist()


def test_summary_and_sensitivity_respond_to_save_rate(campaign):
    common = BASE_ASSUMPTIONS.copy()
    common.pop("incremental_save_rate")
    sensitivity = build_decision_sensitivity(
        campaign,
        save_rates=[0.20, 0.35, 0.50],
        budget=15_000,
        capacity=500,
        objective="Maximize expected net benefit",
        common_assumptions=common,
    )
    assert sensitivity.expected_net_benefit.is_monotonic_increasing
    assert sensitivity.expected_customers_saved.is_monotonic_increasing
    assert sensitivity.selected_customers.le(500).all()

    scored = score_retention_decisions(campaign, **BASE_ASSUMPTIONS)
    selected, ranked = select_decision_portfolio(scored, budget=0, capacity=0)
    summary = summarize_decision_portfolio(selected, ranked)
    assert summary["selected_customers"] == 0
    assert summary["expected_campaign_cost"] == 0


@pytest.mark.parametrize(
    "overrides, message",
    [
        ({"reach_rate": 1.1}, "Rates must be between"),
        ({"retention_horizon_months": 0}, "horizon must be positive"),
        ({"contact_cost_per_target": -1}, "Cost assumptions"),
    ],
)
def test_invalid_economic_assumptions_are_rejected(campaign, overrides, message):
    assumptions = BASE_ASSUMPTIONS.copy()
    assumptions.update(overrides)
    with pytest.raises(ValueError, match=message):
        score_retention_decisions(campaign, **assumptions)


def test_invalid_constraints_and_objective_are_rejected(campaign):
    scored = score_retention_decisions(campaign, **BASE_ASSUMPTIONS)
    with pytest.raises(ValueError, match="Budget"):
        select_decision_portfolio(scored, budget=-1, capacity=10)
    with pytest.raises(ValueError, match="Capacity"):
        select_decision_portfolio(scored, budget=100, capacity=-1)
    with pytest.raises(ValueError, match="Unknown decision objective"):
        select_decision_portfolio(scored, budget=100, capacity=10, objective="Unknown")
