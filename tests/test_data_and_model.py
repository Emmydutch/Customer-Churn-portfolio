"""Data, calculation, prediction, missing-value, and edge-case tests."""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import app


ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def data():
    return app.load_data()


def test_processed_kpis_reconcile_to_source(data):
    raw = pd.read_csv(ROOT / "TelcoCustomerChurn.csv")
    clean = data["customers"]
    assert len(raw) == len(clean) == 7_043
    assert raw.CustomerID.nunique() == clean.customer_id.nunique() == 7_043
    assert raw.ChurnLabel.eq("Yes").sum() == clean.churn_flag.sum() == 1_869
    assert raw.CustomerStatus.ne("Churned").sum() == clean.active_flag.sum() == 5_174
    assert math.isclose(raw.MonthlyCharge.mean(), clean.monthly_charge.mean(), abs_tol=1e-10)
    assert math.isclose(raw.TotalRevenue.sum(), clean.total_revenue.sum(), abs_tol=1e-6)
    assert math.isclose(raw.loc[raw.ChurnLabel.eq("Yes"), "MonthlyCharge"].sum(), 139_130.85, abs_tol=.01)


def test_clean_application_data_is_complete_and_unique(data):
    customers = data["customers"]
    assert not customers.isna().any().any()
    assert not customers.customer_id.duplicated().any()
    assert customers.churn_flag.isin([0, 1]).all()
    assert customers.active_flag.isin([0, 1]).all()


def test_scores_and_retention_population_reconcile(data):
    scores, retention = data["scores"], data["retention"]
    assert len(scores) == 7_043
    assert scores.predicted_churn_probability.between(0, 1).all()
    eligible = scores.customer_status.ne("Churned") & scores.high_risk_at_selected_threshold.eq(1)
    assert eligible.sum() == len(retention) == 947
    assert retention.customer_id.nunique() == 947


def test_dashboard_filter_matches_direct_source_query(data):
    customers = data["customers"]
    selections = {
        "contract": ["Month-to-Month"],
        "internet_type": ["Fiber Optic"],
        "premium_tech_support": ["No"],
    }
    actual = app.filter_customers(customers, selections)
    expected = customers.loc[
        customers.contract.eq("Month-to-Month")
        & customers.internet_type.eq("Fiber Optic")
        & customers.premium_tech_support.eq("No")
    ]
    assert actual.customer_id.tolist() == expected.customer_id.tolist()
    assert len(actual) > 0


def test_filter_empty_state_and_unknown_filter(data):
    customers = data["customers"]
    assert app.filter_customers(customers, {"contract": ["Not a real contract"]}).empty
    with pytest.raises(KeyError, match="Unknown dashboard filter"):
        app.filter_customers(customers, {"unknown": ["value"]})


def test_model_prediction_and_local_explanation(data):
    model = app.load_model()
    row = data["customers"].loc[:, model.feature_names_in_].iloc[[0]].copy()
    probability = float(model.predict_proba(row)[0, 1])
    explanation = app.local_explanation(row, model, data["background"])
    assert 0 <= probability <= 1
    assert len(explanation) == 8
    assert explanation.feature.nunique() == 8
    assert set(explanation.direction).issubset({"Raises risk", "Lowers risk"})


def test_model_handles_missing_and_unseen_inputs(data):
    model = app.load_model()
    row = data["customers"].loc[:, model.feature_names_in_].iloc[[0]].copy()
    row.loc[:, "monthly_charge"] = np.nan
    row.loc[:, "contract"] = np.nan
    row.loc[:, "internet_type"] = "Previously unseen technology"
    probability = float(model.predict_proba(row)[0, 1])
    assert np.isfinite(probability)
    assert 0 <= probability <= 1


def test_retention_base_case_reconciles_to_export(data):
    result = app.calculate_campaign_impact(
        data["retention"], 947, .75, .50, .35, 12, .70, 4.0, 1.0
    )
    expected = pd.read_csv(ROOT / "artifacts" / "retention" / "retention_scenario_results.csv")
    base = expected.loc[expected.scenario.eq("Base")].iloc[0]
    assert math.isclose(result["expected_customers_saved"], base.expected_customers_saved, abs_tol=.01)
    assert math.isclose(result["campaign_cost"], base.total_campaign_cost, abs_tol=.01)
    assert math.isclose(result["net_benefit"], base.estimated_net_benefit, abs_tol=.01)


@pytest.mark.parametrize(
    "kwargs, message",
    [
        ({"campaign_size": 0}, "Campaign size"),
        ({"reach_rate": 1.1}, "rates must be between"),
        ({"contact_cost": -1}, "must be non-negative"),
        ({"horizon_months": 0}, "must be non-negative"),
    ],
)
def test_retention_rejects_invalid_assumptions(data, kwargs, message):
    inputs = dict(
        campaign=data["retention"], campaign_size=100, reach_rate=.75,
        acceptance_rate=.5, incremental_save_rate=.35, horizon_months=12,
        gross_margin_rate=.7, contact_cost=4.0, offer_cost_multiplier=1.0,
    )
    inputs.update(kwargs)
    with pytest.raises(ValueError, match=message):
        app.calculate_campaign_impact(**inputs)
