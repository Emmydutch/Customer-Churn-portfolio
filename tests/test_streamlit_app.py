"""Streamlit rendering and interaction tests."""

from __future__ import annotations

import time
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest


PAGES = [
    "Executive Overview",
    "Customer Analysis",
    "Churn Drivers",
    "Geographic Analysis",
    "Customer Risk Predictor",
    "Decision Centre",
    "Retention Simulator",
    "Methodology",
]
APP_PATH = Path(__file__).resolve().parents[1] / "app.py"


def open_page(page: str) -> AppTest:
    app = AppTest.from_file(APP_PATH, default_timeout=180)
    app.run()
    navigation = next(radio for radio in app.radio if page in radio.options)
    navigation.set_value(page)
    app.run(timeout=180)
    return app


@pytest.mark.parametrize("page", PAGES)
def test_every_page_renders_without_exception(page):
    rendered = open_page(page)
    assert not rendered.exception


def test_customer_filter_interaction_updates_population():
    rendered = open_page("Customer Analysis")
    baseline = int(rendered.metric[0].value.replace(",", ""))
    contract = next(widget for widget in rendered.multiselect if widget.label == "Contract")
    contract.set_value(["Month-to-Month"])
    rendered.run(timeout=180)
    filtered = int(rendered.metric[0].value.replace(",", ""))
    assert not rendered.exception
    assert 0 < filtered < baseline


def test_predictor_form_returns_probability_and_explanation():
    rendered = open_page("Customer Risk Predictor")
    button = next(button for button in rendered.button if button.label == "Calculate churn risk")
    button.click()
    rendered.run(timeout=180)
    assert not rendered.exception
    assert any(metric.label == "Predicted churn probability" for metric in rendered.metric)
    assert len(rendered.get("plotly_chart")) == 1


def test_retention_controls_update_scenario():
    rendered = open_page("Retention Simulator")
    baseline_cost = next(metric.value for metric in rendered.metric if metric.label == "Campaign cost")
    target_slider = next(slider for slider in rendered.slider if slider.label == "Customers targeted")
    target_slider.set_value(100)
    rendered.run(timeout=180)
    updated_cost = next(metric.value for metric in rendered.metric if metric.label == "Campaign cost")
    assert not rendered.exception
    assert updated_cost != baseline_cost


def test_decision_centre_respects_changed_capacity():
    rendered = open_page("Decision Centre")
    baseline = next(metric.value for metric in rendered.metric if metric.label == "Customers selected")
    capacity = next(slider for slider in rendered.slider if slider.label == "Maximum customers to contact")
    capacity.set_value(100)
    apply_button = next(button for button in rendered.button if button.label == "Apply decision policy")
    apply_button.click()
    rendered.run(timeout=180)
    updated = next(metric.value for metric in rendered.metric if metric.label == "Customers selected")
    assert not rendered.exception
    assert int(updated.replace(",", "")) <= 100
    assert updated != baseline


def test_warm_executive_render_performance():
    rendered = open_page("Executive Overview")
    start = time.perf_counter()
    rendered.run(timeout=180)
    elapsed = time.perf_counter() - start
    assert not rendered.exception
    assert elapsed < 5.0, f"Warm render took {elapsed:.2f}s"


def test_dark_theme_toggle_applies_without_exception():
    rendered = open_page("Executive Overview")
    theme_toggle = next(toggle for toggle in rendered.toggle if toggle.label == "Dark theme")
    theme_toggle.set_value(True)
    rendered.run(timeout=180)
    dark_css = [block.value for block in rendered.markdown if "color-scheme:dark" in block.value]
    assert not rendered.exception
    assert theme_toggle.value is True
    assert len(dark_css) == 1
    assert len(rendered.get("plotly_chart")) == 2
