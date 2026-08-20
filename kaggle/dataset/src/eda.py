"""Reusable exploratory-analysis summaries and Plotly charts."""

from __future__ import annotations

from collections.abc import Sequence

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go


NAVY = "#17324D"
BLUE = "#2F75B5"
TEAL = "#19A7A0"
ORANGE = "#F28E2B"
RED = "#D9534F"
LIGHT_BLUE = "#DCEAF7"


def churn_summary(
    df: pd.DataFrame,
    dimension: str,
    order: Sequence | None = None,
) -> pd.DataFrame:
    """Return customer count, churn count, and churn rate by one dimension."""

    required = {"customer_id", "churn_flag", dimension}
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"Missing columns for churn summary: {missing}")

    summary = (
        df.groupby(dimension, observed=True, dropna=False)
        .agg(customers=("customer_id", "size"), churned=("churn_flag", "sum"))
        .reset_index()
    )
    summary["churn_rate"] = summary["churned"] / summary["customers"]
    summary["churn_rate_label"] = summary["churn_rate"].map(lambda value: f"{value:.1%}")

    if order is not None:
        summary[dimension] = pd.Categorical(
            summary[dimension], categories=list(order), ordered=True
        )
        summary = summary.sort_values(dimension)

    return summary.reset_index(drop=True)


def churn_rate_bar(
    summary: pd.DataFrame,
    dimension: str,
    title: str,
    *,
    horizontal: bool = False,
    color: str = BLUE,
) -> go.Figure:
    """Create a consistent churn-rate bar chart with rate and count in hover."""

    if horizontal:
        figure = px.bar(
            summary,
            x="churn_rate",
            y=dimension,
            orientation="h",
            text="churn_rate_label",
            custom_data=[dimension, "customers", "churned"],
        )
        figure.update_traces(marker_color=color, textposition="outside")
        figure.update_yaxes(title=None)
    else:
        figure = px.bar(
            summary,
            x=dimension,
            y="churn_rate",
            text="churn_rate_label",
            custom_data=[dimension, "customers", "churned"],
        )
        figure.update_traces(marker_color=color, textposition="outside")
        figure.update_xaxes(title=None)

    figure.update_traces(
        hovertemplate=(
            "Segment: %{customdata[0]}<br>Churn rate: %{value:.1%}"
            "<br>Customers: %{customdata[1]:,}<br>Churned: %{customdata[2]:,}<extra></extra>"
        )
    )
    figure.update_layout(
        title=title,
        template="plotly_white",
        height=430,
        margin=dict(l=30, r=30, t=75, b=40),
        showlegend=False,
        font=dict(family="Arial", color=NAVY),
    )
    figure.update_xaxes(tickformat=".0%" if horizontal else None)
    figure.update_yaxes(tickformat=".0%" if not horizontal else None, rangemode="tozero")
    return figure


def city_churn_summary(df: pd.DataFrame, minimum_customers: int = 50) -> pd.DataFrame:
    """Aggregate city-level churn with a sample-size threshold."""

    summary = (
        df.groupby("city", observed=True)
        .agg(
            customers=("customer_id", "size"),
            churned=("churn_flag", "sum"),
            latitude=("latitude", "mean"),
            longitude=("longitude", "mean"),
        )
        .reset_index()
    )
    summary["churn_rate"] = summary["churned"] / summary["customers"]
    return summary.loc[summary["customers"].ge(minimum_customers)].sort_values(
        ["churn_rate", "customers"], ascending=[False, False]
    )


def churn_reason_summary(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return category and reason summaries for churned customers only."""

    churned = df.loc[df["churn_flag"].eq(1)].copy()
    category = (
        churned.groupby("churn_category", observed=True)
        .agg(customers=("customer_id", "size"))
        .reset_index()
        .sort_values("customers", ascending=False)
    )
    category["share_of_churn"] = category["customers"] / len(churned)

    reason = (
        churned.groupby("churn_reason", observed=True)
        .agg(customers=("customer_id", "size"))
        .reset_index()
        .sort_values("customers", ascending=False)
    )
    reason["share_of_churn"] = reason["customers"] / len(churned)
    return category, reason
