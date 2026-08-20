"""Executive customer-churn intelligence dashboard.

Developed and Designed by Emmanuel Onuoha.
"""

from __future__ import annotations

import html
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from src.decision_engine import (
    OBJECTIVES,
    build_decision_sensitivity,
    score_retention_decisions,
    select_decision_portfolio,
    summarize_decision_portfolio,
)


ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "data" / "processed" / "telco_customer_churn_features.csv"
MODEL_PATH = ROOT / "artifacts" / "evaluation" / "production_churn_model.joblib"
SCORE_PATH = ROOT / "artifacts" / "evaluation" / "portfolio_risk_scores.csv"
IMPORTANCE_PATH = ROOT / "artifacts" / "explanations" / "global_permutation_importance.csv"
BACKGROUND_PATH = ROOT / "artifacts" / "explanations" / "explanation_background.csv"
RETENTION_PATH = ROOT / "artifacts" / "retention" / "retention_campaign_population.csv"
SEGMENT_PATH = ROOT / "data" / "processed" / "prioritized_risk_segments.csv"
PHOTO_PATH = ROOT / "assets" / "emmanuel-onuoha.jpg"
THRESHOLD = 0.32
NAVIGATION = [
    "Executive Overview",
    "Customer Analysis",
    "Churn Drivers",
    "Geographic Analysis",
    "Customer Risk Predictor",
    "Decision Centre",
    "Retention Simulator",
    "Methodology",
]
BLUE, NAVY, ORANGE, TEAL, RED, MUTED = "#377BB5", "#17345B", "#F28E2B", "#2A9D8F", "#D9534F", "#64748B"


st.set_page_config(
    page_title="Customer Churn Intelligence | Emmanuel Onuoha",
    page_icon="📡",
    layout="wide",
    initial_sidebar_state="auto",
)


def inject_css() -> None:
    st.markdown(
        """
        <style>
        :root { --navy:#17345B; --orange:#F28E2B; --muted:#64748B; }
        html, body, [class*="css"] { font-family:'Segoe UI',Arial,sans-serif; }
        h1, h2, h3 { font-family:'Segoe UI',Arial,sans-serif !important; color:var(--navy); letter-spacing:-.02em; }
        [data-testid="stAppViewContainer"] { background:linear-gradient(180deg,#F7F9FC 0%,#FFFFFF 32%); }
        [data-testid="stSidebar"] { background:linear-gradient(180deg,#112C50 0%,#173E68 100%); }
        [data-testid="stSidebar"] * { color:#F8FAFC; }
        [data-testid="stSidebar"] .stRadio label { padding:.38rem .45rem; border-radius:8px; }
        [data-testid="stSidebar"] .stRadio label:hover { background:rgba(255,255,255,.08); }
        [data-testid="stSidebar"] [data-testid="stMetric"] { background:rgba(255,255,255,.10); border-color:rgba(255,255,255,.20); }
        [data-testid="stSidebar"] [data-testid="stMetricLabel"],
        [data-testid="stSidebar"] [data-testid="stMetricValue"] { color:#FFFFFF; }
        .block-container { padding-top:1.4rem; padding-bottom:4.5rem; max-width:1500px; }
        .brand-strip { border-left:5px solid var(--orange); padding:.2rem 0 .2rem 1rem; margin-bottom:1.2rem; }
        .brand-kicker { color:var(--orange); font-size:.76rem; font-weight:700; letter-spacing:.12em; text-transform:uppercase; }
        .brand-title { color:var(--navy); font:800 2.25rem/1.08 'Segoe UI',Arial,sans-serif; margin:.2rem 0; }
        .brand-subtitle { color:var(--muted); font-size:.98rem; }
        .insight { background:#FFF7ED; border:1px solid #FED7AA; border-left:4px solid var(--orange); border-radius:10px; padding:1rem 1.1rem; margin:.5rem 0 1rem; color:#4A3828; }
        .recommendation { background:#EFF6FF; border:1px solid #BFDBFE; border-radius:12px; padding:1rem; min-height:130px; }
        .recommendation strong { color:var(--navy); }
        .risk-high { color:#B42318; background:#FEE4E2; border-radius:999px; padding:.3rem .65rem; font-weight:700; }
        .risk-low { color:#067647; background:#D1FADF; border-radius:999px; padding:.3rem .65rem; font-weight:700; }
        [data-testid="stMetric"] { background:white; border:1px solid #E2E8F0; border-radius:14px; padding:1rem; box-shadow:0 3px 14px rgba(23,52,91,.06); }
        [data-testid="stMetricLabel"] { color:#64748B; }
        [data-testid="stMetricValue"] { color:#17345B; font-family:'Segoe UI',Arial,sans-serif; }
        .footer { margin-top:3rem; padding:1.2rem 0 .5rem; border-top:1px solid #DDE5EF; text-align:center; color:#64748B; font-size:.82rem; }
        .footer strong { color:#17345B; }
        @media (max-width: 768px) {
            .block-container { padding:1rem .8rem 4rem; }
            .brand-title { font-size:1.65rem; }
            .brand-subtitle { font-size:.9rem; }
            [data-testid="stMetric"] { padding:.75rem; }
            .recommendation { min-height:auto; margin-bottom:.5rem; }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )
    if st.session_state.get("dark_mode", False):
        st.markdown(
            """
            <style>
            :root { color-scheme:dark; --navy:#F1F5F9; --muted:#A8B3C7; }
            [data-testid="stAppViewContainer"], [data-testid="stHeader"] {
                background:linear-gradient(180deg,#0B1220 0%,#0F172A 42%,#111827 100%);
            }
            [data-testid="stAppViewContainer"] p,
            [data-testid="stAppViewContainer"] label,
            [data-testid="stAppViewContainer"] .stCaption,
            [data-testid="stCaptionContainer"] { color:#CBD5E1 !important; }
            h1, h2, h3, .brand-title { color:#F8FAFC !important; }
            .brand-subtitle { color:#A8B3C7; }
            [data-testid="stMetric"] {
                background:#111C2E; border-color:#2A3A55;
                box-shadow:0 5px 18px rgba(0,0,0,.22);
            }
            [data-testid="stMetricLabel"] { color:#A8B3C7; }
            [data-testid="stMetricValue"] { color:#F8FAFC; }
            .insight { background:#2A2118; border-color:#8A5A1F; color:#F6D7A7; }
            .recommendation { background:#111C2E; border-color:#2A3A55; color:#CBD5E1; }
            .recommendation strong { color:#F8FAFC; }
            .footer { border-color:#2A3A55; color:#94A3B8; }
            .footer strong { color:#F8FAFC; }
            [data-testid="stExpander"], [data-testid="stForm"] {
                background:#111C2E; border-color:#2A3A55; border-radius:12px;
            }
            [data-baseweb="select"] > div, [data-baseweb="input"] > div,
            [data-testid="stNumberInput"] input {
                background:#111C2E !important; color:#F8FAFC !important; border-color:#33445F !important;
            }
            [data-testid="stDataFrame"] { border:1px solid #2A3A55; border-radius:10px; }
            [data-testid="stAlert"] { background:#172237; color:#E2E8F0; }
            hr { border-color:#2A3A55 !important; }
            </style>
            """,
            unsafe_allow_html=True,
        )


@st.cache_data(show_spinner=False)
def load_data() -> dict[str, pd.DataFrame]:
    return {
        "customers": pd.read_csv(DATA_PATH, dtype={"customer_id": "string", "zip_code": "string"}),
        "scores": pd.read_csv(SCORE_PATH, dtype={"customer_id": "string"}),
        "importance": pd.read_csv(IMPORTANCE_PATH),
        "background": pd.read_csv(BACKGROUND_PATH),
        "retention": pd.read_csv(RETENTION_PATH, dtype={"customer_id": "string", "zip_code": "string"}),
        "segments": pd.read_csv(SEGMENT_PATH),
    }


@st.cache_resource(show_spinner=False)
def load_model():
    return joblib.load(MODEL_PATH)


def style_figure(fig: go.Figure, height: int = 390) -> go.Figure:
    dark_mode = st.session_state.get("dark_mode", False)
    text_color = "#DCE6F5" if dark_mode else NAVY
    grid_color = "#293750" if dark_mode else "#E8EDF4"
    surface = "#111827" if dark_mode else "#FFFFFF"
    fig.update_layout(
        template="plotly_dark" if dark_mode else "plotly_white",
        height=height, margin=dict(l=20, r=20, t=55, b=20),
        paper_bgcolor=surface, plot_bgcolor=surface,
        font=dict(family="Segoe UI, Arial", color=text_color),
        title_font=dict(family="Segoe UI, Arial", size=18, color=text_color),
        hoverlabel=dict(bgcolor="#172237" if dark_mode else "white", font_color=text_color),
        legend=dict(font=dict(color=text_color)), legend_title_text="",
    )
    fig.update_xaxes(showgrid=False, tickfont=dict(color=text_color), title_font=dict(color=text_color))
    fig.update_yaxes(gridcolor=grid_color, tickfont=dict(color=text_color), title_font=dict(color=text_color))
    fig.update_coloraxes(
        colorbar_tickfont=dict(color=text_color),
        colorbar_title_font=dict(color=text_color),
    )
    return fig


def page_header(title: str, subtitle: str) -> None:
    st.markdown(
        f'<div class="brand-strip"><div class="brand-kicker">Customer Churn Intelligence</div>'
        f'<div class="brand-title">{html.escape(title)}</div>'
        f'<div class="brand-subtitle">{html.escape(subtitle)}</div></div>',
        unsafe_allow_html=True,
    )


def footer() -> None:
    st.markdown(
        '<div class="footer"><strong>Developed and Designed by Emmanuel Onuoha</strong><br>'
        'Portfolio analytics project · Predictions support decisions and do not replace human judgment.</div>',
        unsafe_allow_html=True,
    )


def sidebar() -> str:
    with st.sidebar:
        if PHOTO_PATH.exists():
            st.image(str(PHOTO_PATH), width=112)
        st.markdown("### Emmanuel Onuoha")
        st.caption("Customer Churn Portfolio")
        st.toggle("Dark theme", key="dark_mode", help="Switch between light and dark presentation modes.")
        page = st.radio("Navigate", NAVIGATION, label_visibility="collapsed")
        st.divider()
        st.caption("MODEL OPERATING POINT")
        st.metric("High-risk threshold", f"{THRESHOLD:.0%}")
        st.caption("Selected for ≥80% recall using out-of-fold training predictions.")
    return page


def empty_state(message: str) -> None:
    st.info(f"No customers match the current filters. {message}")


def filter_customers(df: pd.DataFrame, selections: dict[str, list[object]]) -> pd.DataFrame:
    """Apply dashboard filters without mutating the supplied customer frame."""

    filtered = df.copy()
    for column, values in selections.items():
        if column not in filtered.columns:
            raise KeyError(f"Unknown dashboard filter: {column}")
        if values:
            filtered = filtered.loc[filtered[column].isin(values)]
    return filtered


def calculate_campaign_impact(
    campaign: pd.DataFrame,
    campaign_size: int,
    reach_rate: float,
    acceptance_rate: float,
    incremental_save_rate: float,
    horizon_months: int,
    gross_margin_rate: float,
    contact_cost: float,
    offer_cost_multiplier: float,
) -> dict[str, float]:
    """Calculate transparent retention economics for the highest-risk customers."""

    if campaign_size < 1 or campaign_size > len(campaign):
        raise ValueError("Campaign size must be between 1 and the eligible population.")
    rates = [reach_rate, acceptance_rate, incremental_save_rate, gross_margin_rate]
    if any(rate < 0 or rate > 1 for rate in rates):
        raise ValueError("Reach, acceptance, save, and margin rates must be between 0 and 1.")
    if horizon_months < 1 or contact_cost < 0 or offer_cost_multiplier < 0:
        raise ValueError("Horizon and campaign costs must be non-negative, with at least one month.")
    selected = campaign.sort_values("predicted_churn_probability", ascending=False).head(campaign_size)
    expected_churners = float(selected.predicted_churn_probability.sum())
    saved = expected_churners * reach_rate * acceptance_rate * incremental_save_rate
    benefit = (
        float((selected.predicted_churn_probability * selected.monthly_charge).sum())
        * reach_rate * acceptance_rate * incremental_save_rate * horizon_months * gross_margin_rate
    )
    cost = (
        campaign_size * contact_cost
        + float(selected.base_offer_cost.sum()) * reach_rate * acceptance_rate * offer_cost_multiplier
    )
    net = benefit - cost
    return {
        "expected_churners": expected_churners,
        "expected_customers_saved": saved,
        "retained_gross_margin": benefit,
        "campaign_cost": cost,
        "net_benefit": net,
        "roi": net / cost if cost else np.nan,
    }


def executive_page(data: dict[str, pd.DataFrame]) -> None:
    df, scores, retention = data["customers"], data["scores"], data["retention"]
    page_header("Executive Overview", "Portfolio health, risk exposure, and the actions that matter most")
    total, churned = len(df), int(df["churn_flag"].sum())
    c1, c2, c3 = st.columns(3)
    c1.metric("Total customers", f"{total:,}")
    c2.metric("Churn rate", f"{churned/total:.1%}", f"{churned:,} churned", delta_color="inverse")
    c3.metric("Active customers", f"{int(df['active_flag'].sum()):,}")
    c4, c5 = st.columns(2)
    c4.metric("High-risk active", f"{len(retention):,}", f"{len(retention)/df['active_flag'].sum():.1%} of active", delta_color="inverse")
    c5.metric("Risk-weighted MRR", f"${(retention.predicted_churn_probability*retention.monthly_charge).sum():,.0f}")

    left, right = st.columns([1.05, 1], gap="large")
    with left:
        status = df.groupby("customer_status", as_index=False).agg(customers=("customer_id", "size"))
        fig = px.bar(status, x="customer_status", y="customers", color="customer_status",
                     color_discrete_map={"Stayed": BLUE, "Churned": RED, "Joined": TEAL},
                     title="Customer portfolio status", text_auto=",.0f")
        st.plotly_chart(style_figure(fig), width="stretch")
    with right:
        cohort = df.groupby("tenure_group", observed=True, as_index=False).agg(
            customers=("customer_id", "size"), churn_rate=("churn_flag", "mean"))
        fig = px.line(cohort, x="tenure_group", y="churn_rate", markers=True,
                      title="Churn by tenure cohort", color_discrete_sequence=[ORANGE])
        fig.update_yaxes(tickformat=".0%")
        st.plotly_chart(style_figure(fig), width="stretch")
        st.caption("A true time trend is unavailable because the source is a single-quarter snapshot.")

    st.markdown('<div class="insight"><strong>Executive signal:</strong> Churn is concentrated among early-tenure, month-to-month, fiber customers—especially those without premium support. The validated model flags 947 active customers for governed outreach.</div>', unsafe_allow_html=True)
    st.subheader("Priority recommendations")
    cols = st.columns(4)
    recommendations = [
        ("Stabilize early tenure", "Introduce 30/60/90-day onboarding and proactive fiber health checks."),
        ("Fix before discounting", "Resolve fiber and support issues before offering longer contracts."),
        ("Protect customer value", "Route high-value/high-risk customers to specialist outreach."),
        ("Prove incremental impact", "Pilot interventions with randomized holdouts and track net benefit."),
    ]
    for col, (title, body) in zip(cols, recommendations):
        col.markdown(f'<div class="recommendation"><strong>{title}</strong><br><br>{body}</div>', unsafe_allow_html=True)


def customer_analysis_page(data: dict[str, pd.DataFrame]) -> None:
    df = data["customers"]
    page_header("Customer Analysis", "Explore churn patterns across customer, contract, service, and tenure dimensions")
    with st.expander("Filter the customer population", expanded=True):
        c1, c2, c3, c4 = st.columns(4)
        genders = c1.multiselect("Gender", sorted(df.gender.unique()), placeholder="All genders")
        contracts = c2.multiselect("Contract", sorted(df.contract.unique()), placeholder="All contracts")
        internet = c3.multiselect("Internet type", sorted(df.internet_type.unique()), placeholder="All internet types")
        tenure = c4.multiselect("Tenure group", list(df.tenure_group.drop_duplicates()), placeholder="All cohorts")
        c5, c6, c7 = st.columns(3)
        ages = c5.multiselect("Age group", list(df.age_group.drop_duplicates()), placeholder="All age groups")
        support = c6.multiselect("Premium tech support", sorted(df.premium_tech_support.unique()), placeholder="All")
        status = c7.multiselect("Customer status", sorted(df.customer_status.unique()), placeholder="All statuses")
    filtered = filter_customers(df, dict(
        gender=genders, contract=contracts, internet_type=internet,
        tenure_group=tenure, age_group=ages,
        premium_tech_support=support, customer_status=status,
    ))
    if filtered.empty:
        empty_state("Remove one or more filters to continue.")
        return
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Customers", f"{len(filtered):,}")
    k2.metric("Churn rate", f"{filtered.churn_flag.mean():.1%}")
    k3.metric("Average tenure", f"{filtered.tenure_in_months.mean():.1f} months")
    k4.metric("Average monthly charge", f"${filtered.monthly_charge.mean():,.2f}")
    dimension = st.selectbox("Compare churn by", ["contract", "tenure_group", "internet_type", "age_group", "payment_method", "customer_engagement_profile"], format_func=lambda x: x.replace("_", " ").title())
    summary = filtered.groupby(dimension, observed=True, as_index=False).agg(customers=("customer_id", "size"), churn_rate=("churn_flag", "mean"))
    summary = summary.sort_values("churn_rate", ascending=False)
    left, right = st.columns([1.15, .85], gap="large")
    with left:
        fig = px.bar(summary, x=dimension, y="churn_rate", color="churn_rate", text_auto=".1%",
                     color_continuous_scale=["#CFE7F5", ORANGE, RED], title=f"Churn rate by {dimension.replace('_',' ')}")
        fig.update_yaxes(tickformat=".0%")
        st.plotly_chart(style_figure(fig, 430), width="stretch")
    with right:
        display = summary.rename(columns={dimension: "Segment", "customers": "Customers", "churn_rate": "Churn rate"})
        st.dataframe(display, hide_index=True, width="stretch",
                     column_config={"Churn rate": st.column_config.ProgressColumn(format="percent", min_value=0, max_value=1)})


def drivers_page(data: dict[str, pd.DataFrame]) -> None:
    df, importance = data["customers"], data["importance"]
    page_header("Churn Drivers", "Recorded departure reasons, pre-churn risk factors, and model behavior")
    churned = df[df.churn_flag.eq(1)]
    tab1, tab2, tab3 = st.tabs(["Recorded reasons", "Model risk factors", "Interpretation guide"])
    with tab1:
        categories = churned.groupby("churn_category", as_index=False).size().rename(columns={"size": "customers"})
        categories["share"] = categories.customers / categories.customers.sum()
        reasons = churned.groupby("churn_reason", as_index=False).size().nlargest(10, "size").sort_values("size").rename(columns={"size": "customers"})
        fig1 = px.bar(categories.sort_values("share", ascending=False), x="churn_category", y="share", text_auto=".1%", title="Share of churn by category", color_discrete_sequence=[BLUE])
        fig1.update_yaxes(tickformat=".0%")
        fig2 = px.bar(reasons, x="customers", y="churn_reason", orientation="h", title="Top 10 recorded reasons", color_discrete_sequence=[ORANGE])
        st.plotly_chart(style_figure(fig1, 350), width="stretch")
        st.plotly_chart(style_figure(fig2, 480).update_layout(margin=dict(l=220, r=30, t=55, b=35)), width="stretch")
    with tab2:
        imp = importance.head(12).sort_values("pr_auc_importance_mean")
        imp["label"] = imp.feature.str.replace("_", " ").str.title()
        fig = px.bar(imp, x="pr_auc_importance_mean", y="label", orientation="h", error_x="pr_auc_importance_std",
                     title="Holdout permutation importance (PR-AUC loss)", color_discrete_sequence=[TEAL])
        st.plotly_chart(style_figure(fig, 500), width="stretch")
        st.caption("A larger decrease in PR-AUC after shuffling indicates greater predictive reliance; it does not establish causality.")
    with tab3:
        st.info("Recorded churn reasons are post-outcome descriptions and are excluded from prediction. The production model uses only governed pre-churn fields.")
        st.markdown("""
        - **Association is not causation:** model drivers identify predictive patterns, not guaranteed causes.
        - **No outcome leakage:** Customer Status, Churn Reason, Churn Category, Churn Score, and Satisfaction Score are excluded.
        - **Actionable use:** combine model signals with service context and human review before contacting a customer.
        """)


def geography_page(data: dict[str, pd.DataFrame]) -> None:
    df = data["customers"]
    page_header("Geographic Analysis", "Locate churn concentration while separating rate from customer volume")
    minimum = st.slider("Minimum customers per location", 5, 100, 25, 5, help="Avoid overinterpreting unstable rates from very small locations.")
    level = st.radio("Geographic level", ["City", "ZIP code"], horizontal=True)
    group = "city" if level == "City" else "zip_code"
    geo = df.groupby(group, as_index=False).agg(customers=("customer_id", "size"), churned=("churn_flag", "sum"), churn_rate=("churn_flag", "mean"), latitude=("latitude", "mean"), longitude=("longitude", "mean"))
    geo = geo[geo.customers.ge(minimum)]
    if geo.empty:
        empty_state("Lower the minimum-customer requirement.")
        return
    left, right = st.columns([1.3, .7], gap="large")
    with left:
        fig = px.scatter_map(geo, lat="latitude", lon="longitude", size="customers", color="churn_rate",
            hover_name=group, hover_data={"customers": ":,", "churned": ":,", "churn_rate": ":.1%", "latitude": False, "longitude": False},
            color_continuous_scale=["#CFE7F5", ORANGE, RED], zoom=4.4, center={"lat": 36.7, "lon": -119.5},
            map_style="carto-darkmatter" if st.session_state.get("dark_mode", False) else "carto-positron",
            title=f"{level} churn distribution")
        st.plotly_chart(style_figure(fig, 590), width="stretch")
    with right:
        hotspots = geo.sort_values(["churn_rate", "customers"], ascending=False).head(12)
        st.subheader("Highest observed rates")
        st.dataframe(hotspots[[group, "customers", "churned", "churn_rate"]], hide_index=True, width="stretch",
            column_config={"churn_rate": st.column_config.ProgressColumn("Churn rate", format="percent", min_value=0, max_value=1)})
        st.caption("Use volume and rate together. Geographic patterns warrant investigation; they do not prove location caused churn.")


def predictor_defaults(df: pd.DataFrame) -> dict[str, object]:
    features = list(load_model().feature_names_in_)
    defaults = {}
    for feature in features:
        series = df[feature]
        defaults[feature] = float(series.median()) if pd.api.types.is_numeric_dtype(series) else str(series.mode().iloc[0])
    return defaults


def local_explanation(row: pd.DataFrame, model, background: pd.DataFrame) -> pd.DataFrame:
    transformed = np.asarray(model.named_steps["preprocess"].transform(row), dtype=float)[0]
    terms = list(model.named_steps["preprocess"].get_feature_names_out())
    coefficients = model.named_steps["model"].coef_[0]
    bg = background.set_index("term")
    grouped: dict[str, float] = {}
    for term, value, coef in zip(terms, transformed, coefficients):
        clean = term.replace("numeric__", "").replace("categorical__", "")
        original = bg.loc[clean, "original_feature"]
        contribution = (value - float(bg.loc[clean, "background_transformed_mean"])) * coef
        grouped[original] = grouped.get(original, 0.0) + contribution
    result = pd.DataFrame({"feature": grouped.keys(), "contribution": grouped.values()})
    result["direction"] = np.where(result.contribution.gt(0), "Raises risk", "Lowers risk")
    return result.reindex(result.contribution.abs().sort_values(ascending=False).index).head(8)


def predictor_page(data: dict[str, pd.DataFrame]) -> None:
    df, background = data["customers"], data["background"]
    model, defaults = load_model(), predictor_defaults(df)
    page_header("Customer Risk Predictor", "Score a customer using governed pre-churn information and explain the result")
    with st.form("predictor_form"):
        st.subheader("Customer characteristics")
        a, b, c = st.columns(3)
        values = defaults.copy()
        values["number_of_dependents"] = a.number_input("Number of dependents", 0, 10, int(defaults["number_of_dependents"]))
        values["number_of_referrals"] = a.number_input("Number of referrals", 0, 20, int(defaults["number_of_referrals"]))
        values["tenure_in_months"] = a.slider("Tenure (months)", 1, 72, int(defaults["tenure_in_months"]))
        values["monthly_charge"] = a.number_input("Monthly charge ($)", 0.0, 200.0, float(defaults["monthly_charge"]), 1.0)
        values["avg_monthly_long_distance_charges"] = a.number_input("Avg. long-distance charge ($)", 0.0, 100.0, float(defaults["avg_monthly_long_distance_charges"]), 1.0)
        values["avg_monthly_gb_download"] = a.number_input("Avg. monthly download (GB)", 0.0, 100.0, float(defaults["avg_monthly_gb_download"]), 1.0)
        values["total_refunds"] = a.number_input("Total refunds ($)", 0.0, 200.0, float(defaults["total_refunds"]), 1.0)
        values["total_extra_data_charges"] = a.number_input("Extra data charges ($)", 0.0, 500.0, float(defaults["total_extra_data_charges"]), 5.0)
        categorical = [f for f in model.feature_names_in_ if not pd.api.types.is_numeric_dtype(df[f])]
        for index, feature in enumerate(categorical):
            col = [b, c][index % 2]
            options = sorted(df[feature].astype(str).unique())
            default_index = options.index(str(defaults[feature])) if str(defaults[feature]) in options else 0
            values[feature] = col.selectbox(feature.replace("_", " ").title(), options, index=default_index)
        submitted = st.form_submit_button("Calculate churn risk", type="primary", width="stretch")
    if submitted:
        row = pd.DataFrame([{feature: values[feature] for feature in model.feature_names_in_}])
        probability = float(model.predict_proba(row)[0, 1])
        high = probability >= THRESHOLD
        label = "High risk" if high else "Below operating threshold"
        st.subheader("Prediction result")
        m1, m2, m3 = st.columns(3)
        m1.metric("Predicted churn probability", f"{probability:.1%}")
        m2.metric("Operating threshold", f"{THRESHOLD:.0%}")
        m3.markdown(f'<br><span class="{"risk-high" if high else "risk-low"}">{label}</span>', unsafe_allow_html=True)
        explanation = local_explanation(row, model, background).sort_values("contribution")
        explanation["label"] = explanation.feature.str.replace("_", " ").str.title()
        fig = px.bar(explanation, x="contribution", y="label", orientation="h", color="direction",
                     color_discrete_map={"Raises risk": RED, "Lowers risk": TEAL}, title="Main factors relative to the portfolio background")
        st.plotly_chart(style_figure(fig, 430), width="stretch")
        st.caption("Contributions are exact additive logistic-model effects on the log-odds scale. They explain this prediction but do not prove causality.")


def decision_centre_page(data: dict[str, pd.DataFrame]) -> None:
    campaign = data["retention"].copy()
    page_header(
        "Decision Centre",
        "Allocate limited retention resources using transparent expected-value rules",
    )
    st.info(
        "This is a scenario-based decision policy, not a causal uplift model. "
        "Incremental save rates must be validated with a randomized campaign pilot."
    )

    with st.form("decision_policy_form"):
        st.subheader("Decision constraints and assumptions")
        constraint_left, constraint_right = st.columns(2, gap="large")
        with constraint_left:
            objective = st.selectbox("Decision objective", list(OBJECTIVES))
            capacity = st.slider(
                "Maximum customers to contact", 25, len(campaign), min(500, len(campaign)), 25
            )
            budget = st.slider("Expected campaign budget ($)", 1_000, 40_000, 15_000, 500)
            reach_rate = st.slider("Reach rate", 0, 100, 75, 5) / 100
            acceptance_rate = st.slider("Offer acceptance", 0, 100, 50, 5) / 100
        with constraint_right:
            incremental_save_rate = st.slider(
                "Incremental save rate", 5, 70, 35, 5,
                help="Assumed additional retention caused by the intervention among accepting customers.",
            ) / 100
            horizon = st.slider("Retention value horizon (months)", 1, 24, 12)
            margin_rate = st.slider("Gross margin rate", 10, 100, 70, 5) / 100
            contact_cost = st.number_input(
                "Contact cost per targeted customer ($)", 0.0, 100.0, 4.0, 1.0
            )
            offer_multiplier = st.slider("Offer cost multiplier", 0.25, 2.0, 1.0, 0.05)
        st.form_submit_button("Apply decision policy", type="primary", width="stretch")

    assumptions = {
        "reach_rate": reach_rate,
        "acceptance_rate": acceptance_rate,
        "incremental_save_rate": incremental_save_rate,
        "retention_horizon_months": horizon,
        "gross_margin_rate": margin_rate,
        "contact_cost_per_target": contact_cost,
        "offer_cost_multiplier": offer_multiplier,
    }
    scored = score_retention_decisions(campaign, **assumptions)
    selected, ranked = select_decision_portfolio(
        scored, budget=budget, capacity=capacity, objective=objective
    )
    summary = summarize_decision_portfolio(selected, ranked)

    st.subheader("Recommended campaign portfolio")
    a, b, c = st.columns(3)
    a.metric("Customers selected", f"{summary['selected_customers']:,}", f"of {len(campaign):,} eligible")
    b.metric("Expected campaign cost", f"${summary['expected_campaign_cost']:,.0f}", f"${budget - summary['expected_campaign_cost']:,.0f} budget remaining")
    c.metric("Expected customers saved", f"{summary['expected_customers_saved']:,.1f}")
    a, b = st.columns(2)
    a.metric("Expected net benefit", f"${summary['expected_net_benefit']:,.0f}")
    b.metric(
        "Expected ROI",
        "N/A" if np.isnan(summary["expected_roi"]) else f"{summary['expected_roi']:.1%}",
    )

    if selected.empty:
        st.warning("No customer has positive expected economics within the current budget and capacity assumptions.")
        return

    action_summary = (
        selected.groupby("primary_intervention", as_index=False)
        .agg(
            customers=("customer_id", "size"),
            expected_cost=("expected_campaign_cost", "sum"),
            expected_saves=("expected_incremental_saves", "sum"),
            expected_net_benefit=("expected_net_benefit", "sum"),
            recommended_action=("recommended_action", "first"),
        )
        .sort_values("expected_net_benefit", ascending=False)
    )
    chart = px.bar(
        action_summary.sort_values("expected_net_benefit"),
        x="expected_net_benefit",
        y="primary_intervention",
        orientation="h",
        color="expected_saves",
        color_continuous_scale=["#DCEAF6", BLUE],
        title="Expected net benefit by recommended intervention",
        labels={
            "expected_net_benefit": "Expected net benefit ($)",
            "primary_intervention": "Intervention",
            "expected_saves": "Expected saves",
        },
    )
    st.plotly_chart(style_figure(chart, 430), width="stretch")

    st.subheader("Budget allocation and next-best actions")
    st.dataframe(
        action_summary,
        hide_index=True,
        width="stretch",
        column_config={
            "primary_intervention": "Intervention",
            "customers": st.column_config.NumberColumn("Customers", format="%d"),
            "expected_cost": st.column_config.NumberColumn("Expected cost", format="$%.0f"),
            "expected_saves": st.column_config.NumberColumn("Expected saves", format="%.1f"),
            "expected_net_benefit": st.column_config.NumberColumn("Expected net benefit", format="$%.0f"),
            "recommended_action": "Next-best action",
        },
    )
    st.download_button(
        "Download aggregated decision plan",
        action_summary.to_csv(index=False).encode("utf-8"),
        file_name="aggregated_retention_decision_plan.csv",
        mime="text/csv",
        help="The public application exports an aggregated plan without customer identifiers.",
    )

    st.subheader("Highest-priority anonymized candidates")
    candidate_view = selected[
        [
            "decision_rank",
            "decision_category",
            "customer_value_segment",
            "descriptive_risk_segment",
            "primary_intervention",
            "predicted_churn_probability",
            "monthly_charge",
            "expected_campaign_cost",
            "expected_net_benefit",
        ]
    ].head(25)
    st.dataframe(
        candidate_view,
        hide_index=True,
        width="stretch",
        column_config={
            "decision_rank": st.column_config.NumberColumn("Rank", format="%d"),
            "decision_category": "Decision",
            "customer_value_segment": "Value segment",
            "descriptive_risk_segment": "Risk segment",
            "primary_intervention": "Next-best action group",
            "predicted_churn_probability": st.column_config.ProgressColumn(
                "Churn probability", format="percent", min_value=0, max_value=1
            ),
            "monthly_charge": st.column_config.NumberColumn("Monthly charge", format="$%.2f"),
            "expected_campaign_cost": st.column_config.NumberColumn("Expected cost", format="$%.2f"),
            "expected_net_benefit": st.column_config.NumberColumn("Expected net benefit", format="$%.2f"),
        },
    )

    with st.expander("Selection governance check"):
        governance_rows = []
        for field, label in [("gender", "Gender"), ("senior_citizen", "Senior citizen")]:
            for group, subset in ranked.groupby(field, observed=True):
                governance_rows.append({
                    "monitoring_dimension": label,
                    "group": str(group),
                    "eligible_customers": int(len(subset)),
                    "selected_customers": int(subset["selected_for_campaign"].sum()),
                    "selection_rate": float(subset["selected_for_campaign"].mean()),
                })
        governance = pd.DataFrame(governance_rows)
        st.dataframe(
            governance,
            hide_index=True,
            width="stretch",
            column_config={
                "monitoring_dimension": "Monitoring dimension",
                "group": "Group",
                "eligible_customers": st.column_config.NumberColumn("Eligible", format="%d"),
                "selected_customers": st.column_config.NumberColumn("Selected", format="%d"),
                "selection_rate": st.column_config.ProgressColumn(
                    "Selection rate", format="percent", min_value=0, max_value=1
                ),
            },
        )
        st.caption(
            "Gender and senior status are not ranking inputs. These descriptive rates support human review; "
            "they do not by themselves establish fairness or discrimination."
        )

    common_assumptions = assumptions.copy()
    common_assumptions.pop("incremental_save_rate")
    sensitivity_rates = sorted({
        max(0.05, incremental_save_rate - 0.15),
        incremental_save_rate,
        min(0.90, incremental_save_rate + 0.15),
    })
    sensitivity = build_decision_sensitivity(
        campaign,
        save_rates=sensitivity_rates,
        budget=budget,
        capacity=capacity,
        objective=objective,
        common_assumptions=common_assumptions,
    )
    sensitivity["scenario"] = [
        "Selected" if np.isclose(rate, incremental_save_rate)
        else "Lower" if rate < incremental_save_rate
        else "Upper"
        for rate in sensitivity["incremental_save_rate"]
    ]
    sensitivity_chart = px.bar(
        sensitivity,
        x="scenario",
        y="expected_net_benefit",
        color="incremental_save_rate",
        text="selected_customers",
        title="Decision sensitivity to incremental save rate",
        labels={
            "scenario": "Save-rate assumption",
            "expected_net_benefit": "Expected net benefit ($)",
            "incremental_save_rate": "Save rate",
            "selected_customers": "Selected",
        },
    )
    st.plotly_chart(style_figure(sensitivity_chart, 370), width="stretch")

    with st.expander("How the decision policy works"):
        st.markdown(
            """
            1. Estimate each customer's probability-weighted incremental saves using the selected reach, acceptance, and save-rate assumptions.
            2. Convert expected saves into retained gross margin using monthly charge, value horizon, and margin rate.
            3. Subtract expected contact and accepted-offer costs.
            4. Exclude customers with non-positive expected net benefit.
            5. Rank the remaining customers by the selected objective and admit them while budget and capacity remain.

            The policy is deterministic and auditable. It supports planning but does not prove that an intervention will cause retention. Use a holdout control group, monitor subgroup outcomes, and prohibit punitive decisions based on churn risk.
            """
        )


def simulator_page(data: dict[str, pd.DataFrame]) -> None:
    campaign = data["retention"].sort_values("predicted_churn_probability", ascending=False)
    page_header("Retention Simulator", "Test transparent campaign assumptions before committing retention budget")
    left, right = st.columns([.8, 1.2], gap="large")
    with left:
        st.subheader("Campaign assumptions")
        size = st.slider("Customers targeted", 50, len(campaign), min(500, len(campaign)), 25)
        reach = st.slider("Reach rate", 0, 100, 75, 5) / 100
        acceptance = st.slider("Offer acceptance", 0, 100, 50, 5) / 100
        save_rate = st.slider("Incremental save rate", 0, 100, 35, 5, help="Additional retention caused by the campaign among accepting customers.") / 100
        horizon = st.slider("Retention value horizon (months)", 1, 24, 12)
        margin = st.slider("Gross margin rate", 0, 100, 70, 5) / 100
        contact_cost = st.number_input("Contact cost per targeted customer ($)", 0.0, 100.0, 4.0, 1.0)
        offer_multiplier = st.slider("Offer cost multiplier", 0.25, 2.0, 1.0, .05)
    selected = campaign.head(size)
    impact = calculate_campaign_impact(
        campaign, size, reach, acceptance, save_rate, horizon, margin, contact_cost, offer_multiplier
    )
    expected_churners = impact["expected_churners"]
    saved = impact["expected_customers_saved"]
    benefit = impact["retained_gross_margin"]
    cost = impact["campaign_cost"]
    net = impact["net_benefit"]
    roi = impact["roi"]
    with right:
        st.subheader("Estimated campaign impact")
        a, b = st.columns(2)
        a.metric("Expected customers saved", f"{saved:,.1f}")
        b.metric("Model-weighted churners", f"{expected_churners:,.1f}")
        a.metric("Campaign cost", f"${cost:,.0f}")
        b.metric("Retained gross margin", f"${benefit:,.0f}")
        a.metric("Estimated net benefit", f"${net:,.0f}")
        b.metric("Estimated ROI", "N/A" if np.isnan(roi) else f"{roi:.1%}")
        waterfall = go.Figure(go.Waterfall(x=["Retained margin", "Campaign cost", "Net benefit"],
            y=[benefit, -cost, net], measure=["relative", "relative", "total"],
            increasing_marker_color=TEAL, decreasing_marker_color=RED, totals_marker_color=BLUE))
        waterfall.update_layout(title="Scenario economics", yaxis_tickprefix="$")
        st.plotly_chart(style_figure(waterfall, 360), width="stretch")
    by_action = selected.groupby("primary_intervention", as_index=False).agg(customers=("customer_id", "size"), monthly_revenue=("monthly_charge", "sum"), expected_churners=("predicted_churn_probability", "sum"))
    st.subheader("Selected campaign mix")
    st.dataframe(by_action.sort_values("expected_churners", ascending=False), hide_index=True, width="stretch",
        column_config={"monthly_revenue": st.column_config.NumberColumn("Monthly revenue", format="$%.2f"), "expected_churners": st.column_config.NumberColumn("Expected churners", format="%.1f")})
    st.warning("Scenario outputs are planning estimates. Validate incremental save rate and offer economics with a randomized control group before scaling.")


def methodology_page(data: dict[str, pd.DataFrame]) -> None:
    page_header("Methodology", "Definitions, analytical controls, validation evidence, and responsible use")
    tabs = st.tabs(["Data & definitions", "Preparation", "Model", "Decision policy", "Limitations & responsible use"])
    with tabs[0]:
        st.markdown("""
        **Source:** supplied `TelcoCustomerChurn.csv`, containing 7,043 California telecom customer records for Q3.

        **Churn:** `ChurnLabel = Yes`; **active:** customers whose status is Stayed or Joined. The dataset is a customer-level snapshot, not monthly event history.

        **Core metrics:** churn rate is churned customers divided by total customers. Risk-weighted exposure is predicted churn probability multiplied by monthly charge. Supplied CLTV is reported as a dataset field, not independently reconstructed.
        """)
    with tabs[1]:
        st.markdown("""
        - Standardized names and categorical values while preserving all 7,043 records.
        - Converted structural blanks to explicit business states such as No Offer, No Internet, or Not Applicable.
        - Reconciled service relationships, status, identifiers, and revenue components.
        - Added 12 business-readable features for segmentation and reporting.
        - Preserved the raw CSV; all application data comes from reproducible processed exports.
        """)
    with tabs[2]:
        st.markdown("""
        A regularized logistic regression was selected for interpretability and business usefulness. Threshold 0.32 was chosen from out-of-fold training predictions as the highest-precision point retaining at least 80% recall. On the untouched holdout set it achieved **61.58% precision, 83.16% recall, 70.76% F1, and 89.85% ROC-AUC**.

        Outcome-revealing or uncertain fields—including Customer Status, Churn Reason, Churn Category, Churn Score, Satisfaction Score, CLTV, Total Revenue, identifiers, and location coordinates—are excluded from prediction.
        """)
    with tabs[3]:
        st.markdown("""
        The Decision Centre applies a transparent constrained ranking after prediction. Customer-level expected value combines churn probability, editable campaign reach and acceptance, an assumed incremental save rate, retained gross margin, and expected intervention cost.

        Only positive expected-value candidates can be selected. The policy then ranks candidates by the chosen business objective and admits them while expected budget and contact capacity remain. This is a scenario-based prescriptive policy, not an uplift or causal-treatment model.
        """)
    with tabs[4]:
        st.markdown("""
        - Results describe one California snapshot and may not generalize across time or markets.
        - No campaign-history data exists; simulator outputs are assumptions, not realized causal impact.
        - Fairness checks are monitoring signals and do not guarantee equitable outcomes.
        - Do not deny service, change prices, or disadvantage customers solely because of a model score.
        - Require human review, record intervention outcomes, monitor drift and subgroup performance, and provide appropriate governance.
        """)
        st.info("This dashboard supports retention prioritization. It is not an automated decision system and should not be used for punitive customer treatment.")


def main() -> None:
    inject_css()
    missing = [path for path in [DATA_PATH, MODEL_PATH, SCORE_PATH, IMPORTANCE_PATH, BACKGROUND_PATH, RETENTION_PATH, PHOTO_PATH] if not path.exists()]
    if missing:
        st.error("The application cannot start because required project files are missing: " + ", ".join(str(p.relative_to(ROOT)) for p in missing))
        st.stop()
    with st.spinner("Loading validated customer intelligence…"):
        data = load_data()
    page = sidebar()
    pages = {
        "Executive Overview": executive_page,
        "Customer Analysis": customer_analysis_page,
        "Churn Drivers": drivers_page,
        "Geographic Analysis": geography_page,
        "Customer Risk Predictor": predictor_page,
        "Decision Centre": decision_centre_page,
        "Retention Simulator": simulator_page,
        "Methodology": methodology_page,
    }
    pages[page](data)
    footer()


if __name__ == "__main__":
    main()
