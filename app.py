"""Streamlit dashboard: live market KPIs + interactive salary predictor.

Run:  streamlit run app.py
"""
from __future__ import annotations

import logging
from typing import Any

import pandas as pd
import plotly.express as px
import streamlit as st

from config import DB_PATH, MODEL_PATH, SKILL_PATTERNS, setup_logging, skill_column
from database import fetch_jobs_df, fetch_scrape_runs_df
from model import (add_skill_flags, load_model, predict_salary, predict_with_interval,
                   skill_uplift, train_and_save)
from scraper import run_scrape

setup_logging()
logger = logging.getLogger(__name__)

st.set_page_config(page_title="Tech Job Market Intelligence", page_icon="\U0001F4CA", layout="wide")

REFRESH_SECONDS = 30
ACCENT = "#4F6BED"


# --------------------------------------------------------------------------- #
# Cached data access
# --------------------------------------------------------------------------- #
@st.cache_data(ttl=REFRESH_SECONDS, show_spinner=False)
def load_jobs() -> pd.DataFrame:
    """All jobs plus skill flags (cached briefly so the live view stays cheap)."""
    df = fetch_jobs_df(DB_PATH)
    return add_skill_flags(df) if not df.empty else df


@st.cache_data(ttl=REFRESH_SECONDS, show_spinner=False)
def load_runs() -> pd.DataFrame:
    return fetch_scrape_runs_df(DB_PATH)


@st.cache_resource(show_spinner=False)
def _load_model_cached(path: str, mtime: float) -> dict[str, Any]:
    """Keyed on file mtime so retraining invalidates the cached model automatically."""
    return load_model(MODEL_PATH)


def get_model() -> dict[str, Any] | None:
    if not MODEL_PATH.exists():
        return None
    return _load_model_cached(str(MODEL_PATH), MODEL_PATH.stat().st_mtime)


@st.cache_resource(show_spinner="First start: scraping a snapshot and training the model...")
def ensure_ready() -> bool:
    """Bootstrap data + model once per process (makes one-click cloud deploys work)."""
    from database import count_jobs
    if count_jobs(DB_PATH) == 0:
        run_scrape(pages=30, source="simulated", seed=42)
    if not MODEL_PATH.exists():
        train_and_save()
    return True


SOURCE_LABELS = {"All": None, "Simulated": "simulated", "Remotive (real)": "remotive"}


def selected_source_df(df: pd.DataFrame) -> pd.DataFrame:
    """Apply the sidebar data-source filter (read from session state so fragments see it)."""
    wanted = SOURCE_LABELS.get(st.session_state.get("view_source", "All"))
    return df if wanted is None else df[df["source"] == wanted]


def money(value: float) -> str:
    return f"${value:,.0f}"


def style_fig(fig, height: int = 320):
    fig.update_layout(height=height, margin=dict(l=10, r=10, t=40, b=10), showlegend=False,
                      plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)")
    return fig


# --------------------------------------------------------------------------- #
# Sidebar: pipeline controls
# --------------------------------------------------------------------------- #
def render_sidebar() -> None:
    st.sidebar.header("Data pipeline")
    st.sidebar.caption("Simulated job board \u2192 SQLite upsert \u2192 XGBoost")
    pages = st.sidebar.slider("Pages to scrape (25 listings each)", 5, 60, 20, step=5)

    if st.sidebar.button("Run scraper now", width="stretch"):
        with st.spinner("Scraping..."):
            summary = run_scrape(pages=pages, source="simulated")
        st.cache_data.clear()
        st.sidebar.success(
            f"{summary.inserted} new, {summary.updated} updated "
            f"({summary.pages_failed} pages failed) in {summary.duration_s:.1f}s"
        )

    if st.sidebar.button("Retrain model", width="stretch"):
        try:
            with st.spinner("Training..."):
                metrics = train_and_save()
            st.sidebar.success(f"MAE ${metrics['mae']:,.0f} | R\u00b2 {metrics['r2']:.3f}")
        except ValueError as exc:
            st.sidebar.error(str(exc))
        st.cache_data.clear()

    st.sidebar.divider()
    st.sidebar.radio("Market view", list(SOURCE_LABELS), key="view_source",
                     help="Real data comes from Remotive's public API.")
    if st.sidebar.button("Fetch real data (Remotive)", width="stretch",
                         help="Max ~4 fetches/day per Remotive's terms; responses are cached for 6 h."):
        with st.spinner("Contacting Remotive..."):
            summary = run_scrape(source="remotive")
        st.cache_data.clear()
        if summary.pages_failed:
            st.sidebar.error("Remotive fetch failed (see logs). Cached data is unaffected.")
        else:
            st.sidebar.success(f"{summary.inserted} new, {summary.updated} updated real listings")
    st.sidebar.caption("Real listings courtesy of [Remotive](https://remotive.com).")
    st.sidebar.divider()
    st.sidebar.caption(f"Live view refreshes every {REFRESH_SECONDS}s.")


# --------------------------------------------------------------------------- #
# Tab 1: market overview (auto-refreshing fragment)
# --------------------------------------------------------------------------- #
@st.fragment(run_every=f"{REFRESH_SECONDS}s")
def render_market_overview() -> None:
    df, runs = selected_source_df(load_jobs()), load_runs()
    if df.empty:
        st.info("No data for this view yet. Use the sidebar to run the scraper or fetch real data.")
        return

    salaried = df.dropna(subset=["salary_mid"])
    if salaried.empty:
        st.info("No listings with disclosed salary in this view yet; skill demand is still shown below.")
    skill_share = (
        pd.Series({s: df[skill_column(s)].mean() for s in SKILL_PATTERNS}).sort_values(ascending=False)
    )

    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Total jobs scraped", f"{len(df):,}")
    k2.metric("Hiring companies", f"{df['company'].str.lower().nunique():,}")
    k3.metric("Average salary", money(salaried["salary_mid"].mean()) if len(salaried) else "n/a")
    k4.metric("Salary disclosed", f"{len(salaried) / len(df):.0%}")
    if not runs.empty:
        st.caption(f"Last scrape: {runs.iloc[0]['finished_at']} UTC \u00b7 {len(runs)} runs recorded")

    left, right = st.columns(2)
    with left:
        top5 = skill_share.head(5).reset_index()
        top5.columns = ["Skill", "Share of listings"]
        fig = px.bar(top5.iloc[::-1], x="Share of listings", y="Skill", orientation="h",
                     title="Top 5 most in-demand skills", color_discrete_sequence=[ACCENT])
        fig.update_xaxes(tickformat=".0%")
        st.plotly_chart(style_fig(fig), width="stretch")
    if salaried.empty:
        return
    with right:
        by_role = (salaried.groupby("role_category")["salary_mid"].agg(["mean", "count"])
                   .reset_index().sort_values("mean"))
        fig = px.bar(by_role, x="mean", y="role_category", orientation="h", text="mean",
                     title="Average salary by role", color_discrete_sequence=[ACCENT],
                     labels={"mean": "Mean annual salary (USD)", "role_category": ""})
        fig.update_traces(texttemplate="$%{x:,.0f}")
        st.plotly_chart(style_fig(fig), width="stretch")

    left, right = st.columns(2)
    with left:
        by_loc = salaried.groupby("location")["salary_mid"].mean().reset_index().sort_values("salary_mid")
        fig = px.bar(by_loc, x="salary_mid", y="location", orientation="h",
                     title="Average salary by location", color_discrete_sequence=[ACCENT],
                     labels={"salary_mid": "Mean annual salary (USD)", "location": ""})
        st.plotly_chart(style_fig(fig, 340), width="stretch")
    with right:
        order = ["Junior", "Mid", "Senior", "Lead"]
        fig = px.box(salaried, x="experience_level", y="salary_mid", category_orders={"experience_level": order},
                     title="Salary distribution by experience", color_discrete_sequence=[ACCENT],
                     labels={"salary_mid": "Annual salary (USD)", "experience_level": ""})
        st.plotly_chart(style_fig(fig, 340), width="stretch")


# --------------------------------------------------------------------------- #
# Tab 2: salary predictor
# --------------------------------------------------------------------------- #
def render_predictor() -> None:
    bundle = get_model()
    if bundle is None:
        st.warning("No trained model found. Scrape some data, then click **Retrain model** in the sidebar.")
        return

    cats = bundle["categories"]
    with st.form("predictor"):
        c1, c2, c3 = st.columns(3)
        role = c1.selectbox("Role", cats["role_category"])
        location = c2.selectbox("Location", cats["location"], index=cats["location"].index("Remote")
                                if "Remote" in cats["location"] else 0)
        experience = c3.selectbox("Experience level", cats["experience_level"],
                                  index=min(1, len(cats["experience_level"]) - 1))
        skills = st.multiselect("Skills you have", bundle["skills"], default=["Python", "SQL"])
        submitted = st.form_submit_button("Estimate market rate", type="primary")

    if not submitted:
        st.caption("Pick a profile and press the button.")
        return

    estimate, low, high = predict_with_interval(bundle, role, location, experience, skills)
    metrics = bundle["metrics"]
    role_avg = bundle["role_mean_salary"].get(role, estimate)
    target = metrics.get("interval_target_coverage", 0.8)

    m1, m2, m3 = st.columns(3)
    m1.metric("Estimated market rate", money(estimate))
    m2.metric(f"{target:.0%} prediction interval", f"{money(low)} \u2013 {money(high)}")
    m3.metric(f"vs. average {role}", money(role_avg), delta=f"{estimate - role_avg:+,.0f}")
    if "interval_coverage" in metrics:
        st.caption(f"Conformal interval: measured hold-out coverage {metrics['interval_coverage']:.0%} "
                   f"(target {target:.0%}). Trained on {metrics.get('trained_on', 'n/a')} data.")

    uplift = skill_uplift(bundle, role, location, experience, skills)
    if not uplift.empty:
        fig = px.bar(uplift.iloc[::-1], x="uplift", y="skill", orientation="h",
                     title="Highest-value skills to add next (estimated salary change)",
                     color_discrete_sequence=[ACCENT], labels={"uplift": "Change in annual salary (USD)", "skill": ""})
        st.plotly_chart(style_fig(fig, 300), width="stretch")
    st.caption("Estimates are model outputs trained on the scraped dataset, not financial advice.")


# --------------------------------------------------------------------------- #
# Tab 3: model & data
# --------------------------------------------------------------------------- #
def render_model_and_data() -> None:
    bundle = get_model()
    if bundle is not None:
        m = bundle["metrics"]
        st.subheader("Model performance (hold-out set)")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("RMSE", money(m["rmse"]))
        c2.metric("MAE", money(m["mae"]))
        c3.metric("R\u00b2", f"{m['r2']:.3f}")
        c4.metric("5-fold CV RMSE", money(m["cv_rmse_mean"]))
        if "interval_coverage" in m:
            i1, i2, i3 = st.columns(3)
            i1.metric("Interval coverage (CQR)", f"{m['interval_coverage']:.1%}",
                      help=f"Target {m['interval_target_coverage']:.0%}. Share of hold-out salaries inside the interval.")
            i2.metric("Naive \u00b1MAE band coverage", f"{m['naive_mae_band_coverage']:.1%}",
                      help="What a simple point-estimate \u00b1 error band would have achieved.")
            i3.metric("Mean interval width", money(m["interval_mean_width"]))
        st.caption(f"{m['model_type']} \u00b7 trained {m['trained_at']} UTC \u00b7 "
                   f"{m['n_train']} train / {m['n_test']} test rows \u00b7 mean-baseline RMSE {money(m['baseline_rmse'])}")
        imp = pd.DataFrame(bundle["feature_importance"]).head(12).iloc[::-1]
        fig = px.bar(imp, x="importance", y="feature", orientation="h", title="Top feature importances",
                     color_discrete_sequence=[ACCENT])
        st.plotly_chart(style_fig(fig, 380), width="stretch")
    else:
        st.info("Model not trained yet.")

    st.subheader("Pipeline history")
    runs = load_runs()
    if runs.empty:
        st.caption("No scrape runs recorded.")
    else:
        st.dataframe(runs.drop(columns=["id"]).head(10), width="stretch", hide_index=True)

    st.subheader("Latest listings")
    df = selected_source_df(load_jobs())
    if not df.empty:
        view = df[["title", "company", "location", "experience_level", "salary_raw", "salary_mid",
                   "source", "url"]].head(200)
        st.dataframe(
            view, width="stretch", hide_index=True,
            column_config={"url": st.column_config.LinkColumn("Original posting", display_text="open"),
                           "salary_mid": st.column_config.NumberColumn("salary_mid", format="$%d")},
        )
        if (df["source"] == "remotive").any():
            st.caption("Real listings sourced from [Remotive](https://remotive.com) via its public API; "
                       "follow the links to apply on the original posting.")


# --------------------------------------------------------------------------- #
# Page
# --------------------------------------------------------------------------- #
def main() -> None:
    st.title("Tech Job Market Intelligence Engine")
    st.caption("Real-time hiring trends and salary prediction for Data Science, Data Engineering, "
               "ML and Software roles. Data source: built-in job-board simulator.")
    ensure_ready()
    render_sidebar()
    tab_market, tab_predict, tab_model = st.tabs(["Market overview", "Salary predictor", "Model & data"])
    with tab_market:
        render_market_overview()
    with tab_predict:
        render_predictor()
    with tab_model:
        render_model_and_data()


main()
