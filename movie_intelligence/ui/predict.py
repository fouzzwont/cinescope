"""Revenue prediction page."""

from __future__ import annotations

import altair as alt
import pandas as pd
import streamlit as st

from ..modelling import feature_row, predict_revenue
from . import charts
from .components import ACCENT, hero, money, section
from .state import get_model

GENRE_OPTIONS = [
    "Action",
    "Adventure",
    "Animation",
    "Comedy",
    "Crime",
    "Documentary",
    "Drama",
    "Family",
    "Fantasy",
    "History",
    "Horror",
    "Music",
    "Mystery",
    "Romance",
    "Science Fiction",
    "Thriller",
    "War",
    "Western",
]

LANGUAGES = {
    "English": "en",
    "French": "fr",
    "Spanish": "es",
    "German": "de",
    "Hindi": "hi",
    "Italian": "it",
    "Japanese": "ja",
    "Chinese": "zh",
    "Korean": "ko",
}


def _comparables(df: pd.DataFrame, budget: float, genre: str) -> pd.DataFrame:
    band = df[
        (df["budget"] >= budget * 0.6)
        & (df["budget"] <= budget * 1.6)
        & (df["revenue"] > 0)
    ]
    same_genre = band[band["genres"].map(lambda g: genre in g)]
    return same_genre if len(same_genre) >= 5 else band


def render(df: pd.DataFrame) -> None:
    hero(
        "Revenue model",
        "Estimate a film's box office",
        "A random-forest model trained on historical films with reported budgets "
        "and revenues. Outputs are a statistical range, not a studio forecast.",
    )

    artifact, just_trained = get_model()
    if just_trained:
        st.toast("Trained the revenue model on first run and cached it.", icon="🧠")

    with st.form("prediction_form"):
        st.markdown("##### Film profile")
        first, second, third = st.columns(3)
        budget_m = first.number_input(
            "Budget ($ millions)",
            min_value=0.1,
            max_value=600.0,
            value=50.0,
            step=5.0,
            key="predict_budget",
        )
        popularity = second.number_input(
            "TMDb popularity",
            min_value=0.0,
            max_value=900.0,
            value=25.0,
            step=1.0,
            help="TMDb's weekly popularity score. Most films sit between 5 and 60.",
            key="predict_popularity",
        )
        runtime = third.slider(
            "Runtime (minutes)", 60, 240, 110, key="predict_runtime"
        )

        first, second, third = st.columns(3)
        vote_average = first.slider(
            "Expected rating", 0.0, 10.0, 6.5, 0.1, key="predict_rating"
        )
        release_year = second.number_input(
            "Release year",
            min_value=1960,
            max_value=2035,
            value=2025,
            step=1,
            key="predict_year",
        )
        companies = third.slider(
            "Production companies",
            1,
            20,
            3,
            help="Larger co-productions tend to carry bigger budgets.",
            key="predict_companies",
        )

        first, second = st.columns([2, 1])
        genres = first.multiselect(
            "Genres",
            options=GENRE_OPTIONS,
            default=["Action", "Adventure"],
            max_selections=3,
            help="The first genre is used as the primary genre feature.",
            key="predict_genres",
        )
        language = second.selectbox(
            "Original language",
            options=list(LANGUAGES),
            index=0,
            key="predict_language",
        )

        submitted = st.form_submit_button("Predict revenue", type="primary", width="stretch")

    if submitted:
        features = feature_row(
            budget=budget_m * 1_000_000,
            popularity=popularity,
            runtime=runtime,
            vote_average=vote_average,
            release_year=int(release_year),
            genres=genres or ["Drama"],
            companies=int(companies),
            original_language=LANGUAGES[language],
        )
        result = predict_revenue(artifact, features)
        st.session_state["revenue_prediction"] = {
            **result,
            "budget": budget_m * 1_000_000,
            "genres": list(genres),
            "primary_genre": (genres or ["Drama"])[0],
            "year": int(release_year),
        }

    result = st.session_state.get("revenue_prediction")
    if not result:
        section("Model card", f"{artifact.model_name} · trained on {artifact.n_train:,} films")
        _model_card(artifact)
        return

    st.divider()
    section("Prediction", f"Based on a {money(result['budget'])} budget profile.")

    left, middle, right = st.columns([1.6, 1, 1])
    with left:
        st.markdown(
            f'<div class="cs-big-number">{money(result["revenue"], 2)}</div>'
            f'<div class="cs-subtle">Estimated worldwide revenue · 80% range '
            f'{money(result["low"])} – {money(result["high"])}</div>',
            unsafe_allow_html=True,
        )
    multiple = result["revenue"] / result["budget"] if result["budget"] else float("nan")
    middle.metric("Return", f"{multiple:.1f}×", help="Predicted revenue divided by budget")
    right.metric("Implied profit", money(result["revenue"] - result["budget"]))

    band = _comparables(df, result["budget"], result["primary_genre"])
    if not band.empty:
        median = band["revenue"].median()
        delta = (result["revenue"] - median) / median if median else 0
        st.caption(
            f"Benchmark: {len(band):,} films with a similar budget"
            + (f" and the {result['primary_genre']} genre" if result["primary_genre"] else "")
            + f" had a median revenue of {money(median)}. "
            f"This estimate is {delta:+.0%} versus that median."
        )

    st.divider()
    section("What drives the model", "Share of total feature importance.")
    _importance_chart(artifact.importances)
    _model_card(artifact)


def _importance_chart(importances: list[dict]) -> None:
    if not importances:
        return
    frame = pd.DataFrame(importances).sort_values("importance")
    frame["label"] = (
        frame["feature"]
        .str.replace("num__", "", regex=False)
        .str.replace("cat__", "", regex=False)
        .str.replace("_", " ", regex=False)
    )
    chart = (
        alt.Chart(frame)
        .mark_bar(cornerRadiusTopRight=4, cornerRadiusBottomRight=4, color=ACCENT)
        .encode(
            x=alt.X("importance:Q", title="Relative importance", axis=alt.Axis(format="%")),
            y=alt.Y("label:N", sort="x", title=None),
            tooltip=[
                alt.Tooltip("label:N", title="Feature"),
                alt.Tooltip("importance:Q", title="Importance", format=".1%"),
            ],
        )
    )
    st.altair_chart(charts.style(chart, 340), width="stretch")


def _model_card(artifact) -> None:
    metrics = artifact.metrics
    columns = st.columns(4)
    columns[0].metric("R²", f"{metrics.get('r2', 0):.2f}", help="Holdout R²: share of revenue variance explained")
    columns[1].metric("MAE", money(metrics.get("mae")), help="Mean absolute error on held-out films")
    columns[2].metric("Error", f"{metrics.get('median_ape', 0):.0f}%", help="Median absolute percentage error")
    columns[3].metric("±25%", f"{metrics.get('within_25_pct', 0):.0f}%", help="Share of films predicted within 25% of actual revenue")

    st.caption(
        f"{artifact.model_name} · {artifact.n_train:,} training films · {artifact.n_test:,} holdout films · "
        f"5-fold CV log-RMSE {metrics.get('cv_rmse_log', 0):.3f} · trained {artifact.trained_at[:10]}. "
        "Revenue depends on marketing, release timing, competition and word of mouth, none of "
        "which are in this dataset — use the range as a sanity check, not a decision."
    )

    with st.expander("How the model was built"):
        st.markdown(
            "- **Target:** `log1p(revenue)`, transformed back for reporting.\n"
            "- **Features:** log budget, log popularity, runtime, rating, release year, "
            "genre and language, counts of genres/companies, budget per minute.\n"
            "- **Excluded:** vote count and post-release signals, to limit leakage.\n"
            "- **Selection:** Ridge, Random Forest and Histogram Gradient Boosting compared "
            "with 5-fold cross-validation; the best log-RMSE model is refit on the training split.\n"
            "- **Range:** ±1.28 residual standard deviations in log space (80% interval)."
        )
