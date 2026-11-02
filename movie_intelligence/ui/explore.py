"""Catalogue explorer page."""

from __future__ import annotations

import altair as alt
import pandas as pd
import streamlit as st

from . import charts
from .components import ACCENT, hero, money, section


def _filtered(df: pd.DataFrame) -> pd.DataFrame:
    years = df["release_year"].dropna()
    min_year, max_year = int(years.min()), int(years.max())

    with st.container(border=True):
        first, second, third = st.columns([1.5, 1.2, 1])
        query = first.text_input(
            "Search titles",
            placeholder="e.g. Star Wars, Godfather ...",
            help="Case-insensitive substring match.",
            key="explore_query",
        )
        genre_options = sorted({g for genres in df["genres"] for g in genres})
        selected_genres = second.multiselect(
            "Genres",
            options=genre_options,
            placeholder="All genres",
            help="A film matches if it carries any selected genre.",
            key="explore_genres",
        )
        min_rating = third.slider(
            "Minimum rating", 0.0, 10.0, 0.0, 0.1, key="explore_min_rating"
        )

        first, second = st.columns([2, 1])
        year_range = first.slider(
            "Release years",
            min_value=min_year,
            max_value=max_year,
            value=(min_year, max_year),
            key="explore_years",
        )
        min_revenue = second.number_input(
            "Min revenue ($M)",
            min_value=0.0,
            value=0.0,
            step=10.0,
            help="Set to 0 to include films without reported revenue.",
            key="explore_min_revenue",
        )

    mask = df["release_year"].between(*year_range)
    mask &= df["vote_average"] >= min_rating
    mask &= df["revenue_musd"] >= min_revenue
    if query:
        mask &= df["title"].str.contains(query.strip(), case=False, na=False, regex=False)
    if selected_genres:
        mask &= df["genres"].map(lambda g: any(genre in g for genre in selected_genres))

    return df[mask].copy()


def render(df: pd.DataFrame) -> None:
    hero(
        "Catalogue",
        f"Explore {len(df):,} films",
        "Filter the catalogue, compare budgets against box office, and pull the slice "
        "you need as a CSV.",
    )

    filtered = _filtered(df)
    if filtered.empty:
        st.warning("No films match those filters. Try widening the year range or clearing genres.")
        return

    profitable = filtered[filtered["budget"] > 0]
    hit_rate = (profitable["revenue"] > 2 * profitable["budget"]).mean() * 100
    cols = st.columns(4)
    cols[0].metric("Films", f"{len(filtered):,}")
    cols[1].metric(
        "Budget",
        money(filtered.loc[filtered["budget"] > 0, "budget"].median()),
        help="Median budget in this selection",
    )
    cols[2].metric(
        "Revenue",
        money(filtered.loc[filtered["revenue"] > 0, "revenue"].median()),
        help="Median revenue in this selection",
    )
    cols[3].metric("2× returns", f"{hit_rate:.0f}%", help="Revenue above twice the budget")

    section("Budget vs box office", "Logarithmic axes; films with reported figures above $100K.")
    scatter_data = filtered[
        (filtered["budget"] >= 100_000) & (filtered["revenue"] >= 100_000)
    ].copy()
    if scatter_data.empty:
        st.info("No films with both budget and revenue in this selection.")
    else:
        top_genres = scatter_data["primary_genre"].value_counts().head(7).index
        scatter_data["genre_group"] = scatter_data["primary_genre"].where(
            scatter_data["primary_genre"].isin(top_genres), "Other"
        )
        scatter = (
            alt.Chart(scatter_data)
            .mark_circle(size=42, opacity=0.55, stroke="rgba(0,0,0,0.25)", strokeWidth=0.4)
            .encode(
                x=alt.X(
                    "budget:Q",
                    scale=alt.Scale(type="log"),
                    title="Budget (USD, log)",
                    axis=alt.Axis(format="~s"),
                ),
                y=alt.Y(
                    "revenue:Q",
                    scale=alt.Scale(type="log"),
                    title="Revenue (USD, log)",
                    axis=alt.Axis(format="~s"),
                ),
                color=alt.Color("genre_group:N", title="Genre"),
                tooltip=[
                    alt.Tooltip("title:N", title="Film"),
                    alt.Tooltip("release_year:O", title="Year"),
                    alt.Tooltip("primary_genre:N", title="Genre"),
                    alt.Tooltip("budget:Q", title="Budget", format="$,.0f"),
                    alt.Tooltip("revenue:Q", title="Revenue", format="$,.0f"),
                    alt.Tooltip("roi:Q", title="Return", format=".1f"),
                ],
            )
        )
        st.altair_chart(charts.style(scatter, 380), width="stretch")

    left, right = st.columns((1, 1))
    with left:
        section("Return on investment", "Capped at 10× for readability.")
        roi_data = filtered[filtered["roi"].notna() & (filtered["roi"] > 0)].copy()
        roi_data["roi_capped"] = roi_data["roi"].clip(upper=10)
        histogram = (
            alt.Chart(roi_data)
            .mark_bar(color=ACCENT, cornerRadiusTopLeft=3, cornerRadiusTopRight=3)
            .encode(
                x=alt.X("roi_capped:Q", bin=alt.Bin(maxbins=30), title="Revenue ÷ budget"),
                y=alt.Y("count():Q", title="Films"),
                tooltip=[alt.Tooltip("count():Q", title="Films")],
            )
        )
        st.altair_chart(charts.style(histogram, 290), width="stretch")

    with right:
        section("Profit by decade")
        decade = (
            filtered[filtered["release_year"].notna()]
            .groupby("decade", as_index=False)
            .agg(profit_musd=("profit_musd", "median"), films=("id", "size"))
        )
        decade = decade[decade["decade"].notna()]
        decade["decade_label"] = decade["decade"].astype(int).astype(str) + "s"
        bars = (
            alt.Chart(decade)
            .mark_bar(cornerRadiusTopLeft=4, cornerRadiusTopRight=4, color="#5B8DEF")
            .encode(
                x=alt.X("decade_label:N", title="Decade", sort=None),
                y=alt.Y("profit_musd:Q", title="Median profit ($M)"),
                tooltip=[
                    alt.Tooltip("decade_label:N", title="Decade"),
                    alt.Tooltip("films:Q", title="Films"),
                    alt.Tooltip("profit_musd:Q", title="Median profit ($M)", format=".1f"),
                ],
            )
        )
        st.altair_chart(charts.style(bars, 290), width="stretch")

    section("Films in this slice", f"{len(filtered):,} rows · sorted by revenue")
    table = filtered.sort_values("revenue", ascending=False)[
        [
            "title",
            "release_year",
            "primary_genre",
            "director",
            "budget",
            "revenue",
            "roi",
            "vote_average",
        ]
    ].copy()
    table["budget"] = table["budget"].map(money)
    table["revenue"] = table["revenue"].map(money)
    table["roi"] = table["roi"].map(lambda v: f"{v:.1f}×" if pd.notna(v) else "—")
    table["director"] = table["director"].replace("", "—")

    st.dataframe(
        table,
        hide_index=True,
        width="stretch",
        height=420,
        column_config={
            "title": st.column_config.TextColumn("Title", width="large"),
            "release_year": st.column_config.NumberColumn("Year", format="%d"),
            "primary_genre": "Genre",
            "director": "Director",
            "budget": "Budget",
            "revenue": "Revenue",
            "roi": "Return",
            "vote_average": st.column_config.NumberColumn("Rating", format="%.1f"),
        },
    )

    export = filtered.drop(columns=[c for c in ("release_date",) if c in filtered.columns])
    st.download_button(
        "Download filtered catalogue (CSV)",
        data=export.to_csv(index=False).encode("utf-8"),
        file_name="cinescope_selection.csv",
        mime="text/csv",
        icon="⬇️",
    )
