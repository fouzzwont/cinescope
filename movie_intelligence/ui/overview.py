"""Portfolio overview page."""

from __future__ import annotations

import altair as alt
import pandas as pd
import streamlit as st

from . import charts
from .components import ACCENT, hero, money, section


def render(df: pd.DataFrame) -> None:
    hero(
        "Movie intelligence",
        "What the box office looks like",
        f"{len(df):,} films from the TMDb 5000 catalogue, with budgets, revenues, "
        "genres, cast and crew. Use the sidebar to explore, predict or get recommendations.",
    )

    total_revenue = df["revenue"].sum()
    profitable = (df["profit"] > 0).mean() * 100
    median_roi = df.loc[df["roi"].notna(), "roi"].median()

    kpi_cols = st.columns(4)
    kpi_cols[0].metric("Films", f"{len(df):,}", help="Full TMDb 5000 catalogue")
    kpi_cols[1].metric(
        "Revenue",
        money(total_revenue),
        help="Sum of reported revenue across all films",
    )
    kpi_cols[2].metric(
        "ROI",
        f"{median_roi:.1f}×" if pd.notna(median_roi) else "—",
        help="Median revenue divided by budget",
    )
    kpi_cols[3].metric(
        "Profitable",
        f"{profitable:.0f}%",
        help="Share of films whose revenue exceeds their budget",
    )

    section("Box office over time", "Total reported revenue per release year.")
    yearly = charts.by_year(df)
    if yearly.empty:
        st.info("No releases in the selected range.")
    else:
        line = (
            alt.Chart(yearly)
            .mark_area(
                line={"color": ACCENT, "strokeWidth": 2.5},
                color=alt.Gradient(
                    gradient="linear",
                    stops=[
                        alt.GradientStop(color="rgba(229,9,20,0.35)", offset=0),
                        alt.GradientStop(color="rgba(229,9,20,0.02)", offset=1),
                    ],
                    x1=1, x2=1, y1=1, y2=0,
                ),
                interpolate="monotone",
            )
            .encode(
                x=alt.X("release_year:O", title="Release year", axis=alt.Axis(labelAngle=-45)),
                y=alt.Y("revenue_busd:Q", title="Revenue (USD billions)"),
                tooltip=[
                    alt.Tooltip("release_year:O", title="Year"),
                    alt.Tooltip("movies:Q", title="Films"),
                    alt.Tooltip("revenue_busd:Q", title="Total revenue ($B)", format=".1f"),
                ],
            )
        )
        st.altair_chart(charts.style(line, 300), width="stretch")

    left, right = st.columns((1, 1))

    with left:
        section("Genres by volume")
        genres = charts.genre_summary(df, top=10)
        chart = (
            alt.Chart(genres)
            .mark_bar(cornerRadiusTopRight=4, cornerRadiusBottomRight=4, color=ACCENT)
            .encode(
                x=alt.X("movies:Q", title="Films"),
                y=alt.Y("genres:N", sort="-x", title=None),
                tooltip=[
                    alt.Tooltip("genres:N", title="Genre"),
                    alt.Tooltip("movies:Q", title="Films"),
                    alt.Tooltip("median_revenue_musd:Q", title="Median revenue ($M)", format=".1f"),
                ],
            )
        )
        st.altair_chart(charts.style(chart, 320), width="stretch")

    with right:
        section("Highest-grossing films")
        top = df.nlargest(8, "revenue")[
            ["title", "release_year", "primary_genre", "revenue_musd", "roi"]
        ].copy()
        top["revenue"] = top["revenue_musd"].map(lambda v: money(v * 1_000_000))
        top["roi"] = top["roi"].map(lambda v: f"{v:.1f}×" if pd.notna(v) else "—")
        st.dataframe(
            top[["title", "release_year", "primary_genre", "revenue", "roi"]],
            hide_index=True,
            width="stretch",
            column_config={
                "title": st.column_config.TextColumn("Title", width="large"),
                "release_year": st.column_config.NumberColumn("Year", format="%d"),
                "primary_genre": "Genre",
                "revenue": "Revenue",
                "roi": "Return",
            },
        )

    section("Directors with the biggest cumulative box office", "Only directors credited on the film.")
    directors = (
        df[df["director"] != ""]
        .groupby("director", as_index=False)
        .agg(films=("id", "size"), revenue=("revenue", "sum"), median_rating=("vote_average", "median"))
        .sort_values("revenue", ascending=False)
        .head(10)
    )
    directors["revenue"] = directors["revenue"].map(money)
    st.dataframe(
        directors.rename(
            columns={
                "director": "Director",
                "films": "Films",
                "revenue": "Total revenue",
                "median_rating": "Median rating",
            }
        ),
        hide_index=True,
        width="stretch",
        column_config={
            "Median rating": st.column_config.NumberColumn(format="%.1f ⭐"),
        },
    )
