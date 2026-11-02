"""Altair theme helpers."""

from __future__ import annotations

import altair as alt
import pandas as pd

from .components import ACCENT, GOLD, MUTED


def style(chart: alt.Chart, height: int = 330) -> alt.Chart:
    return (
        chart.properties(height=height)
        .configure_view(strokeOpacity=0)
        .configure_axis(
            labelColor=MUTED,
            titleColor=MUTED,
            gridColor="rgba(255,255,255,0.06)",
            domainColor="rgba(255,255,255,0.15)",
            tickColor="rgba(255,255,255,0.15)",
            labelFontSize=11,
            titleFontSize=11,
            labelLimit=200,
        )
        .configure_legend(labelColor=MUTED, titleColor=MUTED, orient="top")
    )


def by_year(df: pd.DataFrame, start_year: int = 1980) -> pd.DataFrame:
    yearly = (
        df[df["release_year"] >= start_year]
        .groupby("release_year", as_index=False)
        .agg(
            movies=("id", "size"),
            revenue=("revenue", "sum"),
            median_revenue=("revenue", "median"),
        )
    )
    yearly["revenue_busd"] = yearly["revenue"] / 1_000_000_000
    return yearly


def genre_summary(df: pd.DataFrame, top: int = 10) -> pd.DataFrame:
    exploded = df.explode("genres")
    exploded = exploded[exploded["genres"].notna() & (exploded["genres"] != "")]
    summary = (
        exploded.groupby("genres", as_index=False)
        .agg(
            movies=("id", "size"),
            median_revenue_musd=("revenue_musd", "median"),
            total_revenue_busd=("revenue", lambda s: s.sum() / 1_000_000_000),
        )
        .sort_values("movies", ascending=False)
        .head(top)
    )
    return summary


def color_scale(field: str) -> alt.Color:
    return alt.Color(field, scale=alt.Scale(scheme="reds"), legend=None)


__all__ = ["style", "by_year", "genre_summary", "color_scale", "ACCENT", "GOLD", "MUTED"]
