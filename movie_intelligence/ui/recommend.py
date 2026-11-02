"""Recommendations page."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from .components import card_grid, empty_state, hero, money, poster_card, section
from .state import get_recommender, get_tmdb


def _default_index(titles: list[str]) -> int:
    for candidate in ("The Dark Knight", "Inception", "Avatar", "The Godfather"):
        if candidate in titles:
            return titles.index(candidate)
    return 0


def render(df: pd.DataFrame) -> None:
    hero(
        "Recommendations",
        "Find your next film",
        "TF-IDF similarity across plot, tagline, genres, keywords, cast and crew, "
        "optionally blended with audience ratings. Pick a film and the catalogue does the rest.",
    )

    rec = get_recommender()
    titles = sorted(rec.titles)

    with st.container(border=True):
        first, second = st.columns([2, 1])
        selected = first.selectbox(
            "Start from a film",
            options=titles,
            index=_default_index(titles),
            help="Type to search the catalogue.",
            key="rec_title",
        )
        how_many = second.select_slider(
            "Number of results",
            options=[6, 9, 12, 18],
            value=12,
            key="rec_count",
        )

        first, second = st.columns([1, 2])
        genre_options = ["Any genre"] + sorted(
            {g for genres in df["genres"] for g in genres}
        )
        genre = first.selectbox(
            "Genre filter", options=genre_options, index=0, key="rec_genre"
        )
        min_year = int(df["release_year"].min())
        max_year = int(df["release_year"].max())
        years = second.slider(
            "Release years",
            min_value=min_year,
            max_value=max_year,
            value=(min_year, max_year),
            key="rec_years",
        )

        first, second = st.columns(2)
        blend = first.slider(
            "Rating influence",
            0.0,
            0.6,
            0.15,
            0.05,
            help="0 = pure content similarity, higher values favour well-rated films.",
            key="rec_blend",
        )
        min_rating = second.slider(
            "Minimum audience rating", 0.0, 10.0, 0.0, 0.5, key="rec_min_rating"
        )

    results = rec.recommend(
        selected,
        n=how_many,
        genre=None if genre == "Any genre" else genre,
        year_from=years[0],
        year_to=years[1],
        min_rating=min_rating,
        quality_weight=blend,
    )

    source_index = rec.resolve_title(selected)
    if source_index is not None:
        _source_panel(df, rec, source_index)

    section("Because you watched", f"{len(results)} titles similar to {selected}.")
    if results.empty:
        empty_state("No films match those filters. Try widening the year range or clearing the genre.")
        return

    tmdb = get_tmdb()
    use_posters = False
    if tmdb.enabled:
        use_posters = st.toggle(
            "Load live posters from TMDB",
            value=False,
            help="Fetches artwork for the visible results. Requires internet access and an API key.",
            key="rec_posters",
        )

    cards = []
    poster_lookup = {}
    if use_posters:
        with st.spinner("Fetching posters ..."):
            poster_lookup = {row.id: tmdb.poster_url(row.id) for row in results.itertuples(index=False)}
    for row in results.itertuples(index=False):
        meta = f"{int(row.release_year) if pd.notna(row.release_year) else '—'} · {row.primary_genre} · ★ {row.vote_average:.1f}"
        reasons = list(row.shared_genres[:2]) + [k.title() for k in row.shared_keywords[:2]]
        if row.shared_cast:
            reasons.append(f"with {row.shared_cast[0]}")
        cards.append(
            poster_card(
                title=row.title,
                meta=meta,
                poster_url=poster_lookup.get(row.id),
                badge=f"{max(row.similarity, 0) * 100:.0f}% similar",
                reasons=reasons,
            )
        )

    card_grid(cards)

    if use_posters:
        st.caption("Posters loaded live from TMDB.")
    st.caption(
        "Badges show raw content similarity; the order blends it with audience ratings, "
        "so a slightly lower badge can rank higher. Similarity is not a quality score — "
        "raise rating influence or set a minimum rating to favour acclaimed films."
    )


def _source_panel(df: pd.DataFrame, rec, index: int) -> None:
    movie = df.iloc[index]
    genres = " · ".join(movie["genres"][:3])
    with st.container(border=True):
        first, second = st.columns([1.3, 2.4])
        with first:
            st.markdown(f"### {movie['title']}")
            st.caption(
                f"{movie['release_year']} · {genres} · ★ {movie['vote_average']:.1f} "
                f"({int(movie['vote_count']):,} votes)"
            )
            st.markdown(
                f"Budget {money(movie['budget'])} · Revenue {money(movie['revenue'])}"
            )
        with second:
            st.write(movie["overview"] or "No overview available.")
