"""Tests for the content-based recommender."""

from __future__ import annotations

import numpy as np
import pandas as pd

from movie_intelligence.recommender import Recommender


def test_resolve_title_handles_exact_and_fuzzy(catalogue) -> None:
    rec = Recommender(catalogue)
    assert rec.resolve_title("Inception") == 0
    assert rec.resolve_title("inception") == 0
    assert rec.resolve_title("Incepton") == 0  # typo
    assert rec.resolve_title("") is None
    assert rec.resolve_title("Not A Real Film") is None


def test_search_titles_returns_prefix_matches(catalogue) -> None:
    rec = Recommender(catalogue)
    results = rec.search_titles("the")
    assert any("The Dark Knight" in title for title in results)
    assert rec.search_titles("") == []


def test_recommendations_exclude_source_and_respect_filters(catalogue) -> None:
    rec = Recommender(catalogue)
    results = rec.recommend("Inception", n=5)
    assert not results.empty
    assert results["title"].iloc[0] != "Inception"
    assert "Inception" not in set(results["title"])
    assert results["score"].is_monotonic_decreasing

    animated = rec.recommend("Toy Story", n=5, genre="Animation")
    assert animated.empty or animated["genres"].map(lambda g: "Animation" in g).all()

    filtered = rec.recommend("Inception", n=5, year_from=2010, min_rating=7.0)
    assert (filtered["release_year"] >= 2010).all()
    assert (filtered["vote_average"] >= 7.0).all()


def test_quality_weight_changes_scores(catalogue) -> None:
    rec = Recommender(catalogue)
    pure = rec.recommend("The Room", n=3, quality_weight=0.0)
    blended = rec.recommend("The Room", n=3, quality_weight=0.6)
    assert pure["score"].is_monotonic_decreasing
    assert blended["score"].is_monotonic_decreasing
    assert len(pure) == len(blended)
    assert not np.allclose(
        pure.set_index("title")["score"].sort_index(),
        blended.set_index("title")["score"].sort_index(),
    )


def test_unknown_title_returns_empty_frame(catalogue) -> None:
    rec = Recommender(catalogue)
    result = rec.recommend("Definitely Not A Movie", n=5)
    assert isinstance(result, pd.DataFrame)
    assert result.empty
