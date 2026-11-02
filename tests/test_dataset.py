"""Tests for raw-data parsing and cleaning."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

from movie_intelligence.dataset import (
    build_movies_dataframe,
    load_movies,
    modelling_subset,
)


def _raw_movies() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "id": 1,
                "title": "Test Movie",
                "original_title": "Test Movie",
                "overview": "A test.",
                "tagline": "Just a test",
                "genres": json.dumps([{"id": 28, "name": "Action"}, {"id": 18, "name": "Drama"}]),
                "keywords": json.dumps([{"id": 1, "name": "friendship"}]),
                "production_companies": json.dumps([{"name": "Studio A"}]),
                "production_countries": json.dumps([{"iso_3166_1": "US", "name": "United States"}]),
                "budget": 10_000_000,
                "revenue": 50_000_000,
                "popularity": 12.5,
                "runtime": 100,
                "vote_average": 7.2,
                "vote_count": 500,
                "release_date": "2010-05-01",
                "original_language": "en",
                "status": "Released",
            },
            {
                "id": 2,
                "title": "Zero Budget",
                "original_title": "Zero Budget",
                "overview": "No money.",
                "tagline": None,
                "genres": "[]",
                "keywords": "",
                "production_companies": "[]",
                "production_countries": "[]",
                "budget": 0,
                "revenue": 0,
                "popularity": 1.0,
                "runtime": np.nan,
                "vote_average": 5.0,
                "vote_count": 5,
                "release_date": None,
                "original_language": "en",
                "status": "Released",
            },
        ]
    )


def _raw_credits() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "movie_id": 1,
                "title": "Test Movie",
                "cast": json.dumps(
                    [{"name": "B Actor", "order": 1}, {"name": "A Actor", "order": 0}]
                ),
                "crew": json.dumps(
                    [{"name": "Editor", "job": "Editor"}, {"name": "Jane Doe", "job": "Director"}]
                ),
            }
        ]
    )


def test_build_dataset_parses_lists_and_credits() -> None:
    clean = build_movies_dataframe(_raw_movies(), _raw_credits())
    assert len(clean) == 2
    first = clean.loc[clean["id"] == 1].iloc[0]
    assert first["genres"] == "Action|Drama"
    assert first["primary_genre"] == "Action"
    assert first["cast"] == "A Actor|B Actor"  # ordered by billing
    assert first["director"] == "Jane Doe"
    assert first["release_year"] == 2010


def test_build_dataset_fills_missing_runtime_and_zero_financials() -> None:
    clean = build_movies_dataframe(_raw_movies(), _raw_credits())
    second = clean.loc[clean["id"] == 2].iloc[0]
    assert second["budget"] == 0 and second["revenue"] == 0
    assert second["runtime"] > 0
    assert np.isnan(second["release_year"]) or pd.isna(second["release_year"])


def test_derived_features_and_modelling_subset(tmp_path, catalogue) -> None:
    path = tmp_path / "movies.csv"
    export = catalogue.copy()
    for column in ("genres", "keywords", "cast", "companies", "countries"):
        export[column] = export[column].map(lambda values: "|".join(values))
    export["release_date"] = export["release_date"].dt.strftime("%Y-%m-%d")
    export.to_csv(path, index=False)

    loaded = load_movies(path)
    assert isinstance(loaded.loc[0, "genres"], list)
    assert loaded.loc[0, "profit"] == loaded.loc[0, "revenue"] - loaded.loc[0, "budget"]
    assert 0 <= loaded.loc[0, "weighted_rating"] <= 10

    subset = modelling_subset(loaded)
    assert (subset["budget"] > 0).all()
    assert (subset["revenue"] > 0).all()
