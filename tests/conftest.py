"""Shared fixtures for the CineScope test-suite."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


@pytest.fixture()
def catalogue() -> pd.DataFrame:
    """A miniature catalogue with the columns the app relies on."""
    rows = [
        ("Inception", ["Action", "Science Fiction"], ["dream", "heist"], ["Leonardo DiCaprio"], "Christopher Nolan", 160_000_000, 825_000_000, 8.4, 30_000),
        ("Interstellar", ["Adventure", "Drama"], ["space", "time"], ["Matthew McConaughey"], "Christopher Nolan", 165_000_000, 677_000_000, 8.4, 28_000),
        ("The Dark Knight", ["Action", "Crime"], ["batman", "joker"], ["Christian Bale"], "Christopher Nolan", 185_000_000, 1_005_000_000, 8.5, 31_000),
        ("Toy Story", ["Animation", "Family"], ["toys", "friendship"], ["Tom Hanks"], "John Lasseter", 30_000_000, 394_000_000, 7.7, 17_000),
        ("The Godfather", ["Crime", "Drama"], ["mafia", "family"], ["Marlon Brando"], "Francis Ford Coppola", 6_000_000, 245_000_000, 8.7, 16_000),
        ("The Room", ["Drama"], ["friendship"], ["Tommy Wiseau"], "Tommy Wiseau", 6_000_000, 1_200, 3.7, 900),
        ("Heat", ["Action", "Crime"], ["heist", "detective"], ["Al Pacino"], "Michael Mann", 60_000_000, 187_000_000, 7.3, 6_000),
        ("Gravity", ["Science Fiction", "Thriller"], ["space", "survival"], ["Sandra Bullock"], "Alfonso Cuaron", 100_000_000, 723_000_000, 7.7, 20_000),
    ]
    columns = [
        "title",
        "genres",
        "keywords",
        "cast",
        "director",
        "budget",
        "revenue",
        "vote_average",
        "vote_count",
    ]
    df = pd.DataFrame(rows, columns=columns)
    df.insert(0, "id", range(1, len(df) + 1))
    df["original_title"] = df["title"]
    df["overview"] = df["title"] + " follows a group of people through a defining conflict."
    df["tagline"] = "A story about " + df["keywords"].map(lambda k: k[0])
    df["companies"] = df["director"].map(lambda d: [f"{d} Productions"])
    df["countries"] = [["United States of America"]] * len(df)
    df["release_year"] = [2010, 2014, 2008, 1995, 1972, 2003, 1995, 2013]
    df["release_date"] = pd.to_datetime(df["release_year"].astype(str) + "-01-01")
    df["runtime"] = [148, 169, 152, 81, 175, 99, 170, 91]
    df["original_language"] = "en"
    df["status"] = "Released"
    df["primary_genre"] = df["genres"].map(lambda g: g[0])
    df["profit"] = df["revenue"] - df["budget"]
    df["roi"] = df["revenue"] / df["budget"]
    df["budget_musd"] = df["budget"] / 1_000_000
    df["revenue_musd"] = df["revenue"] / 1_000_000
    df["profit_musd"] = df["profit"] / 1_000_000
    df["popularity"] = [90.0, 70.0, 120.0, 45.0, 55.0, 3.0, 30.0, 60.0]
    df["weighted_rating"] = [8.3, 8.3, 8.4, 7.6, 8.6, 3.6, 7.2, 7.6]
    df["decade"] = (df["release_year"] // 10 * 10).astype("Int64")
    return df


@pytest.fixture()
def training_frame(catalogue: pd.DataFrame) -> pd.DataFrame:
    """Expand the catalogue into a frame large enough to train on."""
    rng = np.random.default_rng(0)
    frames = []
    for repeat in range(60):
        block = catalogue.copy()
        block["id"] = block["id"] + repeat * 100
        scale = rng.uniform(0.6, 1.6, size=len(block))
        block["budget"] = (block["budget"] * scale).round()
        block["revenue"] = (block["revenue"] * scale * rng.uniform(0.7, 1.3, size=len(block))).round()
        block["popularity"] = block["popularity"] * rng.uniform(0.7, 1.3, size=len(block))
        block["vote_average"] = block["vote_average"] + rng.normal(0, 0.15, size=len(block))
        block["release_year"] = block["release_year"] + (repeat % 3)
        frames.append(block)
    return pd.concat(frames, ignore_index=True)
