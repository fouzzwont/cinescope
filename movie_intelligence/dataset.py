"""Data loading, parsing and feature derivation for the TMDb catalogue.

The raw Kaggle files store list columns (genres, keywords, cast, ...) as JSON
strings. :func:`build_movies_dataframe` normalises them into ``|``-separated
strings so the resulting CSV stays human readable; :func:`load_movies` turns
those back into Python lists for the application.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from .config import MOVIES_PATH

LIST_COLUMNS = ("genres", "keywords", "cast", "companies", "countries")

# Columns kept in the cleaned dataset, in display order.
OUTPUT_COLUMNS = [
    "id",
    "title",
    "original_title",
    "overview",
    "tagline",
    "genres",
    "primary_genre",
    "keywords",
    "cast",
    "director",
    "companies",
    "countries",
    "budget",
    "revenue",
    "popularity",
    "runtime",
    "vote_average",
    "vote_count",
    "release_date",
    "release_year",
    "original_language",
    "status",
]

PERCENTILE_FOR_WEIGHTING = 0.90


def _parse_name_list(value: object, limit: int | None = None) -> list[str]:
    """Parse a TMDb JSON list (or an already-joined string) into names."""
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return []
    if isinstance(value, (list, tuple)):
        items = list(value)
    else:
        text = str(value).strip()
        if not text:
            return []
        if text.startswith("["):
            try:
                items = json.loads(text)
            except json.JSONDecodeError:
                return []
        else:
            items = [part for part in text.split("|")]

    names: list[str] = []
    for item in items:
        if isinstance(item, dict):
            name = item.get("name") or item.get("iso_3166_1") or ""
        else:
            name = str(item)
        name = name.strip()
        if name and name not in names:
            names.append(name)
    return names[:limit] if limit else names


def _extract_cast(value: object, limit: int = 5) -> list[str]:
    """Return the top-billed cast members, ordered by billing position."""
    if not isinstance(value, str) or not value.strip():
        return []
    try:
        records = json.loads(value)
    except json.JSONDecodeError:
        return _parse_name_list(value, limit=limit)
    ordered = sorted(
        (r for r in records if isinstance(r, dict)),
        key=lambda r: r.get("order", 10_000),
    )
    names = [r.get("name", "").strip() for r in ordered]
    return [n for n in dict.fromkeys(names) if n][:limit]


def _extract_director(crew_value: object) -> str:
    """Return the first credited director from a credits ``crew`` payload."""
    if not isinstance(crew_value, str) or not crew_value.strip():
        return ""
    try:
        records = json.loads(crew_value)
    except json.JSONDecodeError:
        return ""
    for member in records:
        if isinstance(member, dict) and member.get("job") == "Director":
            name = str(member.get("name", "")).strip()
            if name:
                return name
    return ""


def build_movies_dataframe(
    movies: pd.DataFrame,
    credits: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Merge the raw movies and credits frames into the cleaned catalogue."""
    df = movies.copy()

    if credits is not None and {"movie_id", "cast", "crew"}.issubset(credits.columns):
        extra = pd.DataFrame(
            {
                "id": credits["movie_id"],
                "cast": credits["cast"].map(_extract_cast),
                "director": credits["crew"].map(_extract_director),
            }
        )
        extra = extra.drop_duplicates(subset="id")
        df = df.merge(extra, on="id", how="left")
    else:
        df["cast"] = [[] for _ in range(len(df))]
        df["director"] = ""

    for column, limit in (
        ("genres", None),
        ("keywords", None),
        ("companies", None),
        ("countries", None),
    ):
        source = df[column] if column in df.columns else pd.Series([""] * len(df))
        df[column] = source.map(lambda v, lim=limit: _parse_name_list(v, lim))

    if "tagline" not in df.columns:
        df["tagline"] = ""
    df["tagline"] = df["tagline"].fillna("").astype(str)
    df["overview"] = df.get("overview", "").fillna("").astype(str)
    df["original_language"] = df.get("original_language", "en").fillna("en")
    df["status"] = df.get("status", "Released").fillna("Released")

    df["release_date"] = pd.to_datetime(df.get("release_date"), errors="coerce")
    df["release_year"] = df["release_date"].dt.year

    df["budget"] = pd.to_numeric(df.get("budget"), errors="coerce").fillna(0).clip(lower=0)
    df["revenue"] = pd.to_numeric(df.get("revenue"), errors="coerce").fillna(0).clip(lower=0)
    df["popularity"] = pd.to_numeric(df.get("popularity"), errors="coerce").fillna(0)
    df["vote_average"] = pd.to_numeric(df.get("vote_average"), errors="coerce").fillna(0)
    df["vote_count"] = pd.to_numeric(df.get("vote_count"), errors="coerce").fillna(0)
    df["runtime"] = pd.to_numeric(df.get("runtime"), errors="coerce")
    runtime_median = df["runtime"].median()
    df["runtime"] = df["runtime"].fillna(runtime_median if pd.notna(runtime_median) else 0)
    df["runtime"] = df["runtime"].where(df["runtime"] > 0, df["runtime"].median())

    df["primary_genre"] = df["genres"].map(lambda g: g[0] if g else "Unknown")
    df["director"] = df["director"].fillna("").astype(str)
    for column in LIST_COLUMNS:
        df[column] = df[column].map(lambda values: values if isinstance(values, list) else [])

    df = df.dropna(subset=["id", "title"])
    df["id"] = df["id"].astype("int64")
    df["title"] = df["title"].astype(str).str.strip()
    df = df[df["title"] != ""]
    df = df.drop_duplicates(subset="id").reset_index(drop=True)

    for column in LIST_COLUMNS:
        df[column] = df[column].map(
            lambda names: "|".join(names) if isinstance(names, list) else ""
        )
    df["release_date"] = df["release_date"].dt.strftime("%Y-%m-%d")

    return df[OUTPUT_COLUMNS].sort_values("id").reset_index(drop=True)


def _split(value: object) -> list[str]:
    if isinstance(value, list):
        return value
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return []
    return [part for part in str(value).split("|") if part]


def add_derived_features(df: pd.DataFrame) -> pd.DataFrame:
    """Attach analysis-friendly columns (profit, ROI, weighted rating)."""
    out = df.copy()
    out["release_date"] = pd.to_datetime(out["release_date"], errors="coerce")
    out["release_year"] = out["release_year"].fillna(
        out["release_date"].dt.year
    ).astype("Int64")

    out["profit"] = out["revenue"] - out["budget"]
    out["roi"] = np.where(out["budget"] > 0, out["revenue"] / out["budget"], np.nan)
    out["budget_musd"] = out["budget"] / 1_000_000
    out["revenue_musd"] = out["revenue"] / 1_000_000
    out["profit_musd"] = out["profit"] / 1_000_000
    out["decade"] = (out["release_year"] // 10 * 10).astype("Int64")

    # IMDb-style weighted rating: pulls low-vote-count films toward the mean.
    mean_rating = out["vote_average"].mean()
    threshold = out["vote_count"].quantile(PERCENTILE_FOR_WEIGHTING)
    votes = out["vote_count"].fillna(0)
    out["weighted_rating"] = np.where(
        votes + threshold > 0,
        (votes / (votes + threshold)) * out["vote_average"]
        + (threshold / (votes + threshold)) * mean_rating,
        mean_rating,
    )
    return out


def load_movies(path: str | Path = MOVIES_PATH) -> pd.DataFrame:
    """Load the cleaned catalogue, restoring list columns and derived metrics."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Dataset not found at {path}. Run `python scripts/build_dataset.py --download`."
        )
    df = pd.read_csv(path)
    for column in LIST_COLUMNS:
        df[column] = df[column].map(_split)
    df["release_date"] = pd.to_datetime(df["release_date"], errors="coerce")
    return add_derived_features(df)


def modelling_subset(df: pd.DataFrame) -> pd.DataFrame:
    """Rows usable for supervised training: reported budget and revenue."""
    mask = (
        (df["budget"] > 0)
        & (df["revenue"] > 0)
        & df["release_year"].notna()
        & (df["runtime"] > 0)
    )
    return df.loc[mask].copy()
