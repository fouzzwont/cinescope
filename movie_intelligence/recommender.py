"""Content-based movie recommendations with TF-IDF and cosine similarity.

Only one similarity row is computed per request (instead of the full 4803x4803
matrix), which keeps memory flat and responses fast.
"""

from __future__ import annotations

import difflib
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize

DEFAULT_COLUMNS = [
    "id",
    "title",
    "release_year",
    "primary_genre",
    "genres",
    "vote_average",
    "vote_count",
    "revenue_musd",
    "similarity",
    "score",
    "shared_genres",
    "shared_keywords",
    "shared_cast",
]


def _tokens(values: object) -> list[str]:
    if isinstance(values, list):
        return [str(v) for v in values if str(v).strip()]
    if values is None or (isinstance(values, float) and np.isnan(values)):
        return []
    return [part.strip() for part in str(values).split("|") if part.strip()]


def _join(values: list[str]) -> str:
    return " ".join(values)


@dataclass
class Recommendation:
    """A single recommendation plus the evidence behind it."""

    index: int
    similarity: float
    score: float
    shared_genres: tuple[str, ...] = ()
    shared_keywords: tuple[str, ...] = ()
    shared_cast: tuple[str, ...] = ()


class Recommender:
    """Fuzzy title lookup + weighted content similarity."""

    def __init__(self, df: pd.DataFrame, text_weight: float = 0.85) -> None:
        if df.empty:
            raise ValueError("Cannot build a recommender from an empty catalogue.")
        self.df = df.reset_index(drop=True).copy()
        self.text_weight = float(np.clip(text_weight, 0.0, 1.0))

        for column in ("genres", "keywords", "cast", "companies"):
            if column not in self.df.columns:
                self.df[column] = [[] for _ in range(len(self.df))]
            self.df[column] = self.df[column].map(_tokens)

        self.titles = self.df["title"].astype(str).tolist()
        self._title_lookup = {title.casefold(): i for i, title in enumerate(self.titles)}
        self._sets = {
            "genres": [set(g) for g in self.df["genres"]],
            "keywords": [set(k) for k in self.df["keywords"]],
            "cast": [set(c) for c in self.df["cast"]],
        }

        soup = [
            self._build_soup(
                overview=row.overview,
                tagline=row.tagline,
                genres=row.genres,
                keywords=row.keywords,
                cast=row.cast,
                director=row.director,
                companies=row.companies,
            )
            for row in self.df.itertuples(index=False)
        ]
        self.vectorizer = TfidfVectorizer(
            stop_words="english",
            ngram_range=(1, 2),
            min_df=2,
            max_features=60_000,
            sublinear_tf=True,
            strip_accents="unicode",
        )
        self.matrix = normalize(self.vectorizer.fit_transform(soup)).tocsr()

        quality = pd.to_numeric(self.df["weighted_rating"], errors="coerce").fillna(0.0)
        span = quality.max() - quality.min()
        self.quality = ((quality - quality.min()) / span if span else quality * 0).to_numpy()

    @staticmethod
    def _build_soup(
        *,
        overview: object,
        tagline: object,
        genres: list[str],
        keywords: list[str],
        cast: list[str],
        director: object,
        companies: list[str],
    ) -> str:
        # Repeated tokens act as weights inside the TF-IDF document.
        director_tokens = [str(director)] * 2 if director else []
        parts = [
            _join(genres + genres),
            str(overview or ""),
            str(tagline or ""),
            _join(keywords),
            _join(cast),
            _join(director_tokens),
            _join(companies),
        ]
        return " ".join(part for part in parts if part)

    # ------------------------------------------------------------------ lookup
    def resolve_title(self, query: str) -> int | None:
        """Return a row index for an exact or fuzzy title match."""
        if not query or not str(query).strip():
            return None
        cleaned = str(query).strip()
        exact = self._title_lookup.get(cleaned.casefold())
        if exact is not None:
            return exact
        close = difflib.get_close_matches(cleaned, self.titles, n=1, cutoff=0.6)
        return self._title_lookup[close[0].casefold()] if close else None

    def search_titles(self, query: str, limit: int = 10) -> list[str]:
        """Fuzzy search used for suggestion chips in the UI."""
        if not query or not str(query).strip():
            return []
        cleaned = str(query).strip().casefold()
        starts = [t for t in self.titles if t.casefold().startswith(cleaned)]
        contains = [
            t
            for t in self.titles
            if cleaned in t.casefold() and t not in starts
        ]
        fuzzy = [
            t
            for t in difflib.get_close_matches(query, self.titles, n=limit * 2, cutoff=0.5)
            if t not in starts and t not in contains
        ]
        return (starts + contains + fuzzy)[:limit]

    # ------------------------------------------------------------- recommend
    def recommend(
        self,
        query: str | int,
        n: int = 10,
        *,
        genre: str | None = None,
        year_from: int | None = None,
        year_to: int | None = None,
        min_rating: float | None = None,
        quality_weight: float | None = None,
    ) -> pd.DataFrame:
        """Return ``n`` similar movies, optionally filtered and quality-blended."""
        index = query if isinstance(query, int) else self.resolve_title(str(query))
        if index is None or not (0 <= index < len(self.df)):
            return pd.DataFrame(columns=DEFAULT_COLUMNS)

        row = self.matrix[index]
        similarity = np.asarray((self.matrix @ row.T).todense()).ravel()

        weight = self.text_weight if quality_weight is None else float(quality_weight)
        weight = float(np.clip(weight, 0.0, 1.0))
        score = (1.0 - weight) * similarity + weight * self.quality

        mask = np.ones(len(self.df), dtype=bool)
        mask[index] = False
        years = pd.to_numeric(self.df["release_year"], errors="coerce")
        if genre:
            mask &= self.df["genres"].map(lambda g: genre in g).to_numpy()
        if year_from is not None:
            mask &= (years.fillna(0) >= year_from).to_numpy()
        if year_to is not None:
            mask &= (years.fillna(9999) <= year_to).to_numpy()
        if min_rating is not None:
            mask &= (
                pd.to_numeric(self.df["vote_average"], errors="coerce").fillna(0)
                >= min_rating
            ).to_numpy()

        candidates = np.where(mask)[0]
        if candidates.size == 0:
            return pd.DataFrame(columns=DEFAULT_COLUMNS)
        ranked = candidates[np.argsort(score[candidates])[::-1][: max(int(n), 1)]]

        genres = self._sets["genres"][index]
        keywords = self._sets["keywords"][index]
        cast = self._sets["cast"][index]
        records = []
        for other in ranked:
            records.append(
                {
                    "id": int(self.df.at[other, "id"]),
                    "title": self.df.at[other, "title"],
                    "release_year": self.df.at[other, "release_year"],
                    "primary_genre": self.df.at[other, "primary_genre"],
                    "genres": self.df.at[other, "genres"],
                    "vote_average": self.df.at[other, "vote_average"],
                    "vote_count": self.df.at[other, "vote_count"],
                    "revenue_musd": self.df.at[other, "revenue_musd"],
                    "similarity": float(similarity[other]),
                    "score": float(score[other]),
                    "shared_genres": tuple(sorted(genres & self._sets["genres"][other])),
                    "shared_keywords": tuple(
                        sorted(keywords & self._sets["keywords"][other])
                    )[:4],
                    "shared_cast": tuple(sorted(cast & self._sets["cast"][other]))[:3],
                }
            )
        return pd.DataFrame.from_records(records, columns=DEFAULT_COLUMNS)
