"""Minimal, defensive TMDB API client.

Posters are not part of the static Kaggle export, so they are fetched live when
``TMDB_API_KEY`` is configured. Every failure degrades gracefully to ``None`` --
the UI falls back to typographic cards -- and repeated failures disable the
client for the rest of the session to avoid slow page loads.
"""

from __future__ import annotations

from typing import Any

import requests

from .config import tmdb_api_key

API_BASE = "https://api.themoviedb.org/3"
IMAGE_BASE = "https://image.tmdb.org/t/p"

MAX_CONSECUTIVE_FAILURES = 3


class TMDBClient:
    def __init__(
        self,
        api_key: str | None = None,
        timeout: float = 6.0,
        session: requests.Session | None = None,
    ) -> None:
        self.api_key = api_key if api_key is not None else tmdb_api_key()
        self.timeout = timeout
        self._session = session or requests.Session()
        self._cache: dict[int, dict[str, Any] | None] = {}
        self._failures = 0

    @property
    def enabled(self) -> bool:
        return bool(self.api_key) and self._failures < MAX_CONSECUTIVE_FAILURES

    def get_movie(self, movie_id: int | str) -> dict[str, Any] | None:
        """Fetch and cache movie details. Returns ``None`` on any error."""
        if not self.enabled:
            return None
        try:
            key = int(movie_id)
        except (TypeError, ValueError):
            return None
        if key in self._cache:
            return self._cache[key]

        try:
            response = self._session.get(
                f"{API_BASE}/movie/{key}",
                params={"api_key": self.api_key, "language": "en-US"},
                timeout=self.timeout,
            )
            response.raise_for_status()
            payload = response.json()
        except (requests.RequestException, ValueError):
            self._failures += 1
            self._cache[key] = None
            return None

        self._failures = 0
        self._cache[key] = payload
        return payload

    def poster_url(self, movie_id: int | str, size: str = "w342") -> str | None:
        """Return a full poster URL, or ``None`` when unavailable."""
        details = self.get_movie(movie_id)
        path = (details or {}).get("poster_path")
        return f"{IMAGE_BASE}/{size}{path}" if path else None
