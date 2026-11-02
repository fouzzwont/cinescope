"""Tests for the optional TMDB client."""

from __future__ import annotations

import requests

from movie_intelligence.tmdb import TMDBClient


class _FakeResponse:
    def __init__(self, payload: dict | None = None, status: int = 200) -> None:
        self._payload = payload or {}
        self.status_code = status

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            raise requests.HTTPError(f"status {self.status_code}")

    def json(self) -> dict:
        return self._payload


class _FakeSession:
    def __init__(self, response: _FakeResponse) -> None:
        self.response = response
        self.calls = 0

    def get(self, *_args, **_kwargs) -> _FakeResponse:
        self.calls += 1
        return self.response


def test_client_without_key_is_disabled() -> None:
    client = TMDBClient(api_key="")
    assert client.enabled is False
    assert client.poster_url(550) is None


def test_client_builds_poster_url_and_caches() -> None:
    session = _FakeSession(_FakeResponse({"poster_path": "/fightclub.jpg"}))
    client = TMDBClient(api_key="test-key", session=session)
    assert client.enabled is True
    assert client.poster_url(550) == "https://image.tmdb.org/t/p/w342/fightclub.jpg"
    assert client.poster_url(550) == "https://image.tmdb.org/t/p/w342/fightclub.jpg"
    assert session.calls == 1  # cached


def test_client_disables_itself_after_repeated_failures() -> None:
    session = _FakeSession(_FakeResponse(status=500))
    client = TMDBClient(api_key="test-key", session=session)
    for movie_id in (1, 2, 3):
        assert client.get_movie(movie_id) is None
    assert client.enabled is False
    assert client.get_movie(4) is None
    assert session.calls == 3
