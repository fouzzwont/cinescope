"""Application-wide paths and configuration."""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
MODELS_DIR = PROJECT_ROOT / "models"

MOVIES_PATH = DATA_DIR / "movies.csv"
MODEL_PATH = MODELS_DIR / "revenue_model.joblib"
METRICS_PATH = MODELS_DIR / "model_metrics.json"

RAW_MOVIES_PATH = RAW_DIR / "tmdb_5000_movies.csv"
RAW_CREDITS_PATH = RAW_DIR / "tmdb_5000_credits.csv"

DATASET_URLS = {
    RAW_MOVIES_PATH: (
        "https://huggingface.co/datasets/maximus007411/tmdb-5000-movie-data/"
        "resolve/main/tmdb_5000_movies.csv"
    ),
    RAW_CREDITS_PATH: (
        "https://huggingface.co/datasets/maximus007411/tmdb-5000-movie-data/"
        "resolve/main/tmdb_5000_credits.csv"
    ),
}

# Loading .env is a no-op when the file is absent, which keeps the app portable.
load_dotenv(PROJECT_ROOT / ".env")

APP_TITLE = "CineScope"
APP_ICON = "🎬"


def tmdb_api_key() -> str | None:
    """Return the configured TMDB API key, or ``None`` when unavailable."""
    key = os.getenv("TMDB_API_KEY", "").strip()
    return key or None
