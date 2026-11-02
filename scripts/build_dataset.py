"""Build ``data/movies.csv`` from the raw TMDb movie and credits files.

Usage
-----
    python scripts/build_dataset.py            # use files already in data/raw
    python scripts/build_dataset.py --download # fetch missing raw files first
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from movie_intelligence.config import (  # noqa: E402
    DATASET_URLS,
    MOVIES_PATH,
    RAW_CREDITS_PATH,
    RAW_MOVIES_PATH,
)
from movie_intelligence.dataset import build_movies_dataframe  # noqa: E402


def download(url: str, destination: Path) -> None:
    import requests

    destination.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading {destination.name} ...", flush=True)
    response = requests.get(url, timeout=120, stream=True)
    response.raise_for_status()
    with destination.open("wb") as handle:
        for chunk in response.iter_content(chunk_size=1 << 20):
            handle.write(chunk)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--download",
        action="store_true",
        help="download raw files that are missing from data/raw",
    )
    parser.add_argument("--output", type=Path, default=MOVIES_PATH)
    args = parser.parse_args()

    if args.download:
        for path, url in DATASET_URLS.items():
            if not path.exists() or path.stat().st_size == 0:
                download(url, path)

    missing = [p for p in (RAW_MOVIES_PATH, RAW_CREDITS_PATH) if not p.exists()]
    if missing:
        names = ", ".join(p.name for p in missing)
        print(f"Missing raw file(s): {names}. Re-run with --download.", file=sys.stderr)
        return 1

    movies = pd.read_csv(RAW_MOVIES_PATH)
    credits = pd.read_csv(RAW_CREDITS_PATH)
    print(f"Raw movies: {movies.shape}, credits: {credits.shape}")

    clean = build_movies_dataframe(movies, credits)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    clean.to_csv(args.output, index=False)

    modelled = clean[(clean["budget"] > 0) & (clean["revenue"] > 0)]
    print(
        f"Wrote {len(clean):,} movies to {args.output} "
        f"({args.output.stat().st_size / 1_000_000:.1f} MB); "
        f"{len(modelled):,} usable for training."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
