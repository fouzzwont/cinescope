"""Train and persist the revenue model.

Usage
-----
    python scripts/train_model.py
    python scripts/train_model.py --report   # also write models/model_metrics.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from movie_intelligence.config import MODEL_PATH  # noqa: E402
from movie_intelligence.dataset import load_movies  # noqa: E402
from movie_intelligence.modelling import (  # noqa: E402
    save_artifact,
    train_model,
    write_metrics_report,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", action="store_true", help="write a JSON metrics report")
    args = parser.parse_args()

    df = load_movies()
    print(f"Loaded {len(df):,} movies; training ...")
    artifact = train_model(df)

    save_artifact(artifact, MODEL_PATH)
    if args.report:
        write_metrics_report(artifact)

    summary = {
        "model": artifact.model_name,
        "rows": f"{artifact.n_train} train / {artifact.n_test} test",
        "metrics": artifact.metrics,
        "candidates_cv_rmse_log": artifact.candidates,
        "top_features": artifact.importances[:5],
    }
    print(json.dumps(summary, indent=2))
    print(f"Saved model to {MODEL_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
