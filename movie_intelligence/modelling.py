"""Revenue model: feature engineering, training and inference.

The savings from this module intentionally have no Streamlit dependency so the
same code can be used from scripts, tests and the web app.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.inspection import permutation_importance
from sklearn.linear_model import RidgeCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .config import MODEL_PATH, MODELS_DIR
from .dataset import modelling_subset

RANDOM_STATE = 42
TEST_SIZE = 0.20
N_FOLDS = 5
INTERVAL_Z = 1.2816  # 80% prediction interval

NUMERIC_FEATURES = [
    "budget_log",
    "popularity_log",
    "runtime",
    "vote_average",
    "release_year",
    "num_genres",
    "num_companies",
    "budget_per_minute_log",
]
CATEGORICAL_FEATURES = ["primary_genre", "original_language", "release_decade"]

FEATURE_COLUMNS = NUMERIC_FEATURES + CATEGORICAL_FEATURES


@dataclass
class ModelArtifact:
    """Everything needed to reproduce and explain a prediction."""

    pipeline: Pipeline
    metrics: dict[str, float]
    importances: list[dict[str, Any]] = field(default_factory=list)
    model_name: str = "model"
    residual_std_log: float = 0.0
    n_train: int = 0
    n_test: int = 0
    trained_at: str = ""
    candidates: dict[str, float] = field(default_factory=dict)
    feature_columns: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "pipeline": self.pipeline,
            "metrics": self.metrics,
            "importances": self.importances,
            "model_name": self.model_name,
            "residual_std_log": self.residual_std_log,
            "n_train": self.n_train,
            "n_test": self.n_test,
            "trained_at": self.trained_at,
            "candidates": self.candidates,
            "feature_columns": self.feature_columns,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> ModelArtifact:
        return cls(**payload)


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create the model input frame. Pure function, safe at inference time."""
    runtime = pd.to_numeric(df.get("runtime"), errors="coerce").fillna(100).clip(lower=1)
    budget = pd.to_numeric(df.get("budget"), errors="coerce").fillna(0).clip(lower=0)

    features = pd.DataFrame(index=df.index)
    features["budget_log"] = np.log1p(budget)
    features["popularity_log"] = np.log1p(
        pd.to_numeric(df.get("popularity"), errors="coerce").fillna(0).clip(lower=0)
    )
    features["runtime"] = runtime
    features["vote_average"] = (
        pd.to_numeric(df.get("vote_average"), errors="coerce").fillna(0).clip(0, 10)
    )
    features["release_year"] = pd.to_numeric(
        df.get("release_year"), errors="coerce"
    ).fillna(2000).astype(float)
    features["num_genres"] = df.get("genres", pd.Series([[]] * len(df))).map(len)
    features["num_companies"] = df.get("companies", pd.Series([[]] * len(df))).map(len)
    features["budget_per_minute_log"] = np.log1p(budget / runtime)

    features["primary_genre"] = (
        df.get("primary_genre", pd.Series(["Unknown"] * len(df))).fillna("Unknown")
    )
    features["original_language"] = (
        df.get("original_language", pd.Series(["en"] * len(df))).fillna("en")
    )
    features["release_decade"] = (
        (features["release_year"] // 10 * 10).astype(int).astype(str) + "s"
    )
    return features[FEATURE_COLUMNS]


def _build_pipeline(regressor: Any) -> Pipeline:
    preprocess = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), NUMERIC_FEATURES),
            (
                "cat",
                OneHotEncoder(handle_unknown="ignore", min_frequency=0.01, sparse_output=False),
                CATEGORICAL_FEATURES,
            ),
        ],
        remainder="drop",
    )
    return Pipeline([("preprocess", preprocess), ("model", regressor)])


def _candidate_models() -> dict[str, Any]:
    return {
        "Ridge regression": RidgeCV(alphas=np.logspace(-2, 3, 12)),
        "Random forest": RandomForestRegressor(
            n_estimators=400,
            min_samples_leaf=2,
            max_features=0.6,
            n_jobs=-1,
            random_state=RANDOM_STATE,
        ),
        "Gradient boosting": HistGradientBoostingRegressor(
            max_iter=500,
            learning_rate=0.05,
            max_leaf_nodes=31,
            l2_regularization=1.0,
            random_state=RANDOM_STATE,
        ),
    }


def _feature_importances(pipeline: Pipeline, X_test: pd.DataFrame, y_test: np.ndarray) -> list[dict[str, Any]]:
    """Return the 12 most influential transformed features for this model."""
    preprocess: ColumnTransformer = pipeline.named_steps["preprocess"]
    model = pipeline.named_steps["model"]
    names = np.asarray(preprocess.get_feature_names_out())

    if hasattr(model, "feature_importances_"):
        values = np.asarray(model.feature_importances_)
    elif hasattr(model, "coef_"):
        values = np.abs(np.asarray(model.coef_)).ravel()
    else:
        sample = X_test.head(800)
        result = permutation_importance(
            pipeline,
            sample,
            y_test[: len(sample)],
            n_repeats=4,
            random_state=RANDOM_STATE,
            scoring="neg_root_mean_squared_error",
        )
        values = np.asarray(result.importances_mean).ravel()
        names = np.asarray(FEATURE_COLUMNS)

    if len(values) != len(names):
        names = np.asarray(FEATURE_COLUMNS) if len(values) == len(FEATURE_COLUMNS) else names[: len(values)]

    order = np.argsort(values)[::-1][:12]
    total = float(values.sum()) or 1.0
    return [
        {
            "feature": str(names[i]).replace("num__", "").replace("cat__", ""),
            "importance": round(float(values[i]) / total, 4),
        }
        for i in order
        if values[i] > 0
    ]


def _dollar_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.clip(np.asarray(y_pred, dtype=float), 0, None)
    absolute_error = np.abs(y_true - y_pred)
    with np.errstate(divide="ignore", invalid="ignore"):
        relative = np.where(y_true > 0, absolute_error / y_true, np.nan)
    return {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(math.sqrt(mean_squared_error(y_true, y_pred))),
        "r2": float(r2_score(y_true, y_pred)),
        "median_ape": float(np.nanmedian(relative) * 100),
        "within_25_pct": float(np.nanmean(relative <= 0.25) * 100),
    }


def train_model(
    df: pd.DataFrame,
    test_size: float = TEST_SIZE,
    random_state: int = RANDOM_STATE,
) -> ModelArtifact:
    """Train, cross-validate, evaluate and package a revenue model."""
    data = modelling_subset(df)
    if len(data) < 200:
        raise ValueError(
            f"Not enough training rows ({len(data)}). Expected at least 200."
        )

    X = engineer_features(data)
    y_log = np.log1p(data["revenue"].to_numpy(dtype=float))

    X_train, X_test, y_train_log, y_test_log, _, revenue_test = train_test_split(
        X,
        y_log,
        data["revenue"].to_numpy(dtype=float),
        test_size=test_size,
        random_state=random_state,
    )

    cv = KFold(n_splits=N_FOLDS, shuffle=True, random_state=random_state)
    candidates: dict[str, float] = {}
    fitted: dict[str, Pipeline] = {}
    for name, regressor in _candidate_models().items():
        pipeline = _build_pipeline(regressor)
        scores = cross_val_score(
            pipeline,
            X_train,
            y_train_log,
            cv=cv,
            scoring="neg_root_mean_squared_error",
            n_jobs=-1,
        )
        candidates[name] = float(-scores.mean())
        pipeline.fit(X_train, y_train_log)
        fitted[name] = pipeline

    best_name = min(candidates, key=candidates.get)
    best = fitted[best_name]

    log_predictions = best.predict(X_test)
    predictions = np.clip(np.expm1(log_predictions), 0, None)
    metrics = _dollar_metrics(revenue_test, predictions)
    metrics["r2_log"] = float(r2_score(y_test_log, log_predictions))
    metrics["cv_rmse_log"] = round(candidates[best_name], 4)

    return ModelArtifact(
        pipeline=best,
        metrics={key: round(value, 4) for key, value in metrics.items()},
        importances=_feature_importances(best, X_test, y_test_log),
        model_name=best_name,
        residual_std_log=float(np.std(y_test_log - log_predictions, ddof=1)),
        n_train=len(X_train),
        n_test=len(X_test),
        trained_at=datetime.now(UTC).isoformat(timespec="seconds"),
        candidates={name: round(score, 4) for name, score in candidates.items()},
        feature_columns=FEATURE_COLUMNS,
    )


def save_artifact(artifact: ModelArtifact, path: str | Path = MODEL_PATH) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(artifact.to_dict(), path)
    return path


def load_artifact(path: str | Path = MODEL_PATH) -> ModelArtifact:
    payload = joblib.load(Path(path))
    if not isinstance(payload, dict):
        raise ValueError("Unrecognised model artifact format.")
    return ModelArtifact.from_dict(payload)


def ensure_model(df: pd.DataFrame, path: str | Path = MODEL_PATH) -> tuple[ModelArtifact, bool]:
    """Load a saved artifact or train a fresh one. Returns ``(model, trained)``."""
    path = Path(path)
    if path.exists():
        try:
            artifact = load_artifact(path)
            if artifact.feature_columns == FEATURE_COLUMNS:
                return artifact, False
        except (OSError, ValueError, KeyError, TypeError):
            pass  # corrupt or outdated artifact: fall through and retrain
    artifact = train_model(df)
    save_artifact(artifact, path)
    return artifact, True


def feature_row(
    *,
    budget: float,
    popularity: float,
    runtime: float,
    vote_average: float,
    release_year: int,
    genres: list[str],
    companies: int,
    original_language: str,
) -> pd.DataFrame:
    """Build a one-row feature frame from raw user input."""
    record = pd.DataFrame(
        [
            {
                "budget": max(float(budget), 0.0),
                "popularity": max(float(popularity), 0.0),
                "runtime": max(float(runtime), 1.0),
                "vote_average": min(max(float(vote_average), 0.0), 10.0),
                "release_year": int(release_year),
                "genres": [list(genres)] if genres else [["Unknown"]],
                "primary_genre": genres[0] if genres else "Unknown",
                "companies": list(range(max(int(companies), 0))),
                "original_language": original_language or "en",
            }
        ]
    )
    return engineer_features(record)


INTERVAL_Z_SCORES = {0.80: 1.2816, 0.90: 1.6449, 0.95: 1.9600}


def predict_revenue(
    artifact: ModelArtifact,
    features: pd.DataFrame,
    interval: float = 0.80,
) -> dict[str, float]:
    """Predict revenue and an approximate prediction interval in dollars."""
    log_prediction = float(artifact.pipeline.predict(features)[0])
    z = INTERVAL_Z_SCORES.get(round(interval, 2), INTERVAL_Z)
    sigma = artifact.residual_std_log or 0.0
    return {
        "revenue": float(max(np.expm1(log_prediction), 0.0)),
        "low": float(max(np.expm1(log_prediction - z * sigma), 0.0)),
        "high": float(max(np.expm1(log_prediction + z * sigma), 0.0)),
        "log_prediction": log_prediction,
    }


def write_metrics_report(artifact: ModelArtifact, path: str | Path | None = None) -> Path:
    path = Path(path) if path else MODELS_DIR / "model_metrics.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    report = {key: value for key, value in artifact.to_dict().items() if key != "pipeline"}
    path.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    return path
