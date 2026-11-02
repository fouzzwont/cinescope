"""Tests for revenue model training and inference."""

from __future__ import annotations

import numpy as np

from movie_intelligence.modelling import (
    feature_row,
    load_artifact,
    predict_revenue,
    save_artifact,
    train_model,
)


def test_training_produces_sane_metrics(training_frame) -> None:
    artifact = train_model(training_frame)
    assert artifact.model_name in {"Ridge regression", "Random forest", "Gradient boosting"}
    assert artifact.n_train > 0 and artifact.n_test > 0
    assert -1 <= artifact.metrics["r2"] <= 1
    assert artifact.metrics["mae"] > 0
    assert artifact.residual_std_log > 0
    assert artifact.importances


def test_prediction_interval_is_ordered(training_frame) -> None:
    artifact = train_model(training_frame)
    features = feature_row(
        budget=80_000_000,
        popularity=40,
        runtime=120,
        vote_average=7.0,
        release_year=2015,
        genres=["Action"],
        companies=3,
        original_language="en",
    )
    result = predict_revenue(artifact, features)
    assert 0 <= result["low"] <= result["revenue"] <= result["high"]
    assert np.isfinite(result["revenue"])


def test_save_and_load_roundtrip(training_frame, tmp_path) -> None:
    artifact = train_model(training_frame)
    path = save_artifact(artifact, tmp_path / "model.joblib")
    restored = load_artifact(path)

    features = feature_row(
        budget=50_000_000,
        popularity=25,
        runtime=110,
        vote_average=6.5,
        release_year=2012,
        genres=["Drama"],
        companies=2,
        original_language="en",
    )
    assert restored.model_name == artifact.model_name
    assert restored.metrics == artifact.metrics
    assert predict_revenue(restored, features)["revenue"] == predict_revenue(artifact, features)["revenue"]
