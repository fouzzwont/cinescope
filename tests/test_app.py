"""Smoke tests: every page renders without raising."""

from __future__ import annotations

from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

APP_PATH = Path(__file__).resolve().parent.parent / "app.py"


@pytest.fixture(scope="module")
def app() -> AppTest:
    instance = AppTest.from_file(str(APP_PATH), default_timeout=300)
    instance.run()
    return instance


def test_overview_renders(app: AppTest) -> None:
    assert not app.exception
    assert app.radio[0].options == [
        "Overview",
        "Explore catalogue",
        "Predict revenue",
        "Recommendations",
    ]
    assert app.metric  # KPI cards are present


@pytest.mark.parametrize(
    "page", ["Explore catalogue", "Predict revenue", "Recommendations"]
)
def test_each_page_renders(app: AppTest, page: str) -> None:
    app.radio[0].set_value(page).run()
    assert not app.exception


def test_prediction_form_returns_a_result(app: AppTest) -> None:
    app.radio[0].set_value("Predict revenue").run()
    assert not app.exception
    labels = [button.label for button in app.button]
    assert "Predict revenue" in labels

    app.button[0].click().run()
    assert not app.exception
    metrics = {metric.label for metric in app.metric}
    assert "Return" in metrics
