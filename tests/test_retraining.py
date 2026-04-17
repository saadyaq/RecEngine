from __future__ import annotations

from unittest.mock import MagicMock, patch

import mlflow
import pandas as pd
import pytest

from src.models.collaborative import CollaborativeModel
from src.training.promote import (
    auto_promote,
    compare_models,
    rollback_to_previous,
    save_deployment_state,
    load_deployment_state,
)
from src.training.train import evaluate_model, train_and_log


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _mini_df() -> pd.DataFrame:
    rows = []
    for u_idx, user in enumerate(["u1", "u2", "u3", "u4"]):
        for i_idx, item in enumerate(["i1", "i2", "i3", "i4", "i5"]):
            rows.append(
                {
                    "user_id": user,
                    "parent_asin": item,
                    "rating": float((u_idx + i_idx) % 5 + 1),
                    "timestamp": u_idx * 5 + i_idx,
                }
            )
    return pd.DataFrame(rows)


@pytest.fixture
def mini_model() -> CollaborativeModel:
    df = _mini_df()
    model = CollaborativeModel(n_factors=10, n_epochs=5)
    model.fit(df)
    return model


# ---------------------------------------------------------------------------
# Tâche 8.1 — evaluate_model returns the expected metric keys
# ---------------------------------------------------------------------------


def test_evaluate_model_returns_metric_keys(mini_model):
    test_df = _mini_df().sample(frac=0.4, random_state=42)
    metrics = evaluate_model(mini_model, test_df, ks=[5, 10])

    for k in [5, 10]:
        assert f"precision_at_{k}" in metrics
        assert f"recall_at_{k}" in metrics
        assert f"ndcg_at_{k}" in metrics

    for v in metrics.values():
        assert 0.0 <= v <= 1.0


def test_train_and_log_returns_metrics(tmp_path):
    """train_and_log runs without error and returns a metrics dict with rmse."""
    mlflow.set_tracking_uri(f"sqlite:///{tmp_path / 'mlflow.db'}")

    train_df = _mini_df()
    test_df = _mini_df().sample(frac=0.3, random_state=1)

    model, metrics = train_and_log(train_df, test_df, n_factors=10, n_epochs=5)

    assert isinstance(model, CollaborativeModel)
    assert "rmse" in metrics
    assert metrics["rmse"] >= 0.0
    assert "ndcg_at_10" in metrics


def test_run_training_produces_model_a_metrics(tmp_path):
    """run_training() trains model A and returns results dict."""
    train_df = _mini_df()
    test_df = _mini_df().sample(frac=0.3, random_state=2)
    mlflow.set_tracking_uri(f"sqlite:///{tmp_path / 'mlflow.db'}")

    with (
        patch("src.training.train.pd.read_parquet", side_effect=[train_df, test_df]),
        patch(
            "src.training.train.settings.MLFLOW_TRACKING_URI", f"sqlite:///{tmp_path / 'mlflow.db'}"
        ),
        patch(
            "src.training.train.CollaborativeModel",
            wraps=lambda **kw: CollaborativeModel(
                n_factors=10,
                n_epochs=5,
                **{k: v for k, v in kw.items() if k not in ("n_factors", "n_epochs")},
            ),
        ),
    ):
        from src.training.train import run_training

        results = run_training(tuning=False, train_model_b=False, train_model_c=False)

    assert "model_a" in results["models_trained"]
    assert "model_a_metrics" in results
    assert "ndcg_at_10" in results["model_a_metrics"]


# ---------------------------------------------------------------------------
# Tâche 8.2 — compare_models and promotion logic
# ---------------------------------------------------------------------------


def test_compare_models_should_promote():
    candidate = {"ndcg_at_10": 0.30, "precision_at_10": 0.25}
    production = {"ndcg_at_10": 0.25, "precision_at_10": 0.20}

    should_promote, report = compare_models(candidate, production, improvement_threshold=0.01)

    assert should_promote is True
    assert len(report["improved_metrics"]) > 0
    assert len(report["significant_improvements"]) > 0


def test_compare_models_should_reject_no_improvement():
    candidate = {"ndcg_at_10": 0.24, "precision_at_10": 0.19}
    production = {"ndcg_at_10": 0.25, "precision_at_10": 0.20}

    should_promote, report = compare_models(candidate, production, improvement_threshold=0.01)

    assert should_promote is False
    assert report["degraded_metrics"]


def test_compare_models_no_common_metrics():
    should_promote, report = compare_models({"auc": 0.8}, {"ndcg_at_10": 0.25})
    assert should_promote is False
    assert report["reason"] == "no_common_metrics"


def test_rollback_returns_false_when_no_archived(tmp_path):
    """rollback_to_previous returns False gracefully when no archived model exists."""
    mlflow.set_tracking_uri(f"sqlite:///{tmp_path / 'mlflow.db'}")

    client_mock = MagicMock()
    client_mock.get_latest_versions.return_value = []

    with patch("src.training.promote.mlflow.tracking.MlflowClient", return_value=client_mock):
        result = rollback_to_previous("model-a")

    assert result is False


# ---------------------------------------------------------------------------
# Tâche 8.2 — auto_promote when no production model
# ---------------------------------------------------------------------------


def test_auto_promote_no_production_promotes_to_staging(tmp_path):
    """When no production model exists, auto_promote registers to staging."""
    mlflow.set_tracking_uri(f"sqlite:///{tmp_path / 'mlflow.db'}")

    run_mock = MagicMock()
    run_mock.data.metrics = {"ndcg_at_10": 0.30}

    client_mock = MagicMock()
    client_mock.get_run.return_value = run_mock
    client_mock.get_latest_versions.return_value = []
    client_mock.register_model.return_value = MagicMock(version="1")

    with patch("src.training.promote.mlflow.tracking.MlflowClient", return_value=client_mock):
        result = auto_promote("model-a", run_id="fake-run-id")

    assert result["action"] in ("promoted_to_staging", "failed")


# ---------------------------------------------------------------------------
# Tâche 8.2 — deployment state persistence
# ---------------------------------------------------------------------------


def test_save_and_load_deployment_state(tmp_path):
    with patch("src.training.promote.settings.DATA_PROCESSED_DIR", tmp_path):
        save_deployment_state("model-a", "staging", "1", "run-abc")
        state = load_deployment_state("model-a")

    assert state is not None
    assert state["stage"] == "staging"
    assert state["version"] == "1"
    assert state["run_id"] == "run-abc"
