import json
from unittest.mock import MagicMock, patch


from src.config import settings
from src.training.promote import (
    auto_promote,
    check_canary_performance,
    compare_models,
    get_model_metrics,
    load_deployment_state,
    promote_canary_to_production,
    promote_to_canary,
    promote_to_staging,
    rollback_to_previous,
    save_deployment_state,
)


class TestModelMetrics:
    def test_get_model_metrics(self):
        with patch("mlflow.tracking.MlflowClient") as mock_client:
            mock_run = MagicMock()
            mock_run.data.metrics = {
                "ndcg_at_10": 0.25,
                "precision_at_10": 0.18,
                "recall_at_10": 0.32,
            }
            mock_client.return_value.get_run.return_value = mock_run

            metrics = get_model_metrics("test_run_id")

            assert metrics["ndcg_at_10"] == 0.25
            assert metrics["precision_at_10"] == 0.18
            assert "recall_at_10" in metrics

    def test_get_model_metrics_specific_names(self):
        with patch("mlflow.tracking.MlflowClient") as mock_client:
            mock_run = MagicMock()
            mock_run.data.metrics = {
                "ndcg_at_10": 0.25,
                "auc_roc": 0.78,
                "other_metric": 0.50,
            }
            mock_client.return_value.get_run.return_value = mock_run

            metrics = get_model_metrics("test_run_id", metric_names=["ndcg_at_10", "auc_roc"])

            assert len(metrics) == 2
            assert "ndcg_at_10" in metrics
            assert "auc_roc" in metrics
            assert "other_metric" not in metrics


class TestModelComparison:
    def test_compare_models_better_candidate(self):
        candidate = {"ndcg_at_10": 0.30, "precision_at_10": 0.25}
        production = {"ndcg_at_10": 0.25, "precision_at_10": 0.20}

        should_promote, report = compare_models(candidate, production, improvement_threshold=0.01)

        assert should_promote is True
        assert report["improved_metrics"] == ["ndcg_at_10", "precision_at_10"]
        assert len(report["degraded_metrics"]) == 0

    def test_compare_models_worse_candidate(self):
        candidate = {"ndcg_at_10": 0.20, "precision_at_10": 0.15}
        production = {"ndcg_at_10": 0.25, "precision_at_10": 0.20}

        should_promote, report = compare_models(candidate, production)

        assert should_promote is False
        assert len(report["improved_metrics"]) == 0
        assert report["degraded_metrics"] == ["ndcg_at_10", "precision_at_10"]

    def test_compare_models_mixed_results(self):
        candidate = {"ndcg_at_10": 0.30, "precision_at_10": 0.15}
        production = {"ndcg_at_10": 0.25, "precision_at_10": 0.20}

        should_promote, report = compare_models(candidate, production)

        assert should_promote is True
        assert "ndcg_at_10" in report["improved_metrics"]
        assert "precision_at_10" in report["degraded_metrics"]

    def test_compare_models_no_common_metrics(self):
        candidate = {"metric_a": 0.30}
        production = {"metric_b": 0.25}

        should_promote, report = compare_models(candidate, production)

        assert should_promote is False
        assert report["reason"] == "no_common_metrics"

    def test_compare_models_significant_improvement(self):
        candidate = {"ndcg_at_10": 0.35, "precision_at_10": 0.30}
        production = {"ndcg_at_10": 0.25, "precision_at_10": 0.20}

        should_promote, report = compare_models(candidate, production, improvement_threshold=0.10)

        assert should_promote is True
        assert len(report["significant_improvements"]) >= 1


class TestDeploymentState:
    def test_save_and_load_deployment_state(self, tmp_path):
        with patch.object(settings, "DATA_PROCESSED_DIR", tmp_path):
            save_deployment_state("model-a", "production", "v1", "run123")

            state_file = tmp_path / "deployment_state.json"
            assert state_file.exists()

            state = load_deployment_state("model-a")
            assert state is not None
            assert state["stage"] == "production"
            assert state["version"] == "v1"
            assert state["run_id"] == "run123"

    def test_load_nonexistent_deployment_state(self):
        state = load_deployment_state("nonexistent-model")
        assert state is None

    def test_save_multiple_deployment_states(self, tmp_path):
        with patch.object(settings, "DATA_PROCESSED_DIR", tmp_path):
            save_deployment_state("model-a", "production", "v1", "run123")
            save_deployment_state("model-c", "staging", "v2", "run456")

            state_a = load_deployment_state("model-a")
            state_c = load_deployment_state("model-c")

            assert state_a["version"] == "v1"
            assert state_c["version"] == "v2"


class TestAutoPromote:
    def test_auto_promote_no_production_model(self):
        with patch("src.training.promote.get_production_model", return_value=None):
            with patch("src.training.promote.promote_to_staging") as mock_promote:
                mock_promote.return_value = True

                result = auto_promote("model-a", "test_run_id")

                assert result["action"] == "promoted_to_staging"
                assert result["reason"] == "no_production_model"
                assert result["success"] is True

    def test_auto_promote_candidate_not_better(self):
        prod_metrics = {"ndcg_at_10": 0.30, "precision_at_10": 0.25}

        with patch(
            "src.training.promote.get_production_model",
            return_value=("run1", prod_metrics),
        ):
            with patch(
                "src.training.promote.get_model_metrics",
                return_value={"ndcg_at_10": 0.20},
            ):
                result = auto_promote("model-a", "test_run_id")

                assert result["action"] == "rejected"
                assert result["reason"] == "no_significant_improvement"

    def test_auto_promote_promotes_to_staging(self):
        prod_metrics = {"ndcg_at_10": 0.25, "precision_at_10": 0.20}
        cand_metrics = {"ndcg_at_10": 0.30, "precision_at_10": 0.25}

        with patch(
            "src.training.promote.get_production_model",
            return_value=("run1", prod_metrics),
        ):
            with patch("src.training.promote.get_model_metrics", return_value=cand_metrics):
                with patch("src.training.promote.promote_to_staging", return_value=True):
                    result = auto_promote("model-a", "test_run_id")

                    assert result["action"] in [
                        "promoted_to_staging",
                        "promoted_to_canary",
                    ]
                    assert result["success"] is True


class TestCanaryPerformance:
    def test_check_canary_no_feedback_data(self, tmp_path):
        result = check_canary_performance("model-a", feedback_log=tmp_path / "nonexistent.jsonl")

        assert result["status"] == "no_data"
        assert result["proceed_to_production"] is True

    def test_check_canary_insufficient_data(self, tmp_path):
        feedback_file = tmp_path / "feedback.jsonl"
        feedback_file.parent.mkdir(parents=True, exist_ok=True)

        with open(feedback_file, "w") as f:
            for i in range(50):
                f.write(json.dumps({"variant": "canary", "action": "click"}) + "\n")

        result = check_canary_performance("model-a", feedback_log=feedback_file)

        assert result["status"] == "insufficient_data"
        assert result["proceed_to_production"] is False

    def test_check_canary_high_error_rate(self, tmp_path):
        feedback_file = tmp_path / "feedback.jsonl"
        feedback_file.parent.mkdir(parents=True, exist_ok=True)

        with open(feedback_file, "w") as f:
            for i in range(150):
                action = "error" if i < 20 else "click"
                f.write(json.dumps({"variant": "canary", "action": action}) + "\n")

        result = check_canary_performance(
            "model-a", feedback_log=feedback_file, error_threshold=0.05
        )

        assert result["status"] == "high_error_rate"
        assert result["proceed_to_production"] is False

    def test_check_canary_healthy(self, tmp_path):
        feedback_file = tmp_path / "feedback.jsonl"
        feedback_file.parent.mkdir(parents=True, exist_ok=True)

        with open(feedback_file, "w") as f:
            for i in range(150):
                action = "error" if i < 2 else "click"
                f.write(json.dumps({"variant": "canary", "action": action}) + "\n")

        result = check_canary_performance(
            "model-a", feedback_log=feedback_file, error_threshold=0.05
        )

        assert result["status"] == "healthy"
        assert result["proceed_to_production"] is True


class TestPromotionWorkflow:
    def test_promote_to_staging_mock(self):
        with patch("mlflow.tracking.MlflowClient") as mock_client:
            mock_model = MagicMock()
            mock_model.version = 1
            mock_client.return_value.register_model.return_value = mock_model

            result = promote_to_staging("model-a", "test_run_id")

            assert result is True
            mock_client.return_value.register_model.assert_called_once()

    def test_promote_to_canary_no_staging_model(self):
        with patch("src.training.promote.get_staging_model", return_value=None):
            result = promote_to_canary("model-a")
            assert result is False

    def test_rollback_no_archived_model(self):
        with patch("mlflow.tracking.MlflowClient") as mock_client:
            mock_client.return_value.get_latest_versions.return_value = []

            result = rollback_to_previous("model-a")

            assert result is False

    def test_promote_canary_to_production_unhealthy(self, tmp_path):
        feedback_file = tmp_path / "feedback.jsonl"
        feedback_file.parent.mkdir(parents=True, exist_ok=True)

        with open(feedback_file, "w") as f:
            for i in range(150):
                action = "error" if i < 30 else "click"
                f.write(json.dumps({"variant": "canary", "action": action}) + "\n")

        with patch.object(settings, "DATA_PROCESSED_DIR", tmp_path):
            result = promote_canary_to_production("model-a")
            assert result is False
