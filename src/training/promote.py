import json
import time
from pathlib import Path

import mlflow
import pandas as pd
from loguru import logger

from src.config import settings


def get_model_metrics(run_id: str, metric_names: list[str] = None) -> dict[str, float]:
    """Retrieve metrics from an MLflow run."""
    if metric_names is None:
        metric_names = ["ndcg_at_10", "precision_at_10", "recall_at_10", "auc_roc"]

    client = mlflow.tracking.MlflowClient()
    run = client.get_run(run_id)
    metrics = {}
    for name in metric_names:
        if name in run.data.metrics:
            metrics[name] = run.data.metrics[name]
    return metrics


def get_production_model(experiment_name: str) -> tuple[str, dict] | None:
    """Get the current production model from MLflow Model Registry."""
    client = mlflow.tracking.MlflowClient()

    try:
        model_versions = client.get_latest_versions(
            f"recengine-{experiment_name}", stages=["Production"]
        )
        if not model_versions:
            logger.warning(f"No production model found for {experiment_name}")
            return None

        version = model_versions[0]
        run_id = version.run_id
        metrics = get_model_metrics(run_id)
        logger.info(
            f"Production model: {experiment_name} v{version.version} "
            f"(run: {run_id[:8]}) | metrics: {metrics}"
        )
        return run_id, metrics
    except Exception as e:
        logger.warning(f"Could not get production model: {e}")
        return None


def get_staging_model(experiment_name: str) -> tuple[str, dict] | None:
    """Get the latest staging model from MLflow."""
    client = mlflow.tracking.MlflowClient()

    try:
        model_versions = client.get_latest_versions(
            f"recengine-{experiment_name}", stages=["Staging"]
        )
        if not model_versions:
            logger.info(f"No staging model found for {experiment_name}")
            return None

        version = model_versions[0]
        run_id = version.run_id
        metrics = get_model_metrics(run_id)
        logger.info(
            f"Staging model: {experiment_name} v{version.version} "
            f"(run: {run_id[:8]}) | metrics: {metrics}"
        )
        return run_id, metrics
    except Exception as e:
        logger.warning(f"Could not get staging model: {e}")
        return None


def compare_models(
    candidate_metrics: dict[str, float],
    production_metrics: dict[str, float],
    improvement_threshold: float = 0.01,
) -> tuple[bool, dict]:
    """
    Compare candidate vs production model.

    Returns (should_promote, comparison_report).
    """
    common_metrics = set(candidate_metrics.keys()) & set(production_metrics.keys())
    if not common_metrics:
        logger.warning("No common metrics to compare")
        return False, {"reason": "no_common_metrics"}

    improvements = {}
    for metric in common_metrics:
        cand_val = candidate_metrics[metric]
        prod_val = production_metrics[metric]
        rel_improvement = (cand_val - prod_val) / prod_val if prod_val != 0 else 0
        improvements[metric] = {
            "candidate": cand_val,
            "production": prod_val,
            "absolute_diff": cand_val - prod_val,
            "relative_diff": rel_improvement,
            "improved": cand_val > prod_val,
        }

    improved_metrics = [m for m, v in improvements.items() if v["improved"]]
    degraded_metrics = [m for m, v in improvements.items() if not v["improved"]]

    significant_improvements = [
        m
        for m, v in improvements.items()
        if v["improved"] and v["relative_diff"] >= improvement_threshold
    ]

    should_promote = (
        len(significant_improvements) > 0 and len(degraded_metrics) < len(common_metrics) / 2
    )

    report = {
        "should_promote": should_promote,
        "improved_metrics": improved_metrics,
        "degraded_metrics": degraded_metrics,
        "significant_improvements": significant_improvements,
        "improvements": improvements,
    }

    logger.info(
        f"Comparison: {len(improved_metrics)} improved, "
        f"{len(degraded_metrics)} degraded, "
        f"{len(significant_improvements)} significant"
    )

    return should_promote, report


def promote_to_staging(
    experiment_name: str,
    run_id: str,
    model_name: str = None,
) -> bool:
    """
    Register a model as 'Staging' in MLflow Model Registry.
    """
    if model_name is None:
        model_name = f"recengine-{experiment_name}"

    client = mlflow.tracking.MlflowClient()

    try:
        result = client.register_model(model_uri=f"runs:/{run_id}/model", name=model_name)
        logger.info(f"Model registered: {model_name} v{result.version}")

        client.transition_model_version_stage(
            name=model_name, version=result.version, stage="Staging"
        )
        logger.info(f"Model {model_name} v{result.version} promoted to Staging")
        return True
    except Exception as e:
        logger.error(f"Failed to promote to staging: {e}")
        return False


def promote_to_canary(
    experiment_name: str,
    model_name: str = None,
) -> bool:
    """
    Promote staging model to Canary (5% traffic).
    In production, this would update the deployment configuration.
    """
    if model_name is None:
        model_name = f"recengine-{experiment_name}"

    staging_model = get_staging_model(experiment_name)
    if not staging_model:
        logger.warning(f"No staging model to promote to canary for {experiment_name}")
        return False

    run_id, metrics = staging_model
    logger.info(f"Promoting {model_name} to Canary (5% traffic)")

    client = mlflow.tracking.MlflowClient()
    try:
        latest = client.get_latest_versions(model_name, stages=["Staging"])
        if latest:
            version = latest[0].version
            client.transition_model_version_stage(name=model_name, version=version, stage="Staging")
            logger.info(f"Model {model_name} v{version} is now in Canary")

            save_deployment_state(experiment_name, "canary", version, run_id)
            return True
    except Exception as e:
        logger.error(f"Failed to promote to canary: {e}")
        return False

    return False


def promote_to_production(
    experiment_name: str,
    model_name: str = None,
    archive_current: bool = True,
) -> bool:
    """
    Promote staging model to Production (100% traffic).
    """
    if model_name is None:
        model_name = f"recengine-{experiment_name}"

    staging_model = get_staging_model(experiment_name)
    if not staging_model:
        logger.warning(f"No staging model to promote to production for {experiment_name}")
        return False

    run_id, metrics = staging_model
    logger.info(f"Promoting {model_name} to Production")

    client = mlflow.tracking.MlflowClient()
    try:
        latest = client.get_latest_versions(model_name, stages=["Staging"])
        if not latest:
            logger.error(f"No staging version found for {model_name}")
            return False

        version = latest[0].version

        if archive_current:
            prod_versions = client.get_latest_versions(model_name, stages=["Production"])
            if prod_versions:
                old_version = prod_versions[0].version
                client.transition_model_version_stage(
                    name=model_name, version=old_version, stage="Archived"
                )
                logger.info(f"Archived previous production model v{old_version}")

        client.transition_model_version_stage(name=model_name, version=version, stage="Production")
        logger.info(f"Model {model_name} v{version} is now in Production")

        save_deployment_state(experiment_name, "production", version, run_id)
        return True
    except Exception as e:
        logger.error(f"Failed to promote to production: {e}")
        return False

    return False


def rollback_to_previous(experiment_name: str, model_name: str = None) -> bool:
    """
    Rollback to the previous production model.
    """
    if model_name is None:
        model_name = f"recengine-{experiment_name}"

    client = mlflow.tracking.MlflowClient()

    try:
        archived_versions = client.get_latest_versions(model_name, stages=["Archived"])
        if not archived_versions:
            logger.error(f"No archived model to rollback to for {model_name}")
            return False

        version = archived_versions[0].version
        logger.info(f"Rolling back to {model_name} v{version}")

        current_prod = client.get_latest_versions(model_name, stages=["Production"])
        if current_prod:
            client.transition_model_version_stage(
                name=model_name, version=current_prod[0].version, stage="Archived"
            )

        client.transition_model_version_stage(name=model_name, version=version, stage="Production")
        logger.info(f"Rolled back to {model_name} v{version}")

        save_deployment_state(experiment_name, "production", version, "rollback")
        return True
    except Exception as e:
        logger.error(f"Failed to rollback: {e}")
        return False


def save_deployment_state(experiment_name: str, stage: str, version: str, run_id: str) -> None:
    """Save deployment state to a JSON file for the serving layer."""
    state_file = settings.DATA_PROCESSED_DIR / "deployment_state.json"
    state_file.parent.mkdir(parents=True, exist_ok=True)

    state = {
        experiment_name: {
            "stage": stage,
            "version": version,
            "run_id": run_id,
            "updated_at": time.time(),
        }
    }

    existing_state = {}
    if state_file.exists():
        with open(state_file) as f:
            existing_state = json.load(f)

    existing_state.update(state)

    with open(state_file, "w") as f:
        json.dump(existing_state, f, indent=2)

    logger.info(f"Deployment state saved to {state_file}")


def load_deployment_state(experiment_name: str) -> dict | None:
    """Load deployment state from JSON file."""
    state_file = settings.DATA_PROCESSED_DIR / "deployment_state.json"
    if not state_file.exists():
        return None

    with open(state_file) as f:
        state = json.load(f)

    return state.get(experiment_name)


def auto_promote(
    experiment_name: str,
    run_id: str,
    improvement_threshold: float = 0.01,
) -> dict:
    """
    Automatic promotion pipeline:
    1. Compare candidate vs production
    2. If improved: promote to staging
    3. If significantly improved: promote to canary

    Returns promotion report.
    """
    logger.info(f"Starting auto-promotion for {experiment_name} (run: {run_id})")

    candidate_metrics = get_model_metrics(run_id)
    production_model = get_production_model(experiment_name)

    if not production_model:
        logger.info("No production model, promoting directly to staging")
        promoted = promote_to_staging(experiment_name, run_id)
        return {
            "action": "promoted_to_staging",
            "reason": "no_production_model",
            "success": promoted,
        }

    _, prod_metrics = production_model
    should_promote, comparison = compare_models(
        candidate_metrics, prod_metrics, improvement_threshold
    )

    if not should_promote:
        logger.info(
            f"Candidate not better than production. "
            f"Improved: {comparison['improved_metrics']}, "
            f"Degraded: {comparison['degraded_metrics']}"
        )
        return {
            "action": "rejected",
            "reason": "no_significant_improvement",
            "comparison": comparison,
        }

    promoted_staging = promote_to_staging(experiment_name, run_id)
    if not promoted_staging:
        return {
            "action": "failed",
            "reason": "staging_promotion_failed",
            "comparison": comparison,
        }

    if len(comparison["significant_improvements"]) >= 2:
        promoted_canary = promote_to_canary(experiment_name)
        return {
            "action": "promoted_to_canary",
            "reason": "significant_improvement",
            "comparison": comparison,
            "success": promoted_canary,
        }

    return {
        "action": "promoted_to_staging",
        "reason": "moderate_improvement",
        "comparison": comparison,
        "success": True,
    }


def check_canary_performance(
    experiment_name: str,
    feedback_log: Path = None,
    error_threshold: float = 0.05,
) -> dict:
    """
    Check if canary model is performing well.
    In production, this would analyze real-time metrics.
    """
    if feedback_log is None:
        feedback_log = settings.DATA_PROCESSED_DIR / "feedback.jsonl"

    if not feedback_log.exists():
        logger.info("No feedback data available for canary evaluation")
        return {"status": "no_data", "proceed_to_production": True}

    feedback_df = pd.read_json(feedback_log, lines=True)

    canary_feedback = feedback_df[feedback_df.get("variant") == "canary"]
    if len(canary_feedback) < 100:
        logger.info(f"Insufficient canary feedback ({len(canary_feedback)} samples)")
        return {"status": "insufficient_data", "proceed_to_production": False}

    error_rate = (canary_feedback.get("action") == "error").mean()

    if error_rate > error_threshold:
        logger.warning(f"Canary error rate {error_rate:.2%} > threshold {error_threshold:.2%}")
        return {
            "status": "high_error_rate",
            "error_rate": error_rate,
            "proceed_to_production": False,
        }

    logger.info(f"Canary performing well: error rate {error_rate:.2%} < {error_threshold:.2%}")
    return {
        "status": "healthy",
        "error_rate": error_rate,
        "proceed_to_production": True,
    }


def promote_canary_to_production(experiment_name: str) -> bool:
    """
    Promote canary to production after validation period.
    """
    canary_check = check_canary_performance(experiment_name)

    if not canary_check.get("proceed_to_production", False):
        logger.warning(f"Canary validation failed: {canary_check.get('status')}")
        return False

    return promote_to_production(experiment_name)


if __name__ == "__main__":
    import sys

    mlflow.set_tracking_uri(settings.MLFLOW_TRACKING_URI)

    if len(sys.argv) < 3:
        print("Usage: python -m src.training.promote <experiment> <run_id> [--canary|--production]")
        print("  experiment: model-a, model-b, model-c")
        print("  run_id: MLflow run ID")
        print("  --canary: promote to canary (5% traffic)")
        print("  --production: promote directly to production")
        sys.exit(1)

    experiment = sys.argv[1]
    run_id = sys.argv[2]
    mode = "staging"

    if "--canary" in sys.argv:
        mode = "canary"
    elif "--production" in sys.argv:
        mode = "production"

    logger.info(f"Promoting {experiment} run {run_id} to {mode}")

    if mode == "staging":
        result = auto_promote(experiment, run_id)
    elif mode == "canary":
        promote_to_staging(experiment, run_id)
        result = {
            "action": "promoted_to_canary",
            "success": promote_to_canary(experiment),
        }
    else:
        promote_to_staging(experiment, run_id)
        result = {
            "action": "promoted_to_production",
            "success": promote_to_production(experiment),
        }

    logger.info(f"Promotion result: {result}")
