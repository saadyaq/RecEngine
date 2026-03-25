import itertools
import time
from pathlib import Path

import numpy as np
import mlflow
import pandas as pd
from loguru import logger

from src.config import settings
from src.models.collaborative import CollaborativeModel
from src.models.semantic import SemanticModel, build_product_texts
from src.training.evaluate import ndcg_at_k, precision_at_k, recall_at_k


def evaluate_model(
    model: CollaborativeModel, test_df: pd.DataFrame, ks: list[int] = None
) -> dict[str, float]:
    if ks is None:
        ks = [5, 10, 20]

    # Per-user evaluation
    user_metrics = {f"precision_at_{k}": [] for k in ks}
    user_metrics.update({f"recall_at_{k}": [] for k in ks})
    user_metrics.update({f"ndcg_at_{k}": [] for k in ks})

    for user_id, group in test_df.groupby("user_id"):
        relevant = group["parent_asin"].tolist()
        recs = model.recommend(user_id, n=max(ks), exclude_seen=True)
        recommended = [item_id for item_id, _ in recs]

        for k in ks:
            user_metrics[f"precision_at_{k}"].append(precision_at_k(recommended, relevant, k))
            user_metrics[f"recall_at_{k}"].append(recall_at_k(recommended, relevant, k))
            user_metrics[f"ndcg_at_{k}"].append(ndcg_at_k(recommended, relevant, k))

    # Average across users
    results = {}
    for metric_name, values in user_metrics.items():
        results[metric_name] = sum(values) / len(values) if values else 0.0

    return results


def evaluate_semantic_model(
    model: SemanticModel,
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    ks: list[int] = None,
) -> dict[str, float]:
    """Evaluate Model B with the same ranking metrics as Model A."""
    if ks is None:
        ks = [5, 10, 20]

    user_metrics = {f"precision_at_{k}": [] for k in ks}
    user_metrics.update({f"recall_at_{k}": [] for k in ks})
    user_metrics.update({f"ndcg_at_{k}": [] for k in ks})

    test_users = test_df["user_id"].unique()
    logger.info(f"Evaluating Model B on {len(test_users)} users...")

    for user_id in test_users:
        relevant = test_df[test_df["user_id"] == user_id]["parent_asin"].tolist()
        recs = model.recommend(user_id, train_df, n=max(ks))
        recommended = [item_id for item_id, _ in recs]

        for k in ks:
            user_metrics[f"precision_at_{k}"].append(precision_at_k(recommended, relevant, k))
            user_metrics[f"recall_at_{k}"].append(recall_at_k(recommended, relevant, k))
            user_metrics[f"ndcg_at_{k}"].append(ndcg_at_k(recommended, relevant, k))

    results = {}
    for metric_name, values in user_metrics.items():
        results[metric_name] = sum(values) / len(values) if values else 0.0

    return results


def train_semantic_and_log(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    metadata_df: pd.DataFrame,
    model_name: str = "all-MiniLM-L6-v2",
) -> tuple[SemanticModel, dict]:
    """Train Model B, evaluate, and log to MLflow."""
    mlflow.set_experiment("recengine-model-b")

    with mlflow.start_run(run_name="semantic-model"):
        mlflow.log_param("model_name", model_name)
        mlflow.log_param("model_type", "SemanticEmbeddings")

        # Build index
        start_time = time.time()
        product_texts = build_product_texts(metadata_df)
        mlflow.log_metric("num_products", len(product_texts))

        model = SemanticModel(model_name=model_name)
        model.build_index(product_texts)
        build_duration = time.time() - start_time
        mlflow.log_metric("build_duration_seconds", build_duration)

        # Evaluate
        metrics = evaluate_semantic_model(model, train_df, test_df)
        mlflow.log_metrics(metrics)

        logger.info(
            f"Model B built in {build_duration:.1f}s | "
            f"NDCG@10: {metrics.get('ndcg_at_10', 0):.4f}"
        )

        return model, metrics


def compare_models(
    metrics_a: dict[str, float],
    metrics_b: dict[str, float],
) -> pd.DataFrame:
    """Create a comparison table between Model A and Model B."""
    all_keys = sorted(set(metrics_a.keys()) | set(metrics_b.keys()))
    rows = []
    for key in all_keys:
        val_a = metrics_a.get(key, float("nan"))
        val_b = metrics_b.get(key, float("nan"))
        diff = val_b - val_a if not (np.isnan(val_a) or np.isnan(val_b)) else float("nan")
        rows.append({"metric": key, "model_a": val_a, "model_b": val_b, "diff_b_minus_a": diff})

    comparison = pd.DataFrame(rows)
    logger.info(f"\n{'='*60}\nModel A vs Model B comparison:\n{comparison.to_string(index=False)}")
    return comparison


def cold_start_analysis(
    model_b: SemanticModel,
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
) -> dict:
    """Analyze Model B performance on cold-start vs warm items."""
    train_items = set(train_df["parent_asin"].unique())
    test_items = set(test_df["parent_asin"].unique())

    cold_items = test_items - train_items
    warm_items = test_items & train_items

    # Split test set into cold-start and warm users/items
    test_cold = test_df[test_df["parent_asin"].isin(cold_items)]
    test_warm = test_df[test_df["parent_asin"].isin(warm_items)]

    results = {
        "total_test_items": len(test_items),
        "cold_start_items": len(cold_items),
        "warm_items": len(warm_items),
        "cold_start_ratio": len(cold_items) / len(test_items) if test_items else 0,
        "cold_start_interactions": len(test_cold),
        "warm_interactions": len(test_warm),
    }

    # Evaluate Model B on warm items
    if not test_warm.empty:
        warm_metrics = evaluate_semantic_model(model_b, train_df, test_warm)
        for k, v in warm_metrics.items():
            results[f"warm_{k}"] = v

    # For cold-start: Model B can still recommend via semantic similarity
    # but the user must have training data (liked items) for the recommendation to work
    if not test_cold.empty:
        cold_metrics = evaluate_semantic_model(model_b, train_df, test_cold)
        for k, v in cold_metrics.items():
            results[f"cold_{k}"] = v

    logger.info(
        f"\nCold Start Analysis:\n"
        f"  Total test items: {results['total_test_items']}\n"
        f"  Cold-start items: {results['cold_start_items']} "
        f"({results['cold_start_ratio']:.1%})\n"
        f"  Warm items: {results['warm_items']}"
    )

    return results


def train_and_log(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    n_factors: int = 100,
    n_epochs: int = 20,
    regularization: float = 0.01,
    alpha: float = 40.0,
) -> tuple[CollaborativeModel, dict]:
    mlflow.set_experiment("recengine-model-a")

    with mlflow.start_run():
        # Log hyperparameters
        mlflow.log_params(
            {
                "n_factors": n_factors,
                "n_epochs": n_epochs,
                "regularization": regularization,
                "alpha": alpha,
            }
        )

        # Train
        start_time = time.time()
        model = CollaborativeModel(
            n_factors=n_factors,
            n_epochs=n_epochs,
            regularization=regularization,
            alpha=alpha,
        )
        model.fit(train_df)
        train_duration = time.time() - start_time
        mlflow.log_metric("train_duration_seconds", train_duration)

        # RMSE on test set
        test_preds = []
        test_actuals = []
        for _, row in test_df.iterrows():
            preds = model.predict(row["user_id"], [row["parent_asin"]])
            test_preds.append(preds[0][1])
            test_actuals.append(row["rating"])
        rmse = float(np.sqrt(np.mean((np.array(test_preds) - np.array(test_actuals)) ** 2)))
        mlflow.log_metric("rmse", rmse)

        # Ranking metrics
        metrics = evaluate_model(model, test_df)
        mlflow.log_metrics(metrics)

        # Log model as artifact
        mlflow.log_param("model_type", "ALS")

        logger.info(
            f"Training done in {train_duration:.1f}s | RMSE: {rmse:.4f} | "
            f"NDCG@10: {metrics.get('ndcg_at_10', 0):.4f}"
        )

        return model, {**metrics, "rmse": rmse}


def hyperparameter_search(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
) -> tuple[CollaborativeModel, dict]:
    param_grid = {
        "n_factors": [50, 100, 200],
        "n_epochs": [15, 30],
        "regularization": [0.01, 0.1],
        "alpha": [20.0, 40.0, 80.0],
    }

    best_model = None
    best_metrics = None
    best_ndcg = -1.0

    keys = param_grid.keys()
    combinations = list(itertools.product(*param_grid.values()))
    logger.info(f"Starting hyperparameter search: {len(combinations)} combinations")

    for i, values in enumerate(combinations):
        params = dict(zip(keys, values))
        logger.info(f"[{i + 1}/{len(combinations)}] {params}")

        model, metrics = train_and_log(train_df, test_df, **params)

        if metrics["ndcg_at_10"] > best_ndcg:
            best_ndcg = metrics["ndcg_at_10"]
            best_model = model
            best_metrics = metrics

    logger.info(f"Best NDCG@10: {best_ndcg:.4f}")

    # Register best model
    mlflow.set_experiment("recengine-model-a")
    with mlflow.start_run(run_name="best-model-a"):
        mlflow.log_params(
            {
                "n_factors": best_model.n_factors,
                "n_epochs": best_model.n_epochs,
                "regularization": best_model.regularization,
                "alpha": best_model.alpha,
            }
        )
        mlflow.log_metrics(best_metrics)
        mlflow.set_tag("best_model", "true")

    return best_model, best_metrics


def run_training(tuning: bool = False, train_model_b: bool = False):
    data_dir = Path("data/processed")
    train_df = pd.read_parquet(data_dir / "train.parquet")
    test_df = pd.read_parquet(data_dir / "test.parquet")

    logger.info(f"Train: {len(train_df):,} | Test: {len(test_df):,}")

    # Use local file storage by default, remote server if configured
    tracking_uri = settings.MLFLOW_TRACKING_URI
    if tracking_uri.startswith("http"):
        # Check if server is reachable, fallback to local if not
        try:
            import requests

            requests.get(tracking_uri, timeout=2)
        except Exception:
            tracking_uri = "sqlite:///mlflow.db"
            logger.warning(f"MLflow server not reachable, using local storage: {tracking_uri}")
    mlflow.set_tracking_uri(tracking_uri)

    # Model A
    if tuning:
        model_a, metrics_a = hyperparameter_search(train_df, test_df)
    else:
        model_a, metrics_a = train_and_log(train_df, test_df)

    logger.info(f"Model A metrics: {metrics_a}")

    # Model B
    if train_model_b:
        metadata_path = data_dir / "metadata.parquet"
        if not metadata_path.exists():
            logger.warning("metadata.parquet not found, skipping Model B")
            return model_a, metrics_a

        metadata_df = pd.read_parquet(metadata_path)
        model_b, metrics_b = train_semantic_and_log(train_df, test_df, metadata_df)

        # Comparison
        comparison = compare_models(metrics_a, metrics_b)
        comparison.to_csv(data_dir / "model_comparison.csv", index=False)

        # Cold start analysis
        cold_results = cold_start_analysis(model_b, train_df, test_df)

        # Log comparison in MLflow
        mlflow.set_experiment("recengine-comparison")
        with mlflow.start_run(run_name="model-a-vs-b"):
            for k, v in metrics_a.items():
                mlflow.log_metric(f"model_a_{k}", v)
            for k, v in metrics_b.items():
                mlflow.log_metric(f"model_b_{k}", v)
            for k, v in cold_results.items():
                if isinstance(v, (int, float)):
                    mlflow.log_metric(f"cold_start_{k}", v)

        return (model_a, metrics_a), (model_b, metrics_b)

    return model_a, metrics_a


if __name__ == "__main__":
    import sys

    train_b = "--model-b" in sys.argv
    tuning = "--tuning" in sys.argv
    run_training(tuning=tuning, train_model_b=train_b)
