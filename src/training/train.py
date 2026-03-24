import itertools
import time
from pathlib import Path

import numpy as np
import mlflow
import pandas as pd
from loguru import logger

from src.config import settings
from src.models.collaborative import CollaborativeModel
from src.training.evaluate import ndcg_at_k, precision_at_k, recall_at_k


def evaluate_model(
    model: CollaborativeModel, test_df: pd.DataFrame, ks: list[int] = None
) -> dict[str, float]:
    if ks is None:
        ks = [5, 10, 20]

    # Per-user evaluation
    user_metrics = {f"precision@{k}": [] for k in ks}
    user_metrics.update({f"recall@{k}": [] for k in ks})
    user_metrics.update({f"ndcg@{k}": [] for k in ks})

    for user_id, group in test_df.groupby("user_id"):
        relevant = group["parent_asin"].tolist()
        recs = model.recommend(user_id, n=max(ks), exclude_seen=True)
        recommended = [item_id for item_id, _ in recs]

        for k in ks:
            user_metrics[f"precision@{k}"].append(precision_at_k(recommended, relevant, k))
            user_metrics[f"recall@{k}"].append(recall_at_k(recommended, relevant, k))
            user_metrics[f"ndcg@{k}"].append(ndcg_at_k(recommended, relevant, k))

    # Average across users
    results = {}
    for metric_name, values in user_metrics.items():
        results[metric_name] = sum(values) / len(values) if values else 0.0

    return results


def train_and_log(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    n_factors: int = 100,
    n_epochs: int = 20,
    lr_all: float = 0.005,
    reg_all: float = 0.02,
) -> tuple[CollaborativeModel, dict]:
    mlflow.set_experiment("recengine-model-a")

    with mlflow.start_run():
        # Log hyperparameters
        mlflow.log_params(
            {
                "n_factors": n_factors,
                "n_epochs": n_epochs,
                "lr_all": lr_all,
                "reg_all": reg_all,
            }
        )

        # Train
        start_time = time.time()
        model = CollaborativeModel(
            n_factors=n_factors,
            n_epochs=n_epochs,
            lr_all=lr_all,
            reg_all=reg_all,
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
        mlflow.log_param("model_type", "SVD")

        logger.info(
            f"Training done in {train_duration:.1f}s | RMSE: {rmse:.4f} | "
            f"NDCG@10: {metrics.get('ndcg@10', 0):.4f}"
        )

        return model, {**metrics, "rmse": rmse}


def hyperparameter_search(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
) -> tuple[CollaborativeModel, dict]:
    param_grid = {
        "n_factors": [50, 100, 150],
        "n_epochs": [20, 30],
        "lr_all": [0.005, 0.01],
        "reg_all": [0.02, 0.05],
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

        if metrics["ndcg@10"] > best_ndcg:
            best_ndcg = metrics["ndcg@10"]
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
                "lr_all": best_model.lr_all,
                "reg_all": best_model.reg_all,
            }
        )
        mlflow.log_metrics(best_metrics)
        mlflow.set_tag("best_model", "true")

    return best_model, best_metrics


def run_training(tuning: bool = False):
    data_dir = Path("data/processed")
    train_df = pd.read_parquet(data_dir / "train.parquet")
    test_df = pd.read_parquet(data_dir / "test.parquet")

    logger.info(f"Train: {len(train_df):,} | Test: {len(test_df):,}")

    mlflow.set_tracking_uri(settings.MLFLOW_TRACKING_URI)

    if tuning:
        model, metrics = hyperparameter_search(train_df, test_df)
    else:
        model, metrics = train_and_log(train_df, test_df)

    logger.info(f"Final metrics: {metrics}")
    return model, metrics


if __name__ == "__main__":
    run_training(tuning=False)
