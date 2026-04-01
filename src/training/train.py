import itertools
import time
from pathlib import Path

import numpy as np
import mlflow
import pandas as pd
from loguru import logger
from sklearn.metrics import roc_auc_score, log_loss
from sklearn.model_selection import train_test_split

from src.config import settings
from src.models.collaborative import CollaborativeModel
from src.models.semantic import SemanticModel, build_product_texts
from src.models.ctr import CTRModel, build_ctr_dataset
from src.models.features import build_user_features, build_item_features
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
    sample_users: int = 1000,
) -> dict[str, float]:
    """Evaluate Model B with the same ranking metrics as Model A."""
    if ks is None:
        ks = [5, 10, 20]

    user_metrics = {f"precision_at_{k}": [] for k in ks}
    user_metrics.update({f"recall_at_{k}": [] for k in ks})
    user_metrics.update({f"ndcg_at_{k}": [] for k in ks})

    test_users = test_df["user_id"].unique()
    if sample_users and len(test_users) > sample_users:
        rng = np.random.default_rng(42)
        test_users = rng.choice(test_users, size=sample_users, replace=False)
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

        # Build index — filter metadata to only items present in train
        train_items = set(train_df["parent_asin"].unique())
        metadata_filtered = metadata_df[metadata_df["parent_asin"].isin(train_items)]
        logger.info(
            f"Metadata filtered: {len(metadata_df):,} → {len(metadata_filtered):,} items (train only)"
        )

        start_time = time.time()
        product_texts = build_product_texts(metadata_filtered)
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


def train_ctr_and_log(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    metadata_df: pd.DataFrame,
    model_a: CollaborativeModel,
    model_b: SemanticModel,
    sample_users: int = 2000,
) -> tuple[CTRModel, dict]:
    """Train Model C (CTR XGBoost), evaluate, and log to MLflow."""
    mlflow.set_experiment("recengine-model-c")

    with mlflow.start_run(run_name="ctr-model"):
        # Build features
        logger.info("Building user and item features...")
        user_features = build_user_features(train_df)
        item_features = build_item_features(train_df, metadata_df)

        # Build CTR dataset (sample users for speed)
        logger.info(f"Building CTR dataset (sample_users={sample_users})...")
        ctr_df = build_ctr_dataset(
            train_df,
            user_features,
            item_features,
            model_a,
            model_b,
            neg_ratio=4,
            sample_users=sample_users,
        )

        # Feature columns (exclude id and label cols)
        drop_cols = ["user_id", "parent_asin", "label"]
        feature_cols = [c for c in ctr_df.columns if c not in drop_cols]

        X = ctr_df[feature_cols]
        y = ctr_df["label"]

        # Train/val split
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        mlflow.log_params(
            {
                "n_train": len(X_train),
                "n_val": len(X_val),
                "n_features": len(feature_cols),
                "neg_ratio": 4,
                "sample_users": sample_users,
            }
        )

        # Train
        start_time = time.time()
        model = CTRModel()
        model.fit(X_train, y_train, X_val, y_val)
        train_duration = time.time() - start_time

        # Evaluate
        val_probs = model.predict(X_val)
        auc = roc_auc_score(y_val, val_probs)
        logloss = log_loss(y_val, val_probs)

        mlflow.log_metrics(
            {
                "auc_roc": auc,
                "log_loss": logloss,
                "train_duration_seconds": train_duration,
            }
        )

        # Feature importance
        importance = model.get_feature_importance()
        top_features = sorted(importance.items(), key=lambda x: x[1], reverse=True)[:5]
        logger.info(f"Top features: {top_features}")

        logger.info(
            f"CTR Model trained in {train_duration:.1f}s | "
            f"AUC-ROC: {auc:.4f} | Log-loss: {logloss:.4f}"
        )

        metrics = {"auc_roc": auc, "log_loss": logloss}
        return model, metrics


def get_recommendations(
    user_id: str,
    model_a: CollaborativeModel,
    model_b: SemanticModel,
    model_c: CTRModel,
    user_features: pd.DataFrame,
    item_features: pd.DataFrame,
    train_df: pd.DataFrame,
    n: int = 10,
    n_candidates: int = 100,
) -> list[tuple[str, float]]:
    """Full pipeline: retrieval (A+B) -> re-ranking (C) -> top N."""
    # Retrieval
    candidates_a = model_a.recommend(user_id, n=n_candidates, exclude_seen=True)
    candidates_b = model_b.recommend(user_id, train_df, n=n_candidates) if model_b else []

    # Union and deduplicate
    seen_candidates: dict[str, float] = {}
    for item_id, score in candidates_a:
        seen_candidates[item_id] = score
    for item_id, score in candidates_b:
        if item_id not in seen_candidates:
            seen_candidates[item_id] = score

    if not seen_candidates:
        return []

    # Build feature rows for each candidate
    uf_row = user_features[user_features["user_id"] == user_id]
    candidate_rows = []
    for item_id in seen_candidates:
        row = {"user_id": user_id, "parent_asin": item_id}
        # User features
        if not uf_row.empty:
            for col in uf_row.columns:
                if col != "user_id":
                    row[col] = float(uf_row[col].iloc[0])
        # Item features
        if_row = item_features[item_features["parent_asin"] == item_id]
        if not if_row.empty:
            for col in if_row.columns:
                if col != "parent_asin":
                    row[col] = float(if_row[col].iloc[0])
        # Cross features (scores from A and B)
        row["model_a_score"] = seen_candidates[item_id]
        row["model_b_score"] = 0.0
        candidate_rows.append(row)

    # Re-rank with Model C
    reranked = model_c.rerank(candidate_rows)
    return [(r["parent_asin"], r["ctr_score"]) for r in reranked[:n]]


def evaluate_full_pipeline(
    model_a: CollaborativeModel,
    model_b: SemanticModel,
    model_c: CTRModel,
    user_features: pd.DataFrame,
    item_features: pd.DataFrame,
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    ks: list[int] = None,
    sample_users: int = 500,
) -> dict[str, float]:
    """Evaluate the full pipeline (A+B+C) on the test set."""
    if ks is None:
        ks = [5, 10, 20]

    user_metrics = {f"pipeline_precision_at_{k}": [] for k in ks}
    user_metrics.update({f"pipeline_recall_at_{k}": [] for k in ks})
    user_metrics.update({f"pipeline_ndcg_at_{k}": [] for k in ks})

    test_users = test_df["user_id"].unique()
    if sample_users:
        rng = np.random.default_rng(42)
        test_users = rng.choice(test_users, size=min(sample_users, len(test_users)), replace=False)

    logger.info(f"Evaluating full pipeline on {len(test_users)} users...")

    for user_id in test_users:
        relevant = test_df[test_df["user_id"] == user_id]["parent_asin"].tolist()
        recs = get_recommendations(
            user_id,
            model_a,
            model_b,
            model_c,
            user_features,
            item_features,
            train_df,
            n=max(ks),
        )
        recommended = [item_id for item_id, _ in recs]

        for k in ks:
            user_metrics[f"pipeline_precision_at_{k}"].append(
                precision_at_k(recommended, relevant, k)
            )
            user_metrics[f"pipeline_recall_at_{k}"].append(recall_at_k(recommended, relevant, k))
            user_metrics[f"pipeline_ndcg_at_{k}"].append(ndcg_at_k(recommended, relevant, k))

    results = {}
    for metric_name, values in user_metrics.items():
        results[metric_name] = sum(values) / len(values) if values else 0.0

    return results


def run_training(tuning: bool = False, train_model_b: bool = False, train_model_c: bool = False):
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

    metadata_df = None
    if train_model_b or train_model_c:
        metadata_path = data_dir / "metadata.parquet"
        if metadata_path.exists():
            metadata_df = pd.read_parquet(metadata_path)
        else:
            logger.warning("metadata.parquet not found")

    # Model A
    if tuning:
        model_a, metrics_a = hyperparameter_search(train_df, test_df)
    else:
        model_a, metrics_a = train_and_log(train_df, test_df)

    logger.info(f"Model A metrics: {metrics_a}")

    model_b = None
    metrics_b = {}

    # Model B
    if train_model_b and metadata_df is not None:
        model_b, metrics_b = train_semantic_and_log(train_df, test_df, metadata_df)
        comparison = compare_models(metrics_a, metrics_b)
        comparison.to_csv(data_dir / "model_comparison.csv", index=False)
        cold_results = cold_start_analysis(model_b, train_df, test_df)

        mlflow.set_experiment("recengine-comparison")
        with mlflow.start_run(run_name="model-a-vs-b"):
            for k, v in metrics_a.items():
                mlflow.log_metric(f"model_a_{k}", v)
            for k, v in metrics_b.items():
                mlflow.log_metric(f"model_b_{k}", v)
            for k, v in cold_results.items():
                if isinstance(v, (int, float)):
                    mlflow.log_metric(f"cold_start_{k}", v)

    # Model C
    if train_model_c and metadata_df is not None:
        model_c, metrics_c = train_ctr_and_log(train_df, test_df, metadata_df, model_a, model_b)
        logger.info(f"Model C metrics: {metrics_c}")

        # Full pipeline evaluation
        user_features = build_user_features(train_df)
        item_features = build_item_features(train_df, metadata_df)
        pipeline_metrics = evaluate_full_pipeline(
            model_a, model_b, model_c, user_features, item_features, train_df, test_df
        )
        logger.info(f"Full pipeline metrics: {pipeline_metrics}")

        # Compare pipeline vs Model A alone
        mlflow.set_experiment("recengine-comparison")
        with mlflow.start_run(run_name="pipeline-vs-model-a"):
            for k, v in metrics_a.items():
                mlflow.log_metric(f"model_a_{k}", v)
            for k, v in pipeline_metrics.items():
                mlflow.log_metric(k, v)

        return model_a, model_b, model_c, pipeline_metrics

    return model_a, metrics_a


if __name__ == "__main__":
    import sys

    train_b = "--model-b" in sys.argv
    train_c = "--model-c" in sys.argv
    tuning = "--tuning" in sys.argv
    run_training(tuning=tuning, train_model_b=train_b, train_model_c=train_c)
