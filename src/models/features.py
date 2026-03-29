import pandas as pd
import numpy as np
from loguru import logger


def build_user_features(train_df: pd.DataFrame) -> pd.DataFrame:
    """Build per-user features from training interactions."""
    grp = train_df.groupby("user_id")

    user_features = pd.DataFrame(
        {
            "num_ratings": grp["rating"].count(),
            "avg_rating": grp["rating"].mean(),
            "rating_std": grp["rating"].std().fillna(0.0),
            "days_active": grp["timestamp"].apply(
                lambda x: (x.max() - x.min()) / 86400 if len(x) > 1 else 0
            ),
        }
    ).reset_index()

    user_features = user_features.rename(columns={"index": "user_id"})
    user_features = user_features.fillna(0.0)

    logger.info(
        f"Built user features: {len(user_features)} users, {user_features.shape[1]-1} features"
    )
    return user_features


def build_item_features(train_df: pd.DataFrame, metadata_df: pd.DataFrame) -> pd.DataFrame:
    """Build per-item features from training interactions and metadata."""
    grp = train_df.groupby("parent_asin")

    item_features = pd.DataFrame(
        {
            "avg_rating_received": grp["rating"].mean(),
            "num_ratings_received": grp["rating"].count(),
            "rating_std_received": grp["rating"].std().fillna(0.0),
        }
    ).reset_index()

    # Join metadata features
    meta = metadata_df[["parent_asin"]].copy()

    # Price
    if "price" in metadata_df.columns:
        meta["price"] = pd.to_numeric(metadata_df["price"], errors="coerce")
    else:
        meta["price"] = np.nan

    # Description length
    if "description" in metadata_df.columns:
        meta["description_length"] = metadata_df["description"].apply(
            lambda x: len(
                " ".join(str(i) for i in x)
                if isinstance(x, list)
                else str(x) if isinstance(x, str) else ""
            )
        )
    else:
        meta["description_length"] = 0

    item_features = item_features.merge(meta, on="parent_asin", how="left")
    item_features["price"] = item_features["price"].fillna(item_features["price"].median())
    item_features = item_features.fillna(0.0)

    logger.info(
        f"Built item features: {len(item_features)} items, {item_features.shape[1]-1} features"
    )
    return item_features


def build_cross_features(
    user_id: str,
    item_id: str,
    model_a,
    model_b,
    train_df: pd.DataFrame,
    user_features: pd.DataFrame,
) -> dict:
    """Build cross features for a (user, item) pair using Model A and B scores."""
    # Model A score
    preds_a = model_a.predict(user_id, [item_id])
    model_a_score = preds_a[0][1] if preds_a else 3.0

    # Model B score via find_similar (proxy for relevance)
    model_b_score = 0.0
    if model_b is not None and model_b.index is not None:
        user_data = train_df[train_df["user_id"] == user_id]
        liked = user_data[user_data["rating"] >= 4]["parent_asin"].tolist()
        if not liked:
            liked = user_data["parent_asin"].tolist()
        scores = []
        for liked_item in liked[:5]:  # limit to 5 for speed
            similar = model_b.find_similar(liked_item, n=20)
            for asin, score in similar:
                if asin == item_id:
                    scores.append(score)
                    break
        model_b_score = float(np.mean(scores)) if scores else 0.0

    return {
        "model_a_score": model_a_score,
        "model_b_score": model_b_score,
    }
