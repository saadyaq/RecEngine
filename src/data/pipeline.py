from pathlib import Path

import pandas as pd
from datasets import load_dataset
from loguru import logger

from src.config import settings


def load_reviews(category: str = "Electronics", n: int = 2_000_000) -> pd.DataFrame:
    logger.info(f"Loading reviews for category: {category} (n={n:,})")
    dataset = load_dataset(
        "McAuley-Lab/Amazon-Reviews-2023",
        f"raw_review_{category}",
        split="full",
        trust_remote_code=True,
        streaming=True,
    )
    df = pd.DataFrame(list(dataset.take(n)))
    logger.info(f"Loaded {len(df):,} reviews")
    return df


def load_metadata(category: str = "Electronics", n: int = 500_000) -> pd.DataFrame:
    logger.info(f"Loading metadata for category: {category} (n={n:,})")
    dataset = load_dataset(
        "McAuley-Lab/Amazon-Reviews-2023",
        f"raw_meta_{category}",
        split="full",
        trust_remote_code=True,
        streaming=True,
    )
    df = pd.DataFrame(list(dataset.take(n)))
    logger.info(f"Loaded {len(df):,} metadata items")
    return df


def preprocess_reviews(df: pd.DataFrame, min_interactions: int = 5) -> pd.DataFrame:
    try:
        logger.info(f"Raw reviews: {len(df):,}")
        cols = ["user_id", "parent_asin", "rating", "timestamp"]
        df = df[cols].copy()
        df = df.drop_duplicates(subset=["user_id", "parent_asin"], keep="last")

        user_counts = df["user_id"].value_counts()
        valid_users = user_counts[user_counts >= min_interactions].index
        df = df[df["user_id"].isin(valid_users)]

        item_counts = df["parent_asin"].value_counts()
        valid_items = item_counts[item_counts >= min_interactions].index
        df = df[df["parent_asin"].isin(valid_items)]

        logger.info(
            f"After filtering: {len(df):,} reviews, "
            f"{df['user_id'].nunique():,} users, "
            f"{df['parent_asin'].nunique():,} items"
        )
        return df
    except KeyError as e:
        logger.error(f"Missing expected column: {e}")
        raise
    except Exception as e:
        logger.error(f"Error during preprocessing: {e}")
        raise


def temporal_train_test_split(
    df: pd.DataFrame, test_ratio: float = 0.2
) -> tuple[pd.DataFrame, pd.DataFrame]:
    df = df.sort_values("timestamp")
    train_frames = []
    test_frames = []

    for user_id, group in df.groupby("user_id"):
        n = len(group)
        split_idx = int(n * (1 - test_ratio))
        if split_idx < 1:
            train_frames.append(group)
            continue
        train_frames.append(group.iloc[:split_idx])
        test_frames.append(group.iloc[split_idx:])

    train = pd.concat(train_frames, ignore_index=True)
    test = pd.concat(test_frames, ignore_index=True)

    logger.info(f"Train: {len(train):,} | Test: {len(test):,}")
    return train, test


def run_pipeline(
    output_dir: str = "data/processed",
    category: str = None,
) -> None:
    if category is None:
        category = settings.DATASET_CATEGORY

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Load
    reviews = load_reviews(category)
    metadata = load_metadata(category)

    # Align: keep only items present in both
    common_items = set(reviews["parent_asin"]) & set(metadata["parent_asin"])
    reviews = reviews[reviews["parent_asin"].isin(common_items)]
    metadata = metadata[metadata["parent_asin"].isin(common_items)]
    logger.info(f"After alignment: {len(reviews):,} reviews, {len(metadata):,} items")

    # Preprocess
    reviews_clean = preprocess_reviews(reviews, min_interactions=settings.MIN_INTERACTIONS)

    # Validate
    from src.data.validation import validate_reviews

    assert validate_reviews(reviews_clean), "Review validation failed"

    # Split
    train, test = temporal_train_test_split(reviews_clean, test_ratio=settings.TEST_RATIO)

    # Validate no leakage
    from src.data.validation import validate_no_leakage

    assert validate_no_leakage(train, test), "Data leakage detected"

    # Save
    train.to_parquet(output_path / "train.parquet", index=False)
    test.to_parquet(output_path / "test.parquet", index=False)
    metadata.to_parquet(output_path / "metadata.parquet", index=False)

    logger.info(f"Pipeline complete. Saved to {output_path}")
    logger.info(
        f"Summary: {train['user_id'].nunique():,} users, "
        f"{train['parent_asin'].nunique():,} items, "
        f"{len(train):,} train / {len(test):,} test"
    )


if __name__ == "__main__":
    run_pipeline()
