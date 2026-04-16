from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

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


def preprocess_reviews_chunked(
    df: pd.DataFrame, min_interactions: int = 5, chunk_size: int = 500_000
) -> pd.DataFrame:
    import dask.dataframe as dd

    logger.info(
        f"Chunked preprocessing with dask: {len(df):,} raw reviews " f"(chunk_size={chunk_size:,})"
    )
    cols = ["user_id", "parent_asin", "rating", "timestamp"]
    df = df[cols].copy()

    npartitions = max(1, len(df) // chunk_size)
    ddf = dd.from_pandas(df, npartitions=npartitions)

    ddf = ddf.sort_values(["user_id", "parent_asin", "timestamp"])
    ddf = ddf.drop_duplicates(subset=["user_id", "parent_asin"], keep="last")

    user_counts = ddf.groupby("user_id").size().compute()
    valid_users = set(user_counts[user_counts >= min_interactions].index)
    ddf = ddf[ddf["user_id"].isin(valid_users)]

    item_counts = ddf.groupby("parent_asin").size().compute()
    valid_items = set(item_counts[item_counts >= min_interactions].index)
    ddf = ddf[ddf["parent_asin"].isin(valid_items)]

    result = ddf.compute().reset_index(drop=True)
    logger.info(
        f"After chunked filtering: {len(result):,} reviews, "
        f"{result['user_id'].nunique():,} users, "
        f"{result['parent_asin'].nunique():,} items"
    )
    return result


def _validate_split_temporal_integrity(train: pd.DataFrame, test: pd.DataFrame) -> None:
    train_max = train.groupby("user_id")["timestamp"].max()
    test_min = test.groupby("user_id")["timestamp"].min()
    common_users = train_max.index.intersection(test_min.index)
    if len(common_users) > 0:
        violations = train_max[common_users] > test_min[common_users]
        if violations.any():
            bad_users = violations[violations].index.tolist()
            raise ValueError(
                f"Data leakage: {len(bad_users)} users have test interactions "
                f"predating their training interactions (e.g., {bad_users[:5]})"
            )
    logger.info("User-level time gap validation passed")


def temporal_train_test_split(
    df: pd.DataFrame, test_ratio: float = 0.2
) -> tuple[pd.DataFrame, pd.DataFrame]:
    df = df.sort_values("timestamp")
    train_frames = []
    test_frames = []

    skipped = 0
    for user_id, group in df.groupby("user_id"):
        n = len(group)
        split_idx = int(n * (1 - test_ratio))
        if n < 2:
            train_frames.append(group)
            skipped += 1
            continue
        if split_idx < 1:
            split_idx = 1
        train_frames.append(group.iloc[:split_idx])
        test_frames.append(group.iloc[split_idx:])

    if skipped:
        logger.warning(f"Skipped {skipped} users with only 1 interaction (train-only)")

    train = pd.concat(train_frames, ignore_index=True)
    test = pd.concat(test_frames, ignore_index=True)

    _validate_split_temporal_integrity(train, test)

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

    # Parallel loading
    logger.info("Loading reviews and metadata in parallel...")
    with ThreadPoolExecutor(max_workers=2) as executor:
        reviews_future = executor.submit(load_reviews, category)
        metadata_future = executor.submit(load_metadata, category)
        reviews = reviews_future.result()
        metadata = metadata_future.result()

    # Parallel alignment
    common_items = set(reviews["parent_asin"]) & set(metadata["parent_asin"])
    logger.info(f"Common items for alignment: {len(common_items):,}")

    with ThreadPoolExecutor(max_workers=2) as executor:
        reviews_fut = executor.submit(
            reviews.__getitem__, reviews["parent_asin"].isin(common_items)
        )
        metadata_fut = executor.submit(
            metadata.__getitem__, metadata["parent_asin"].isin(common_items)
        )
        reviews = reviews_fut.result()
        metadata = metadata_fut.result()
    logger.info(f"After alignment: {len(reviews):,} reviews, {len(metadata):,} items")

    # Preprocess (use dask for large datasets)
    if settings.USE_DASK and len(reviews) > settings.DASK_CHUNK_SIZE:
        reviews_clean = preprocess_reviews_chunked(
            reviews,
            min_interactions=settings.MIN_INTERACTIONS,
            chunk_size=settings.DASK_CHUNK_SIZE,
        )
    else:
        reviews_clean = preprocess_reviews(reviews, min_interactions=settings.MIN_INTERACTIONS)

    # Validate
    from src.data.validation import (
        validate_reviews,
        validate_metadata,
        validate_no_leakage,
    )

    assert validate_reviews(reviews_clean), "Review validation failed"
    assert validate_metadata(metadata), "Metadata validation failed"

    # Split
    train, test = temporal_train_test_split(reviews_clean, test_ratio=settings.TEST_RATIO)

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
