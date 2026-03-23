import pandas as pd
from datasets import load_dataset
from loguru import logger


def load_reviews(category: str = "Electronics") -> pd.DataFrame:
    logger.info(f"Loading reviews for category: {category}")
    dataset = load_dataset(
        "McAuley-Lab/Amazon-Reviews-2023",
        f"raw_review_{category}",
        split="full",
        trust_remote_code=True,
    )
    df = dataset.to_pandas()
    logger.info(f"Loaded {len(df):,} reviews")
    return df


def load_metadata(category: str = "Electronics") -> pd.DataFrame:
    logger.info(f"Loading metadata for category: {category}")
    dataset = load_dataset(
        "McAuley-Lab/Amazon-Reviews-2023",
        f"raw_meta_{category}",
        split="full",
        trust_remote_code=True,
    )
    df = dataset.to_pandas()
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
