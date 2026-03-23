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
