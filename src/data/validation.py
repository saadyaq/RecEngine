import pandas as pd
from loguru import logger


REQUIRED_REVIEW_COLS = ["user_id", "parent_asin", "rating", "timestamp"]


def validate_reviews(df: pd.DataFrame) -> bool:
    # Check required columns
    missing = set(REQUIRED_REVIEW_COLS) - set(df.columns)
    if missing:
        logger.error(f"Missing columns: {missing}")
        return False

    # Check no nulls in required columns
    nulls = df[REQUIRED_REVIEW_COLS].isnull().sum()
    if nulls.any():
        logger.error(f"Null values found:\n{nulls[nulls > 0]}")
        return False

    # Check rating range [1, 5]
    if df["rating"].min() < 1 or df["rating"].max() > 5:
        logger.error(f"Ratings out of range: [{df['rating'].min()}, {df['rating'].max()}]")
        return False

    # Check no duplicates
    dupes = df.duplicated(subset=["user_id", "parent_asin"]).sum()
    if dupes > 0:
        logger.error(f"Found {dupes} duplicate (user, item) pairs")
        return False

    logger.info("Review validation passed")
    return True


def validate_no_leakage(train: pd.DataFrame, test: pd.DataFrame) -> bool:
    # Check no overlap of (user_id, parent_asin) pairs
    train_pairs = set(zip(train["user_id"], train["parent_asin"]))
    test_pairs = set(zip(test["user_id"], test["parent_asin"]))
    overlap = train_pairs & test_pairs
    if overlap:
        logger.error(f"Found {len(overlap)} overlapping (user, item) pairs")
        return False

    # Check timestamps are coherent: for each user, max train timestamp <= min test timestamp
    train_max = train.groupby("user_id")["timestamp"].max()
    test_min = test.groupby("user_id")["timestamp"].min()
    common_users = train_max.index.intersection(test_min.index)
    if len(common_users) > 0:
        violations = (train_max[common_users] > test_min[common_users]).sum()
        if violations > 0:
            logger.error(f"Found {violations} users with train timestamps after test timestamps")
            return False

    logger.info("No leakage detected")
    return True


def validate_metadata(df: pd.DataFrame) -> bool:
    # Check parent_asin exists
    if "parent_asin" not in df.columns:
        logger.error("Missing column: parent_asin")
        return False

    # Check no null parent_asin
    if df["parent_asin"].isnull().any():
        logger.error("Null values in parent_asin")
        return False

    # Check titles non-empty
    if "title" in df.columns:
        empty_titles = (df["title"].isnull() | (df["title"].str.strip() == "")).sum()
        if empty_titles > 0:
            logger.warning(f"{empty_titles} items with empty titles")

    logger.info("Metadata validation passed")
    return True
