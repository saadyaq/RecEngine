import pandas as pd
from loguru import logger


REQUIRED_REVIEW_COLS = ["user_id", "parent_asin", "rating", "timestamp"]

AMAZON_TIMESTAMP_MIN_MS = 820454400000  # 1996-01-01 00:00:00 UTC
AMAZON_TIMESTAMP_MAX_MS = 1798761599000  # 2026-12-31 23:59:59 UTC


def validate_timestamp_range(df: pd.DataFrame) -> bool:
    if "timestamp" not in df.columns:
        logger.error("Missing column: timestamp")
        return False

    ts = df["timestamp"]

    if ts.max() < 1_000_000_000:
        logger.info("Timestamps appear to be synthetic; skipping range validation")
        return True

    if ts.max() > 1e12:
        ts_min, ts_max = AMAZON_TIMESTAMP_MIN_MS, AMAZON_TIMESTAMP_MAX_MS
        unit = "milliseconds"
    else:
        ts_min = AMAZON_TIMESTAMP_MIN_MS // 1000
        ts_max = AMAZON_TIMESTAMP_MAX_MS // 1000
        unit = "seconds"

    out_of_range = ((ts < ts_min) | (ts > ts_max)).sum()
    if out_of_range > 0:
        logger.error(
            f"Found {out_of_range:,} timestamps outside valid Amazon range "
            f"({ts_min} - {ts_max} {unit})"
        )
        return False

    logger.info(f"Timestamp range validation passed ({unit})")
    return True


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

    # Check timestamp range
    if not validate_timestamp_range(df):
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

    # Check category assignments
    if "main_category" in df.columns:
        null_cats = df["main_category"].isnull().sum()
        empty_cats = 0
        if df["main_category"].dtype == object:
            empty_cats = (df["main_category"].fillna("").str.strip() == "").sum()
        invalid_cats = null_cats + empty_cats
        if invalid_cats > 0:
            logger.error(f"Found {invalid_cats:,} items with missing or empty category assignments")
            return False
        unique_cats = df["main_category"].nunique()
        logger.info(f"Category validation passed: {unique_cats} unique categories")
    else:
        logger.warning("Column 'main_category' not found; skipping category validation")

    logger.info("Metadata validation passed")
    return True
