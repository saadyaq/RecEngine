import pandas as pd
import pytest

from src.data.pipeline import (
    preprocess_reviews,
    temporal_train_test_split,
    _validate_split_temporal_integrity,
)
from src.data.validation import (
    validate_no_leakage,
    validate_reviews,
    validate_timestamp_range,
    validate_metadata,
)


def _make_reviews(**overrides):
    data = {
        "user_id": ["u1", "u1", "u1", "u1", "u1", "u2", "u2", "u2", "u2", "u2"],
        "parent_asin": ["i1", "i2", "i3", "i4", "i5", "i1", "i2", "i3", "i4", "i5"],
        "rating": [4.0, 3.0, 5.0, 2.0, 1.0, 5.0, 4.0, 3.0, 2.0, 1.0],
        "timestamp": list(range(10)),
    }
    data.update(overrides)
    return pd.DataFrame(data)


def _make_metadata(**overrides):
    data = {
        "parent_asin": ["i1", "i2", "i3"],
        "title": ["Product A", "Product B", "Product C"],
        "main_category": ["Electronics", "Electronics", "Books"],
    }
    data.update(overrides)
    return pd.DataFrame(data)


# --- validate_reviews ---


def test_validate_reviews_valid_data():
    df = _make_reviews()
    assert validate_reviews(df) is True


def test_validate_reviews_missing_column():
    df = _make_reviews()
    df = df.drop(columns=["rating"])
    assert validate_reviews(df) is False


def test_validate_reviews_rating_out_of_range():
    df = _make_reviews(rating=[6.0, 3.0, 5.0, 2.0, 1.0, 5.0, 4.0, 3.0, 2.0, 1.0])
    assert validate_reviews(df) is False


def test_validate_reviews_duplicates():
    df = _make_reviews(
        user_id=["u1", "u1", "u1", "u1", "u1", "u1", "u2", "u2", "u2", "u2"],
        parent_asin=["i1", "i1", "i3", "i4", "i5", "i6", "i1", "i2", "i3", "i4"],
    )
    assert validate_reviews(df) is False


# --- validate_timestamp_range ---


def test_validate_timestamp_range_synthetic():
    df = _make_reviews()
    assert validate_timestamp_range(df) is True


def test_validate_timestamp_range_valid_ms():
    ts = pd.Series([820454400000, 1609459200000, 1798761599000])
    df = pd.DataFrame({"timestamp": ts})
    assert validate_timestamp_range(df) is True


def test_validate_timestamp_range_invalid_ms():
    ts = pd.Series([100000000000, 820454400000])
    df = pd.DataFrame({"timestamp": ts})
    assert validate_timestamp_range(df) is False


# --- validate_no_leakage ---


def test_validate_no_leakage_clean():
    train = pd.DataFrame(
        {
            "user_id": ["u1", "u1"],
            "parent_asin": ["i1", "i2"],
            "timestamp": [1, 2],
        }
    )
    test = pd.DataFrame(
        {
            "user_id": ["u1"],
            "parent_asin": ["i3"],
            "timestamp": [3],
        }
    )
    assert validate_no_leakage(train, test) is True


def test_validate_no_leakage_overlap():
    train = pd.DataFrame(
        {
            "user_id": ["u1", "u1"],
            "parent_asin": ["i1", "i2"],
            "timestamp": [1, 2],
        }
    )
    test = pd.DataFrame(
        {
            "user_id": ["u1"],
            "parent_asin": ["i1"],
            "timestamp": [3],
        }
    )
    assert validate_no_leakage(train, test) is False


# --- validate_metadata ---


def test_validate_metadata_valid():
    df = _make_metadata()
    assert validate_metadata(df) is True


def test_validate_metadata_missing_category():
    df = _make_metadata(main_category=["Electronics", "", None])
    assert validate_metadata(df) is False


def test_validate_metadata_no_category_column():
    df = _make_metadata().drop(columns=["main_category"])
    assert validate_metadata(df) is True


# --- preprocess_reviews ---


def test_preprocess_removes_duplicates():
    df = pd.DataFrame(
        {
            "user_id": ["u1"] * 6 + ["u2"] * 6,
            "parent_asin": ["i1", "i1", "i2", "i3", "i4", "i5"]
            + ["i1", "i2", "i3", "i4", "i5", "i6"],
            "rating": [3.0, 5.0, 4.0, 3.0, 2.0, 1.0] + [5.0, 4.0, 3.0, 2.0, 1.0, 4.0],
            "timestamp": list(range(12)),
        }
    )
    result = preprocess_reviews(df, min_interactions=2)
    dupes = result.duplicated(subset=["user_id", "parent_asin"]).sum()
    assert dupes == 0


def test_preprocess_filters_low_interaction_users():
    df = pd.DataFrame(
        {
            "user_id": ["u1"] * 5 + ["u2"],
            "parent_asin": ["i1", "i2", "i3", "i4", "i5", "i1"],
            "rating": [4.0, 3.0, 5.0, 2.0, 1.0, 5.0],
            "timestamp": list(range(6)),
        }
    )
    result = preprocess_reviews(df, min_interactions=5)
    assert "u2" not in result["user_id"].values


# --- temporal_train_test_split ---


def test_temporal_split_ordering():
    df = _make_reviews()
    train, test = temporal_train_test_split(df, test_ratio=0.4)
    for user_id in train["user_id"].unique():
        user_train = train[train["user_id"] == user_id]
        user_test = test[test["user_id"] == user_id]
        if len(user_test) > 0:
            assert user_train["timestamp"].max() <= user_test["timestamp"].min()


def test_temporal_split_ratio():
    df = _make_reviews()
    train, test = temporal_train_test_split(df, test_ratio=0.2)
    total = len(train) + len(test)
    actual_ratio = len(test) / total
    assert 0.1 <= actual_ratio <= 0.4


def test_temporal_split_leakage_raises():
    train = pd.DataFrame(
        {
            "user_id": ["u1", "u1"],
            "parent_asin": ["i1", "i2"],
            "timestamp": [100, 200],
        }
    )
    test = pd.DataFrame(
        {
            "user_id": ["u1"],
            "parent_asin": ["i3"],
            "timestamp": [50],
        }
    )
    with pytest.raises(ValueError, match="Data leakage"):
        _validate_split_temporal_integrity(train, test)
