import numpy as np
import pandas as pd

from src.models.ctr import CTRModel, build_ctr_dataset
from src.models.features import build_item_features, build_user_features
from src.models.collaborative import CollaborativeModel


def _make_train_df():
    rng = np.random.default_rng(0)
    users = [f"u{i}" for i in range(10)]
    items = [f"i{i}" for i in range(20)]
    rows = []
    for u in users:
        for _ in range(5):
            item = rng.choice(items)
            rows.append(
                {
                    "user_id": u,
                    "parent_asin": item,
                    "rating": float(rng.integers(1, 6)),
                    "timestamp": int(rng.integers(1000, 9999)),
                }
            )
    df = pd.DataFrame(rows).drop_duplicates(subset=["user_id", "parent_asin"])
    return df


def _make_metadata_df(items):
    return pd.DataFrame(
        {
            "parent_asin": items,
            "title": [f"Product {i}" for i in items],
            "description": [f"Description for {i}" for i in items],
            "features": [["feature1", "feature2"] for _ in items],
            "price": [float(i[1:]) * 10 for i in items],
        }
    )


def _make_model_a(train_df):
    model = CollaborativeModel(n_factors=5, n_epochs=3)
    model.fit(train_df)
    return model


def test_build_user_features_no_nan():
    train_df = _make_train_df()
    user_features = build_user_features(train_df)
    assert user_features.isnull().sum().sum() == 0
    assert "user_id" in user_features.columns
    assert "num_ratings" in user_features.columns


def test_build_item_features_no_nan():
    train_df = _make_train_df()
    items = train_df["parent_asin"].unique().tolist()
    metadata_df = _make_metadata_df(items)
    item_features = build_item_features(train_df, metadata_df)
    assert item_features.isnull().sum().sum() == 0
    assert "parent_asin" in item_features.columns


def test_build_ctr_dataset_ratio():
    train_df = _make_train_df()
    items = train_df["parent_asin"].unique().tolist()
    metadata_df = _make_metadata_df(items)
    model_a = _make_model_a(train_df)
    user_features = build_user_features(train_df)
    item_features = build_item_features(train_df, metadata_df)

    ctr_df = build_ctr_dataset(
        train_df,
        user_features,
        item_features,
        model_a,
        None,
        neg_ratio=4,
        sample_users=5,
    )
    pos = (ctr_df["label"] == 1).sum()
    neg = (ctr_df["label"] == 0).sum()
    assert pos > 0
    assert neg > 0
    assert neg / pos <= 4.5


def test_negative_sampling_excludes_seen():
    train_df = _make_train_df()
    items = train_df["parent_asin"].unique().tolist()
    metadata_df = _make_metadata_df(items)
    model_a = _make_model_a(train_df)
    user_features = build_user_features(train_df)
    item_features = build_item_features(train_df, metadata_df)

    ctr_df = build_ctr_dataset(
        train_df,
        user_features,
        item_features,
        model_a,
        None,
        neg_ratio=2,
        sample_users=3,
    )
    negatives = ctr_df[ctr_df["label"] == 0]
    seen_pairs = set(zip(train_df["user_id"], train_df["parent_asin"]))
    for _, row in negatives.iterrows():
        assert (row["user_id"], row["parent_asin"]) not in seen_pairs


def test_ctr_model_predicts_probabilities():
    train_df = _make_train_df()
    items = train_df["parent_asin"].unique().tolist()
    metadata_df = _make_metadata_df(items)
    model_a = _make_model_a(train_df)
    user_features = build_user_features(train_df)
    item_features = build_item_features(train_df, metadata_df)

    ctr_df = build_ctr_dataset(
        train_df,
        user_features,
        item_features,
        model_a,
        None,
        neg_ratio=2,
        sample_users=5,
    )
    drop_cols = ["user_id", "parent_asin", "label"]
    feature_cols = [c for c in ctr_df.columns if c not in drop_cols]
    X = ctr_df[feature_cols]
    y = ctr_df["label"]

    from sklearn.model_selection import train_test_split

    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.3, random_state=42)

    model = CTRModel({"n_estimators": 10, "max_depth": 3})
    model.fit(X_train, y_train, X_val, y_val)
    probs = model.predict(X_val)

    assert probs.shape == (len(X_val),)
    assert (probs >= 0).all() and (probs <= 1).all()


def test_reranking_changes_order():
    train_df = _make_train_df()
    items = train_df["parent_asin"].unique().tolist()
    metadata_df = _make_metadata_df(items)
    model_a = _make_model_a(train_df)
    user_features = build_user_features(train_df)
    item_features = build_item_features(train_df, metadata_df)

    ctr_df = build_ctr_dataset(
        train_df,
        user_features,
        item_features,
        model_a,
        None,
        neg_ratio=2,
        sample_users=5,
    )
    drop_cols = ["user_id", "parent_asin", "label"]
    feature_cols = [c for c in ctr_df.columns if c not in drop_cols]
    X = ctr_df[feature_cols]
    y = ctr_df["label"]

    from sklearn.model_selection import train_test_split

    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.3, random_state=42)

    model = CTRModel({"n_estimators": 10, "max_depth": 3})
    model.fit(X_train, y_train, X_val, y_val)

    candidates = [
        {col: float(X_val.iloc[i][col]) for col in feature_cols} for i in range(min(5, len(X_val)))
    ]
    reranked = model.rerank(candidates)
    scores = [c["ctr_score"] for c in reranked]
    assert scores == sorted(scores, reverse=True)
