import pandas as pd

from src.models.collaborative import CollaborativeModel
from src.training.evaluate import ndcg_at_k, precision_at_k


def _make_train_df():
    """Small dataset with 3 users and 5 items, each user has 5 interactions."""
    rows = []
    users = ["u1", "u2", "u3"]
    items = ["i1", "i2", "i3", "i4", "i5"]
    ratings = [
        [5, 4, 3, 2, 1],
        [1, 2, 3, 4, 5],
        [3, 3, 3, 4, 4],
    ]
    for u_idx, user in enumerate(users):
        for i_idx, item in enumerate(items):
            rows.append(
                {
                    "user_id": user,
                    "parent_asin": item,
                    "rating": float(ratings[u_idx][i_idx]),
                    "timestamp": u_idx * 5 + i_idx,
                }
            )
    return pd.DataFrame(rows)


# --- CollaborativeModel tests ---


def test_collaborative_model_fits():
    df = _make_train_df()
    model = CollaborativeModel(n_factors=10, n_epochs=5)
    model.fit(df)
    assert model.train_df is not None
    assert model.all_items is not None


def test_collaborative_model_predicts():
    df = _make_train_df()
    model = CollaborativeModel(n_factors=10, n_epochs=5)
    model.fit(df)
    predictions = model.predict("u1", ["i1", "i2", "i3"])
    assert len(predictions) == 3
    for item_id, score in predictions:
        assert isinstance(score, float)
        assert 1.0 <= score <= 5.0


def test_collaborative_model_recommends_n_items():
    df = _make_train_df()
    model = CollaborativeModel(n_factors=10, n_epochs=5)
    model.fit(df)
    recs = model.recommend("u1", n=3, exclude_seen=False)
    assert len(recs) == 3
    # Should be sorted by score descending
    scores = [score for _, score in recs]
    assert scores == sorted(scores, reverse=True)


# --- Metrics tests ---


def test_metrics_precision_at_k():
    assert precision_at_k([1, 2, 3], [2, 3, 5], 3) == 2 / 3
    assert precision_at_k([1, 2, 3], [4, 5, 6], 3) == 0.0
    assert precision_at_k([1, 2, 3], [1, 2, 3], 3) == 1.0


def test_metrics_ndcg_at_k():
    # All relevant at top -> NDCG = 1.0
    assert ndcg_at_k([1, 2, 3], [1, 2, 3], 3) == 1.0
    # No relevant items -> NDCG = 0.0
    assert ndcg_at_k([1, 2, 3], [4, 5, 6], 3) == 0.0
    # Partial match
    result = ndcg_at_k([1, 2, 3], [2, 3, 5], 3)
    assert 0.0 < result < 1.0
