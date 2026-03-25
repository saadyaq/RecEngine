import pandas as pd

from src.models.collaborative import CollaborativeModel
from src.models.semantic import SemanticModel, build_product_texts
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
    model = CollaborativeModel(n_factors=10, n_epochs=5, regularization=0.01, alpha=40.0)
    model.fit(df)
    assert model.train_df is not None
    assert model.all_items is not None


def test_collaborative_model_predicts():
    df = _make_train_df()
    model = CollaborativeModel(n_factors=10, n_epochs=5, regularization=0.01, alpha=40.0)
    model.fit(df)
    predictions = model.predict("u1", ["i1", "i2", "i3"])
    assert len(predictions) == 3
    for item_id, score in predictions:
        assert isinstance(score, float)
        assert 1.0 <= score <= 5.0


def test_collaborative_model_recommends_n_items():
    df = _make_train_df()
    model = CollaborativeModel(n_factors=10, n_epochs=5, regularization=0.01, alpha=40.0)
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


# --- SemanticModel tests ---


def _make_metadata_df():
    """Small metadata for testing semantic model."""
    return pd.DataFrame(
        {
            "parent_asin": ["i1", "i2", "i3", "i4", "i5"],
            "title": [
                "Wireless Bluetooth Headphones",
                "Bluetooth Earbuds Wireless",
                "USB-C Charging Cable",
                "Laptop Stand Adjustable",
                "Wireless Mouse Ergonomic",
            ],
            "description": [
                "Noise cancelling over-ear headphones",
                "In-ear wireless earbuds with microphone",
                "Fast charging USB type C cable",
                "Aluminum stand for laptop and tablet",
                "Ergonomic wireless mouse with USB receiver",
            ],
            "features": [
                ["Active noise cancellation", "40h battery"],
                ["Bluetooth 5.0", "Touch controls"],
                ["3ft length", "Nylon braided"],
                ["Adjustable height", "Foldable"],
                ["2.4GHz wireless", "Silent clicks"],
            ],
        }
    )


def test_semantic_model_builds_index():
    metadata_df = _make_metadata_df()
    product_texts = build_product_texts(metadata_df)
    model = SemanticModel()
    model.build_index(product_texts)
    assert model.index.ntotal == 5


def test_semantic_model_find_similar():
    metadata_df = _make_metadata_df()
    product_texts = build_product_texts(metadata_df)
    model = SemanticModel()
    model.build_index(product_texts)
    results = model.find_similar("i1", n=3)
    assert len(results) == 3
    # Should not contain the query item itself
    assert all(asin != "i1" for asin, _ in results)


def test_semantic_model_recommend():
    metadata_df = _make_metadata_df()
    product_texts = build_product_texts(metadata_df)
    model = SemanticModel()
    model.build_index(product_texts)

    train_df = pd.DataFrame(
        {
            "user_id": ["u1", "u1", "u1"],
            "parent_asin": ["i1", "i2", "i3"],
            "rating": [5.0, 4.0, 2.0],
            "timestamp": [1, 2, 3],
        }
    )
    recs = model.recommend("u1", train_df, n=2)
    assert len(recs) <= 2
    # Should not recommend items the user has already seen
    seen = {"i1", "i2", "i3"}
    for asin, _ in recs:
        assert asin not in seen


def test_embeddings_dimension():
    metadata_df = _make_metadata_df()
    product_texts = build_product_texts(metadata_df)
    model = SemanticModel(model_name="all-MiniLM-L6-v2")
    model.build_index(product_texts)
    # MiniLM produces 384-dimensional embeddings
    vec = model.index.reconstruct(0)
    assert vec.shape[0] == 384
