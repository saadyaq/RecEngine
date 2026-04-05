import numpy as np
import pandas as pd
from implicit.als import AlternatingLeastSquares
from loguru import logger
from scipy.sparse import csr_matrix

REQUIRED_COLUMNS = {"user_id", "parent_asin", "rating"}


class CollaborativeModel:
    def __init__(
        self,
        n_factors: int = 100,
        n_epochs: int = 20,
        regularization: float = 0.01,
        alpha: float = 40.0,
    ):
        self.n_factors = n_factors
        self.n_epochs = n_epochs
        self.regularization = regularization
        self.alpha = alpha  # confidence scaling for implicit feedback

        self.model = None
        self.user_item_matrix = None
        self.global_mean = 0.0
        self.user_map: dict[str, int] = {}
        self.item_map: dict[str, int] = {}
        self.reverse_item_map: dict[int, str] = {}
        self.train_df = None
        self.all_items: list[str] = []
        self._popular_items: list[str] = []  # items sorted by interaction count

    def fit(self, train_df: pd.DataFrame) -> "CollaborativeModel":
        # --- Input validation ---
        if train_df is None or train_df.empty:
            raise ValueError("Cannot fit model on empty DataFrame")

        missing = REQUIRED_COLUMNS - set(train_df.columns)
        if missing:
            raise ValueError(f"Training data missing columns: {missing}")

        if not np.isfinite(self.alpha) or self.alpha <= 0:
            raise ValueError(f"Invalid alpha={self.alpha}: must be positive and finite")

        # Drop rows with NaN ratings
        n_before = len(train_df)
        train_df = train_df.dropna(subset=["rating"]).copy()
        if len(train_df) < n_before:
            logger.warning(f"Dropped {n_before - len(train_df)} rows with NaN ratings")

        if train_df.empty:
            raise ValueError("No valid ratings after dropping NaN values")

        self.train_df = train_df
        self.all_items = train_df["parent_asin"].unique().tolist()

        if not self.all_items:
            raise ValueError("Training data contains no items")

        self.global_mean = float(train_df["rating"].mean())
        if not np.isfinite(self.global_mean):
            raise ValueError("global_mean is not finite after computation")

        # Popularity-ranked items (for cold-start fallback)
        self._popular_items = train_df["parent_asin"].value_counts().index.tolist()

        # Build user/item mappings
        users = train_df["user_id"].unique()
        items = train_df["parent_asin"].unique()
        self.user_map = {u: i for i, u in enumerate(users)}
        self.item_map = {item: i for i, item in enumerate(items)}
        self.reverse_item_map = {i: item for item, i in self.item_map.items()}

        # Build sparse user-item matrix
        row = train_df["user_id"].map(self.user_map)
        col = train_df["parent_asin"].map(self.item_map)

        if row.isna().any() or col.isna().any():
            raise ValueError("Unmapped user or item IDs during matrix construction")

        data = train_df["rating"].values.astype(np.float32)
        user_item = csr_matrix((data, (row.values, col.values)), shape=(len(users), len(items)))
        self.user_item_matrix = (user_item * self.alpha).tocsr()

        self.model = AlternatingLeastSquares(
            factors=self.n_factors,
            iterations=self.n_epochs,
            regularization=self.regularization,
            random_state=42,
        )
        self.model.fit(self.user_item_matrix)

        logger.info(
            f"CollaborativeModel (ALS) trained on {len(users)} users, "
            f"{len(items)} items, {len(train_df)} ratings"
        )
        return self

    def predict(self, user_id: str, item_ids: list[str]) -> list[tuple[str, float]]:
        if self.model is None:
            raise RuntimeError("Model not fitted. Call fit() before predict()")

        if user_id not in self.user_map:
            return [(item_id, self.global_mean) for item_id in item_ids]

        user_idx = self.user_map[user_id]
        user_vec = self.model.user_factors[user_idx]
        predictions = []

        for item_id in item_ids:
            if item_id not in self.item_map:
                score = self.global_mean
            else:
                item_idx = self.item_map[item_id]
                item_vec = self.model.item_factors[item_idx]
                score = float(np.dot(user_vec, item_vec))
            predictions.append((item_id, score))
        return predictions

    def recommend(
        self, user_id: str, n: int = 10, exclude_seen: bool = True
    ) -> list[tuple[str, float]]:
        if self.model is None:
            raise RuntimeError("Model not fitted. Call fit() before recommend()")

        if n <= 0:
            raise ValueError(f"n must be positive, got {n}")

        # Cold-start: unknown user → return popular items
        if user_id not in self.user_map:
            logger.warning(f"Unknown user '{user_id}': returning popularity-based fallback")
            return [(item, self.global_mean) for item in self._popular_items[:n]]

        user_idx = self.user_map[user_id]

        # Clamp n to the number of available unseen items
        actual_n = min(n, len(self.all_items))

        item_indices, scores = self.model.recommend(
            user_idx,
            self.user_item_matrix[user_idx],
            N=actual_n,
            filter_already_liked_items=exclude_seen,
        )

        results = []
        for idx, score in zip(item_indices, scores):
            int_idx = int(idx)
            item_id = self.reverse_item_map.get(int_idx)
            if item_id is None:
                logger.warning(
                    f"Index {int_idx} not found in reverse_item_map "
                    f"(map size={len(self.reverse_item_map)})"
                )
                continue
            results.append((item_id, float(score)))

        return results
