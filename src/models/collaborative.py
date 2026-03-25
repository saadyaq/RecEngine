import numpy as np
import pandas as pd
from implicit.als import AlternatingLeastSquares
from loguru import logger
from scipy.sparse import csr_matrix


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

    def fit(self, train_df: pd.DataFrame) -> "CollaborativeModel":
        self.train_df = train_df
        self.all_items = train_df["parent_asin"].unique().tolist()
        self.global_mean = train_df["rating"].mean()

        # Build user/item mappings
        users = train_df["user_id"].unique()
        items = train_df["parent_asin"].unique()
        self.user_map = {u: i for i, u in enumerate(users)}
        self.item_map = {item: i for i, item in enumerate(items)}
        self.reverse_item_map = {i: item for item, i in self.item_map.items()}

        # Build sparse user-item matrix
        row = train_df["user_id"].map(self.user_map).values
        col = train_df["parent_asin"].map(self.item_map).values
        data = train_df["rating"].values.astype(np.float32)

        # Build user-item confidence matrix
        user_item = csr_matrix((data, (row, col)), shape=(len(users), len(items)))
        # Apply confidence weighting: confidence = alpha * rating
        self.user_item_matrix = (user_item * self.alpha).tocsr()

        self.model = AlternatingLeastSquares(
            factors=self.n_factors,
            iterations=self.n_epochs,
            regularization=self.regularization,
            random_state=42,
        )
        # implicit.fit expects user-item matrix
        self.model.fit(self.user_item_matrix)

        logger.info(
            f"CollaborativeModel (ALS) trained on {len(users)} users, "
            f"{len(items)} items, {len(train_df)} ratings"
        )
        return self

    def predict(self, user_id: str, item_ids: list[str]) -> list[tuple[str, float]]:
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
                # Clip to rating range
                score = max(1.0, min(5.0, score))
            predictions.append((item_id, score))
        return predictions

    def recommend(
        self, user_id: str, n: int = 10, exclude_seen: bool = True
    ) -> list[tuple[str, float]]:
        if user_id not in self.user_map:
            return []

        user_idx = self.user_map[user_id]

        # Use implicit's optimized recommend method
        if exclude_seen:
            item_indices, scores = self.model.recommend(
                user_idx,
                self.user_item_matrix[user_idx],
                N=n,
                filter_already_liked_items=True,
            )
        else:
            item_indices, scores = self.model.recommend(
                user_idx,
                self.user_item_matrix[user_idx],
                N=n,
                filter_already_liked_items=False,
            )

        results = []
        for idx, score in zip(item_indices, scores):
            item_id = self.reverse_item_map.get(int(idx))
            if item_id:
                # Map score to 1-5 range
                clipped = max(1.0, min(5.0, float(score)))
                results.append((item_id, clipped))
        return results
