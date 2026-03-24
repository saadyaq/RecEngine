import numpy as np
import pandas as pd
from loguru import logger
from scipy.sparse import csr_matrix
from sklearn.decomposition import TruncatedSVD


class CollaborativeModel:
    def __init__(
        self,
        n_factors: int = 100,
        n_epochs: int = 20,
        lr_all: float = 0.005,
        reg_all: float = 0.02,
    ):
        self.n_factors = n_factors
        self.n_epochs = n_epochs
        self.lr_all = lr_all
        self.reg_all = reg_all

        self.svd = None
        self.user_factors = None
        self.item_factors = None
        self.global_mean = 0.0
        self.user_map = {}
        self.item_map = {}
        self.reverse_item_map = {}
        self.train_df = None
        self.all_items = None

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
        data = train_df["rating"].values
        matrix = csr_matrix((data, (row, col)), shape=(len(users), len(items)))

        # SVD decomposition
        n_components = min(self.n_factors, min(matrix.shape) - 1)
        self.svd = TruncatedSVD(n_components=n_components, n_iter=self.n_epochs)
        self.user_factors = self.svd.fit_transform(matrix)
        self.item_factors = self.svd.components_.T

        logger.info(
            f"CollaborativeModel trained on {len(users)} users, "
            f"{len(items)} items, {len(train_df)} ratings"
        )
        return self

    def predict(self, user_id: str, item_ids: list[str]) -> list[tuple[str, float]]:
        predictions = []
        if user_id not in self.user_map:
            return [(item_id, self.global_mean) for item_id in item_ids]

        user_idx = self.user_map[user_id]
        user_vec = self.user_factors[user_idx]

        for item_id in item_ids:
            if item_id not in self.item_map:
                score = self.global_mean
            else:
                item_idx = self.item_map[item_id]
                item_vec = self.item_factors[item_idx]
                score = float(np.dot(user_vec, item_vec))
                score = max(1.0, min(5.0, score))
            predictions.append((item_id, score))
        return predictions

    def recommend(
        self, user_id: str, n: int = 10, exclude_seen: bool = True
    ) -> list[tuple[str, float]]:
        if exclude_seen and self.train_df is not None:
            seen_items = set(self.train_df[self.train_df["user_id"] == user_id]["parent_asin"])
            candidate_items = [i for i in self.all_items if i not in seen_items]
        else:
            candidate_items = self.all_items

        predictions = self.predict(user_id, candidate_items)
        predictions.sort(key=lambda x: x[1], reverse=True)
        return predictions[:n]
