import numpy as np
import pandas as pd
from loguru import logger
from xgboost import XGBClassifier

from src.models.features import build_cross_features


def build_ctr_dataset(
    train_df: pd.DataFrame,
    user_features: pd.DataFrame,
    item_features: pd.DataFrame,
    model_a,
    model_b,
    neg_ratio: int = 4,
    sample_users: int = None,
) -> pd.DataFrame:
    """
    Build CTR dataset with positive (label=1) and negative (label=0) pairs.
    For each positive interaction (rating >= 4), sample neg_ratio random unseen items.
    """
    rng = np.random.default_rng(42)
    all_items = train_df["parent_asin"].unique()

    positives = train_df[train_df["rating"] >= 4][["user_id", "parent_asin"]].copy()
    positives["label"] = 1

    if sample_users is not None:
        users_sample = rng.choice(
            positives["user_id"].unique(),
            size=min(sample_users, positives["user_id"].nunique()),
            replace=False,
        )
        positives = positives[positives["user_id"].isin(users_sample)]

    rows = []
    seen_by_user: dict[str, set] = {}
    for _, row in train_df.iterrows():
        uid = row["user_id"]
        if uid not in seen_by_user:
            seen_by_user[uid] = set()
        seen_by_user[uid].add(row["parent_asin"])

    logger.info(f"Building CTR dataset: {len(positives)} positives, neg_ratio={neg_ratio}")

    for _, pos_row in positives.iterrows():
        user_id = pos_row["user_id"]
        item_id = pos_row["parent_asin"]

        # Positive pair
        cross = build_cross_features(user_id, item_id, model_a, model_b, train_df, user_features)
        rows.append({"user_id": user_id, "parent_asin": item_id, "label": 1, **cross})

        # Negative samples
        seen = seen_by_user.get(user_id, set())
        unseen = [i for i in all_items if i not in seen and i != item_id]
        n_neg = min(neg_ratio, len(unseen))
        neg_items = rng.choice(unseen, size=n_neg, replace=False)

        for neg_item in neg_items:
            cross_neg = build_cross_features(
                user_id, neg_item, model_a, model_b, train_df, user_features
            )
            rows.append({"user_id": user_id, "parent_asin": neg_item, "label": 0, **cross_neg})

    ctr_df = pd.DataFrame(rows)

    # Join user and item features
    ctr_df = ctr_df.merge(user_features, on="user_id", how="left")
    ctr_df = ctr_df.merge(item_features, on="parent_asin", how="left")
    ctr_df = ctr_df.fillna(0.0)

    pos_count = (ctr_df["label"] == 1).sum()
    neg_count = (ctr_df["label"] == 0).sum()
    logger.info(
        f"CTR dataset: {len(ctr_df)} rows | positives={pos_count}, negatives={neg_count}, ratio={neg_count/pos_count:.1f}:1"
    )

    return ctr_df


class CTRModel:
    def __init__(self, params: dict = None):
        default_params = {
            "n_estimators": 200,
            "max_depth": 6,
            "learning_rate": 0.05,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "use_label_encoder": False,
            "eval_metric": "logloss",
            "random_state": 42,
            "n_jobs": -1,
        }
        if params:
            default_params.update(params)
        self.params = default_params
        self.model = XGBClassifier(**self.params)
        self.feature_cols: list[str] = []

    def fit(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: pd.DataFrame,
        y_val: pd.Series,
    ) -> "CTRModel":
        self.feature_cols = X_train.columns.tolist()
        self.model.fit(
            X_train,
            y_train,
            eval_set=[(X_val, y_val)],
            verbose=False,
        )
        logger.info(
            f"CTRModel trained on {len(X_train)} samples, {len(self.feature_cols)} features"
        )
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Return P(click) probabilities."""
        if isinstance(X, pd.DataFrame):
            X = X[self.feature_cols]
        return self.model.predict_proba(X)[:, 1]

    def rerank(self, candidates: list[dict]) -> list[dict]:
        """
        Re-rank a list of candidate dicts by P(click).
        Each dict must contain the feature columns.
        """
        if not candidates:
            return []
        df = pd.DataFrame(candidates)
        # Fill missing feature cols with 0
        for col in self.feature_cols:
            if col not in df.columns:
                df[col] = 0.0
        scores = self.predict(df[self.feature_cols])
        for i, score in enumerate(scores):
            candidates[i]["ctr_score"] = float(score)
        candidates.sort(key=lambda x: x["ctr_score"], reverse=True)
        return candidates

    def get_feature_importance(self) -> dict[str, float]:
        """Return feature importances as a dict."""
        return dict(zip(self.feature_cols, self.model.feature_importances_))
