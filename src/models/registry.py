import os
import pickle
from pathlib import Path

from loguru import logger

from src.config import settings
from src.models.collaborative import CollaborativeModel
from src.models.ctr import CTRModel

MODEL_DIR = Path(os.getenv("MODEL_DIR", str(settings.PROJECT_ROOT / "data" / "models")))


class ModelRegistry:
    def __init__(self):
        self.model_a: CollaborativeModel | None = None
        self.model_c: CTRModel | None = None
        self.versions: dict[str, str] = {}

    def load(self) -> None:
        """Load models from pickle files, training model_a from scratch if needed."""
        MODEL_DIR.mkdir(parents=True, exist_ok=True)
        self._load_model_a()
        self._load_model_c()

    def _load_model_a(self) -> None:
        path = MODEL_DIR / "model_a.pkl"
        if path.exists():
            with open(path, "rb") as f:
                self.model_a = pickle.load(f)
            self.versions["model_a"] = "local_pickle"
            logger.info(f"Model A loaded from {path}")
        else:
            logger.warning("model_a.pkl not found — training from train.parquet")
            self._train_and_save_model_a(path)

    def _train_and_save_model_a(self, save_path: Path) -> None:
        import pandas as pd

        train_path = settings.DATA_PROCESSED_DIR / "train.parquet"
        if not train_path.exists():
            logger.error(f"No training data at {train_path}; model_a will be None")
            self.versions["model_a"] = "not_loaded"
            return

        train_df = pd.read_parquet(train_path)
        self.model_a = CollaborativeModel(n_factors=50, n_epochs=10)
        self.model_a.fit(train_df)
        with open(save_path, "wb") as f:
            pickle.dump(self.model_a, f)
        self.versions["model_a"] = "trained_from_parquet"
        logger.info(f"Model A trained and saved to {save_path}")

    def _load_model_c(self) -> None:
        path = MODEL_DIR / "model_c.pkl"
        if path.exists():
            with open(path, "rb") as f:
                self.model_c = pickle.load(f)
            self.versions["model_c"] = "local_pickle"
            logger.info(f"Model C loaded from {path}")
        else:
            logger.warning(
                "model_c.pkl not found — CTR re-ranking disabled (Variant B = Variant A)"
            )
            self.versions["model_c"] = "not_loaded"

    def save(self) -> None:
        """Persist current models to pickle files."""
        MODEL_DIR.mkdir(parents=True, exist_ok=True)
        if self.model_a is not None:
            with open(MODEL_DIR / "model_a.pkl", "wb") as f:
                pickle.dump(self.model_a, f)
            logger.info("Model A saved")
        if self.model_c is not None:
            with open(MODEL_DIR / "model_c.pkl", "wb") as f:
                pickle.dump(self.model_c, f)
            logger.info("Model C saved")


registry = ModelRegistry()
