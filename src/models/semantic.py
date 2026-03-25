import json
from pathlib import Path

import faiss
import numpy as np
import pandas as pd
from loguru import logger
from sentence_transformers import SentenceTransformer


def build_product_texts(metadata_df: pd.DataFrame) -> dict[str, str]:
    """Concatenate title + description + features for each product."""
    product_texts = {}
    for _, row in metadata_df.iterrows():
        asin = row["parent_asin"]
        parts = []

        title = row.get("title", "")
        if pd.notna(title) and str(title).strip():
            parts.append(str(title).strip())

        description = row.get("description", "")
        if isinstance(description, list):
            description = " ".join(str(d) for d in description if d)
        elif pd.isna(description):
            description = ""
        description = str(description).strip()
        if description:
            parts.append(description)

        features = row.get("features", "")
        if isinstance(features, list):
            features = " ".join(str(f) for f in features if f)
        elif pd.isna(features):
            features = ""
        features = str(features).strip()
        if features:
            parts.append(features)

        text = " ".join(parts) if parts else "unknown product"
        product_texts[asin] = text

    return product_texts


class SemanticModel:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model_name = model_name
        self.encoder = SentenceTransformer(model_name)
        self.index = None
        self.asin_list: list[str] = []
        self.asin_to_idx: dict[str, int] = {}

    def build_index(self, product_texts: dict[str, str]) -> "SemanticModel":
        """Encode all products and build a FAISS index for cosine similarity."""
        self.asin_list = list(product_texts.keys())
        self.asin_to_idx = {asin: i for i, asin in enumerate(self.asin_list)}
        texts = [product_texts[asin] for asin in self.asin_list]

        logger.info(f"Encoding {len(texts)} products with {self.model_name}...")
        embeddings = self.encoder.encode(
            texts, batch_size=64, show_progress_bar=True, normalize_embeddings=True
        )
        embeddings = np.array(embeddings, dtype=np.float32)

        # FAISS inner product index (cosine similarity since embeddings are normalized)
        dim = embeddings.shape[1]
        self.index = faiss.IndexFlatIP(dim)
        self.index.add(embeddings)

        logger.info(f"FAISS index built: {self.index.ntotal} vectors, dim={dim}")
        return self

    def find_similar(self, product_id: str, n: int = 10) -> list[tuple[str, float]]:
        """Find the N most similar products to a given product."""
        if product_id not in self.asin_to_idx:
            return []

        idx = self.asin_to_idx[product_id]
        # Reconstruct the vector for the query product
        query = self.index.reconstruct(idx).reshape(1, -1)
        # Search n+1 because the product itself will be the top result
        scores, indices = self.index.search(query, n + 1)

        results = []
        for score, idx_found in zip(scores[0], indices[0]):
            if idx_found < 0 or idx_found >= len(self.asin_list):
                continue
            found_asin = self.asin_list[idx_found]
            if found_asin == product_id:
                continue
            results.append((found_asin, float(score)))
            if len(results) >= n:
                break

        return results

    def recommend(
        self,
        user_id: str,
        train_df: pd.DataFrame,
        n: int = 10,
    ) -> list[tuple[str, float]]:
        """Recommend products based on semantic similarity to user's liked items."""
        user_data = train_df[train_df["user_id"] == user_id]
        if user_data.empty:
            return []

        # Get products the user rated >= 4
        liked = user_data[user_data["rating"] >= 4]["parent_asin"].tolist()
        if not liked:
            # Fall back to all rated products
            liked = user_data["parent_asin"].tolist()

        seen_items = set(user_data["parent_asin"])
        candidate_scores: dict[str, list[float]] = {}

        for product_id in liked:
            similar = self.find_similar(product_id, n=20)
            for asin, score in similar:
                if asin in seen_items:
                    continue
                if asin not in candidate_scores:
                    candidate_scores[asin] = []
                candidate_scores[asin].append(score)

        # Aggregate: average score across all liked items that found this candidate
        aggregated = [
            (asin, sum(scores) / len(scores)) for asin, scores in candidate_scores.items()
        ]
        aggregated.sort(key=lambda x: x[1], reverse=True)
        return aggregated[:n]

    def save_index(self, path: str | Path) -> None:
        """Save the FAISS index and asin mapping to disk."""
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self.index, str(path / "faiss.index"))
        with open(path / "asin_list.json", "w") as f:
            json.dump(self.asin_list, f)
        logger.info(f"Index saved to {path}")

    def load_index(self, path: str | Path) -> "SemanticModel":
        """Load a previously saved FAISS index."""
        path = Path(path)
        self.index = faiss.read_index(str(path / "faiss.index"))
        with open(path / "asin_list.json") as f:
            self.asin_list = json.load(f)
        self.asin_to_idx = {asin: i for i, asin in enumerate(self.asin_list)}
        logger.info(f"Index loaded from {path}: {self.index.ntotal} vectors")
        return self
