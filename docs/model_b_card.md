# Model Card — Model B (Semantic Embeddings)

## Overview

| | |
|---|---|
| **Type** | Dense retrieval via sentence embeddings + FAISS ANN index |
| **Backbone** | `all-MiniLM-L6-v2` (sentence-transformers) |
| **Task** | Top-N item retrieval via semantic similarity |
| **Role in pipeline** | Stage 1 retrieval — generates ~100 candidates per user |

## How it works

1. Each product is represented as a text: `title + description + features` (truncated/padded as needed).
2. All product texts are encoded into 384-dimensional L2-normalized vectors using `all-MiniLM-L6-v2`.
3. Vectors are indexed in a FAISS `IndexFlatIP` (inner product = cosine similarity on normalized vectors).
4. At inference: for each item the user has rated ≥ 4, find the K most similar items in the FAISS index. Aggregate scores (max pooling), exclude seen items, return top N.

## Training Data

- Same metadata as the rest of the pipeline: Amazon Reviews 2023, Electronics category
- **Items indexed:** 5,956 products from the training split
- Embedding model weights are frozen (no fine-tuning)

## Evaluation Results (test set, sample of 1,000 users)

| Metric | Value |
|---|---|
| NDCG@10 | 0.0031 |
| Precision@10 | 0.0011 |
| Recall@10 | 0.0053 |

## Cold Start Performance

Model B is the only model in this system that can handle item cold start. Items absent from the training interaction data can still be retrieved via their description.

| Segment | NDCG@10 |
|---|---|
| Warm items (seen in train) | 0.0035 |
| Cold-start items (unseen in train) | 0.0018 |

## Limitations

- **User cold start:** Still requires the user to have rated at least one item ≥ 4 to build a preference profile.
- **Text quality:** Products with missing or generic descriptions produce poor embeddings.
- **Language:** English-only. Non-English product titles degrade quality.

## Usage

```python
from src.models.semantic import SemanticModel, build_product_texts

model = SemanticModel(model_name="all-MiniLM-L6-v2")
product_texts = build_product_texts(metadata_df)
model.build_index(product_texts)
model.save_index("data/models/semantic_index")

recs = model.recommend("user_id_123", train_df, n=10)
# Returns: [("item_id", similarity_score), ...]
```
