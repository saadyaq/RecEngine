# Model Card — Model A (Collaborative Filtering)

## Overview

| | |
|---|---|
| **Type** | Implicit Alternating Least Squares (ALS) |
| **Library** | `implicit` |
| **Task** | Top-N item retrieval from user-item interaction history |
| **Role in pipeline** | Stage 1 retrieval — generates ~100 candidates per user |

## Training Data

- **Dataset:** Amazon Reviews 2023, Electronics category
- **Size:** ~141K user-item interactions (train split)
- **Users:** 19,340 | **Items:** 5,956
- **Split:** Temporal — oldest 80% per user in train, most recent 20% in test

## Hyperparameters (best run)

| Parameter | Value |
|---|---|
| n_factors | 100 |
| n_epochs | 20 |
| regularization | 0.01 |
| alpha (confidence scaling) | 40.0 |

## Evaluation Results (test set)

| Metric | Value |
|---|---|
| NDCG@10 | 0.0132 |
| Precision@10 | 0.0045 |
| Recall@10 | 0.0207 |
| NDCG@20 | 0.0161 |

## Limitations

- **Cold start:** Cannot generate recommendations for users or items with no interaction history. New users receive popular-item fallbacks.
- **Context-free:** Does not use item content (title, description, category). Two items with identical interaction patterns look identical to this model.
- **Popularity bias:** Popular items tend to be over-recommended because they have more interactions.

## Usage

```python
from src.models.collaborative import CollaborativeModel

model = CollaborativeModel(n_factors=100, n_epochs=20)
model.fit(train_df)
recs = model.recommend("user_id_123", n=10, exclude_seen=True)
# Returns: [("item_id", score), ...]
```
