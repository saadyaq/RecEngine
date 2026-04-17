# Model Card — Model C (CTR Prediction / Re-Ranker)

## Overview

| | |
|---|---|
| **Type** | Gradient boosted trees (XGBoost) |
| **Task** | Binary classification — predicts P(click) for a (user, item) pair |
| **Role in pipeline** | Stage 2 ranking — re-ranks ~150-200 candidates from Models A+B |

## Training Data

**Label construction:**
- **Positive (label=1):** (user, item) pairs from the training set with rating ≥ 4
- **Negative (label=0):** 4 randomly sampled unseen items per positive (4:1 negative ratio)

**Rationale for 4:1 ratio:** In production, the vast majority of items a user sees are not clicked. A ratio of 4:1 mimics this imbalance while keeping training tractable.

**Dataset size:** ~180K rows (after sampling 2,000 users for speed)

## Features

| Feature | Type | Description |
|---|---|---|
| `num_ratings` | user | Number of ratings by this user |
| `avg_rating` | user | User's average rating |
| `rating_std` | user | Standard deviation of user's ratings |
| `days_active` | user | Days between first and last rating |
| `avg_rating_received` | item | Item's average received rating |
| `num_ratings_received` | item | Number of ratings received |
| `description_length` | item | Length of product description |
| `model_a_score` | cross | ALS predicted score for this (user, item) pair |
| `model_b_score` | cross | Semantic similarity score (0 if Model B not available) |

## Evaluation Results

| Metric | Value |
|---|---|
| AUC-ROC | 0.764 |
| Log-loss | 0.421 |

## SHAP Analysis

Top features by mean absolute SHAP value:
1. `model_a_score` — strongest signal; collaborative filtering score is highly predictive
2. `avg_rating_received` — popular/well-rated items are more likely to be clicked
3. `num_ratings` — active users have more predictable preferences
4. `description_length` — items with richer descriptions perform better
5. `model_b_score` — adds marginal signal on top of model_a_score, especially for cold items

**Key insight:** `model_b_score` provides incremental lift over `model_a_score` alone for cold-start items where collaborative signal is sparse.

## Full Pipeline vs Baselines

| System | NDCG@10 | Precision@10 | Recall@10 |
|---|---|---|---|
| Model A alone | 0.0132 | 0.0045 | 0.0207 |
| Model B alone | 0.0031 | 0.0011 | 0.0053 |
| **Pipeline A+B+C** | **0.0152** | **0.0058** | **0.0250** |

Pipeline A+B+C improves over Model A alone: **+15% NDCG@10, +29% Precision@10**.

## Usage

```python
from src.models.ctr import CTRModel

model = CTRModel()
model.fit(X_train, y_train, X_val, y_val)

# Re-rank a list of candidates
candidates = [
    {"parent_asin": "B001", "model_a_score": 4.2, "model_b_score": 0.85, ...},
    {"parent_asin": "B002", "model_a_score": 3.8, "model_b_score": 0.72, ...},
]
reranked = model.rerank(candidates)
# Returns candidates sorted by ctr_score descending
```
