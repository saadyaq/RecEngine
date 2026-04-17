# ADR 004 — Two-Stage Architecture: Retrieval + Ranking

## Context

A catalog of thousands of items cannot be scored by a complex model for every request in real time — the latency would be unacceptable. A single model that is fast enough to score all items (e.g., a dot product) lacks the expressive power of a CTR model with rich cross features.

## Decision

We split the recommendation task into two stages:

**Stage 1 — Retrieval:** Two fast models each generate ~100 candidates independently.
- Model A (ALS collaborative filtering): dot product between user and item factor vectors, O(items) but vectorized.
- Model B (semantic FAISS search): approximate nearest-neighbor on L2-normalized embeddings, O(log items).

The ~150-200 unique candidates from both models are merged and passed to Stage 2.

**Stage 2 — Ranking:** Model C (XGBoost) scores each candidate with rich features (user stats, item stats, model_a_score, model_b_score, category match) and returns the top 10 by predicted P(click).

## Consequences

**Why:** This is the standard industrial approach (used by YouTube, Pinterest, Netflix). Retrieval is cheap and has high recall. Ranking is expensive but only runs on a small candidate set, so it can afford richer features and higher accuracy.

**How to apply:** The `get_recommendations` function in `src/training/train.py` implements the full pipeline. The serving layer in `src/serving/app.py` exposes variant B (full pipeline) vs variant A (retrieval only) for the A/B test.

**Trade-off:** Two-stage systems are harder to debug than a single model. The retrieval ceiling bounds ranking quality: if the right item is not in the candidate set, no amount of ranking improvement helps. We monitor this via recall@200 on the retrieval stage.
