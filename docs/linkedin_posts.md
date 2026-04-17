# LinkedIn Posts — RecEngine

---

## Post 1: A/B Testing for ML Models

**"I built an A/B testing system for ML models. Here's what I learned."**

Most ML tutorials end at model training. I spent 9 weeks building what comes *after* — and A/B testing was the hardest part.

Here's what tripped me up:

**1. Random assignment is wrong.**
If you randomly assign users to variants per request, the same user sees both models. Your metrics are contaminated. The fix: hash the user_id deterministically. Same user → always same variant.

**2. You need three modes, not one.**
→ Shadow: new model runs silently, only A is returned. Zero risk.
→ Canary: new model serves 5% of traffic. Limited blast radius.
→ A/B test: 50/50 split with statistical significance.

Deploying directly to A/B without shadow mode is like shipping untested code to prod.

**3. Statistical significance takes time.**
With a 5% baseline CTR and a 1% MDE, you need ~3,800 users per variant to reach 80% power. Most teams declare winners too early.

I built a sample size calculator directly into the Streamlit dashboard. Before launching any test, I know exactly how long it needs to run.

**The result:** A FastAPI service with deterministic routing, full prediction logging, and a z-test dashboard — all production-ready.

What's your approach to A/B testing ML models?

---

## Post 2: Shadow vs Canary vs A/B Testing

**"Shadow deployment vs canary vs A/B testing for ML. When to use which."**

I implemented all three in my recommendation engine. Here's the decision framework I arrived at:

**Shadow deployment**
→ Use when: you're not confident the new model is production-safe
→ New model runs on every request, output is logged but never shown
→ Cost: double compute. Benefit: zero user impact, full latency/error visibility
→ Gate: passes if latency ≈ baseline and no inference errors

**Canary deployment**
→ Use when: shadow passed, but you want real user signal before full rollout
→ 5% of users get the new model
→ Gate: CTR not degraded, error rate < 5%, latency p99 OK

**A/B test**
→ Use when: canary passed, you want to measure impact at scale
→ 50/50 split, run until statistical significance (p < 0.05)
→ Gate: significant positive lift, or null result → rollback

The key insight: each stage is a go/no-go gate. You never skip stages, because a bug that slips through shadow mode reaches 5% of users in canary, not 100%.

In my system, this is controlled by a single env variable: `DEPLOYMENT_MODE=shadow|canary|ab_test`.

---

## Post 3: The Gap Between Training and Production

**"The gap between training a model and running it in production."**

Training a recommendation model took me 2 days. Making it production-ready took 7 more weeks.

Here's everything that's missing from the "train → evaluate → done" narrative:

**Infrastructure**
- FastAPI endpoint that loads models at startup
- Health checks, graceful shutdown, error handling
- Prometheus metrics for latency and throughput

**Deployment**
- Docker multi-stage build to keep image < 1GB
- CI/CD pipeline: push to main → build → deploy to Cloud Run
- Environment variable management across dev/staging/prod

**Model lifecycle**
- MLflow tracking for every training run
- Model Registry with staging/canary/production stages
- Rollback: if error rate spikes after promotion, revert in one command

**Monitoring**
- Evidently for data drift detection on incoming requests
- Alert when >30% of features drift from training distribution
- Weekly automated retraining if drift is detected

**A/B testing**
- Deterministic variant assignment
- Prediction and feedback logging to JSONL
- Statistical significance calculation before declaring a winner

None of this is in Kaggle notebooks. All of it exists in production systems.

The model is 5% of the work.

---

## Post 4: How a CTR Re-Ranking Model Improved My Recommendations

**"How a CTR re-ranking model improved my recommendations. SHAP told me why."**

My recommendation system has two stages:
1. Retrieval: two fast models generate ~200 candidates
2. Ranking: XGBoost predicts P(click) and returns top 10

The ranking model (Model C) improved NDCG@10 by **+15%** over retrieval alone.

SHAP told me why — and the results were surprising.

**Top features by importance:**
1. `model_a_score` — collaborative filtering signal dominates
2. `avg_rating_received` — popular items are more likely to be clicked
3. `num_ratings` — active users are more predictable
4. `description_length` — richer descriptions → higher CTR
5. `model_b_score` — semantic embeddings add marginal but consistent lift

The key insight: `model_b_score` (semantic similarity) adds value *specifically* for users who interacted with few items. When collaborative filtering has sparse signal, semantic embeddings fill the gap.

This validated the two-stage architecture. Model B isn't redundant — it's complementary.

**How I built the training data:**
- Positives: (user, item) pairs with rating ≥ 4
- Negatives: 4 randomly sampled unseen items per positive
- Result: ~180K rows, AUC-ROC = 0.764

The 4:1 negative ratio mimics real-world CTR distributions where most impressions aren't clicked.

---

## Post 5: My End-to-End ML Pipeline

**"My end-to-end ML pipeline: the full architecture."**

I spent 9 weeks building a production-ready recommendation system from scratch. Here's the full architecture:

**Data layer**
→ Amazon Reviews 2023 (Electronics) from HuggingFace, loaded via streaming
→ Temporal train/test split (no data leakage)
→ DVC for data versioning, parquet for storage

**Model layer (two-stage)**
→ Stage 1 — Retrieval:
  - Model A: ALS collaborative filtering (implicit library)
  - Model B: sentence-transformers + FAISS index
→ Stage 2 — Ranking:
  - Model C: XGBoost CTR prediction
  - Features: user stats, item stats, model scores
  - SHAP for explainability

**Serving layer**
→ FastAPI with /recommend, /feedback, /health, /metrics
→ A/B router: shadow / canary / ab_test modes
→ Deterministic user assignment via MD5 hashing
→ Prometheus metrics + JSONL prediction logging

**MLOps layer**
→ MLflow: experiment tracking + model registry
→ Evidently: data drift detection with HTML reports
→ Streamlit dashboard: A/B results, z-test, sample size calculator
→ Automated weekly retraining via GitHub Actions (CT pipeline)
→ Staging → canary → production promotion with rollback

**Infrastructure**
→ Docker multi-stage build
→ docker-compose: API + MLflow + dashboard
→ GCP Cloud Run deployment
→ GitHub Actions: CI (lint/test) + CD (build/deploy) + CT (retrain)

**Results**
→ Pipeline (A+B+C) vs Model A alone: +15% NDCG@10, +29% Precision@10
→ Model C AUC-ROC: 0.764
→ 81 tests, all passing
→ Full CI/CD/CT in GitHub Actions

The phrase that sums it up: retrieval + ranking + A/B testing + drift monitoring + automated retraining = a system that doesn't need a human to babysit it.
