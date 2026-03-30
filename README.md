# RecEngine

Production-ready recommendation engine with two-stage architecture, A/B testing, and full MLOps infrastructure.

## Architecture

```
                    ┌─────────────────────────────────────────────┐
                    │              RETRIEVAL (Stage 1)             │
                    │                                              │
  User Request ───► │  Model A (ALS)        Model B (Semantic)    │
                    │  Collaborative        Sentence-Transformers  │
                    │  Filtering            + FAISS Index          │
                    │  ~100 candidates      ~100 candidates        │
                    └──────────────┬──────────────────────────────┘
                                   │ ~150-200 unique candidates
                    ┌──────────────▼──────────────────────────────┐
                    │              RANKING (Stage 2)               │
                    │                                              │
                    │  Model C (XGBoost CTR)                       │
                    │  Predicts P(click) per candidate             │
                    │  Features: user stats, item stats,           │
                    │            model_a_score, model_b_score      │
                    └──────────────┬──────────────────────────────┘
                                   │ Top 10 recommendations
                    ┌──────────────▼──────────────────────────────┐
                    │           FastAPI /recommend                 │
                    │           A/B Router (shadow/canary/ab)      │
                    └─────────────────────────────────────────────┘
```

## Stack

| Layer | Technology |
|---|---|
| Data | Amazon Reviews 2023 (Electronics), HuggingFace, DVC |
| Model A | Implicit ALS (collaborative filtering) |
| Model B | sentence-transformers (all-MiniLM-L6-v2) + FAISS |
| Model C | XGBoost (CTR prediction) + SHAP |
| Tracking | MLflow |
| Serving | FastAPI + Uvicorn |
| Monitoring | Evidently (drift) + Prometheus |
| Dashboard | Streamlit |
| CI/CD | GitHub Actions |
| Infrastructure | Docker, GCP Cloud Run |

## Results

| Model | NDCG@10 | Precision@10 | Recall@10 |
|---|---|---|---|
| Model A (ALS) | 0.0132 | 0.0045 | 0.0207 |
| Model B (Semantic) | 0.0031 | 0.0011 | 0.0053 |
| Pipeline A+C | **0.0152** | **0.0058** | **0.0250** |

Model C AUC-ROC: **0.764**

Pipeline A+C improves over Model A alone: +15% NDCG@10, +29% Precision@10.

## Quick Start

### Local

```bash
# Setup
git clone https://github.com/your-username/recengine.git
cd recengine
python -m venv venv && source venv/bin/activate
pip install -e ".[dev]"

# Run data pipeline
python -m src.data.pipeline

# Train models
python -m src.training.train              # Model A only
python -m src.training.train --model-b   # Model A + B
python -m src.training.train --model-c   # Model A + C (full pipeline)

# Start API
uvicorn src.serving.app:app --port 8000
```

### Docker

```bash
docker-compose up
```

API: http://localhost:8000
MLflow UI: http://localhost:5000
Dashboard: http://localhost:8501

## API

```bash
# Get recommendations
curl -X POST http://localhost:8000/recommend \
  -H "Content-Type: application/json" \
  -d '{"user_id": "USER123", "num_results": 10}'

# Send feedback
curl -X POST http://localhost:8000/feedback \
  -H "Content-Type: application/json" \
  -d '{"user_id": "USER123", "item_id": "ITEM456", "action": "click"}'

# Health check
curl http://localhost:8000/health
```

## A/B Testing

Three deployment modes controlled by `DEPLOYMENT_MODE` env variable:

- **shadow**: Both models run, only variant A is returned. B is logged silently.
- **canary**: Variant B serves 5% of traffic.
- **ab_test**: Configurable split (default 50/50). Statistical significance calculated via two-proportion z-test.

## Project Status

- [x] Week 1 — Data pipeline (load, preprocess, temporal split, validation)
- [x] Week 2 — Model A (ALS collaborative filtering, MLflow tracking, hyperparameter tuning)
- [x] Week 3 — Model B (semantic embeddings, FAISS index, cold start analysis)
- [x] Week 4 — Model C (XGBoost CTR, feature engineering, SHAP, re-ranking pipeline)
- [ ] Week 5 — FastAPI + A/B testing router
- [ ] Week 6 — Docker + GCP Cloud Run deployment
- [ ] Week 7 — Monitoring (Evidently drift, Prometheus, Streamlit dashboard)
- [ ] Week 8 — Automated retraining + CI/CT/CD
- [ ] Week 9 — Documentation + polish + v1.0.0 release

## Dataset

Amazon Reviews 2023 — Electronics category
Source: [McAuley-Lab/Amazon-Reviews-2023](https://huggingface.co/datasets/McAuley-Lab/Amazon-Reviews-2023)

- 2M reviews loaded (streaming), ~186K after filtering (MIN_INTERACTIONS=10)
- 19,340 users, 5,956 items
- Train: 141,575 ratings | Test: 44,856 ratings
- Temporal split: 80% oldest per user → train, 20% most recent → test
