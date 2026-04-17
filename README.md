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
git clone https://github.com/saadyaq/RecEngine.git
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
- [x] Week 5 — FastAPI + A/B testing router
- [x] Week 6 — Docker + GCP Cloud Run deployment
- [x] Week 7 — Monitoring (Evidently drift, Prometheus, Streamlit dashboard)
- [x] Week 8 — Automated retraining + CI/CT/CD
- [x] Week 9 — Documentation + polish + v1.0.0 release

## Docker & Deployment

### Docker Compose (Local Development)

```bash
# Start all services (API, MLflow, Dashboard)
docker-compose up --build

# Access services:
# - API: http://localhost:8000
# - MLflow UI: http://localhost:5000
# - Dashboard: http://localhost:8501

# View logs
docker-compose logs -f api

# Stop all services
docker-compose down
```

### Docker (Single Container)

```bash
# Build image
docker build -t recengine .

# Run container
docker run -p 8000:8000 \
  -e DEPLOYMENT_MODE=ab_test \
  -e MLFLOW_TRACKING_URI=http://localhost:5000 \
  recengine

# Test health endpoint
curl http://localhost:8000/health
```

### CI/CD Pipeline

**CI (Continuous Integration)** - Runs on every push/PR:
- Linting with ruff
- Formatting check with black
- Type checking with mypy
- Tests with pytest

**CD (Continuous Deployment)** - Runs on merge to main:
- Builds Docker image
- Pushes to GCP Artifact Registry
- Deploys to Cloud Run
- Verifies deployment health

**CT (Continuous Training)** - Runs weekly (Sunday midnight UTC):
- Retrains all models
- Evaluates against current production
- Promotes to staging if improved
- Uploads training artifacts

### GCP Cloud Run Deployment

**Prerequisites:**
1. GCP project with Cloud Run API enabled
2. Artifact Registry configured
3. Service account with appropriate permissions

**GitHub Secrets Required:**
- `GCP_CREDENTIALS`: Service account JSON key
- `GCP_PROJECT_ID`: Your GCP project ID
- `GCP_REGION`: Deployment region (e.g., `us-central1`)

**GitHub Variables Required:**
- `DEPLOYMENT_MODE`: shadow, canary, or ab_test
- `SPLIT_RATIO`: Traffic split (0.0 to 1.0)
- `MLFLOW_TRACKING_URI`: MLflow server URL

**Deploy manually:**
```bash
gcloud run deploy recengine \
  --image us-central1-docker.pkg.dev/YOUR_PROJECT/recengine/recengine:latest \
  --platform managed \
  --region us-central1 \
  --allow-unauthenticated \
  --set-env-vars DEPLOYMENT_MODE=ab_test,SPLIT_RATIO=0.5
```

## Dataset

Amazon Reviews 2023 — Electronics category
Source: [McAuley-Lab/Amazon-Reviews-2023](https://huggingface.co/datasets/McAuley-Lab/Amazon-Reviews-2023)

- 2M reviews loaded (streaming), ~186K after filtering (MIN_INTERACTIONS=10)
- 19,340 users, 5,956 items
- Train: 141,575 ratings | Test: 44,856 ratings
- Temporal split: 80% oldest per user → train, 20% most recent → test
