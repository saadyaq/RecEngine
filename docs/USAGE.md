# RecEngine — Usage Guide

## Table of Contents

1. [Prerequisites](#prerequisites)
2. [Local Setup](#local-setup)
3. [Data Pipeline](#data-pipeline)
4. [Training Models](#training-models)
5. [Running the API](#running-the-api)
6. [API Reference](#api-reference)
7. [A/B Testing](#ab-testing)
8. [Monitoring](#monitoring)
9. [Docker](#docker)
10. [Retraining](#retraining)

---

## Prerequisites

- Python 3.11
- Git
- Docker (optional, for containerized setup)

---

## Local Setup

```bash
# 1. Clone the repo
git clone https://github.com/saadyaq/recengine.git
cd recengine

# 2. Create and activate a virtual environment
python -m venv venv
source venv/bin/activate        # Linux/Mac
# or
venv\Scripts\activate           # Windows

# 3. Install dependencies
pip install -e ".[dev]"

# 4. Install pre-commit hooks
pre-commit install

# 5. Copy environment variables
cp .env.example .env
```

**.env keys:**

```
DEPLOYMENT_MODE=shadow          # shadow | canary | ab_test
SPLIT_RATIO=0.5                 # fraction of traffic to variant B
MLFLOW_TRACKING_URI=http://localhost:5000
MIN_INTERACTIONS=10
```

---

## Data Pipeline

Downloads Amazon Reviews 2023 (Electronics) from HuggingFace, preprocesses, splits temporally, validates, and saves to parquet.

```bash
python -m src.data.pipeline
```

**Output files:**

| File | Description |
|---|---|
| `data/processed/train.parquet` | ~141K user-item interactions |
| `data/processed/test.parquet` | ~45K interactions (most recent per user) |
| `data/processed/metadata.parquet` | Product titles, descriptions, categories |

**Custom settings** via `.env`:

```
DATASET_CATEGORY=Electronics    # HuggingFace dataset category
MIN_INTERACTIONS=10             # Minimum ratings per user and item
TEST_RATIO=0.2                  # Fraction of each user's history in test
```

---

## Training Models

### Model A only (fast, ~5 min)

```bash
python -m src.training.train
```

### Model A + B (adds semantic embeddings, ~20 min)

```bash
python -m src.training.train --model-b
```

### Full pipeline A + B + C (adds CTR re-ranker, ~40 min)

```bash
python -m src.training.train --model-c
```

### With hyperparameter tuning

```bash
python -m src.training.train --tuning
```

### With automatic promotion to MLflow staging

```bash
python -m src.training.train --auto-promote
```

**View results in MLflow:**

```bash
mlflow ui --port 5000
# Open http://localhost:5000
```

Experiments created:
- `recengine-model-a` — ALS collaborative filtering runs
- `recengine-model-b` — Semantic embedding runs
- `recengine-model-c` — CTR XGBoost runs
- `recengine-comparison` — Model A vs B vs full pipeline

---

## Running the API

```bash
uvicorn src.serving.app:app --host 0.0.0.0 --port 8000 --reload
```

The API loads `data/models/model_a.pkl` at startup. Set `DEPLOYMENT_MODE` before starting:

```bash
# Git Bash / Linux / Mac
DEPLOYMENT_MODE=ab_test uvicorn src.serving.app:app --port 8000
```

```cmd
# Windows CMD
set DEPLOYMENT_MODE=ab_test && uvicorn src.serving.app:app --port 8000
```

```powershell
# PowerShell
$env:DEPLOYMENT_MODE="ab_test"; uvicorn src.serving.app:app --port 8000
```

---

## API Reference

### `POST /recommend`

Get top-N recommendations for a user.

**Request:**
```json
{
  "user_id": "AHXC7MR65XTNI6JTPQXOZKBPXHHA",
  "num_results": 10
}
```

**Response:**
```json
{
  "items": [
    {"item_id": "B09X4BPKWK", "score": 4.82, "rank": 1},
    {"item_id": "B08N5WRWNW", "score": 4.71, "rank": 2}
  ],
  "variant": "A",
  "latency_ms": 12.4
}
```

**curl example (Git Bash / Linux / Mac):**
```bash
curl -s -X POST http://localhost:8000/recommend \
  -H "Content-Type: application/json" \
  -d '{"user_id": "AHXC7MR65XTNI6JTPQXOZKBPXHHA", "num_results": 5}' | python -m json.tool
```

**curl example (Windows CMD):**
```cmd
curl -s -X POST http://localhost:8000/recommend -H "Content-Type: application/json" -d "{\"user_id\": \"AHXC7MR65XTNI6JTPQXOZKBPXHHA\", \"num_results\": 5}"
```

**PowerShell:**
```powershell
Invoke-RestMethod -Method Post -Uri http://localhost:8000/recommend -ContentType "application/json" -Body '{"user_id": "AHXC7MR65XTNI6JTPQXOZKBPXHHA", "num_results": 5}'
```

---

### `POST /feedback`

Log a user interaction (click, purchase, or ignore).

**Request:**
```json
{
  "user_id": "AHXC7MR65XTNI6JTPQXOZKBPXHHA",
  "item_id": "B09X4BPKWK",
  "action": "click"
}
```

`action` must be one of: `click` | `purchase` | `ignore`

**curl (Git Bash / Linux / Mac):**
```bash
curl -s -X POST http://localhost:8000/feedback \
  -H "Content-Type: application/json" \
  -d '{"user_id": "USER123", "item_id": "B09X4BPKWK", "action": "purchase"}'
```

**curl (Windows CMD):**
```cmd
curl -s -X POST http://localhost:8000/feedback -H "Content-Type: application/json" -d "{\"user_id\": \"USER123\", \"item_id\": \"B09X4BPKWK\", \"action\": \"purchase\"}"
```

**PowerShell:**
```powershell
Invoke-RestMethod -Method Post -Uri http://localhost:8000/feedback -ContentType "application/json" -Body '{"user_id": "USER123", "item_id": "B09X4BPKWK", "action": "purchase"}'
```

---

### `GET /health`

Check API and model status.

```bash
curl http://localhost:8000/health
```

**Response:**
```json
{
  "status": "ok",
  "model_versions": {
    "model_a": "1.0",
    "model_c": "not_loaded"
  },
  "uptime_seconds": 142.3
}
```

---

### `GET /metrics`

Prometheus metrics endpoint.

```bash
curl http://localhost:8000/metrics
```

Key metrics exposed:
- `recengine_recommend_total{variant, status}` — request count
- `recengine_recommend_latency_seconds{variant}` — latency histogram
- `recengine_item_score{variant}` — score distribution
- `recengine_feedback_total{action}` — feedback event count
- `recengine_errors_total{endpoint}` — error count

---

## A/B Testing

Control the deployment mode via the `DEPLOYMENT_MODE` environment variable:

| Mode | Behaviour |
|---|---|
| `shadow` | Both models run; only variant A is returned. B is logged silently. |
| `canary` | Variant B serves 5% of traffic. |
| `ab_test` | Configurable split (default 50/50). |

**Set the split ratio:**

```bash
# Git Bash / Linux / Mac
DEPLOYMENT_MODE=ab_test SPLIT_RATIO=0.2 uvicorn src.serving.app:app --port 8000
```

```cmd
# Windows CMD
set DEPLOYMENT_MODE=ab_test && set SPLIT_RATIO=0.2 && uvicorn src.serving.app:app --port 8000
```

**Assignment is deterministic:** the same `user_id` always maps to the same variant. Verify:

```cmd
curl -s -X POST http://localhost:8000/recommend -H "Content-Type: application/json" -d "{\"user_id\": \"USER123\"}"
```

**Prediction logs** are written to `data/logs/predictions.jsonl`. Each line:

```json
{
  "timestamp": "2024-01-15T10:23:41+00:00",
  "user_id": "USER123",
  "variant": "A",
  "items": ["B09X4BPKWK", "B08N5WRWNW"],
  "latency_ms": 14.2
}
```

---

## Monitoring

### Streamlit Dashboard

```bash
streamlit run src/dashboard/app.py
# Open http://localhost:8501
```

The dashboard shows:
- KPIs: total recommendations, avg latency, click rate
- Prediction and feedback activity charts
- **A/B test analysis:** conversion rate per variant, z-test significance calculator, sample size calculator
- **Drift report:** Evidently HTML report embedded inline

### Generate a Drift Report (CLI)

```bash
python -m src.monitoring.drift
```

Reports are saved to:
- `data/reports/drift_report.html` — full Evidently HTML report
- `data/reports/drift_summary.json` — machine-readable summary

An alert fires (logged via loguru) when `drift_share > 0.3`.

### Statistical Significance

Use the calculator in the dashboard, or compute manually:

```python
from src.dashboard.app import _two_proportion_z_test, _sample_size_per_variant

# Test if variant B is significantly better
z, p = _two_proportion_z_test(clicks_a=50, n_a=1000, clicks_b=70, n_b=1000)
print(f"z={z:.3f}, p={p:.4f}, significant={p < 0.05}")

# How many users do you need?
n = _sample_size_per_variant(baseline_rate=0.05, mde=0.01, power=0.80)
print(f"Need {n:,} users per variant")
```

---

## Docker

### Start all services

```bash
docker-compose up --build
```

| Service | URL |
|---|---|
| API | http://localhost:8000 |
| MLflow UI | http://localhost:5000 |
| Streamlit Dashboard | http://localhost:8501 |

### Single container

```bash
docker build -t recengine .
docker run -p 8000:8000 -e DEPLOYMENT_MODE=ab_test recengine
```

---

## Retraining

### Manual retraining

```bash
python -m src.training.train --model-c --auto-promote
```

### Promote a run to staging

```bash
python -m src.training.promote model-a <RUN_ID>
```

### Promote staging → canary

```bash
python -m src.training.promote model-a <RUN_ID> --canary
```

### Promote canary → production

```bash
python -m src.training.promote model-a <RUN_ID> --production
```

### Rollback

```python
from src.training.promote import rollback_to_previous
rollback_to_previous("model-a")  # Restores the last archived version
```

### Automated weekly retraining (GitHub Actions)

The CT pipeline runs every Sunday at 2AM UTC. To trigger manually:

1. Go to **Actions → Continuous Training → Run workflow**
2. Select model (`all`, `model-a`, `model-b`, `model-c`) and whether to run tuning

---

## Running Tests

```bash
# All tests
pytest tests/ -v

# Single file
pytest tests/test_api.py -v

# With coverage
pytest tests/ --cov=src --cov-report=term-missing
```

Current status: **81 tests, all passing.**
