import json
from pathlib import Path
from unittest.mock import patch

import pandas as pd
import pytest
from httpx import ASGITransport, AsyncClient

from src.models.collaborative import CollaborativeModel
from src.models.registry import ModelRegistry
from src.serving.app import app
from src.serving.middleware import PredictionLogger


def _make_mini_registry() -> ModelRegistry:
    """Train a tiny Model A on 3 users × 5 items for fast test startup."""
    rows = []
    for u_idx, user in enumerate(["u1", "u2", "u3"]):
        for i_idx, item in enumerate(["i1", "i2", "i3", "i4", "i5"]):
            rows.append(
                {
                    "user_id": user,
                    "parent_asin": item,
                    "rating": float((u_idx + i_idx) % 5 + 1),
                    "timestamp": u_idx * 5 + i_idx,
                }
            )
    df = pd.DataFrame(rows)
    model_a = CollaborativeModel(n_factors=10, n_epochs=5)
    model_a.fit(df)
    reg = ModelRegistry()
    reg.model_a = model_a
    reg.model_c = None
    reg.versions = {"model_a": "test", "model_c": "not_loaded"}
    return reg


@pytest.fixture
async def client(tmp_path: Path):
    mini_reg = _make_mini_registry()
    with patch.object(ModelRegistry, "load", lambda self: None):
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
            app.state.registry = mini_reg
            app.state.pred_logger = PredictionLogger(log_dir=tmp_path)
            yield ac, tmp_path


async def test_health_endpoint(client):
    ac, _ = client
    resp = await ac.get("/health")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "ok"
    assert "model_a" in data["model_versions"]
    assert data["uptime_seconds"] >= 0


async def test_recommend_returns_items(client):
    ac, _ = client
    resp = await ac.post("/recommend", json={"user_id": "u1", "num_results": 3})
    assert resp.status_code == 200
    data = resp.json()
    assert len(data["items"]) == 3
    for item in data["items"]:
        assert "item_id" in item
        assert "score" in item
        assert "rank" in item
    assert data["variant"] in ("A", "B")
    assert data["latency_ms"] >= 0


async def test_recommend_logs_prediction(client):
    ac, tmp_path = client
    await ac.post("/recommend", json={"user_id": "u2", "num_results": 2})
    log_path = tmp_path / "predictions.jsonl"
    assert log_path.exists()
    lines = log_path.read_text().strip().split("\n")
    assert len(lines) >= 1
    record = json.loads(lines[0])
    assert record["user_id"] == "u2"
    assert "variant" in record
    assert "items" in record
    assert "latency_ms" in record


async def test_feedback_endpoint(client):
    ac, _ = client
    resp = await ac.post(
        "/feedback",
        json={
            "user_id": "u1",
            "item_id": "i1",
            "action": "click",
            "timestamp": "2024-01-01T00:00:00",
        },
    )
    assert resp.status_code == 200
    assert resp.json()["status"] == "ok"


async def test_feedback_logged(client):
    ac, tmp_path = client
    await ac.post(
        "/feedback",
        json={"user_id": "u3", "item_id": "i2", "action": "purchase"},
    )
    log_path = tmp_path / "feedback.jsonl"
    assert log_path.exists()
    record = json.loads(log_path.read_text().strip())
    assert record["user_id"] == "u3"
    assert record["action"] == "purchase"


async def test_recommend_unknown_user_returns_fallback(client):
    """Unknown users should get fallback items, not a 500 error."""
    ac, _ = client
    resp = await ac.post("/recommend", json={"user_id": "unknown_user_xyz", "num_results": 5})
    assert resp.status_code == 200
    data = resp.json()
    assert isinstance(data["items"], list)
