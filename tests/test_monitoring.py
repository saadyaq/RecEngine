from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import pandas as pd

from src.dashboard.app import _sample_size_per_variant, _two_proportion_z_test
from src.monitoring.alerts import check_drift_alert, check_latency_alert
from src.monitoring.drift import DriftAnalysis, analyze_prediction_drift


# ---------------------------------------------------------------------------
# Tâche 7.1 — drift report generates without error
# ---------------------------------------------------------------------------


def test_drift_report_generates(tmp_path: Path) -> None:
    log_dir = tmp_path / "logs"
    log_dir.mkdir()
    report_dir = tmp_path / "reports"

    rows = []
    for i in range(50):
        rows.append(
            json.dumps(
                {
                    "timestamp": f"2024-01-01T{i % 24:02d}:00:00+00:00",
                    "user_id": f"u{i}",
                    "variant": "A" if i % 2 == 0 else "B",
                    "items": [f"item_{j}" for j in range(5)],
                    "latency_ms": 50.0 + i,
                }
            )
        )
    (log_dir / "predictions.jsonl").write_text("\n".join(rows), encoding="utf-8")

    with patch("src.monitoring.drift.REPORT_DIR", report_dir):
        analysis = analyze_prediction_drift(log_dir=log_dir, report_dir=report_dir, min_rows=10)

    assert analysis.status in ("ok", "fallback", "not_enough_data")
    assert analysis.reference_rows >= 0
    assert analysis.current_rows >= 0
    if analysis.status in ("ok", "fallback"):
        summary_path = report_dir / "drift_summary.json"
        assert summary_path.exists()
        summary = json.loads(summary_path.read_text())
        assert "drift_detected" in summary


# ---------------------------------------------------------------------------
# Tâche 7.2 — /metrics endpoint returns Prometheus text
# ---------------------------------------------------------------------------


def test_metrics_endpoint() -> None:
    from starlette.testclient import TestClient
    from unittest.mock import patch

    from src.models.collaborative import CollaborativeModel
    from src.models.registry import ModelRegistry
    from src.serving.app import app
    from src.serving.middleware import PredictionLogger

    rows = []
    for u_idx, user in enumerate(["u1", "u2"]):
        for i_idx, item in enumerate(["i1", "i2", "i3"]):
            rows.append(
                {
                    "user_id": user,
                    "parent_asin": item,
                    "rating": float((u_idx + i_idx) % 5 + 1),
                    "timestamp": u_idx * 3 + i_idx,
                }
            )
    df = pd.DataFrame(rows)
    model_a = CollaborativeModel(n_factors=5, n_epochs=3)
    model_a.fit(df)
    reg = ModelRegistry()
    reg.model_a = model_a
    reg.model_c = None
    reg.versions = {"model_a": "test", "model_c": "not_loaded"}

    with patch.object(ModelRegistry, "load", lambda self: None):
        with TestClient(app) as tc:
            app.state.registry = reg
            app.state.pred_logger = PredictionLogger(log_dir=Path("/tmp/test_logs"))
            resp = tc.get("/metrics")

    assert resp.status_code == 200
    body = resp.text
    assert "recengine_recommend_total" in body or "recengine_recommend" in body


# ---------------------------------------------------------------------------
# Tâche 7.3 — z-test calculation is correct
# ---------------------------------------------------------------------------


def test_z_test_calculation() -> None:
    # identical rates → not significant
    z, p = _two_proportion_z_test(50, 1000, 50, 1000)
    assert z == 0.0
    assert p == 1.0

    # large difference → significant
    z2, p2 = _two_proportion_z_test(10, 1000, 100, 1000)
    assert abs(z2) > 1.96
    assert p2 < 0.05

    # zero impressions → safe fallback
    z3, p3 = _two_proportion_z_test(0, 0, 10, 100)
    assert p3 == 1.0


def test_sample_size_calculator() -> None:
    n = _sample_size_per_variant(baseline_rate=0.05, mde=0.01, power=0.80)
    assert n > 0
    # larger MDE → smaller sample needed
    n_large_mde = _sample_size_per_variant(baseline_rate=0.05, mde=0.05, power=0.80)
    assert n_large_mde < n


def test_alerts_drift() -> None:
    analysis_no_drift = DriftAnalysis(
        status="ok", drift_detected=False, drift_share=0.1, reference_rows=100, current_rows=100
    )
    alert = check_drift_alert(analysis_no_drift)
    assert not alert.triggered

    analysis_drift = DriftAnalysis(
        status="ok", drift_detected=True, drift_share=0.5, reference_rows=100, current_rows=100
    )
    alert2 = check_drift_alert(analysis_drift)
    assert alert2.triggered


def test_alerts_latency() -> None:
    alert_ok = check_latency_alert(avg_latency_ms=100.0)
    assert not alert_ok.triggered

    alert_high = check_latency_alert(avg_latency_ms=600.0)
    assert alert_high.triggered
