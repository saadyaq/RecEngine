# ADR 005 — Evidently for Data and Prediction Drift Monitoring

## Context

A deployed model can silently degrade when the distribution of incoming requests shifts away from the training distribution (data drift) or when the model's output distribution changes (prediction drift). We need a monitoring layer that detects these shifts and alerts before they cause user-visible quality degradation.

## Decision

We use [Evidently](https://github.com/evidentlyai/evidently) for generating drift reports, backed by a lightweight pure-Python fallback (normalized mean shift for numeric columns, total variation distance for categorical) for environments where Evidently is unavailable.

The monitoring pipeline (`src/monitoring/drift.py`) reads prediction logs (`data/logs/predictions.jsonl`), splits them into a reference window (first 50%) and a current window (last 50%), and computes drift per feature. Reports are saved as HTML and as a JSON summary that the Streamlit dashboard reads.

## Consequences

**Why:** Evidently provides production-grade statistical tests (Kolmogorov-Smirnov, chi-squared, Wasserstein distance) with an interpretable HTML report. It is open-source, actively maintained, and has a clean Python API. The fallback ensures monitoring never completely fails even without the library.

**How to apply:** Run `python -m src.monitoring.drift` or click "Generate drift report" in the Streamlit dashboard. Alerts fire via `src/monitoring/alerts.py` when `drift_share > 0.3`. Wire this to a scheduler (cron or Cloud Scheduler) for daily automated checks.

**Trade-off:** Evidently's HTML reports are large (~1MB) and not suitable for high-frequency generation. We generate them on demand or daily, not per-request. For real-time alerting, the Prometheus metrics (`RECOMMEND_LATENCY`, `RECOMMEND_COUNTER`) provide the first signal.
