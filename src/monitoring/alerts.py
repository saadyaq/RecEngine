from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from loguru import logger

if TYPE_CHECKING:
    from src.monitoring.drift import DriftAnalysis

DRIFT_SHARE_THRESHOLD = 0.3
LATENCY_THRESHOLD_MS = 500.0
ERROR_RATE_THRESHOLD = 0.05


@dataclass
class Alert:
    name: str
    triggered: bool
    message: str


def check_drift_alert(
    analysis: DriftAnalysis,
    threshold: float = DRIFT_SHARE_THRESHOLD,
) -> Alert:
    triggered = bool(analysis.drift_detected or (analysis.drift_share or 0.0) > threshold)
    message = (
        f"Drift detected: {analysis.drift_share:.1%} of features drifted "
        f"(threshold={threshold:.0%})."
        if triggered
        else "No significant drift detected."
    )
    if triggered:
        logger.warning(f"[ALERT] {message}")
    return Alert(name="drift", triggered=triggered, message=message)


def check_latency_alert(
    avg_latency_ms: float,
    threshold: float = LATENCY_THRESHOLD_MS,
) -> Alert:
    triggered = avg_latency_ms > threshold
    message = (
        f"High latency: {avg_latency_ms:.1f} ms (threshold={threshold:.0f} ms)."
        if triggered
        else f"Latency OK: {avg_latency_ms:.1f} ms."
    )
    if triggered:
        logger.warning(f"[ALERT] {message}")
    return Alert(name="latency", triggered=triggered, message=message)


def check_error_rate_alert(
    error_count: int,
    total_count: int,
    threshold: float = ERROR_RATE_THRESHOLD,
) -> Alert:
    rate = error_count / max(total_count, 1)
    triggered = rate > threshold
    message = (
        f"High error rate: {rate:.1%} (threshold={threshold:.0%})."
        if triggered
        else f"Error rate OK: {rate:.1%}."
    )
    if triggered:
        logger.warning(f"[ALERT] {message}")
    return Alert(name="error_rate", triggered=triggered, message=message)


def run_all_alerts(
    analysis: DriftAnalysis | None = None,
    avg_latency_ms: float = 0.0,
    error_count: int = 0,
    total_count: int = 0,
) -> list[Alert]:
    alerts: list[Alert] = []
    if analysis is not None and analysis.status == "ok":
        alerts.append(check_drift_alert(analysis))
    alerts.append(check_latency_alert(avg_latency_ms))
    alerts.append(check_error_rate_alert(error_count, total_count))
    return alerts
