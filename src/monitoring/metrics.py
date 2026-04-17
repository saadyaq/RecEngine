from __future__ import annotations

from prometheus_client import Counter, Histogram

RECOMMEND_COUNTER = Counter(
    "recengine_recommend_total",
    "Total recommend calls",
    ["variant", "status"],
)

RECOMMEND_LATENCY = Histogram(
    "recengine_recommend_latency_seconds",
    "Recommend latency in seconds by variant",
    ["variant"],
    buckets=[0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5],
)

ITEM_SCORE = Histogram(
    "recengine_item_score",
    "Distribution of item scores returned by variant",
    ["variant"],
    buckets=[0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0],
)

FEEDBACK_COUNTER = Counter(
    "recengine_feedback_total",
    "Feedback events by action",
    ["action"],
)

ERROR_COUNTER = Counter(
    "recengine_errors_total",
    "API errors by endpoint",
    ["endpoint"],
)
