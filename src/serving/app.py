import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from loguru import logger
from prometheus_client import Counter, Histogram, make_asgi_app

from src.models.registry import ModelRegistry
from src.serving.middleware import PredictionLogger
from src.serving.router import ABRouter
from src.serving.schemas import (
    FeedbackRequest,
    HealthResponse,
    RecommendedItem,
    RecommendRequest,
    RecommendResponse,
)

# Prometheus metrics
RECOMMEND_COUNTER = Counter("recengine_recommend_total", "Total recommend calls", ["variant"])
RECOMMEND_LATENCY = Histogram("recengine_recommend_latency_seconds", "Recommend latency in seconds")
FEEDBACK_COUNTER = Counter("recengine_feedback_total", "Total feedback events", ["action"])


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.startup_time = time.time()
    app.state.registry = ModelRegistry()
    app.state.registry.load()
    app.state.router = ABRouter()
    app.state.pred_logger = PredictionLogger()
    logger.info("RecEngine API started")
    yield
    logger.info("RecEngine API shutting down")


app = FastAPI(title="RecEngine", version="0.1.0", lifespan=lifespan)

# Mount Prometheus /metrics endpoint
metrics_app = make_asgi_app()
app.mount("/metrics", metrics_app)


@app.post("/recommend", response_model=RecommendResponse)
def recommend(req: RecommendRequest) -> RecommendResponse:
    registry: ModelRegistry = app.state.registry
    if registry.model_a is None:
        raise HTTPException(status_code=503, detail="Model A not available")

    t0 = time.perf_counter()
    decision = app.state.router.route(req.user_id)
    serving_variant = decision.serving_variant

    items = _run_variant(serving_variant, req.user_id, req.num_results, registry)

    # Shadow mode: also run B and log it (fire-and-forget)
    if decision.shadow_variant:
        try:
            shadow_items = _run_variant(
                decision.shadow_variant, req.user_id, req.num_results, registry
            )
            app.state.pred_logger.log_prediction(
                req.user_id,
                decision.shadow_variant,
                [i.item_id for i in shadow_items],
                0.0,
            )
        except Exception as e:
            logger.warning(f"Shadow variant failed: {e}")

    latency_ms = (time.perf_counter() - t0) * 1000
    RECOMMEND_COUNTER.labels(variant=serving_variant).inc()
    RECOMMEND_LATENCY.observe(latency_ms / 1000)

    app.state.pred_logger.log_prediction(
        req.user_id,
        serving_variant,
        [i.item_id for i in items],
        latency_ms,
    )

    return RecommendResponse(items=items, variant=serving_variant, latency_ms=latency_ms)


def _run_variant(
    variant: str, user_id: str, num_results: int, registry: ModelRegistry
) -> list[RecommendedItem]:
    """Execute Variant A (ALS only) or Variant B (ALS + CTR re-ranking)."""
    if variant == "A" or registry.model_c is None:
        # Variant A: ALS recommendations
        recs = registry.model_a.recommend(user_id, n=num_results, exclude_seen=True)
        if not recs:
            recs = _fallback_recs(registry, num_results)
    else:
        # Variant B: ALS retrieval (100 candidates) + CTR re-ranking
        candidates_raw = registry.model_a.recommend(user_id, n=100, exclude_seen=True)
        if not candidates_raw:
            candidates_raw = _fallback_recs(registry, 100)
        candidate_rows = [
            {"parent_asin": item_id, "model_a_score": score, "model_b_score": 0.0}
            for item_id, score in candidates_raw
        ]
        reranked = registry.model_c.rerank(candidate_rows)
        recs = [(r["parent_asin"], r["ctr_score"]) for r in reranked[:num_results]]

    return [
        RecommendedItem(item_id=item_id, score=score, rank=i + 1)
        for i, (item_id, score) in enumerate(recs)
    ]


def _fallback_recs(registry: ModelRegistry, n: int) -> list[tuple[str, float]]:
    """Return top-N popular items as cold-start fallback."""
    items = (registry.model_a.all_items or [])[:n]
    return [(item, 0.0) for item in items]


@app.post("/feedback")
def feedback(req: FeedbackRequest) -> dict:
    app.state.pred_logger.log_feedback(req.user_id, req.item_id, req.action, req.timestamp)
    FEEDBACK_COUNTER.labels(action=req.action).inc()
    return {"status": "ok"}


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    registry: ModelRegistry = app.state.registry
    uptime = time.time() - app.state.startup_time
    return HealthResponse(
        status="ok",
        model_versions=registry.versions,
        uptime_seconds=uptime,
    )
