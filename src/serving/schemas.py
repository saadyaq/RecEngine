from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field


class RecommendedItem(BaseModel):
    item_id: str
    score: float
    rank: int  # 1-based


class RecommendRequest(BaseModel):
    user_id: str
    num_results: int = Field(default=10, ge=1, le=100)


class RecommendResponse(BaseModel):
    items: list[RecommendedItem]
    variant: str
    latency_ms: float


class FeedbackRequest(BaseModel):
    user_id: str
    item_id: str
    action: Literal["click", "purchase", "ignore"]
    timestamp: datetime = Field(default_factory=datetime.utcnow)


class HealthResponse(BaseModel):
    status: str
    model_versions: dict[str, str]
    uptime_seconds: float
