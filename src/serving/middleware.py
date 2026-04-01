import json
import os
import threading
from datetime import datetime
from pathlib import Path

from loguru import logger as loguru_logger

from src.config import settings

LOG_DIR = Path(os.getenv("LOG_DIR", str(settings.PROJECT_ROOT / "data" / "logs")))


class PredictionLogger:
    def __init__(self, log_dir: Path = LOG_DIR):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self._pred_path = self.log_dir / "predictions.jsonl"
        self._feedback_path = self.log_dir / "feedback.jsonl"
        self._lock = threading.Lock()

    def log_prediction(
        self,
        user_id: str,
        variant: str,
        item_ids: list[str],
        latency_ms: float,
    ) -> None:
        record = {
            "timestamp": datetime.utcnow().isoformat(),
            "user_id": user_id,
            "variant": variant,
            "items": item_ids,
            "latency_ms": round(latency_ms, 2),
        }
        self._write(self._pred_path, record)
        loguru_logger.info(
            f"predict | user={user_id} variant={variant} "
            f"latency={latency_ms:.1f}ms items={item_ids[:3]}"
        )

    def log_feedback(self, user_id: str, item_id: str, action: str, timestamp: datetime) -> None:
        record = {
            "timestamp": timestamp.isoformat(),
            "logged_at": datetime.utcnow().isoformat(),
            "user_id": user_id,
            "item_id": item_id,
            "action": action,
        }
        self._write(self._feedback_path, record)

    def _write(self, path: Path, record: dict) -> None:
        with self._lock:
            with open(path, "a", encoding="utf-8") as f:
                f.write(json.dumps(record) + "\n")
