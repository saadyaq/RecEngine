import hashlib
import os
from dataclasses import dataclass


@dataclass
class RoutingDecision:
    serving_variant: str  # "A" or "B" — what the user receives
    shadow_variant: str | None  # "B" in shadow mode, else None


def assign_variant(user_id: str, split_ratio: float) -> str:
    """Deterministic assignment: same user_id always returns same variant."""
    digest = hashlib.md5(user_id.encode()).hexdigest()
    bucket = int(digest[:8], 16) % 100
    return "B" if bucket < int(split_ratio * 100) else "A"


class ABRouter:
    VALID_MODES = {"shadow", "canary", "ab_test"}

    def __init__(
        self,
        mode: str | None = None,
        split_ratio: float | None = None,
    ):
        self.mode = (mode or os.getenv("DEPLOYMENT_MODE", "ab_test")).lower()
        if self.mode not in self.VALID_MODES:
            raise ValueError(
                f"Invalid DEPLOYMENT_MODE: {self.mode!r}. Must be one of {self.VALID_MODES}"
            )

        if self.mode == "canary":
            self.split_ratio = 0.05
        elif self.mode == "shadow":
            self.split_ratio = 1.0
        else:  # ab_test
            self.split_ratio = split_ratio or float(os.getenv("SPLIT_RATIO", "0.5"))

    def route(self, user_id: str) -> RoutingDecision:
        if self.mode == "shadow":
            # Always serve A; also run B silently in background
            return RoutingDecision(serving_variant="A", shadow_variant="B")
        variant = assign_variant(user_id, self.split_ratio)
        return RoutingDecision(serving_variant=variant, shadow_variant=None)
