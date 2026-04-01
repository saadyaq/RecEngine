import pytest

from src.serving.router import ABRouter, assign_variant


def test_deterministic_assignment():
    """Same user_id always gets the same variant."""
    v1 = assign_variant("user_abc", 0.5)
    v2 = assign_variant("user_abc", 0.5)
    assert v1 == v2


def test_assignment_returns_valid_variant():
    for uid in ["u1", "u2", "u3", "user_100"]:
        v = assign_variant(uid, 0.5)
        assert v in ("A", "B")


def test_split_ratio_approximately_correct():
    """Over 10,000 users, B assignment ratio should be within 5% of target."""
    n = 10_000
    users = [f"user_{i}" for i in range(n)]
    b_count = sum(1 for u in users if assign_variant(u, 0.5) == "B")
    ratio = b_count / n
    assert abs(ratio - 0.5) < 0.05, f"Expected ~0.50, got {ratio:.3f}"


def test_shadow_mode_always_serves_a():
    router = ABRouter(mode="shadow")
    for uid in [f"user_{i}" for i in range(50)]:
        decision = router.route(uid)
        assert decision.serving_variant == "A"
        assert decision.shadow_variant == "B"


def test_canary_mode_low_b_ratio():
    """Canary is 5% — fewer than 15% of users should get B."""
    router = ABRouter(mode="canary")
    n = 10_000
    b_count = sum(1 for i in range(n) if router.route(f"user_{i}").serving_variant == "B")
    assert b_count / n < 0.15, f"Canary B ratio too high: {b_count / n:.3f}"


def test_ab_test_mode_no_shadow():
    """ab_test mode should not run shadow variants."""
    router = ABRouter(mode="ab_test", split_ratio=0.5)
    for uid in [f"user_{i}" for i in range(20)]:
        decision = router.route(uid)
        assert decision.shadow_variant is None


def test_invalid_mode_raises():
    with pytest.raises(ValueError):
        ABRouter(mode="invalid_mode")
