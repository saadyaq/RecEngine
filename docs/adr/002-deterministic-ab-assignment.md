# ADR 002 — Deterministic A/B Variant Assignment via Hashing

## Context

In an A/B test, each user must be consistently assigned to the same variant across all requests. A stateful database mapping user → variant would work but adds latency, a new dependency, and a failure mode.

## Decision

Variant assignment is computed by hashing the `user_id` with MD5, converting the first 8 hex digits to an integer, and taking modulo 100. Values below the split threshold go to variant A, the rest to variant B.

```python
bucket = int(hashlib.md5(user_id.encode()).hexdigest()[:8], 16) % 100
variant = "A" if bucket < split_ratio * 100 else "B"
```

## Consequences

**Why:** The assignment is stateless, O(1), requires no storage, and is reproducible — the same user_id always maps to the same bucket regardless of which API instance handles the request. This is critical for consistent UX and for valid statistical analysis (mixing variants corrupts the test).

**How to apply:** The `ABRouter` class in `src/serving/router.py` implements this. Never change the hash function mid-experiment, as it would reassign users and invalidate the test.

**Trade-off:** Hash-based assignment distributes users pseudo-randomly but not perfectly uniformly at small scale. For large user populations (>10K), the actual split ratio converges to the configured value within ±0.5%.
