# ADR 001 — Temporal Train/Test Split

## Context

We need to evaluate recommendation models on held-out data. The standard approach is a random split, but recommendations are inherently sequential: a model deployed in production will always predict future interactions from past behavior.

## Decision

For each user, we sort their interactions by timestamp and assign the oldest 80% to train and the most recent 20% to test. Users with a single interaction are kept entirely in train.

## Consequences

**Why:** A random split would allow the model to "see" future interactions during training, creating data leakage. If a user rated item A in January and item B in February, a random split might put B in train and A in test — the model then implicitly learns from future data. This inflates offline metrics and produces models that perform worse in production than expected.

**How to apply:** Always validate after the split that `max(train.timestamp) < min(test.timestamp)` per user. The `validate_no_leakage` function in `src/data/validation.py` enforces this.

**Trade-off:** Temporal splits produce lower absolute metric values than random splits (the model genuinely has less signal), but the numbers are honest and correlate with production performance.
