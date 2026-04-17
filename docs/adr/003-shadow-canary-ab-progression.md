# ADR 003 — Progressive Deployment: Shadow → Canary → A/B Test

## Context

Deploying a new ML model directly to 100% of traffic is risky: bugs in the inference path, degraded latency, or silent quality regressions are only discovered after impacting all users.

## Decision

We implement three deployment modes, used in progressive order:

1. **Shadow mode** (`DEPLOYMENT_MODE=shadow`): The new model runs on every request but its output is only logged, never returned to the user. Zero user impact, full observability.
2. **Canary mode** (`DEPLOYMENT_MODE=canary`): The new model serves 5% of traffic. Monitors latency, error rate, and basic engagement.
3. **A/B test mode** (`DEPLOYMENT_MODE=ab_test`): Configurable split (default 50/50). Statistical significance tested with a two-proportion z-test. A winner is declared when p < 0.05.

## Consequences

**Why:** Shadow mode catches inference bugs and latency regressions before any user is affected. Canary limits the blast radius of a bad model to 5% of users. Full A/B testing produces statistically valid comparisons. Each stage provides a go/no-go gate before the next.

**How to apply:** The `DEPLOYMENT_MODE` environment variable controls which mode is active. The `ABRouter` in `src/serving/router.py` reads this at startup. The Streamlit dashboard's z-test calculator determines when to graduate from canary to full A/B.

**Trade-off:** The progression takes longer than a direct rollout. This is intentional: the cost of a bad model reaching 100% of users outweighs the delay.
