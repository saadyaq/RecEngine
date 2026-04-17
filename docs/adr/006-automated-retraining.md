# ADR 006: Automated Retraining Pipeline

**Date:** 2026-04-17
**Status:** Accepted
**Author:** RecEngine Team

## Context

After deploying the recommendation system to production with A/B testing capabilities, we need a strategy for keeping models fresh as user behavior and product catalogs evolve. Manual retraining is error-prone and doesn't scale.

## Decision

Implement an automated retraining pipeline with the following characteristics:

### 1. Scheduled Retraining
- **Frequency:** Weekly (Sunday at 2 AM UTC)
- **Trigger:** GitHub Actions cron schedule + manual dispatch
- **Scope:** All three models (A, B, C) or selective retraining

### 2. Automatic Promotion Pipeline
Models progress through stages automatically:

```
Training → Staging → Canary (5%) → Production (100%)
```

**Promotion criteria:**
- **Staging:** Candidate model shows improvement on at least one key metric (NDCG@10, AUC-ROC)
- **Canary:** Significant improvement (>1% relative) on multiple metrics
- **Production:** Canary validation period (24h) with acceptable error rate (<5%)

### 3. Rollback Mechanism
Automatic rollback if:
- Canary error rate exceeds threshold
- Manual trigger from operations team
- Production model shows degradation in monitoring

### 4. MLflow Integration
- All training runs logged to MLflow
- Model Registry tracks versions and stages
- Deployment state persisted to JSON for serving layer

## Consequences

### Positive
- **Reduced operational burden:** No manual intervention for routine retraining
- **Faster iteration:** New models reach production in hours, not days
- **Safety:** Gradual rollout with automatic rollback reduces risk
- **Reproducibility:** All experiments tracked in MLflow

### Negative
- **Complexity:** More moving parts to maintain
- **Cost:** Additional compute for weekly retraining
- **Risk of overfitting:** Automated promotion might overfit to validation metrics

### Risks and Mitigations
| Risk | Mitigation |
|------|------------|
| Bad model reaches production | Canary deployment limits blast radius; rollback available |
| Training pipeline breaks | CI/CD tests; manual fallback procedure |
| Data drift undetected | Evidently monitoring triggers ad-hoc retraining |
| MLflow server unavailable | Fallback to SQLite local storage |

## Implementation

### Files
- `src/training/train.py`: Main training pipeline with `--auto-promote` flag
- `src/training/promote.py`: Promotion logic (staging → canary → production)
- `.github/workflows/ct.yml`: GitHub Actions workflow for continuous training
- `tests/test_promotion.py`: Unit tests for promotion logic

### Usage

**Manual training:**
```bash
# Train all models
python -m src.training.train --model-b --model-c

# Train with auto-promotion
python -m src.training.train --model-c --auto-promote

# Hyperparameter tuning
python -m src.training.train --tuning
```

**Manual promotion:**
```bash
# Promote specific run to staging
python -m src.training.promote model-a <run_id>

# Promote directly to canary
python -m src.training.promote model-c <run_id> --canary

# Promote directly to production
python -m src.training.promote model-a <run_id> --production
```

**GitHub Actions:**
```bash
# Trigger manually via CLI
gh workflow run ct.yml --field model=all --field tuning=false
```

## Comparison with Alternatives

### Alternative 1: Manual retraining only
**Rejected because:** Too slow, error-prone, doesn't scale with multiple models

### Alternative 2: Full automation without stages
**Rejected because:** Too risky; bad models could affect all users immediately

### Alternative 3: External orchestration (Airflow, Prefect)
**Rejected because:** Overkill for current scale; GitHub Actions is sufficient and simpler

## Future Considerations

1. **Incremental training:** Instead of full retraining, consider online learning for Model C
2. **Multi-armed bandit:** Replace fixed A/B test with adaptive traffic allocation
3. **Feature store:** Centralize feature computation for consistency between training and serving
4. **Model ensemble:** Consider blending predictions from multiple model versions

## References

- MLflow Model Registry: https://mlflow.org/docs/latest/model-registry.html
- GitHub Actions scheduled workflows: https://docs.github.com/en/actions/writing-workflows/choosing-when-your-workflow-runs/events-that-trigger-workflows#schedule
- Canary deployments: https://martinfowler.com/bliki/CanaryRelease.html
