# Semaine 8: Automated Retraining + CI/CT/CD - Résumé

## ✅ Livrables implémentés

### 1. Pipeline de Retraining Automatique

**Fichier:** `src/training/train.py`

**Fonctionnalités:**
- Fonction `run_training()` avec paramètre `auto_promote`
- Entraînement des 3 modèles (A, B, C) avec logging MLflow
- Comparaison automatique avec le modèle en production
- Flag `--auto-promote` pour déclencher la promotion automatique

**Usage:**
```bash
# Entraînement simple
python -m src.training.train

# Avec Models B et C
python -m src.training.train --model-b --model-c

# Avec auto-promotion
python -m src.training.train --model-c --auto-promote

# Hyperparameter tuning
python -m src.training.train --tuning
```

---

### 2. Système de Promotion de Modèles

**Fichier:** `src/training/promote.py` (15,516 bytes)

**Fonctionnalités:**
- `get_model_metrics(run_id)`: Récupère les métriques d'un run MLflow
- `get_production_model(experiment)`: Obtient le modèle en production
- `get_staging_model(experiment)`: Obtient le modèle en staging
- `compare_models(candidate, production)`: Compare les performances
- `promote_to_staging(experiment, run_id)`: Promotion vers staging
- `promote_to_canary(experiment)`: Promotion vers canary (5% traffic)
- `promote_to_production(experiment)`: Promotion vers production (100%)
- `rollback_to_previous(experiment)`: Rollback vers version précédente
- `auto_promote(experiment, run_id)`: Pipeline automatique complet
- `check_canary_performance(experiment)`: Vérifie la santé du canary
- `save_deployment_state()`: Persiste l'état dans un JSON

**Critères de promotion:**
- **Staging:** Amélioration sur au moins une métrique
- **Canary:** Amélioration significative (>1%) sur plusieurs métriques
- **Production:** Canary healthy pendant 24h (error rate < 5%)

**Usage:**
```bash
# Promotion manuelle
python -m src.training.promote model-a <run_id>
python -m src.training.promote model-c <run_id> --canary
python -m src.training.promote model-a <run_id> --production
```

---

### 3. Workflow GitHub Actions CT

**Fichier:** `.github/workflows/ct.yml`

**Déclencheurs:**
- Schedule: Tous les dimanches à 2 AM UTC
- Manuel: Via GitHub UI avec paramètres (model, tuning)

**Steps:**
1. Checkout du code
2. Setup Python 3.11
3. Installation des dépendances avec uv
4. Vérification des données d'entraînement
5. Entraînement des modèles (A, B, C)
6. Évaluation et promotion vers canary
7. Upload des artifacts MLflow
8. Déploiement vers staging

---

### 4. Tests Unitaires

**Fichier:** `tests/test_promotion.py` (21 tests)

**Couverture:**
- `TestModelMetrics`: Récupération des métriques MLflow
- `TestModelComparison`: Logique de comparaison
- `TestDeploymentState`: Persistance de l'état
- `TestAutoPromote`: Pipeline automatique
- `TestCanaryPerformance`: Validation du canary
- `TestPromotionWorkflow`: Workflows de promotion

**Résultats:**
```
============================= 21 passed ==============================
```

---

### 5. Documentation

**Fichier:** `docs/adr/006-automated-retraining.md`

**Contenu:**
- Contexte et décision
- Pipeline de promotion (staging → canary → production)
- Critères de promotion et rollback
- Conséquences positives/négatives
- Tableau des risques et mitigations
- Guide d'utilisation complet
- Comparaison avec les alternatives
- Considérations futures

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    GitHub Actions CT                         │
│  Schedule: Weekly | Manual: model selection                 │
└─────────────────────┬───────────────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────────────┐
│                   Training Pipeline                          │
│  train.py: Load data → Train A/B/C → Evaluate → Log         │
└─────────────────────┬───────────────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────────────┐
│                   MLflow Tracking                            │
│  Experiments | Metrics | Model Registry (Staging/Prod)      │
└─────────────────────┬───────────────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────────────┐
│                   Auto-Promotion                             │
│  promote.py: Compare → Stage → Canary → Validate → Prod     │
└─────────────────────┬───────────────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────────────┐
│                   Deployment State                           │
│  JSON file: {model: stage, version, run_id, timestamp}      │
└─────────────────────┬───────────────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────────────┐
│                   Serving Layer                              │
│  FastAPI reads deployment state for A/B routing             │
└─────────────────────────────────────────────────────────────┘
```

---

## 📊 Métriques et Critères

### Comparison Logic (`compare_models`)
```python
# Improvement threshold: 1% relative improvement
significant = relative_diff >= 0.01

# Promotion criteria:
should_promote = (
    len(significant_improvements) > 0 and
    len(degraded_metrics) < len(common_metrics) / 2
)
```

### Canary Validation (`check_canary_performance`)
```python
# Minimum samples: 100 feedback events
# Error threshold: 5%
if error_rate > 0.05:
    rollback()
```

---

## 🧪 Tests

**Commande:**
```bash
pytest tests/test_promotion.py -v
```

**Résultats:**
- 21 tests passants
- Couverture: metrics, comparison, deployment, auto-promote, canary

---

## 🔧 Variables d'Environnement

| Variable | Description | Default |
|----------|-------------|---------|
| `MLFLOW_TRACKING_URI` | MLflow server URL | `http://localhost:5000` |
| `MODEL_DIR` | Directory for saved models | `data/models` |
| `DATA_PROCESSED_DIR` | Processed data location | `data/processed` |

---

## 📁 Fichiers Clés

| Fichier | Lignes | Description |
|---------|--------|-------------|
| `src/training/train.py` | 582 | Pipeline d'entraînement |
| `src/training/promote.py` | 481 | Logique de promotion |
| `.github/workflows/ct.yml` | 94 | Workflow CT |
| `tests/test_promotion.py` | 288 | Tests unitaires |
| `docs/adr/006-automated-retraining.md` | 120+ | Documentation |

---

## 🚀 Prochaines Étapes

1. **Semaine 9:** Documentation finale + polish
2. README complet avec architecture diagram
3. Démo vidéo/GIF
4. 5 LinkedIn posts
5. Release v1.0.0

---

## ✅ Checklist Semaine 8

- [x] `train.py` avec auto-promotion
- [x] `promote.py` avec staging/canary/production/rollback
- [x] Workflow GitHub Actions CT
- [x] 21 tests unitaires passants
- [x] ADR 006 documenté
- [x] Help commands fonctionnels
- [x] Dépendances à jour (implicit, faiss, xgboost)
