# CLAUDE.md - RecEngine Project Guide

## Vue d'ensemble

RecEngine est un système de recommandation de produits avec A/B testing et infrastructure MLOps complète.

**Architecture deux étages:**
1. Retrieval: Model A (collaborative filtering) + Model B (semantic embeddings) génèrent ~200 candidats
2. Ranking: Model C (CTR prediction, XGBoost) re-rank par P(click), retourne top 10

**Stack:** Python, scikit-learn, Surprise, sentence-transformers, FAISS, XGBoost, SHAP, FastAPI, MLflow, Docker, Evidently, Streamlit, GitHub Actions, GCP Cloud Run

**Dataset:** Amazon Reviews 2023 (Electronics) depuis Hugging Face

**Durée:** 9 semaines, 4-5h/jour

---

## Structure du projet

```
recengine/
├── src/
│   ├── __init__.py
│   ├── config.py
│   ├── data/
│   │   ├── __init__.py
│   │   ├── pipeline.py
│   │   └── validation.py
│   ├── models/
│   │   ├── __init__.py
│   │   ├── collaborative.py    # Model A
│   │   ├── semantic.py         # Model B
│   │   ├── ctr.py              # Model C
│   │   ├── features.py         # Feature engineering Model C
│   │   └── registry.py         # Chargement depuis MLflow
│   ├── serving/
│   │   ├── __init__.py
│   │   ├── app.py              # FastAPI
│   │   ├── router.py           # A/B routing
│   │   ├── schemas.py          # Pydantic models
│   │   └── middleware.py       # Logging, metrics
│   ├── monitoring/
│   │   ├── __init__.py
│   │   ├── drift.py            # Evidently
│   │   ├── metrics.py          # Prometheus
│   │   └── alerts.py
│   ├── training/
│   │   ├── __init__.py
│   │   ├── train.py
│   │   ├── evaluate.py
│   │   └── promote.py
│   └── dashboard/
│       ├── __init__.py
│       └── app.py              # Streamlit
├── tests/
├── configs/
├── docs/
├── notebooks/
├── data/
│   ├── raw/
│   └── processed/
├── .github/workflows/
├── pyproject.toml
├── Dockerfile
├── docker-compose.yml
├── dvc.yaml
└── README.md
```

---
---

# SEMAINE 1: Foundation + Data Pipeline

## Objectifs
- Structure projet + tooling (git, pre-commit, CI)
- Charger Amazon Reviews 2023 depuis Hugging Face
- Preprocessing: filtrage, déduplication
- Split temporel train/test
- Validation des données
- Sauvegarde en parquet + DVC

## Bibliothèques
pandas, numpy, datasets, pydantic-settings, python-dotenv, loguru, dvc, pytest, black, ruff, mypy, pre-commit

---

### Tâche 1.1: Setup du projet

**Fichiers:** pyproject.toml, .gitignore, .env.example, .pre-commit-config.yaml, .github/workflows/ci.yml

**Instructions:**
1. Créer toute la structure de dossiers (les modules des semaines futures: juste des fichiers vides avec un commentaire TODO)
2. pyproject.toml: pandas, numpy, datasets, pydantic-settings, python-dotenv, loguru, dvc en dépendances. pytest, pytest-cov, black, ruff, mypy, pre-commit en dev.
3. .gitignore: Python, data/, mlruns/, .env, IDE
4. .pre-commit-config.yaml: black (line-length=100), ruff, trailing-whitespace, end-of-file-fixer
5. CI: push/PR sur main, Python 3.11, ruff + mypy + pytest

**Validation:**
- [ ] `pip install -e ".[dev]"` fonctionne
- [ ] `pre-commit install` fonctionne
- [ ] `pytest tests/ -v` s'exécute

---

### Tâche 1.2: Configuration centralisée

**Fichier:** `src/config.py`

**Instructions:**
1. Classe Settings (pydantic-settings BaseSettings) avec:
    - PROJECT_ROOT, DATA_RAW_DIR, DATA_PROCESSED_DIR (Paths)
    - DATASET_CATEGORY = "Electronics"
    - MIN_INTERACTIONS = 5
    - TEST_RATIO = 0.2
    - CHUNK_SIZE = 512, CHUNK_OVERLAP = 50 (pour plus tard)
    - TOKENIZER_MODEL = "cl100k_base"
2. Charger depuis .env si existant
3. Instance `settings = Settings()` en bas du module

**Validation:**
- [ ] `from src.config import settings` fonctionne
- [ ] settings.DATA_RAW_DIR retourne un Path valide

---

### Tâche 1.3: Chargement du dataset

**Fichier:** `src/data/pipeline.py`

**Instructions:**
1. Fonction `load_reviews(category, n)`: charger depuis Hugging Face avec streaming si espace limité. Retourner un DataFrame.
2. Fonction `load_metadata(category, n)`: même logique pour les metadata produits.
3. Tester manuellement: vérifier les colonnes, les types, les premières lignes.

**Attention:** les reviews ont user_id, parent_asin, rating, timestamp, text. La metadata a parent_asin, title, description, features, price, categories. Les deux se joignent sur parent_asin.

**Validation:**
- [ ] load_reviews retourne un DataFrame avec les bonnes colonnes
- [ ] load_metadata retourne un DataFrame avec les bonnes colonnes
- [ ] Les deux datasets ont des parent_asin en commun

---

### Tâche 1.4: Preprocessing

**Fichier:** `src/data/pipeline.py`

**Instructions:**
1. Fonction `preprocess_reviews(df, min_interactions)`:
    - Garder uniquement: user_id, parent_asin, rating, timestamp
    - Supprimer doublons (user + item, garder le plus récent)
    - Filtrer users avec < min_interactions ratings
    - Filtrer items avec < min_interactions ratings
    - Logger avant/après

**Validation:**
- [ ] Pas de doublons (user_id, parent_asin)
- [ ] Chaque user a >= min_interactions ratings
- [ ] Chaque item a >= min_interactions ratings

---

### Tâche 1.5: Split temporel

**Fichier:** `src/data/pipeline.py`

**Instructions:**
1. Fonction `temporal_train_test_split(df, test_ratio)`:
    - Trier par timestamp
    - Par user: 80% plus ancien en train, 20% plus récent en test
    - Si user n'a qu'une interaction: tout en train
2. Documenter la décision dans `docs/adr/001-temporal-split.md`: pourquoi temporel et pas random (éviter le data leakage)

**Validation:**
- [ ] Pour chaque user dans les deux sets: max(train.timestamp) < min(test.timestamp)
- [ ] Pas de paires (user, item) identiques entre train et test

---

### Tâche 1.6: Validation des données

**Fichier:** `src/data/validation.py`

**Instructions:**
1. `validate_reviews(df)`: colonnes requises, pas de nulls, rating [1,5], pas de doublons. Retourne bool.
2. `validate_no_leakage(train, test)`: pas d'overlap de paires, timestamps cohérents. Retourne bool.
3. `validate_metadata(df)`: parent_asin existe, titles non vides. Retourne bool.

**Validation:**
- [ ] validate_reviews catch les données invalides
- [ ] validate_no_leakage catch le leakage
- [ ] Retournent True sur des données propres

---

### Tâche 1.7: Pipeline complet

**Fichier:** `src/data/pipeline.py`

**Instructions:**
1. Fonction `run_pipeline()`:
    - Charger reviews + metadata
    - Aligner (garder seulement les items présents dans les deux)
    - Preprocesser
    - Valider
    - Split temporel
    - Valider le split
    - Sauvegarder en parquet (train.parquet, test.parquet, metadata.parquet)
2. Exécutable avec `python -m src.data.pipeline`
3. Créer dvc.yaml avec le stage "preprocess"

**Validation:**
- [ ] Le pipeline produit 3 fichiers parquet
- [ ] Les validations passent toutes
- [ ] `dvc repro` fonctionne

---

### Tâche 1.8: Tests

**Fichier:** `tests/test_data.py`

**Instructions:**
Écrire 10 tests avec des DataFrames de test (PAS le vrai dataset):
1. test_validate_reviews_valid_data
2. test_validate_reviews_missing_column
3. test_validate_reviews_rating_out_of_range
4. test_validate_reviews_duplicates
5. test_validate_no_leakage_clean
6. test_validate_no_leakage_overlap
7. test_preprocess_removes_duplicates
8. test_preprocess_filters_low_interaction_users
9. test_temporal_split_ordering
10. test_temporal_split_ratio

**Validation:**
- [ ] `pytest tests/test_data.py -v` passe les 10 tests
- [ ] Chaque test < 1 seconde

---

### Tâche 1.9: README + Documentation

**Instructions:**
1. README.md: nom, description, architecture ASCII, quick start, stack, status
2. docs/adr/001-temporal-split.md: contexte, décision, conséquences

**Validation:**
- [ ] README clair
- [ ] ADR explique le split temporel

---

### Livrables Semaine 1
- [ ] Structure projet complète
- [ ] CI GitHub Actions fonctionnel
- [ ] Pipeline data: chargement, preprocessing, split, validation, sauvegarde
- [ ] 10 tests passants
- [ ] README + ADR

---
---

# SEMAINE 2: Model A - Collaborative Filtering

## Objectifs
- Implémenter SVD avec la bibliothèque Surprise
- Tracker les expériences avec MLflow
- Hyperparameter tuning
- Métriques de ranking: Precision@K, Recall@K, NDCG@K
- Enregistrer le meilleur modèle dans MLflow Model Registry

## Bibliothèques à ajouter
scikit-surprise, mlflow, optuna (optionnel)

---

### Tâche 2.1: Métriques d'évaluation

**Fichier:** `src/training/evaluate.py`

**Instructions:**
1. Implémenter `precision_at_k(recommended: list, relevant: list, k: int) -> float`
    - Sur les K items recommandés, combien sont dans les relevant?
    - Formule: |recommended[:k] ∩ relevant| / k
2. Implémenter `recall_at_k(recommended: list, relevant: list, k: int) -> float`
    - Sur tous les relevant, combien sont dans les K recommandés?
    - Formule: |recommended[:k] ∩ relevant| / |relevant|
3. Implémenter `ndcg_at_k(recommended: list, relevant: list, k: int) -> float`
    - Normalized Discounted Cumulative Gain
    - Prend en compte la position: un item pertinent en position 1 vaut plus qu'en position 10
4. Implémenter `mean_reciprocal_rank(recommended: list, relevant: list) -> float`
    - 1 / position du premier item pertinent

**Validation:**
- [ ] precision_at_k([1,2,3], [2,3,5], 3) retourne 2/3
- [ ] Les métriques retournent 0.0 si aucun item pertinent
- [ ] Tests unitaires pour chaque métrique

---

### Tâche 2.2: Model A - Collaborative Filtering

**Fichier:** `src/models/collaborative.py`

**Instructions:**
1. Créer une classe `CollaborativeModel` avec:
    - `__init__(self, n_factors, n_epochs, lr_all, reg_all)`: paramètres SVD
    - `fit(self, train_df)`: convertir le DataFrame en format Surprise, entraîner SVD
    - `predict(self, user_id, item_ids) -> list[tuple[str, float]]`: prédire les ratings pour une liste d'items
    - `recommend(self, user_id, n=10, exclude_seen=True) -> list[tuple[str, float]]`: top-N recommendations
2. La bibliothèque Surprise attend un format spécifique: Reader + Dataset. Regarde la doc.
3. Pour `recommend`: prédire le rating pour tous les items non vus par l'user, trier par score décroissant, retourner les top N.

**Attention:** Surprise a sa propre API. Tu dois convertir ton DataFrame pandas en Dataset Surprise. Lis la documentation: https://surprise.readthedocs.io/

**Validation:**
- [ ] Le modèle s'entraîne sur le train set
- [ ] predict retourne des scores entre 1 et 5
- [ ] recommend retourne N items triés par score décroissant

---

### Tâche 2.3: MLflow Tracking

**Instructions:**
1. Setup MLflow:
    - `mlflow.set_experiment("recengine-model-a")`
    - Logger: hyperparamètres, métriques (RMSE, Precision@10, Recall@10, NDCG@10), durée d'entraînement
    - Sauvegarder le modèle comme artifact
2. Lancer le serveur MLflow: `mlflow ui --port 5000`
3. Créer un script ou notebook d'entraînement qui log tout

**Validation:**
- [ ] MLflow UI montre les expériences
- [ ] Hyperparamètres et métriques loggés
- [ ] Modèle sauvegardé comme artifact

---

### Tâche 2.4: Hyperparameter Tuning

**Instructions:**
1. Tester plusieurs combinaisons:
    - n_factors: [50, 100, 150]
    - n_epochs: [20, 30]
    - lr_all: [0.005, 0.01]
    - reg_all: [0.02, 0.05]
2. Logger chaque run dans MLflow
3. Sélectionner le meilleur modèle par NDCG@10
4. Enregistrer le meilleur dans MLflow Model Registry

**Validation:**
- [ ] Plusieurs runs visibles dans MLflow
- [ ] Meilleur modèle identifié et enregistré
- [ ] Model card: docs/model_a_card.md

---

### Tâche 2.5: Évaluation complète

**Instructions:**
1. Évaluer sur le test set:
    - Pour chaque user du test: générer des recommandations avec le modèle
    - Comparer avec les items réels du test (relevant items)
    - Calculer Precision@K, Recall@K, NDCG@K pour K=5,10,20
2. Logger les résultats dans MLflow
3. Sauvegarder un rapport dans notebooks/02_model_a_evaluation.ipynb

**Validation:**
- [ ] Métriques calculées sur le test set complet
- [ ] Résultats loggés dans MLflow
- [ ] Rapport avec les chiffres

---

### Tâche 2.6: Tests Model A

**Fichier:** `tests/test_models.py`

**Instructions:**
1. test_collaborative_model_fits: le modèle s'entraîne sans erreur
2. test_collaborative_model_predicts: predict retourne des scores valides
3. test_collaborative_model_recommends_n_items: recommend retourne exactement N items
4. test_metrics_precision_at_k: vérifie le calcul
5. test_metrics_ndcg_at_k: vérifie le calcul

**Validation:**
- [ ] Tous les tests passent
- [ ] Utiliser des petits DataFrames de test (pas le vrai dataset)

---

### Livrables Semaine 2
- [ ] Métriques de ranking implémentées et testées
- [ ] CollaborativeModel fonctionnel
- [ ] MLflow tracking avec plusieurs runs
- [ ] Meilleur modèle enregistré dans le Model Registry
- [ ] Évaluation sur test set avec rapport
- [ ] Model card documenté

---
---

# SEMAINE 3: Model B - Semantic Embeddings

## Objectifs
- Générer des embeddings produits avec sentence-transformers
- Construire un index FAISS pour recherche rapide
- Recommander par similarité sémantique
- Comparer Model A vs Model B

## Bibliothèques à ajouter
sentence-transformers, faiss-cpu

---

### Tâche 3.1: Génération d'embeddings

**Fichier:** `src/models/semantic.py`

**Instructions:**
1. Créer une fonction `build_product_texts(metadata_df) -> dict[str, str]`:
    - Pour chaque produit, concaténer: title + description + features
    - Retourner un dict {parent_asin: texte_combiné}
    - Gérer les valeurs manquantes (description vide, features vides)
2. Créer une classe `SemanticModel` avec:
    - `__init__(self, model_name="all-MiniLM-L6-v2")`: charger le sentence-transformer
    - `build_index(self, product_texts: dict[str, str])`: encoder tous les produits, construire l'index FAISS
    - Les embeddings doivent être normalisés (pour cosine similarity via inner product)

**Attention:** sentence-transformers encode en batch. Utilise `model.encode(texts, batch_size=64, show_progress_bar=True)`.

**Validation:**
- [ ] Les embeddings ont la bonne dimension (384 pour MiniLM)
- [ ] L'index FAISS contient autant de vecteurs que de produits
- [ ] Les produits sans titre/description sont gérés

---

### Tâche 3.2: Recherche par similarité

**Fichier:** `src/models/semantic.py`

**Instructions:**
1. Ajouter à SemanticModel:
    - `find_similar(self, product_id, n=10) -> list[tuple[str, float]]`: trouver les N produits les plus similaires
    - `recommend(self, user_id, train_df, n=10) -> list[tuple[str, float]]`:
        - Récupérer les produits que l'user a bien notés (rating >= 4)
        - Pour chaque produit aimé, trouver les produits similaires
        - Agréger les scores (moyenne ou max)
        - Exclure les produits déjà vus
        - Retourner top N
2. Sauvegarder/charger l'index: `save_index(path)` et `load_index(path)`

**Validation:**
- [ ] find_similar retourne des produits pertinents (vérifier manuellement sur quelques exemples)
- [ ] recommend exclut les produits déjà vus
- [ ] L'index se sauvegarde et se recharge

---

### Tâche 3.3: MLflow + Évaluation

**Instructions:**
1. Évaluer Model B avec les mêmes métriques que Model A (Precision@K, Recall@K, NDCG@K)
2. Logger dans MLflow sous l'expérience "recengine-model-b"
3. Créer un tableau comparatif Model A vs Model B

**Validation:**
- [ ] Métriques loggées dans MLflow
- [ ] Comparaison A vs B documentée
- [ ] Model card: docs/model_b_card.md

---

### Tâche 3.4: Cold Start Analysis

**Instructions:**
1. Identifier les produits dans le test set qui n'ont PAS d'interactions dans le train set
2. Model A ne peut rien recommander pour ces produits. Model B si (il utilise la description).
3. Mesurer la performance de Model B sur les items cold start vs items non-cold-start
4. Documenter cette analyse: c'est un argument fort pour les interviews

**Validation:**
- [ ] Nombre d'items cold start quantifié
- [ ] Performance de Model B sur cold start mesurée
- [ ] Analyse documentée dans un notebook

---

### Tâche 3.5: Tests Model B

**Fichier:** `tests/test_models.py` (ajouter)

**Instructions:**
1. test_semantic_model_builds_index: l'index a la bonne taille
2. test_semantic_model_find_similar: retourne N résultats
3. test_semantic_model_recommend: retourne des items non vus
4. test_embeddings_dimension: bonne dimension

**Validation:**
- [ ] Tous les tests passent

---

### Livrables Semaine 3
- [ ] SemanticModel avec embeddings + FAISS
- [ ] Recommandation par similarité sémantique
- [ ] Comparaison Model A vs Model B
- [ ] Analyse cold start
- [ ] MLflow tracking
- [ ] Tests passants

---
---

# SEMAINE 4: Model C - CTR Prediction

## Objectifs
- Feature engineering: user, product, cross features
- Construire le dataset d'entraînement (positifs + négatifs)
- Entraîner XGBoost pour prédire P(click)
- SHAP pour l'explainabilité
- Re-ranking pipeline

## Bibliothèques à ajouter
xgboost, shap

---

### Tâche 4.1: Feature Engineering

**Fichier:** `src/models/features.py`

**Instructions:**
1. `build_user_features(train_df) -> pd.DataFrame`: pour chaque user:
    - num_ratings, avg_rating, rating_std
    - days_active (entre premier et dernier rating)
    - favorite_category (si metadata dispo, sinon skip)
2. `build_item_features(train_df, metadata_df) -> pd.DataFrame`: pour chaque item:
    - avg_rating_received, num_ratings_received
    - price (si dispo), description_length
    - category (main_category de metadata)
3. `build_cross_features(user_id, item_id, model_a, model_b) -> dict`: pour une paire:
    - model_a_score: score prédit par collaborative filtering
    - model_b_score: score de similarité sémantique
    - category_match: l'item est dans la catégorie préférée du user? (0/1)

**Validation:**
- [ ] User features: un row par user, pas de NaN
- [ ] Item features: un row par item, NaN gérés (prix manquant etc.)
- [ ] Cross features: retourne un dict avec les 3 features

---

### Tâche 4.2: Construction du dataset CTR

**Fichier:** `src/models/ctr.py`

**Instructions:**
1. Fonction `build_ctr_dataset(train_df, user_features, item_features, model_a, model_b) -> pd.DataFrame`:
    - Positifs (label=1): paires (user, item) existantes dans train avec rating >= 4
    - Négatifs (label=0): pour chaque positif, sampler 4 items aléatoires que le user n'a PAS vus
    - Pour chaque paire: joindre user_features + item_features + cross_features
    - Retourner un DataFrame avec toutes les features et la colonne "label"
2. Attention au ratio: 4 négatifs pour 1 positif. Documente pourquoi (les interactions sont rares dans la réalité).

**Validation:**
- [ ] Le ratio négatifs/positifs est environ 4:1
- [ ] Pas de NaN dans les features critiques
- [ ] Les négatifs sont des items que l'user n'a pas vus

---

### Tâche 4.3: Entraînement XGBoost

**Fichier:** `src/models/ctr.py`

**Instructions:**
1. Créer une classe `CTRModel` avec:
    - `__init__(self, params)`: paramètres XGBoost
    - `fit(self, X_train, y_train, X_val, y_val)`: entraîner avec early stopping
    - `predict(self, X) -> np.ndarray`: retourner les probabilités P(click)
    - `rerank(self, candidates: list[dict]) -> list[dict]`: re-rank une liste de candidats par P(click)
2. Split le CTR dataset: 80% train, 20% validation
3. Logger dans MLflow: AUC-ROC, log-loss, precision-recall curve
4. Hyperparameter tuning: max_depth, learning_rate, n_estimators, subsample

**Validation:**
- [ ] AUC-ROC > 0.7 (sinon les features ne sont pas informatives)
- [ ] Le modèle ne surfit pas (val AUC proche du train AUC)
- [ ] Résultats loggés dans MLflow

---

### Tâche 4.4: SHAP Analysis

**Instructions:**
1. Calculer les SHAP values sur le validation set
2. Créer les plots:
    - shap.summary_plot: quelles features importent le plus?
    - shap.dependence_plot: relation entre model_a_score et la prédiction
3. Interpréter: est-ce que model_b_score (embeddings) ajoute de la valeur par rapport à model_a_score seul?
4. Sauvegarder les plots et l'analyse dans notebooks/03_shap_analysis.ipynb

**Validation:**
- [ ] SHAP values calculées
- [ ] Plots générés et interprétables
- [ ] Analyse documentée

---

### Tâche 4.5: Pipeline de re-ranking

**Instructions:**
1. Créer une fonction complète `get_recommendations(user_id, model_a, model_b, model_c, n=10)`:
    - Model A: top 100 candidats
    - Model B: top 100 candidats
    - Merge (union, dédupliqué)
    - Pour chaque candidat: calculer les features
    - Model C: prédire P(click), trier, retourner top N
2. Comparer la performance du pipeline complet vs Model A seul vs Model B seul
3. Logger les résultats comparatifs dans MLflow

**Validation:**
- [ ] Le pipeline retourne 10 recommendations
- [ ] Performance comparée: pipeline complet vs baselines
- [ ] Model card: docs/model_c_card.md

---

### Tâche 4.6: Tests Model C

**Fichier:** `tests/test_ctr.py`

**Instructions:**
1. test_build_ctr_dataset_ratio: vérifie le ratio 4:1
2. test_ctr_model_predicts_probabilities: output entre 0 et 1
3. test_reranking_changes_order: l'ordre change après re-ranking
4. test_user_features_no_nan: pas de NaN dans les features
5. test_negative_sampling_excludes_seen: les négatifs ne sont pas des items vus

**Validation:**
- [ ] Tous les tests passent

---

### Livrables Semaine 4
- [ ] Feature engineering pipeline
- [ ] Dataset CTR construit (positifs + négatifs)
- [ ] XGBoost entraîné et évalué
- [ ] SHAP analysis
- [ ] Pipeline de re-ranking complet
- [ ] Comparaison: pipeline vs baselines
- [ ] Tests passants

---
---

# SEMAINE 5: FastAPI + A/B Testing Router

## Objectifs
- API FastAPI servant les recommandations
- Router A/B avec assignment déterministe
- Logging des prédictions et du feedback
- Trois variantes: baseline, full pipeline, semantic only

## Bibliothèques à ajouter
fastapi, uvicorn, httpx (pour tests)

---

### Tâche 5.1: Schemas Pydantic

**Fichier:** `src/serving/schemas.py`

**Instructions:**
1. `RecommendRequest`: user_id (str), num_results (int, default=10)
2. `RecommendResponse`: items (list[RecommendedItem]), variant (str), latency_ms (float)
3. `RecommendedItem`: item_id (str), score (float), rank (int)
4. `FeedbackRequest`: user_id (str), item_id (str), action (Literal["click", "purchase", "ignore"]), timestamp (datetime)
5. `HealthResponse`: status (str), model_versions (dict), uptime_seconds (float)

**Validation:**
- [ ] Validation Pydantic fonctionne
- [ ] Sérialisation JSON correcte

---

### Tâche 5.2: A/B Router

**Fichier:** `src/serving/router.py`

**Instructions:**
1. Fonction `assign_variant(user_id: str, split_ratio: float) -> str`:
    - Hash le user_id avec MD5
    - Convertir en nombre, modulo 100
    - Assigner "A" ou "B" selon le ratio
    - Déterministe: même user = toujours même variante
2. Support des modes de déploiement:
    - "shadow": les deux tournent, seul A est retourné. B est loggé.
    - "canary": B sert 5% du trafic
    - "ab_test": split configurable
3. Lire le mode depuis la variable d'environnement DEPLOYMENT_MODE

**Validation:**
- [ ] Même user_id donne toujours la même variante
- [ ] Le ratio est approximativement respecté sur 10000 users
- [ ] Le mode shadow ne retourne que A

---

### Tâche 5.3: FastAPI Application

**Fichier:** `src/serving/app.py`

**Instructions:**
1. Endpoints:
    - `POST /recommend`: reçoit RecommendRequest, route via A/B, retourne RecommendResponse
    - `POST /feedback`: reçoit FeedbackRequest, log l'interaction
    - `GET /health`: retourne HealthResponse
    - `GET /metrics`: métriques Prometheus
2. Au startup: charger les modèles depuis MLflow (ou depuis des fichiers locaux en dev)
3. Logger chaque prédiction: user_id, variante, items retournés, latence

**Variantes:**
- Variant A: Model A seul (collaborative filtering, pas de re-ranking)
- Variant B: Model A + B retrieval, Model C re-ranking (pipeline complet)

**Validation:**
- [ ] `/recommend` retourne des items
- [ ] `/health` retourne le status
- [ ] Les logs montrent la variante utilisée

---

### Tâche 5.4: Middleware et logging

**Fichier:** `src/serving/middleware.py`

**Instructions:**
1. Logger chaque requête: timestamp, user_id, variant, items, latency_ms
2. Stocker les logs dans SQLite (ou un fichier JSONL pour simplifier)
3. Le feedback (clicks/purchases) alimente le retraining futur de Model C

**Validation:**
- [ ] Les prédictions sont loggées
- [ ] Le feedback est stocké
- [ ] On peut relire les logs

---

### Tâche 5.5: Tests API

**Fichier:** `tests/test_api.py` et `tests/test_routing.py`

**Instructions:**
1. test_health_endpoint: retourne 200
2. test_recommend_returns_items: retourne le bon nombre d'items
3. test_recommend_logs_prediction: vérifie que le log est écrit
4. test_deterministic_assignment: même user = même variante
5. test_split_ratio: le ratio est approximativement correct sur un large échantillon
6. test_shadow_mode: ne retourne que les résultats de A

**Validation:**
- [ ] Tous les tests passent
- [ ] Utiliser httpx.AsyncClient pour tester FastAPI

---

### Livrables Semaine 5
- [ ] FastAPI avec /recommend, /feedback, /health
- [ ] A/B router avec assignment déterministe
- [ ] Support shadow/canary/ab_test
- [ ] Logging des prédictions et feedback
- [ ] Tests passants

---
---

# SEMAINE 6: Docker + GCP Deployment

## Objectifs
- Dockerfile multi-stage
- docker-compose: API + MLflow + dashboard
- Déployer sur GCP Cloud Run
- CI/CD: GitHub Actions build + deploy

---

### Tâche 6.1: Dockerfile

**Instructions:**
1. Multi-stage build:
    - Stage 1 (builder): installer les dépendances
    - Stage 2 (runtime): image slim, copier seulement le nécessaire
2. Exposer le port 8000
3. CMD: uvicorn
4. L'image doit être < 1GB

**Validation:**
- [ ] `docker build -t recengine .` fonctionne
- [ ] `docker run -p 8000:8000 recengine` démarre l'API
- [ ] L'API répond sur localhost:8000/health

---

### Tâche 6.2: docker-compose

**Instructions:**
1. Services: api, mlflow, dashboard (streamlit)
2. Volumes pour les données et les modèles
3. Variables d'environnement via .env

**Validation:**
- [ ] `docker-compose up` démarre les 3 services
- [ ] L'API est accessible sur :8000
- [ ] MLflow UI sur :5000

---

### Tâche 6.3: GCP Cloud Run

**Instructions:**
1. Créer un projet GCP
2. Pousser l'image Docker sur GCP Artifact Registry
3. Déployer sur Cloud Run
4. Tester l'URL publique

**Validation:**
- [ ] API accessible via URL Cloud Run
- [ ] /health répond correctement

---

### Tâche 6.4: CI/CD Pipeline

**Fichier:** `.github/workflows/cd.yml`

**Instructions:**
1. Trigger: merge sur main
2. Steps: checkout, build Docker, push to Artifact Registry, deploy to Cloud Run
3. Utiliser les GitHub secrets pour les credentials GCP

**Validation:**
- [ ] Un push sur main déclenche le déploiement
- [ ] La nouvelle version est live sur Cloud Run

---

### Livrables Semaine 6
- [ ] Dockerfile optimisé
- [ ] docker-compose fonctionnel
- [ ] API déployée sur GCP Cloud Run
- [ ] CI/CD automatisé

---
---

# SEMAINE 7: Monitoring + Dashboard

## Objectifs
- Drift detection avec Evidently
- Métriques Prometheus
- Dashboard Streamlit pour les résultats A/B
- Calcul de significativité statistique

## Bibliothèques à ajouter
evidently, prometheus-client (si pas déjà)

---

### Tâche 7.1: Evidently Drift Detection

**Fichier:** `src/monitoring/drift.py`

**Instructions:**
1. Data drift report: comparer les distributions des features de requêtes entrantes vs données d'entraînement
2. Prediction drift: comparer les distributions de scores entre Model A et Model B
3. Générer des rapports HTML
4. Scheduler: exécuter quotidiennement (cron ou Cloud Scheduler)

**Validation:**
- [ ] Rapport de drift généré
- [ ] Alerte si drift > seuil

---

### Tâche 7.2: Prometheus Metrics

**Fichier:** `src/monitoring/metrics.py`

**Instructions:**
1. Compteur de requêtes par variante et status
2. Histogramme de latence par variante
3. Distribution des scores par variante
4. Taux d'erreur
5. Exposer sur l'endpoint /metrics

**Validation:**
- [ ] /metrics retourne des métriques au format Prometheus

---

### Tâche 7.3: Streamlit Dashboard

**Fichier:** `src/dashboard/app.py`

**Instructions:**
1. Conversion rate par variante au fil du temps (graphique)
2. Calculateur de significativité statistique (two-proportion z-test)
3. Viewer des rapports Evidently
4. Comparaison des métriques Model A vs pipeline complet
5. Sample size calculator: combien de données faut-il pour conclure?

**Validation:**
- [ ] Dashboard affiche les résultats A/B
- [ ] Z-test fonctionne
- [ ] Rapports Evidently visibles

---

### Tâche 7.4: Tests Monitoring

**Instructions:**
1. test_drift_report_generates: le rapport est créé sans erreur
2. test_metrics_endpoint: /metrics retourne du texte Prometheus valide
3. test_z_test_calculation: vérifie le calcul statistique

**Validation:**
- [ ] Tests passants

---

### Livrables Semaine 7
- [ ] Drift detection fonctionnel
- [ ] Métriques Prometheus
- [ ] Dashboard Streamlit
- [ ] Significativité statistique calculée
- [ ] Tests passants

---
---

# SEMAINE 8: Automated Retraining + CI/CT/CD

## Objectifs
- Pipeline de retraining automatique
- Promotion de modèles: staging > canary > production
- Rollback automatique
- GitHub Actions pour continuous training

---

### Tâche 8.1: Retraining Pipeline

**Fichier:** `src/training/train.py`

**Instructions:**
1. Script qui:
    - Pull les dernières données (ou simule un refresh)
    - Retrain Models A, B, C
    - Évalue sur le holdout set
    - Compare avec le modèle en production
    - Si amélioration > seuil: enregistre comme "staging" dans MLflow
2. Doit être exécutable: `python -m src.training.train`

**Validation:**
- [ ] Le script retrain les 3 modèles
- [ ] Le nouveau modèle est enregistré si meilleur
- [ ] Logs clairs

---

### Tâche 8.2: Model Promotion

**Fichier:** `src/training/promote.py`

**Instructions:**
1. staging > canary: déployer automatiquement à 5% du trafic
2. canary > production: après validation manuelle (ou automatique si métriques OK pendant 24h)
3. Rollback: si le taux d'erreur augmente après promotion, revenir au modèle précédent

**Validation:**
- [ ] Promotion staging > canary fonctionne
- [ ] Rollback fonctionne

---

### Tâche 8.3: GitHub Actions CT

**Fichier:** `.github/workflows/ct.yml`

**Instructions:**
1. Trigger: schedule hebdomadaire (cron) ou manuel
2. Steps: pull data, retrain, evaluate, register si amélioré, deploy staging
3. Notifications: log le résultat (amélioré ou pas)

**Validation:**
- [ ] Le workflow se déclenche
- [ ] Le retraining s'exécute dans CI

---

### Tâche 8.4: Test end-to-end

**Instructions:**
1. Simuler le cycle complet:
    - Données changent (simuler un drift)
    - Drift détecté
    - Retraining déclenché
    - Nouveau modèle évalué
    - Promotion si meilleur
    - Vérification que l'API sert le nouveau modèle

**Validation:**
- [ ] Le cycle complet fonctionne

---

### Livrables Semaine 8
- [ ] Retraining pipeline automatisé
- [ ] Promotion staging > canary > production
- [ ] Rollback fonctionnel
- [ ] GitHub Actions CT
- [ ] Test end-to-end

---
---

# SEMAINE 9: Documentation + Polish

## Objectifs
- README complet avec architecture diagram
- ADRs pour les décisions clés
- Démo vidéo ou GIF
- 5 LinkedIn posts
- Code propre, tous les tests passent
- Release v1.0.0

---

### Tâche 9.1: README complet

**Instructions:**
1. Architecture diagram (Mermaid ou ASCII)
2. Setup instructions (local + Docker + cloud)
3. API documentation avec exemples curl
4. A/B testing methodology
5. Résultats: métriques, comparaisons, SHAP plots
6. Project status: toutes les checkboxes cochées

---

### Tâche 9.2: ADRs

**Fichier:** `docs/adr/`

**Instructions:**
1. 001-temporal-split.md (fait en Semaine 1)
2. 002-deterministic-ab-assignment.md: pourquoi le hashing
3. 003-shadow-canary-ab-progression.md: pourquoi ce déploiement progressif
4. 004-two-stage-architecture.md: pourquoi retrieval + ranking
5. 005-evidently-for-monitoring.md: pourquoi Evidently

---

### Tâche 9.3: Démo

**Instructions:**
1. Enregistrer une vidéo 2-3 min ou créer un GIF montrant:
    - L'API qui répond
    - Le dashboard Streamlit
    - MLflow avec les expériences
    - Le A/B test en action
2. Ajouter au README

---

### Tâche 9.4: LinkedIn Posts

**Instructions:**
Rédiger 5 posts:
1. "I built an A/B testing system for ML models. Here's what I learned."
2. "Shadow deployment vs canary vs A/B testing for ML. When to use which."
3. "The gap between training a model and running it in production."
4. "How a CTR re-ranking model improved my recommendations. SHAP told me why."
5. "My end-to-end ML pipeline: the full architecture."

---

### Tâche 9.5: Cleanup

**Instructions:**
1. Supprimer le code mort
2. Ajouter des docstrings à toutes les fonctions publiques
3. Vérifier que tous les tests passent: `pytest tests/ -v`
4. Vérifier le linting: `ruff check src/`
5. Tag `git tag v1.0.0`

---

### Livrables Semaine 9
- [ ] README complet et professionnel
- [ ] 5 ADRs
- [ ] Démo vidéo/GIF
- [ ] 5 LinkedIn posts rédigés
- [ ] Code propre, tous les tests passent
- [ ] Release v1.0.0

---
---

# Phrase d'interview

"J'ai construit un système de recommandation production-ready avec une architecture deux étages: retrieval par collaborative filtering et embeddings sémantiques, puis ranking par un modèle CTR supervisé. J'ai implémenté trois modes de déploiement (shadow, canary, A/B test) pour valider les mises à jour de modèle. Le pipeline inclut du drift monitoring avec Evidently, du tracking avec MLflow, du CI/CD avec GitHub Actions, le tout conteneurisé et déployé sur GCP Cloud Run."
