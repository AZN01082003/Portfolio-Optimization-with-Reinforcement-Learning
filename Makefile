# Makefile pour Portfolio RL
.PHONY: help install dev test api monitoring docker clean

# Variables
PYTHON := python3
PIP := pip3
PROJECT_NAME := portfolio-rl
API_PORT := 8000
METRICS_PORT := 8001

# Couleurs pour l'affichage
RED := \033[31m
GREEN := \033[32m
YELLOW := \033[33m
BLUE := \033[34m
NC := \033[0m # No Color

help: ## Affiche l'aide
	@echo "$(BLUE)Portfolio RL - Commandes disponibles:$(NC)"
	@echo ""
	@grep -E '^[a-zA-Z_-]+:.*?## .*$' $(MAKEFILE_LIST) | sort | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "  $(GREEN)%-20s$(NC) %s\n", $1, $2}'
	@echo ""

# =============================================================================
# INSTALLATION ET CONFIGURATION
# =============================================================================

install: ## Installe toutes les dépendances
	@echo "$(BLUE)Installation des dépendances...$(NC)"
	$(PIP) install --upgrade pip
	$(PIP) install -r requirements.txt
	$(PIP) install -r requirements-dev.txt
	$(PIP) install -r requirements-api.txt
	@echo "$(GREEN)✅ Installation terminée$(NC)"

install-minimal: ## Installation minimale (API seulement)
	@echo "$(BLUE)Installation minimale...$(NC)"
	$(PIP) install fastapi uvicorn pydantic
	$(PIP) install stable-baselines3 gymnasium
	$(PIP) install prometheus-client
	@echo "$(GREEN)✅ Installation minimale terminée$(NC)"

setup: ## Configuration initiale du projet
	@echo "$(BLUE)Configuration du projet...$(NC)"
	mkdir -p data/{raw,processed} logs/{api,training,evaluation} models mlruns artifacts
	chmod +x scripts/*.sh
	@echo "$(GREEN)✅ Configuration terminée$(NC)"

dev: install setup ## Installation complète pour développement
	@echo "$(GREEN)✅ Environnement de développement prêt$(NC)"

# =============================================================================
# API ET SERVICES
# =============================================================================

api: ## Démarre l'API Portfolio RL
	@echo "$(BLUE)Démarrage de l'API...$(NC)"
	$(PYTHON) start_api.py --host 0.0.0.0 --port $(API_PORT) --reload

api-prod: ## Démarre l'API en mode production
	@echo "$(BLUE)Démarrage de l'API (production)...$(NC)"
	$(PYTHON) start_api.py --host 0.0.0.0 --port $(API_PORT) --no-metrics

api-check: ## Vérifie les dépendances de l'API
	@echo "$(BLUE)Vérification de l'API...$(NC)"
	$(PYTHON) start_api.py --check-only

api-test: ## Lance les tests de l'API
	@echo "$(BLUE)Test de l'API...$(NC)"
	$(PYTHON) test_api_complete.py --url http://localhost:$(API_PORT)

api-test-quick: ## Tests rapides de l'API
	@echo "$(BLUE)Tests rapides de l'API...$(NC)"
	$(PYTHON) test_api_complete.py --url http://localhost:$(API_PORT) --quick

# =============================================================================
# DONNÉES ET MODÈLES
# =============================================================================

data-ingest: ## Lance l'ingestion de données
	@echo "$(BLUE)Ingestion des données...$(NC)"
	$(PYTHON) -m src.data.ingestion

data-preprocess: ## Lance le préprocessing
	@echo "$(BLUE)Préprocessing des données...$(NC)"
	$(PYTHON) -m src.data.preprocessing

data-pipeline: data-ingest data-preprocess ## Pipeline complet de données
	@echo "$(GREEN)✅ Pipeline de données terminé$(NC)"

train: ## Entraîne le modèle
	@echo "$(BLUE)Entraînement du modèle...$(NC)"
	$(PYTHON) -m src.models.train

train-config: ## Entraîne avec configuration personnalisée
	@echo "$(BLUE)Entraînement avec config personnalisée...$(NC)"
	$(PYTHON) -m src.models.train --config config/custom.json

evaluate: ## Évalue le dernier modèle
	@echo "$(BLUE)Évaluation du modèle...$(NC)"
	$(PYTHON) -m src.models.evaluate

# =============================================================================
# MONITORING ET DOCKER
# =============================================================================

monitoring: ## Démarre la stack de monitoring
	@echo "$(BLUE)Démarrage du monitoring...$(NC)"
	docker-compose -f docker-compose-monitoring.yml up -d
	@echo "$(GREEN)✅ Monitoring démarré:$(NC)"
	@echo "  📊 Prometheus: http://localhost:9090"
	@echo "  📈 Grafana: http://localhost:3000 (admin/admin)"
	@echo "  🚨 AlertManager: http://localhost:9093"

monitoring-stop: ## Arrête la stack de monitoring
	@echo "$(BLUE)Arrêt du monitoring...$(NC)"
	docker-compose -f docker-compose-monitoring.yml down

monitoring-logs: ## Affiche les logs du monitoring
	docker-compose -f docker-compose-monitoring.yml logs -f

docker-build: ## Construit l'image Docker de l'API
	@echo "$(BLUE)Construction de l'image Docker...$(NC)"
	docker build -f api.Dockerfile -t $(PROJECT_NAME)-api:latest .

docker-run: docker-build ## Lance l'API dans Docker
	@echo "$(BLUE)Démarrage de l'API Docker...$(NC)"
	docker run -p $(API_PORT):$(API_PORT) -p $(METRICS_PORT):$(METRICS_PORT) \
		-v $(PWD)/data:/app/data \
		-v $(PWD)/models:/app/models \
		-v $(PWD)/config:/app/config \
		$(PROJECT_NAME)-api:latest

docker-stack: ## Démarre la stack complète Docker
	@echo "$(BLUE)Démarrage de la stack complète...$(NC)"
	docker-compose up -d
	@echo "$(GREEN)✅ Stack démarrée:$(NC)"
	@echo "  🔧 API: http://localhost:$(API_PORT)"
	@echo "  📊 Métriques: http://localhost:$(METRICS_PORT)/metrics"
	@echo "  🧪 MLflow: http://localhost:5000"

docker-stack-stop: ## Arrête la stack Docker
	@echo "$(BLUE)Arrêt de la stack...$(NC)"
	docker-compose down

docker-logs: ## Affiche les logs Docker
	docker-compose logs -f

# =============================================================================
# TESTS ET QUALITÉ
# =============================================================================

test: ## Lance tous les tests
	@echo "$(BLUE)Lancement des tests...$(NC)"
	pytest tests/ -v --tb=short

test-unit: ## Tests unitaires seulement
	@echo "$(BLUE)Tests unitaires...$(NC)"
	pytest tests/test_*.py -v

test-integration: ## Tests d'intégration
	@echo "$(BLUE)Tests d'intégration...$(NC)"
	pytest tests/test_*_integration.py -v

test-api-live: ## Test l'API en fonctionnement
	@echo "$(BLUE)Test de l'API live...$(NC)"
	curl -f http://localhost:$(API_PORT)/health || (echo "$(RED)❌ API non accessible$(NC)" && exit 1)
	@echo "$(GREEN)✅ API accessible$(NC)"

lint: ## Vérifie le style de code
	@echo "$(BLUE)Vérification du code...$(NC)"
	black --check src/ tests/
	isort --check-only src/ tests/
	flake8 src/ tests/

format: ## Formate le code
	@echo "$(BLUE)Formatage du code...$(NC)"
	black src/ tests/
	isort src/ tests/
	@echo "$(GREEN)✅ Code formaté$(NC)"

# =============================================================================
# UTILITAIRES
# =============================================================================

clean: ## Nettoie les fichiers temporaires
	@echo "$(BLUE)Nettoyage...$(NC)"
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.pyo" -delete
	find . -type f -name "*.pyd" -delete
	find . -type f -name ".coverage" -delete
	find . -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true
	rm -rf .pytest_cache/ .mypy_cache/ dist/ build/
	@echo "$(GREEN)✅ Nettoyage terminé$(NC)"

reset-data: ## Supprime toutes les données
	@echo "$(YELLOW)⚠️  Suppression des données...$(NC)"
	@read -p "Êtes-vous sûr? [y/N] " -n 1 -r; \
	if [[ $REPLY =~ ^[Yy]$ ]]; then \
		rm -rf data/raw/* data/processed/* logs/* models/* mlruns/*; \
		echo "$(GREEN)✅ Données supprimées$(NC)"; \
	else \
		echo "$(BLUE)Annulé$(NC)"; \
	fi

status: ## Affiche le statut des services
	@echo "$(BLUE)Statut des services:$(NC)"
	@echo -n "  API ($(API_PORT)): "
	@curl -s -f http://localhost:$(API_PORT)/health >/dev/null && echo "$(GREEN)✅ UP$(NC)" || echo "$(RED)❌ DOWN$(NC)"
	@echo -n "  Métriques ($(METRICS_PORT)): "
	@curl -s -f http://localhost:$(METRICS_PORT)/metrics >/dev/null && echo "$(GREEN)✅ UP$(NC)" || echo "$(RED)❌ DOWN$(NC)"
	@echo -n "  MLflow (5000): "
	@curl -s -f http://localhost:5000/health >/dev/null && echo "$(GREEN)✅ UP$(NC)" || echo "$(RED)❌ DOWN$(NC)"
	@echo -n "  Prometheus (9090): "
	@curl -s -f http://localhost:9090/-/healthy >/dev/null && echo "$(GREEN)✅ UP$(NC)" || echo "$(RED)❌ DOWN$(NC)"
	@echo -n "  Grafana (3000): "
	@curl -s -f http://localhost:3000/api/health >/dev/null && echo "$(GREEN)✅ UP$(NC)" || echo "$(RED)❌ DOWN$(NC)"

logs-api: ## Affiche les logs de l'API
	tail -f logs/api/*.log 2>/dev/null || echo "$(YELLOW)Pas de logs API trouvés$(NC)"

logs-training: ## Affiche les logs d'entraînement
	tail -f logs/training/*.log 2>/dev/null || echo "$(YELLOW)Pas de logs d'entraînement trouvés$(NC)"

info: ## Affiche les informations du projet
	@echo "$(BLUE)Portfolio RL - Informations:$(NC)"
	@echo "  📁 Projet: $(PWD)"
	@echo "  🐍 Python: $(shell $(PYTHON) --version)"
	@echo "  📦 Pip: $(shell $(PIP) --version)"
	@echo "  🔧 API: http://localhost:$(API_PORT)"
	@echo "  📊 Métriques: http://localhost:$(METRICS_PORT)/metrics"
	@echo "  📚 Documentation: http://localhost:$(API_PORT)/docs"
	@echo ""
	@echo "$(BLUE)Répertoires:$(NC)"
	@echo "  📂 Données brutes: data/raw/"
	@echo "  📂 Données traitées: data/processed/"
	@echo "  📂 Modèles: models/"
	@echo "  📂 Logs: logs/"
	@echo "  📂 MLflow: mlruns/"

# =============================================================================
# RACCOURCIS DE DÉVELOPPEMENT
# =============================================================================

dev-start: setup api ## Démarrage rapide pour développement

dev-test: api-test test ## Tests complets de développement

dev-full: data-pipeline train evaluate api ## Pipeline complet de développement

demo: docker-stack monitoring ## Démonstration complète
	@echo "$(GREEN)🎉 Démonstration démarrée!$(NC)"
	@echo "$(BLUE)Accédez aux services:$(NC)"
	@echo "  🔧 API: http://localhost:$(API_PORT)/docs"
	@echo "  📊 Prometheus: http://localhost:9090"
	@echo "  📈 Grafana: http://localhost:3000"
	@echo "  🧪 MLflow: http://localhost:5000"

# =============================================================================
# PRODUCTION
# =============================================================================

prod-check: ## Vérifications avant production
	@echo "$(BLUE)Vérifications de production...$(NC)"
	$(PYTHON) start_api.py --check-only
	pytest tests/ -q
	@echo "$(GREEN)✅ Prêt pour la production$(NC)"

prod-deploy: prod-check docker-build ## Déploiement en production
	@echo "$(BLUE)Déploiement en production...$(NC)"
	docker-compose -f docker-compose.prod.yml up -d
	@echo "$(GREEN)✅ Déployé en production$(NC)"

# Valeur par défaut
.DEFAULT_GOAL := help