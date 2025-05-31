#!/usr/bin/env python3
"""
Version Windows-compatible du script de déploiement MLOps Portfolio RL.
Cette version évite les problèmes d'encodage Unicode sur Windows.
"""
import os
import sys
import platform

# Forcer l'encodage UTF-8 sur Windows
if platform.system() == "Windows":
    os.environ["PYTHONIOENCODING"] = "utf-8"
    # Essayer de configurer la console pour UTF-8
    try:
        import codecs
        sys.stdout = codecs.getwriter('utf-8')(sys.stdout.buffer, 'strict')
        sys.stderr = codecs.getwriter('utf-8')(sys.stderr.buffer, 'strict')
    except:
        pass

# Patch temporaire pour remplacer les émojis par du texte
def patch_for_windows():
    """Remplace les émojis par du texte pour compatibilité Windows."""
    global logger
    
    # Créer un handler de logging compatible Windows
    import logging
    
    class WindowsFormatter(logging.Formatter):
        """Formatter qui remplace les émojis par du texte."""
        
        emoji_map = {
            '🏗️': '[INIT]',
            '📦': '[DEPS]',
            '📝': '[CONFIG]',
            '📁': '[DIR]',
            '✅': '[OK]',
            '❌': '[ERROR]',
            '⚠️': '[WARN]',
            '🔧': '[SETUP]',
            '🚀': '[DEPLOY]',
            '🛠️': '[DEV]',
            '⏳': '[WAIT]',
            '📊': '[MLFLOW]',
            '📈': '[GRAFANA]',
            '🐳': '[DOCKER]',
            '🔨': '[BUILD]',
            '⚙️': '[CONFIG]',
            '🧪': '[TEST]',
            '🧹': '[CLEANUP]',
            '🎉': '[SUCCESS]',
            '💥': '[FAILED]',
            '🛑': '[STOP]',
            '🌐': '[API]',
            '📉': '[METRICS]',
            '🔐': '[SECURITY]'
        }
        
        def format(self, record):
            msg = super().format(record)
            for emoji, text in self.emoji_map.items():
                msg = msg.replace(emoji, text)
            return msg
    
    # Reconfigurer le logger
    for handler in logging.getLogger().handlers[:]:
        logging.getLogger().removeHandler(handler)
    
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(WindowsFormatter('%(asctime)s - %(levelname)s - %(message)s'))
    
    logging.getLogger().addHandler(handler)
    logging.getLogger().setLevel(logging.INFO)

# Appliquer le patch pour Windows
if platform.system() == "Windows":
    patch_for_windows()

# Importer le module principal après le patch
import time
import subprocess
import json
import logging
import requests
from pathlib import Path
from typing import Dict, List, Any
import signal
import atexit

logger = logging.getLogger(__name__)

class MLOpsDeployerWindows:
    """Version Windows du déployeur MLOps."""
    
    def __init__(self, project_root: str = "."):
        self.project_root = Path(project_root).resolve()
        self.deployment_status = {}
        self.running_processes = []
        
        logger.info("Initialisation du déployeur MLOps dans: %s", self.project_root)
        
        # Enregistrer le nettoyage
        atexit.register(self.cleanup)
        
        # Vérifier la structure du projet
        self._validate_project_structure()
    
    def cleanup(self):
        """Nettoie les ressources."""
        logger.info("Nettoyage des ressources...")
        for proc in self.running_processes:
            try:
                proc.terminate()
                proc.wait(timeout=5)
            except:
                try:
                    proc.kill()
                except:
                    pass
    
    def _validate_project_structure(self):
        """Valide et crée la structure du projet."""
        required_dirs = [
            "src", "src/models", "src/data", "src/api", "src/monitoring",
            "config", "dags", "monitoring", "monitoring/grafana", 
            "monitoring/grafana/dashboards", "monitoring/grafana/provisioning",
            "monitoring/grafana/provisioning/datasources",
            "monitoring/grafana/provisioning/dashboards",
            "data", "data/raw", "data/processed", "logs", "logs/api",
            "logs/training", "logs/evaluation", "models", "artifacts", "mlruns"
        ]
        
        missing_dirs = []
        for dir_name in required_dirs:
            dir_path = self.project_root / dir_name
            if not dir_path.exists():
                missing_dirs.append(dir_name)
                dir_path.mkdir(parents=True, exist_ok=True)
                logger.info("Créé: %s", dir_name)
        
        if missing_dirs:
            logger.info("%d répertoires créés", len(missing_dirs))
        else:
            logger.info("Structure de projet validée")
    
    def install_dependencies(self) -> bool:
        """Installe toutes les dépendances Python."""
        logger.info("Installation des dépendances Python...")
        
        dependencies = [
            "fastapi>=0.100.0",
            "uvicorn[standard]>=0.20.0", 
            "pydantic>=2.0.0",
            "stable-baselines3[extra]>=2.0.0",
            "gymnasium>=0.28.1",
            "torch",
            "numpy>=1.20.0",
            "pandas>=1.3.0",
            "yfinance>=0.1.70",
            "scikit-learn>=1.0.0",
            "mlflow>=2.0.0",
            "dvc>=2.10.0",
            "prometheus-client>=0.15.0",
            "psutil>=5.9.0",
            "matplotlib>=3.5.0",
            "seaborn>=0.11.2",
            "requests>=2.25.1",
            "python-dotenv>=0.20.0",
            "pyyaml>=6.0"
        ]
        
        try:
            failed_packages = []
            
            for dep in dependencies:
                logger.info("   Installation: %s", dep)
                try:
                    result = subprocess.run([
                        sys.executable, "-m", "pip", "install", dep
                    ], capture_output=True, text=True, timeout=300)
                    
                    if result.returncode != 0:
                        logger.warning("Échec: %s", dep)
                        failed_packages.append(dep)
                    else:
                        logger.debug("OK: %s", dep)
                        
                except subprocess.TimeoutExpired:
                    logger.warning("Timeout: %s", dep)
                    failed_packages.append(dep)
            
            if failed_packages:
                logger.warning("%d packages ont échoué: %s", len(failed_packages), failed_packages)
                return len(failed_packages) < len(dependencies) * 0.2
            
            logger.info("Toutes les dépendances installées")
            return True
            
        except Exception as e:
            logger.error("Erreur installation dépendances: %s", e)
            return False
    
    def create_configuration_files(self) -> bool:
        """Crée tous les fichiers de configuration."""
        logger.info("Création des fichiers de configuration...")
        
        try:
            # Configuration principale
            main_config = {
                "data": {
                    "tickers": ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA"],
                    "lookback_years": 2,
                    "train_ratio": 0.7,
                    "output_dir": "data/processed",
                    "cache_duration_minutes": 30
                },
                "environment": {
                    "portfolio_value": 10000,
                    "window_size": 30,
                    "trans_cost": 0.0005,
                    "return_rate": 0.0001,
                    "reward_scaling": 100.0,
                    "max_reward_clip": 5.0,
                    "min_reward_clip": -5.0,
                    "normalize_observations": True,
                    "risk_penalty": 0.1
                },
                "training": {
                    "algorithm": "PPO",
                    "learning_rate": 0.0003,
                    "n_steps": 2048,
                    "batch_size": 64,
                    "ent_coef": 0.01,
                    "clip_range": 0.2,
                    "max_grad_norm": 0.5,
                    "gae_lambda": 0.95,
                    "gamma": 0.99,
                    "n_epochs": 10,
                    "total_timesteps": 100000,
                    "log_dir": "logs/training"
                },
                "evaluation": {
                    "results_dir": "logs/evaluation",
                    "min_reward_threshold": 0.1,
                    "min_return_threshold": 5.0
                },
                "api": {
                    "host": "0.0.0.0",
                    "port": 8000,
                    "metrics_port": 8001
                },
                "mlflow": {
                    "tracking_uri": "http://localhost:5000",
                    "experiment_name": "portfolio_optimization"
                }
            }
            
            config_file = self.project_root / "config" / "default.json"
            config_file.parent.mkdir(exist_ok=True)
            with open(config_file, 'w', encoding='utf-8') as f:
                json.dump(main_config, f, indent=2)
            logger.info("Configuration principale: %s", config_file)
            
            self._create_docker_compose_production()
            self._create_prometheus_config()
            self._create_dockerfiles()
            self._create_requirements_file()
            self._create_startup_scripts()
            self._create_grafana_config()
            
            logger.info("Tous les fichiers de configuration créés")
            return True
            
        except Exception as e:
            logger.error("Erreur création configuration: %s", e)
            return False
    
    def _create_docker_compose_production(self):
        """Crée le docker-compose de production."""
        docker_compose = """version: '3.8'

services:
  portfolio-api:
    build:
      context: .
      dockerfile: api.Dockerfile
    ports:
      - "8000:8000"
      - "8001:8001"
    environment:
      - PYTHONPATH=/app
      - MLFLOW_TRACKING_URI=http://mlflow:5000
      - PROMETHEUS_MONITORING=true
    volumes:
      - ./src:/app/src
      - ./config:/app/config
      - ./models:/app/models
      - ./data:/app/data
      - ./logs:/app/logs
    depends_on:
      - mlflow
      - prometheus
    networks:
      - portfolio-network
    restart: unless-stopped

  mlflow:
    image: python:3.9-slim
    ports:
      - "5000:5000"
    volumes:
      - ./mlruns:/mlflow/mlruns
      - ./artifacts:/mlflow/artifacts
    environment:
      - MLFLOW_BACKEND_STORE_URI=sqlite:///mlflow/mlruns/mlflow.db
      - MLFLOW_DEFAULT_ARTIFACT_ROOT=/mlflow/artifacts
    command: >
      bash -c "
      pip install mlflow &&
      mkdir -p /mlflow/mlruns /mlflow/artifacts &&
      mlflow server --host 0.0.0.0 --port 5000 
        --backend-store-uri sqlite:///mlflow/mlruns/mlflow.db 
        --default-artifact-root /mlflow/artifacts
      "
    networks:
      - portfolio-network
    restart: unless-stopped

  prometheus:
    image: prom/prometheus:latest
    ports:
      - "9090:9090"
    volumes:
      - ./monitoring/prometheus.yml:/etc/prometheus/prometheus.yml
      - prometheus_data:/prometheus
    command:
      - '--config.file=/etc/prometheus/prometheus.yml'
      - '--storage.tsdb.path=/prometheus'
      - '--storage.tsdb.retention.time=30d'
    networks:
      - portfolio-network
    restart: unless-stopped

  grafana:
    image: grafana/grafana:latest
    ports:
      - "3000:3000"
    environment:
      - GF_SECURITY_ADMIN_PASSWORD=admin
      - GF_USERS_ALLOW_SIGN_UP=false
    volumes:
      - grafana_data:/var/lib/grafana
      - ./monitoring/grafana/provisioning:/etc/grafana/provisioning
    networks:
      - portfolio-network
    restart: unless-stopped

volumes:
  prometheus_data:
  grafana_data:

networks:
  portfolio-network:
    driver: bridge
"""
        
        compose_file = self.project_root / "docker-compose-production.yml"
        with open(compose_file, 'w', encoding='utf-8') as f:
            f.write(docker_compose)
        logger.info("Docker Compose créé: %s", compose_file)
    
    def _create_prometheus_config(self):
        """Crée la configuration Prometheus."""
        prometheus_config = """global:
  scrape_interval: 15s

scrape_configs:
  - job_name: 'prometheus'
    static_configs:
      - targets: ['localhost:9090']

  - job_name: 'portfolio-api'
    static_configs:
      - targets: ['portfolio-api:8001']
    metrics_path: '/metrics'
    scrape_interval: 5s
"""
        
        prom_file = self.project_root / "monitoring" / "prometheus.yml"
        prom_file.parent.mkdir(exist_ok=True)
        with open(prom_file, 'w', encoding='utf-8') as f:
            f.write(prometheus_config)
        logger.info("Configuration Prometheus: %s", prom_file)
    
    def _create_dockerfiles(self):
        """Crée les Dockerfiles."""
        api_dockerfile = """FROM python:3.9-slim

WORKDIR /app

RUN apt-get update && apt-get install -y curl git build-essential && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && pip install --no-cache-dir -r requirements.txt

COPY src/ ./src/
COPY config/ ./config/

RUN mkdir -p logs/api data/processed models

RUN useradd -m -u 1000 apiuser && chown -R apiuser:apiuser /app
USER apiuser

EXPOSE 8000 8001

ENV PYTHONPATH=/app
ENV API_HOST=0.0.0.0
ENV API_PORT=8000

CMD ["python", "src/main_minimal.py"]
"""
        
        api_dockerfile_file = self.project_root / "api.Dockerfile"
        with open(api_dockerfile_file, 'w', encoding='utf-8') as f:
            f.write(api_dockerfile)
        
        logger.info("Dockerfiles créés")
    
    def _create_requirements_file(self):
        """Crée requirements.txt."""
        requirements = """fastapi>=0.100.0
uvicorn[standard]>=0.20.0
pydantic>=2.0.0
stable-baselines3[extra]>=2.0.0
gymnasium>=0.28.1
torch
numpy>=1.20.0
pandas>=1.3.0
yfinance>=0.1.70
scikit-learn>=1.0.0
mlflow>=2.0.0
prometheus-client>=0.15.0
psutil>=5.9.0
matplotlib>=3.5.0
seaborn>=0.11.2
requests>=2.25.1
python-dotenv>=0.20.0
pyyaml>=6.0
"""
        
        req_file = self.project_root / "requirements.txt"
        with open(req_file, 'w', encoding='utf-8') as f:
            f.write(requirements)
        logger.info("Requirements créé: %s", req_file)
    
    def _create_startup_scripts(self):
        """Crée les scripts de démarrage."""
        # Script Windows (.bat)
        start_dev_script = """@echo off
echo [DEPLOY] Démarrage en mode développement...

python -c "import fastapi, mlflow, prometheus_client" 2>nul || (
    echo [ERROR] Dépendances manquantes. Installation...
    pip install -r requirements.txt
)

echo [DEPLOY] Démarrage des services...
docker-compose -f docker-compose-production.yml up -d mlflow prometheus grafana

echo [WAIT] Attente des services...
timeout /t 10 /nobreak > nul

echo [SUCCESS] Services démarrés!
echo [INFO] API: http://localhost:8000
echo [INFO] MLflow: http://localhost:5000
echo [INFO] Prometheus: http://localhost:9090
echo [INFO] Grafana: http://localhost:3000 (admin/admin)
echo.
echo [INFO] Pour arrêter: Ctrl+C puis docker-compose down

pause
"""
        
        dev_script_file = self.project_root / "start_development.bat"
        with open(dev_script_file, 'w', encoding='utf-8') as f:
            f.write(start_dev_script)
        
        # Script Unix/Linux
        start_dev_unix = """#!/bin/bash
echo "[DEPLOY] Démarrage en mode développement..."

python -c "import fastapi, mlflow, prometheus_client" 2>/dev/null || {
    echo "[ERROR] Dépendances manquantes. Installation..."
    pip install -r requirements.txt
}

echo "[DEPLOY] Démarrage des services..."
docker-compose -f docker-compose-production.yml up -d mlflow prometheus grafana

echo "[WAIT] Attente des services..."
sleep 10

echo "[SUCCESS] Services démarrés!"
echo "[INFO] API: http://localhost:8000"
echo "[INFO] MLflow: http://localhost:5000"
echo "[INFO] Prometheus: http://localhost:9090"
echo "[INFO] Grafana: http://localhost:3000 (admin/admin)"
echo ""
echo "[INFO] Pour arrêter: Ctrl+C puis docker-compose down"
"""
        
        dev_script_unix = self.project_root / "start_development.sh"
        with open(dev_script_unix, 'w', encoding='utf-8') as f:
            f.write(start_dev_unix)
        
        try:
            os.chmod(dev_script_unix, 0o755)
        except:
            pass
        
        logger.info("Scripts de démarrage créés")
    
    def _create_grafana_config(self):
        """Crée la configuration Grafana."""
        try:
            import yaml
            
            datasource_config = {
                "apiVersion": 1,
                "datasources": [
                    {
                        "name": "Prometheus",
                        "type": "prometheus", 
                        "access": "proxy",
                        "url": "http://prometheus:9090",
                        "isDefault": True
                    }
                ]
            }
            
            datasource_file = self.project_root / "monitoring" / "grafana" / "provisioning" / "datasources" / "prometheus.yml"
            datasource_file.parent.mkdir(parents=True, exist_ok=True)
            
            with open(datasource_file, 'w', encoding='utf-8') as f:
                yaml.dump(datasource_config, f)
            
            logger.info("Configuration Grafana créée")
        except ImportError:
            logger.warning("PyYAML non disponible - configuration Grafana simplifiée")
    
    def setup_data_directories(self) -> bool:
        """Configure les répertoires de données."""
        logger.info("Configuration des répertoires de données...")
        
        directories = [
            "data/raw", "data/processed", 
            "logs/api", "logs/training", "logs/evaluation",
            "models", "artifacts", "mlruns", "reports"
        ]
        
        try:
            for directory in directories:
                dir_path = self.project_root / directory
                dir_path.mkdir(parents=True, exist_ok=True)
            
            logger.info("%d répertoires configurés", len(directories))
            return True
            
        except Exception as e:
            logger.error("Erreur création répertoires: %s", e)
            return False
    
    def initialize_git_repository(self) -> bool:
        """Initialise Git."""
        logger.info("Configuration Git...")
        
        try:
            if not (self.project_root / ".git").exists():
                result = subprocess.run(["git", "init"], cwd=self.project_root, 
                                       capture_output=True, text=True)
                if result.returncode == 0:
                    logger.info("Repository Git initialisé")
                else:
                    logger.warning("Git non disponible")
            
            gitignore_content = """# Données et modèles
data/raw/*.csv
data/processed/*.npy
models/*.zip
mlruns/
artifacts/
*.db

# Logs
logs/
*.log

# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
env/
venv/
.venv/
.env

# IDE
.vscode/
.idea/
*.swp
*.swo

# OS
.DS_Store
Thumbs.db

# Docker
.docker/
docker-compose.override.yml

# Secrets
*.key
*.pem
secrets/

# Tests
.pytest_cache/
.coverage
htmlcov/
"""
            
            gitignore_file = self.project_root / ".gitignore"
            with open(gitignore_file, 'w', encoding='utf-8') as f:
                f.write(gitignore_content)
            
            logger.info(".gitignore configuré")
            return True
            
        except Exception as e:
            logger.warning("Erreur configuration Git: %s", e)
            return False
    
    def _check_docker_availability(self) -> bool:
        """Vérifie Docker."""
        logger.info("Vérification de Docker...")
        
        try:
            result = subprocess.run(["docker", "--version"], 
                                   capture_output=True, text=True, timeout=10)
            if result.returncode != 0:
                logger.error("Docker non installé")
                return False
            
            result = subprocess.run(["docker-compose", "--version"], 
                                   capture_output=True, text=True, timeout=10)
            if result.returncode != 0:
                logger.error("Docker Compose non installé")
                return False
            
            result = subprocess.run(["docker", "info"], 
                                   capture_output=True, text=True, timeout=10)
            if result.returncode != 0:
                logger.error("Docker daemon non démarré")
                return False
            
            logger.info("Docker disponible et fonctionnel")
            return True
            
        except Exception as e:
            logger.error("Docker non disponible: %s", e)
            return False
    
    def deploy_infrastructure(self, mode: str = "development") -> Dict[str, bool]:
        """Déploie l'infrastructure."""
        logger.info("Déploiement infrastructure en mode %s", mode)
        
        results = {}
        
        # Validation de l'environnement
        results["environment"] = self._check_docker_availability()
        
        if not results["environment"]:
            logger.error("Docker requis pour le déploiement")
            return results
        
        # Construction des images Docker
        results["docker_build"] = self._build_docker_images()
        
        # Démarrage des services
        if mode == "production":
            results["services"] = self._start_production_services()
        else:
            results["services"] = self._start_development_services()
        
        # Attendre que les services soient prêts
        if results["services"]:
            results["services_ready"] = self._wait_for_services()
        else:
            results["services_ready"] = False
        
        # Configuration post-déploiement
        if results["services_ready"]:
            results["post_config"] = self._post_deployment_configuration()
        else:
            results["post_config"] = False
        
        # Tests de validation
        if results["post_config"]:
            results["validation"] = self._validate_deployment()
        else:
            results["validation"] = False
        
        return results
    
    def _build_docker_images(self) -> bool:
        """Construit les images Docker."""
        logger.info("Construction des images Docker...")
        
        try:
            logger.info("   Construction image API...")
            result = subprocess.run([
                "docker", "build", "-f", "api.Dockerfile", 
                "-t", "portfolio-api", "."
            ], cwd=self.project_root, capture_output=True, text=True)
            
            if result.returncode != 0:
                logger.warning("Erreur image API: %s", result.stderr)
                return False
            
            logger.info("Images Docker construites")
            return True
            
        except Exception as e:
            logger.error("Erreur construction Docker: %s", e)
            return False
    
    def _start_production_services(self) -> bool:
        """Démarre les services de production."""
        logger.info("Démarrage des services de production...")
        
        try:
            result = subprocess.run([
                "docker-compose", "-f", "docker-compose-production.yml", 
                "up", "-d"
            ], cwd=self.project_root, capture_output=True, text=True)
            
            if result.returncode != 0:
                logger.error("Erreur démarrage services: %s", result.stderr)
                return False
            
            logger.info("Services de production démarrés")
            return True
            
        except Exception as e:
            logger.error("Erreur démarrage services: %s", e)
            return False
    
    def _start_development_services(self) -> bool:
        """Démarre les services de développement."""
        logger.info("Démarrage des services de développement...")
        
        essential_services = ["mlflow", "prometheus", "grafana"]
        
        try:
            for service in essential_services:
                logger.info("   Démarrage %s...", service)
                result = subprocess.run([
                    "docker-compose", "-f", "docker-compose-production.yml", 
                    "up", "-d", service
                ], cwd=self.project_root, capture_output=True, text=True)
                
                if result.returncode != 0:
                    logger.warning("Erreur démarrage %s: %s", service, result.stderr)
                    return False
            
            logger.info("Services de développement démarrés")
            return True
            
        except Exception as e:
            logger.error("Erreur démarrage services développement: %s", e)
            return False
    
    def _wait_for_services(self, timeout: int = 120) -> bool:
        """Attend que les services soient prêts."""
        logger.info("Attente que les services soient prêts...")
        
        services_to_check = [
            {"name": "MLflow", "url": "http://localhost:5000", "path": "/"},
            {"name": "Prometheus", "url": "http://localhost:9090", "path": "/-/ready"},
            {"name": "Grafana", "url": "http://localhost:3000", "path": "/api/health"}
        ]
        
        start_time = time.time()
        
        while time.time() - start_time < timeout:
            all_ready = True
            
            for service in services_to_check:
                try:
                    response = requests.get(
                        f"{service['url']}{service['path']}", 
                        timeout=5
                    )
                    if response.status_code not in [200, 401]:
                        all_ready = False
                        logger.debug("%s pas encore prêt", service['name'])
                        break
                    else:
                        logger.debug("%s prêt", service['name'])
                        
                except requests.exceptions.RequestException:
                    all_ready = False
                    logger.debug("%s pas encore accessible", service['name'])
                    break
            
            if all_ready:
                logger.info("Tous les services sont prêts")
                return True
            
            time.sleep(5)
        
        logger.warning("Timeout après %ds - certains services peuvent ne pas être prêts", timeout)
        return False
    
    def _post_deployment_configuration(self) -> bool:
        """Configuration post-déploiement."""
        logger.info("Configuration post-déploiement...")
        
        try:
            self._setup_mlflow_experiment()
            self._create_initial_data_structure()
            self._setup_permissions()
            
            logger.info("Configuration post-déploiement terminée")
            return True
            
        except Exception as e:
            logger.error("Erreur configuration post-déploiement: %s", e)
            return False
    
    def _setup_mlflow_experiment(self) -> bool:
        """Configure MLflow."""
        logger.info("   Configuration MLflow...")
        
        try:
            import mlflow
            
            for attempt in range(10):
                try:
                    mlflow.set_tracking_uri("http://localhost:5000")
                    mlflow.set_experiment("portfolio_optimization")
                    logger.info("Expérimentation MLflow configurée")
                    return True
                except Exception:
                    time.sleep(5)
            
            logger.warning("MLflow non accessible pour configuration")
            return False
            
        except ImportError:
            logger.warning("MLflow non installé")
            return False
    
    def _create_initial_data_structure(self) -> bool:
        """Crée la structure initiale."""
        logger.info("   Structure de données initiale...")
        
        try:
            readme_content = """# Portfolio RL MLOps

## Structure des données

- `raw/`: Données brutes téléchargées
- `processed/`: Données préprocessées pour l'entraînement
- `models/`: Modèles entraînés sauvegardés
- `logs/`: Logs d'entraînement et d'évaluation

## Usage

1. Démarrer l'infrastructure: `start_development.bat` (Windows) ou `./start_development.sh` (Linux)
2. Accéder aux services:
   - API: http://localhost:8000
   - MLflow: http://localhost:5000
   - Grafana: http://localhost:3000 (admin/admin)
   - Prometheus: http://localhost:9090
"""
            
            readme_file = self.project_root / "README.md"
            with open(readme_file, 'w', encoding='utf-8') as f:
                f.write(readme_content)
            
            data_config = {
                "last_update": None,
                "tickers_configured": ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA"],
                "data_sources": {
                    "yahoo_finance": True,
                    "cache_enabled": True
                }
            }
            
            data_config_file = self.project_root / "data" / "config.json"
            with open(data_config_file, 'w', encoding='utf-8') as f:
                json.dump(data_config, f, indent=2)
            
            logger.info("Structure de données initiale créée")
            return True
            
        except Exception as e:
            logger.error("Erreur structure données: %s", e)
            return False
    
    def _setup_permissions(self) -> bool:
        """Configure les permissions."""
        logger.info("   Configuration des permissions...")
        
        try:
            if platform.system() != "Windows":
                scripts = ["start_development.sh"]
                for script in scripts:
                    script_path = self.project_root / script
                    if script_path.exists():
                        os.chmod(script_path, 0o755)
            
            logger.info("Permissions configurées")
            return True
            
        except Exception as e:
            logger.warning("Erreur permissions: %s", e)
            return True
    
    def _validate_deployment(self) -> bool:
        """Valide le déploiement."""
        logger.info("Validation du déploiement...")
        
        validation_results = {
            "mlflow_connection": self._test_mlflow_connection(),
            "prometheus_metrics": self._test_prometheus_metrics(),
            "grafana_access": self._test_grafana_access(),
            "file_structure": self._test_file_structure()
        }
        
        successful_tests = sum(validation_results.values())
        total_tests = len(validation_results)
        
        logger.info("Tests réussis: %d/%d", successful_tests, total_tests)
        
        for test, result in validation_results.items():
            status = "[OK]" if result else "[ERROR]"
            logger.info("   %s %s", status, test)
        
        is_valid = successful_tests >= total_tests * 0.8
        
        if is_valid:
            logger.info("Déploiement validé avec succès")
        else:
            logger.warning("Déploiement partiellement validé")
        
        return is_valid
    
    def _test_mlflow_connection(self) -> bool:
        """Test MLflow."""
        try:
            response = requests.get("http://localhost:5000", timeout=5)
            return response.status_code == 200
        except Exception:
            return False
    
    def _test_prometheus_metrics(self) -> bool:
        """Test Prometheus."""
        try:
            response = requests.get("http://localhost:9090/-/ready", timeout=5)
            return response.status_code == 200
        except Exception:
            return False
    
    def _test_grafana_access(self) -> bool:
        """Test Grafana."""
        try:
            response = requests.get("http://localhost:3000/api/health", timeout=5)
            return response.status_code in [200, 401]
        except Exception:
            return False
    
    def _test_file_structure(self) -> bool:
        """Test structure fichiers."""
        required_files = [
            "config/default.json",
            "docker-compose-production.yml",
            "requirements.txt",
            "monitoring/prometheus.yml"
        ]
        
        return all(
            (self.project_root / file_path).exists() 
            for file_path in required_files
        )
    
    def full_deployment(self, mode: str = "development") -> bool:
        """Déploiement complet."""
        logger.info("DÉPLOIEMENT COMPLET EN MODE %s", mode.upper())
        
        steps = [
            ("Installation dépendances", self.install_dependencies),
            ("Création configuration", self.create_configuration_files),
            ("Configuration répertoires", self.setup_data_directories),
            ("Initialisation Git", self.initialize_git_repository),
        ]
        
        for step_name, step_func in steps:
            logger.info("")
            logger.info("%s...", step_name)
            try:
                if not step_func():
                    logger.error("Échec: %s", step_name)
                    return False
                logger.info("%s terminé", step_name)
            except Exception as e:
                logger.error("Erreur %s: %s", step_name, e)
                return False
        
        logger.info("")
        logger.info("Déploiement infrastructure (%s)...", mode)
        deployment_results = self.deploy_infrastructure(mode)
        
        logger.info("")
        logger.info("RÉSULTATS DU DÉPLOIEMENT:")
        for step, success in deployment_results.items():
            status = "[OK]" if success else "[ERROR]"
            logger.info("   %s %s", status, step)
        
        overall_success = all(deployment_results.values())
        
        if overall_success:
            logger.info("")
            logger.info("DÉPLOIEMENT RÉUSSI!")
            self._display_deployment_summary(mode)
        else:
            logger.error("")
            logger.error("DÉPLOIEMENT ÉCHOUÉ!")
            self._display_troubleshooting_guide()
        
        return overall_success
    
    def _display_deployment_summary(self, mode: str):
        """Affiche le résumé."""
        logger.info("RÉSUMÉ DU DÉPLOIEMENT:")
        logger.info("   Mode: %s", mode)
        logger.info("   Services disponibles:")
        logger.info("     API: http://localhost:8000")
        logger.info("     MLflow: http://localhost:5000")
        logger.info("     Prometheus: http://localhost:9090")
        logger.info("     Grafana: http://localhost:3000 (admin/admin)")
        logger.info("")
        logger.info("PROCHAINES ÉTAPES:")
        logger.info("   1. Tester l'API: curl http://localhost:8000/health")
        logger.info("   2. Accéder à MLflow pour voir les expérimentations")
        logger.info("   3. Configurer les dashboards Grafana")
        if mode == "development":
            logger.info("   4. Démarrer l'API: python src/main_minimal.py")
        logger.info("")
        logger.info("DOCUMENTATION:")
        logger.info("   README.md créé avec les instructions détaillées")
    
    def _display_troubleshooting_guide(self):
        """Guide de dépannage."""
        logger.error("GUIDE DE DÉPANNAGE:")
        logger.error("   1. Vérifier Docker: docker --version")
        logger.error("   2. Vérifier les ports: netstat -an | findstr :8000")
        logger.error("   3. Voir les logs: docker-compose logs")
        logger.error("   4. Redémarrer: docker-compose down && docker-compose up")
        logger.error("   5. Nettoyer: docker system prune -a")


def main():
    """Point d'entrée principal."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Déployeur MLOps Portfolio RL (Windows)")
    parser.add_argument("--mode", choices=["development", "production"], 
                       default="development", help="Mode de déploiement")
    parser.add_argument("--project-root", default=".", 
                       help="Répertoire racine du projet")
    parser.add_argument("--config-only", action="store_true",
                       help="Créer seulement les fichiers de configuration")
    
    args = parser.parse_args()
    
    # Initialiser le déployeur
    deployer = MLOpsDeployerWindows(args.project_root)
    
    try:
        if args.config_only:
            logger.info("Mode configuration seulement")
            success = (
                deployer.create_configuration_files() and
                deployer.setup_data_directories() and
                deployer.initialize_git_repository()
            )
        else:
            success = deployer.full_deployment(args.mode)
        
        if success:
            logger.info("Opération terminée avec succès!")
            sys.exit(0)
        else:
            logger.error("Opération échouée!")
            sys.exit(1)
            
    except KeyboardInterrupt:
        logger.info("Opération interrompue par l'utilisateur")
        deployer.cleanup()
        sys.exit(130)
    except Exception as e:
        logger.error("Erreur inattendue: %s", e)
        deployer.cleanup()
        sys.exit(1)


if __name__ == "__main__":
    main()