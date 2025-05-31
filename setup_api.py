#!/usr/bin/env python3
"""
Script de setup pour l'API FastAPI.
À exécuter à la racine du projet : python setup_api.py
"""
import os
import sys
import subprocess

def install_api_dependencies():
    """Installe les dépendances nécessaires pour l'API."""
    
    print("📦 Installation des dépendances API...")
    
    dependencies = [
        "fastapi>=0.100.0",
        "uvicorn[standard]>=0.20.0",
        "pydantic>=2.0.0",
        "requests>=2.25.1",
        "python-multipart>=0.0.6"  # Pour les uploads de fichiers
    ]
    
    for dep in dependencies:
        try:
            print(f"   Installing {dep}...")
            subprocess.check_call([
                sys.executable, "-m", "pip", "install", dep
            ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            print(f"   ✅ {dep}")
        except subprocess.CalledProcessError:
            print(f"   ❌ Échec: {dep}")
            return False
    
    print("✅ Toutes les dépendances API installées")
    return True

def test_api_imports():
    """Test des imports API."""
    print("🧪 Test des imports API...")
    
    try:
        import fastapi
        import uvicorn
        import pydantic
        import requests
        print("✅ Tous les imports API fonctionnent")
        return True
    except ImportError as e:
        print(f"❌ Import échoué: {e}")
        return False

def create_api_config():
    """Crée un fichier de configuration pour l'API."""
    print("📝 Création de la configuration API...")
    
    api_config = {
        "api": {
            "host": "0.0.0.0",
            "port": 8000,
            "reload": True,
            "workers": 1
        },
        "cors": {
            "allow_origins": ["*"],
            "allow_methods": ["GET", "POST"],
            "allow_headers": ["*"]
        },
        "models": {
            "default_version": "latest",
            "cache_models": True,
            "prediction_timeout": 30
        }
    }
    
    config_dir = "config"
    os.makedirs(config_dir, exist_ok=True)
    
    config_path = os.path.join(config_dir, "api_config.json")
    
    import json
    with open(config_path, 'w') as f:
        json.dump(api_config, f, indent=2)
    
    print(f"✅ Configuration API créée: {config_path}")
    return True

def create_docker_files():
    """Crée les fichiers Docker pour l'API."""
    print("🐳 Création des fichiers Docker...")
    
    # Dockerfile pour l'API
    dockerfile_content = """FROM python:3.9-slim

WORKDIR /app

# Installer les dépendances système
RUN apt-get update && apt-get install -y \\
    gcc \\
    && rm -rf /var/lib/apt/lists/*

# Copier les requirements
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copier le code source
COPY src/ ./src/
COPY config/ ./config/

# Exposer le port
EXPOSE 8000

# Variables d'environnement
ENV PYTHONPATH=/app
ENV API_HOST=0.0.0.0
ENV API_PORT=8000

# Commande de démarrage
CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
"""
    
    with open("api.Dockerfile", "w") as f:
        f.write(dockerfile_content)
    
    # Docker-compose pour l'API
    compose_content = """version: '3.8'

services:
  portfolio-api:
    build:
      context: .
      dockerfile: api.Dockerfile
    container_name: portfolio-api
    ports:
      - "8000:8000"
    environment:
      - PYTHONPATH=/app
      - MLFLOW_TRACKING_URI=http://mlflow:5000
    volumes:
      - ./data:/app/data
      - ./models:/app/models
      - ./logs:/app/logs
    depends_on:
      - mlflow
    restart: unless-stopped
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/health"]
      interval: 30s
      timeout: 10s
      retries: 3

  mlflow:
    image: python:3.9-slim
    container_name: portfolio-mlflow
    ports:
      - "5000:5000"
    command: >
      bash -c "pip install mlflow &&
               mlflow server --host 0.0.0.0 --port 5000 
               --backend-store-uri sqlite:///mlflow.db 
               --default-artifact-root ./artifacts"
    volumes:
      - ./mlruns:/mlruns
      - ./artifacts:/artifacts
    restart: unless-stopped
"""
    
    with open("docker-compose-api.yml", "w") as f:
        f.write(compose_content)
    
    print("✅ Fichiers Docker créés")
    return True

def create_requirements_api():
    """Crée un fichier requirements spécifique à l'API."""
    print("📋 Création de requirements-api.txt...")
    
    requirements_content = """# API FastAPI
fastapi>=0.100.0
uvicorn[standard]>=0.20.0
pydantic>=2.0.0
python-multipart>=0.0.6

# ML et données
numpy>=1.20.0
pandas>=1.3.0
scikit-learn>=1.0.0

# MLflow
mlflow>=2.0.0

# RL (optionnel pour prédictions)
stable-baselines3>=2.0.0
gymnasium>=0.28.1

# Utilitaires
requests>=2.25.1
python-dotenv>=0.20.0
"""
    
    with open("requirements-api.txt", "w") as f:
        f.write(requirements_content)
    
    print("✅ requirements-api.txt créé")
    return True

def create_api_scripts():
    """Crée des scripts utilitaires pour l'API.""" 
    print("📜 Création de scripts utilitaires...")
    
    # Script de démarrage
    start_script = """#!/bin/bash
# Script de démarrage de l'API Portfolio RL

echo "🚀 Démarrage de l'API Portfolio RL..."

# Activer l'environnement virtuel si présent
if [ -d "venv" ]; then
    source venv/bin/activate
    echo "✅ Environnement virtuel activé"
fi

# Vérifier les dépendances
python -c "import fastapi, uvicorn" 2>/dev/null
if [ $? -ne 0 ]; then
    echo "❌ Dépendances manquantes. Installation..."
    pip install -r requirements-api.txt
fi

# Démarrer l'API
echo "🌐 API disponible sur: http://localhost:8000"
echo "📚 Documentation: http://localhost:8000/docs"
echo "⏹️  Arrêter avec: Ctrl+C"

uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --reload
"""
    
    with open("start_api.sh", "w") as f:
        f.write(start_script)
    
    # Rendre exécutable sur Unix
    try:
        os.chmod("start_api.sh", 0o755)
    except:
        pass
    
    # Script Windows
    start_script_win = """@echo off
REM Script de démarrage de l'API Portfolio RL

echo 🚀 Démarrage de l'API Portfolio RL...

REM Activer l'environnement virtuel si présent
if exist venv\\Scripts\\activate.bat (
    call venv\\Scripts\\activate.bat
    echo ✅ Environnement virtuel activé
)

REM Vérifier les dépendances
python -c "import fastapi, uvicorn" 2>nul
if errorlevel 1 (
    echo ❌ Dépendances manquantes. Installation...
    pip install -r requirements-api.txt
)

REM Démarrer l'API
echo 🌐 API disponible sur: http://localhost:8000
echo 📚 Documentation: http://localhost:8000/docs
echo ⏹️  Arrêter avec: Ctrl+C

uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --reload
"""
    
    with open("start_api.bat", "w") as f:
        f.write(start_script_win)
    
    print("✅ Scripts de démarrage créés")
    return True

def display_api_info():
    """Affiche les informations sur l'API."""
    print("\n" + "="*50)
    print("🎯 SETUP API TERMINÉ!")
    print("="*50)
    
    print("\n📋 FICHIERS CRÉÉS:")
    print("   ✅ config/api_config.json - Configuration API")
    print("   ✅ requirements-api.txt - Dépendances API")
    print("   ✅ api.Dockerfile - Image Docker API")
    print("   ✅ docker-compose-api.yml - Orchestration Docker")
    print("   ✅ start_api.sh/.bat - Scripts de démarrage")
    
    print("\n🚀 DÉMARRAGE DE L'API:")
    print("1. 🐍 Mode Python direct:")
    print("   python src/api/main.py")
    
    print("\n2. 📜 Avec script de démarrage:")
    print("   ./start_api.sh  # Linux/Mac")
    print("   start_api.bat   # Windows")
    
    print("\n3. 🐳 Avec Docker:")
    print("   docker-compose -f docker-compose-api.yml up")
    
    print("\n📚 ENDPOINTS DISPONIBLES:")
    print("   🌐 API: http://localhost:8000")
    print("   📖 Documentation: http://localhost:8000/docs")
    print("   📋 ReDoc: http://localhost:8000/redoc")
    print("   🏥 Santé: http://localhost:8000/health")
    print("   🤖 Modèles: http://localhost:8000/models")
    print("   🎯 Prédiction: POST http://localhost:8000/predict")
    print("   📈 Backtest: POST http://localhost:8000/backtest")
    
    print("\n🧪 TESTER L'API:")
    print("   python test_api.py")

def main():
    """Fonction principale de setup."""
    print("🚀 Setup de l'API FastAPI pour Portfolio RL")
    print("="*50)
    
    steps = [
        ("Installation dépendances", install_api_dependencies),
        ("Test imports", test_api_imports),
        ("Configuration API", create_api_config),
        ("Requirements API", create_requirements_api),
        ("Fichiers Docker", create_docker_files),
        ("Scripts utilitaires", create_api_scripts),
    ]
    
    results = []
    
    for step_name, step_func in steps:
        print(f"\n{'='*20} {step_name} {'='*20}")
        
        try:
            result = step_func()
            results.append(result)
            
            if not result and step_name in ["Installation dépendances", "Test imports"]:
                print(f"❌ Étape critique échouée: {step_name}")
                return False
                
        except Exception as e:
            print(f"❌ Erreur lors de {step_name}: {e}")
            results.append(False)
    
    # Résumé
    passed = sum(1 for r in results if r)
    total = len(results)
    
    print(f"\n📊 Résumé: {passed}/{total} étapes réussies")
    
    if passed >= total - 1:  # Accepter 1 échec non critique
        display_api_info()
        return True
    else:
        print(f"❌ Trop d'échecs ({total-passed}). Vérifiez les erreurs ci-dessus.")
        return False

if __name__ == "__main__":
    success = main()
    if not success:
        sys.exit(1)