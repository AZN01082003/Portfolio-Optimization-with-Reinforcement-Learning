#!/usr/bin/env python3
"""
Script de démarrage pour l'API Portfolio RL
Gère les dépendances et démarre l'API avec ou sans monitoring
"""
import os
import sys
import subprocess
import time
import logging
from pathlib import Path

# Configuration du logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def check_dependencies():
    """Vérifie et installe les dépendances si nécessaire."""
    required_packages = {
        'fastapi': 'fastapi',
        'uvicorn': 'uvicorn[standard]',
        'stable_baselines3': 'stable-baselines3[extra]',
        'prometheus_client': 'prometheus-client'
    }
    
    missing_packages = []
    
    for package, install_name in required_packages.items():
        try:
            __import__(package)
            logger.info(f"✅ {package} disponible")
        except ImportError:
            logger.warning(f"⚠️ {package} manquant")
            missing_packages.append(install_name)
    
    if missing_packages:
        logger.info(f"Installation des packages manquants: {missing_packages}")
        try:
            subprocess.check_call([
                sys.executable, '-m', 'pip', 'install'
            ] + missing_packages)
            logger.info("✅ Installation terminée")
        except subprocess.CalledProcessError as e:
            logger.error(f"❌ Erreur installation: {e}")
            return False
    
    return True

def setup_environment():
    """Configure l'environnement."""
    # Ajouter le répertoire racine au PYTHONPATH
    project_root = Path(__file__).parent
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
    
    # Variables d'environnement pour l'API
    os.environ.setdefault('API_HOST', '0.0.0.0')
    os.environ.setdefault('API_PORT', '8000')
    os.environ.setdefault('MLFLOW_TRACKING_URI', 'http://localhost:5000')
    
    # Créer les répertoires nécessaires
    directories = [
        'logs/api',
        'logs/training',
        'data/processed',
        'models'
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
        logger.debug(f"📁 Répertoire créé/vérifié: {directory}")

def start_metrics_server():
    """Démarre le serveur de métriques Prometheus."""
    try:
        # Import dynamique pour éviter les erreurs si pas disponible
        from src.monitoring.simple_metrics import start_metrics_server, simple_portfolio_metrics
        
        if simple_portfolio_metrics and simple_portfolio_metrics.enabled:
            success = start_metrics_server(port=8001, metrics=simple_portfolio_metrics)
            if success:
                logger.info("📊 Serveur de métriques démarré sur le port 8001")
                return True
        
        logger.warning("⚠️ Serveur de métriques non démarré")
        return False
        
    except ImportError as e:
        logger.warning(f"⚠️ Module de métriques non trouvé: {e}")
        return False
    except Exception as e:
        logger.warning(f"⚠️ Impossible de démarrer le serveur de métriques: {e}")
        return False

def start_api(host='0.0.0.0', port=8000, reload=False, with_metrics=True):
    """Démarre l'API."""
    try:
        # Import dynamique pour vérifier la disponibilité
        try:
            import uvicorn
            logger.info("✅ uvicorn disponible")
        except ImportError:
            logger.error("❌ uvicorn non disponible")
            return False
        
        # Vérifier que le module principal existe
        try:
            import src.main
            app = src.main.app
            logger.info("✅ Module API trouvé")
        except ImportError as e:
            logger.error(f"❌ Erreur import src.main: {e}")
            logger.error("💡 Vérifiez que le fichier src/main.py existe et contient 'app'")
            return False
        
        logger.info(f"🚀 Démarrage de l'API sur {host}:{port}")
        
        # Démarrer le serveur de métriques si demandé
        if with_metrics:
            start_metrics_server()
        
        # Configuration uvicorn
        config = uvicorn.Config(
            "src.main:app",  # Utiliser la chaîne d'import
            host=host,
            port=port,
            reload=reload,
            log_level="info",
            access_log=True
        )
        
        server = uvicorn.Server(config)
        server.run()
        
        return True
        
    except KeyboardInterrupt:
        logger.info("🛑 Arrêt demandé par l'utilisateur")
        return True
    except Exception as e:
        logger.error(f"❌ Erreur démarrage API: {e}")
        return False

def test_imports():
    """Teste les imports nécessaires."""
    logger.info("🔍 Test des imports...")
    
    imports_to_test = [
        ('fastapi', 'FastAPI'),
        ('uvicorn', 'uvicorn'),
        ('src.main', 'Module API principal'),
        ('src.monitoring.simple_metrics', 'Module de métriques')
    ]
    
    success = True
    
    for module_name, description in imports_to_test:
        try:
            __import__(module_name)
            logger.info(f"✅ {description} OK")
        except ImportError as e:
            logger.error(f"❌ {description} FAILED: {e}")
            success = False
    
    return success

def main():
    """Fonction principale."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Démarrage de l'API Portfolio RL")
    parser.add_argument("--host", default="0.0.0.0", help="Host de l'API")
    parser.add_argument("--port", type=int, default=8000, help="Port de l'API")
    parser.add_argument("--reload", action="store_true", help="Mode reload pour développement")
    parser.add_argument("--no-metrics", action="store_true", help="Désactiver le serveur de métriques")
    parser.add_argument("--check-only", action="store_true", help="Vérifier seulement les dépendances")
    parser.add_argument("--test-imports", action="store_true", help="Tester les imports")
    
    args = parser.parse_args()
    
    logger.info("🔧 Configuration de l'environnement...")
    setup_environment()
    
    if args.test_imports:
        logger.info("🧪 Test des imports...")
        success = test_imports()
        return 0 if success else 1
    
    logger.info("📦 Vérification des dépendances...")
    if not check_dependencies():
        logger.error("❌ Problème avec les dépendances")
        return 1
    
    if args.check_only:
        logger.info("✅ Vérification terminée avec succès")
        logger.info("💡 Pour tester les imports: python start_api.py --test-imports")
        return 0
    
    logger.info("🚀 Démarrage de l'API Portfolio RL...")
    success = start_api(
        host=args.host,
        port=args.port,
        reload=args.reload,
        with_metrics=not args.no_metrics
    )
    
    return 0 if success else 1

if __name__ == "__main__":
    try:
        exit_code = main()
        sys.exit(exit_code)
    except KeyboardInterrupt:
        logger.info("🛑 Arrêt demandé par l'utilisateur")
        sys.exit(0)
    except Exception as e:
        logger.error(f"❌ Erreur inattendue: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)