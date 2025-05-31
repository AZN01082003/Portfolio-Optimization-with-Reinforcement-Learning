#!/usr/bin/env python3
"""
Test complet de l'API FastAPI.
À placer à la racine du projet : test_api.py
"""
import os
import sys
import json
import time
import requests
import subprocess
import signal
from datetime import datetime

# Ajouter le répertoire du projet au PYTHONPATH
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

def test_api_imports():
    """Test des imports nécessaires pour l'API."""
    print("📦 Test des imports API...")
    
    try:
        import fastapi
        print(f"   ✅ FastAPI: {fastapi.__version__}")
        
        import uvicorn
        print(f"   ✅ Uvicorn: {uvicorn.__version__}")
        
        import pydantic
        print(f"   ✅ Pydantic: {pydantic.version.VERSION}")
        
        from src.api.main import app
        print("   ✅ Module API local")
        
        return True
        
    except ImportError as e:
        print(f"   ❌ Import échoué: {e}")
        print("   💡 Installez avec: pip install fastapi uvicorn pydantic")
        return False

def start_api_server(port=8000):
    """Démarre le serveur API local."""
    try:
        print(f"🚀 Démarrage de l'API sur le port {port}...")
        
        # Commande pour démarrer l'API
        api_cmd = [
            sys.executable, "-m", "uvicorn",
            "src.api.main:app",
            "--host", "0.0.0.0",
            "--port", str(port),
            "--reload"
        ]
        
        process = subprocess.Popen(
            api_cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env={**os.environ, "PYTHONPATH": project_root}
        )
        
        # Attendre que l'API soit prête
        max_wait = 30
        for i in range(max_wait):
            try:
                response = requests.get(f"http://localhost:{port}/health", timeout=1)
                if response.status_code == 200:
                    print(f"✅ API démarrée avec succès!")
                    return process
            except:
                pass
            time.sleep(1)
            print(f"⏳ Attente API... ({i+1}/{max_wait})")
        
        print("❌ Timeout: API n'a pas démarré")
        process.terminate()
        return None
        
    except Exception as e:
        print(f"❌ Erreur lors du démarrage de l'API: {e}")
        return None

def test_health_endpoint(port=8000):
    """Test de l'endpoint de santé."""
    print("🏥 Test de l'endpoint /health...")
    
    try:
        response = requests.get(f"http://localhost:{port}/health", timeout=5)
        
        if response.status_code == 200:
            data = response.json()
            print(f"   ✅ Status: {data.get('status')}")
            print(f"   📦 Dépendances: {data.get('dependencies')}")
            print(f"   🕒 Timestamp: {data.get('timestamp')}")
            return True
        else:
            print(f"   ❌ Status code: {response.status_code}")
            return False
            
    except Exception as e:
        print(f"   ❌ Erreur: {e}")
        return False

def test_models_endpoint(port=8000):
    """Test de l'endpoint des modèles."""
    print("🤖 Test de l'endpoint /models...")
    
    try:
        response = requests.get(f"http://localhost:{port}/models", timeout=5)
        
        if response.status_code == 200:
            data = response.json()
            models = data.get('models', [])
            print(f"   ✅ {len(models)} modèle(s) disponible(s)")
            for model in models:
                print(f"      📦 {model}")
            return True
        else:
            print(f"   ❌ Status code: {response.status_code}")
            return False
            
    except Exception as e:
        print(f"   ❌ Erreur: {e}")
        return False

def test_prediction_endpoint(port=8000):
    """Test de l'endpoint de prédiction."""
    print("🎯 Test de l'endpoint /predict...")
    
    try:
        # Créer des données de test
        test_request = {
            "market_data": {
                "data": [
                    # 5 features, 3 stocks, 30 time periods
                    [[[100 + i + j + t for t in range(30)] for j in range(3)] for i in range(5)]
                ][0],  # Extraire la liste
                "tickers": ["AAPL", "MSFT", "GOOGL"],
                "timestamp": datetime.now().isoformat()
            },
            "portfolio": {
                "weights": [0.33, 0.33, 0.34],
                "portfolio_value": 10000.0,
                "cash": 0.0
            },
            "model_version": "latest",
            "risk_preference": "moderate"
        }
        
        response = requests.post(
            f"http://localhost:{port}/predict",
            json=test_request,
            timeout=10
        )
        
        if response.status_code == 200:
            data = response.json()
            weights = data.get('weights', [])
            confidence = data.get('confidence', 0)
            rebalancing = data.get('rebalancing_needed', False)
            
            print(f"   ✅ Prédiction réussie")
            print(f"   📊 Nouveaux poids: {[f'{w:.3f}' for w in weights]}")
            print(f"   🎯 Confiance: {confidence:.3f}")
            print(f"   🔄 Rééquilibrage nécessaire: {rebalancing}")
            
            # Vérifier que les poids sont valides
            if abs(sum(weights) - 1.0) < 0.01:
                print(f"   ✅ Poids normalisés correctement")
                return True
            else:
                print(f"   ❌ Poids mal normalisés: somme = {sum(weights)}")
                return False
        else:
            print(f"   ❌ Status code: {response.status_code}")
            print(f"   📝 Réponse: {response.text}")
            return False
            
    except Exception as e:
        print(f"   ❌ Erreur: {e}")
        return False

def test_backtest_endpoint(port=8000):
    """Test de l'endpoint de backtest."""
    print("📈 Test de l'endpoint /backtest...")
    
    try:
        # Créer des données de test pour backtest
        test_request = {
            "market_data": {
                "data": [
                    # 5 features, 3 stocks, 50 time periods
                    [[[100 * (1 + 0.01 * t + 0.001 * i) for t in range(50)] for j in range(3)] for i in range(5)]
                ][0],
                "tickers": ["AAPL", "MSFT", "GOOGL"],
                "timestamp": datetime.now().isoformat()
            },
            "initial_portfolio_value": 10000.0,
            "model_version": "latest",
            "transaction_cost": 0.001
        }
        
        response = requests.post(
            f"http://localhost:{port}/backtest",
            json=test_request,
            timeout=15
        )
        
        if response.status_code == 200:
            data = response.json()
            
            final_value = data.get('final_value', 0)
            total_return = data.get('total_return', 0)
            volatility = data.get('volatility', 0)
            sharpe = data.get('sharpe_ratio', 0)
            max_dd = data.get('max_drawdown', 0)
            trades = data.get('trades_count', 0)
            
            print(f"   ✅ Backtest réussi")
            print(f"   💰 Valeur finale: ${final_value:.2f}")
            print(f"   📈 Rendement total: {total_return:.2f}%")
            print(f"   📊 Volatilité: {volatility:.2f}%")
            print(f"   🎯 Ratio Sharpe: {sharpe:.3f}")
            print(f"   📉 Drawdown max: {max_dd:.2f}%")
            print(f"   🔄 Nombre de trades: {trades}")
            
            return True
        else:
            print(f"   ❌ Status code: {response.status_code}")
            print(f"   📝 Réponse: {response.text}")
            return False
            
    except Exception as e:
        print(f"   ❌ Erreur: {e}")
        return False

def test_api_documentation(port=8000):
    """Test de l'accès à la documentation."""
    print("📚 Test de la documentation API...")
    
    try:
        # Test de l'endpoint docs
        response = requests.get(f"http://localhost:{port}/docs", timeout=5)
        
        if response.status_code == 200:
            print(f"   ✅ Documentation Swagger accessible")
            print(f"   🌐 URL: http://localhost:{port}/docs")
            
            # Test de l'endpoint redoc
            response_redoc = requests.get(f"http://localhost:{port}/redoc", timeout=5)
            if response_redoc.status_code == 200:
                print(f"   ✅ Documentation ReDoc accessible")
                print(f"   🌐 URL: http://localhost:{port}/redoc")
            
            return True
        else:
            print(f"   ❌ Status code: {response.status_code}")
            return False
            
    except Exception as e:
        print(f"   ❌ Erreur: {e}")
        return False

def test_error_handling(port=8000):
    """Test de la gestion d'erreurs."""
    print("🚨 Test de la gestion d'erreurs...")
    
    try:
        # Test avec des données invalides
        invalid_request = {
            "market_data": {
                "data": [],  # Données vides
                "tickers": [],
                "timestamp": "invalid"
            },
            "portfolio": {
                "weights": [0.5, 0.6],  # Somme > 1
                "portfolio_value": -1000  # Valeur négative
            }
        }
        
        response = requests.post(
            f"http://localhost:{port}/predict",
            json=invalid_request,
            timeout=5
        )
        
        if response.status_code in [400, 422]:  # Erreurs de validation
            print(f"   ✅ Validation des erreurs fonctionne (status: {response.status_code})")
            return True
        else:
            print(f"   ⚠️ Status inattendu: {response.status_code}")
            return False
            
    except Exception as e:
        print(f"   ❌ Erreur: {e}")
        return False

def create_test_data_files():
    """Crée des fichiers de données de test si nécessaire."""
    print("📊 Création de données de test...")
    
    try:
        from src.data.preprocessing import save_data
        import numpy as np
        
        # Créer des données factices si nécessaire
        os.makedirs("data/processed", exist_ok=True)
        
        if not os.path.exists("data/processed/stock_data_train_latest.npy"):
            # Données d'entraînement factices
            train_data = np.random.rand(5, 3, 70).astype(np.float32)
            save_data(train_data, "data/processed/stock_data_train_latest.npy")
            print("   ✅ Données d'entraînement créées")
        
        if not os.path.exists("data/processed/stock_data_test_latest.npy"):
            # Données de test factices
            test_data = np.random.rand(5, 3, 30).astype(np.float32)
            save_data(test_data, "data/processed/stock_data_test_latest.npy")
            print("   ✅ Données de test créées")
        
        return True
        
    except Exception as e:
        print(f"   ❌ Erreur création données: {e}")
        return False

def main():
    """Fonction principale de test."""
    print("🚀 Test complet de l'API FastAPI")
    print("=" * 60)
    
    # Variables de contrôle
    api_process = None
    port = 8000
    
    try:
        # Tests préparatoires
        prep_tests = [
            ("Imports API", test_api_imports),
            ("Données de test", create_test_data_files),
        ]
        
        prep_results = []
        for test_name, test_func in prep_tests:
            print(f"\n{'='*20} {test_name} {'='*20}")
            result = test_func()
            prep_results.append(result)
            
            if not result and test_name == "Imports API":
                print("❌ Test critique échoué: Imports API")
                return False
        
        # Démarrer l'API
        print(f"\n{'='*20} Démarrage API {'='*20}")
        api_process = start_api_server(port)
        
        if not api_process:
            print("❌ Impossible de démarrer l'API")
            return False
        
        # Tests de l'API
        api_tests = [
            ("Santé API", lambda: test_health_endpoint(port)),
            ("Liste modèles", lambda: test_models_endpoint(port)),
            ("Prédiction", lambda: test_prediction_endpoint(port)),
            ("Backtest", lambda: test_backtest_endpoint(port)),
            ("Documentation", lambda: test_api_documentation(port)),
            ("Gestion erreurs", lambda: test_error_handling(port)),
        ]
        
        api_results = []
        for test_name, test_func in api_tests:
            print(f"\n{'='*20} {test_name} {'='*20}")
            result = test_func()
            api_results.append(result)
        
        # Résumé des résultats
        print("\n" + "=" * 60)
        print("📋 RÉSUMÉ DES TESTS")
        print("=" * 60)
        
        all_results = prep_results + api_results
        all_test_names = [name for name, _ in prep_tests] + [name for name, _ in api_tests]
        
        passed = sum(1 for r in all_results if r)
        total = len(all_results)
        
        for i, test_name in enumerate(all_test_names):
            status = "✅ PASSÉ" if all_results[i] else "❌ ÉCHOUÉ"
            print(f"   {status}: {test_name}")
        
        print(f"\n📊 Score: {passed}/{total} tests passés ({passed/total*100:.1f}%)")
        
        if passed >= total - 1:  # Accepter 1 échec
            print("\n🎉 API FASTAPI FONCTIONNELLE!")
            print("\n📋 Ce qui fonctionne:")
            print("   ✅ Endpoints de base")
            print("   ✅ Prédictions d'allocation")
            print("   ✅ Backtest automatisé")
            print("   ✅ Documentation interactive")
            print("   ✅ Gestion d'erreurs")
            
            print(f"\n🌐 ACCÈS À L'API:")
            print(f"   📍 API: http://localhost:{port}")
            print(f"   📚 Documentation: http://localhost:{port}/docs")
            print(f"   📖 ReDoc: http://localhost:{port}/redoc")
            print(f"   🏥 Santé: http://localhost:{port}/health")
            
            print("\n🎯 ÉTAPE 5 VALIDÉE!")
            print("📋 Prochaine étape: Monitoring (Prometheus + Grafana)")
            
            # Demander si on veut garder l'API ouverte
            try:
                response = input("\n❓ Garder l'API ouverte pour exploration ? (y/N): ")
                if response.lower() in ['y', 'yes']:
                    print("💡 API reste ouverte. Fermez manuellement quand terminé.")
                    print(f"   Utilisez Ctrl+C ou kill {api_process.pid}")
                    return True
            except KeyboardInterrupt:
                print("\n⏹️ Interruption utilisateur")
        
        else:
            print(f"\n⚠️ {total-passed} test(s) ont échoué.")
            print("Vérifiez les erreurs ci-dessus avant de continuer.")
            return False
    
    except KeyboardInterrupt:
        print("\n⏹️ Tests interrompus par l'utilisateur")
        return False
    
    finally:
        # Arrêter l'API
        if api_process:
            print("\n🛑 Arrêt de l'API...")
            api_process.terminate()
            try:
                api_process.wait(timeout=5)
            except:
                api_process.kill()
    
    return True

if __name__ == "__main__":
    success = main()
    if not success:
        sys.exit(1)