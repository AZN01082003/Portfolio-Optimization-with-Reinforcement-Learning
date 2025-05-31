#!/usr/bin/env python3
"""
Test final complet de l'API Portfolio RL avec monitoring.
Script pour valider que tout fonctionne ensemble.
"""
import os
import sys
import time
import json
import requests
import subprocess
import numpy as np
from datetime import datetime
from pathlib import Path

# Ajouter le répertoire du projet au PYTHONPATH
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

def install_dependencies():
    """Installe les dépendances manquantes."""
    print("📦 Installation des dépendances...")
    
    dependencies = [
        "fastapi>=0.100.0",
        "uvicorn[standard]>=0.20.0",
        "prometheus-client>=0.15.0",
        "requests>=2.25.1",
        "numpy>=1.20.0"
    ]
    
    for dep in dependencies:
        try:
            print(f"   Installing {dep}...")
            subprocess.check_call([
                sys.executable, "-m", "pip", "install", dep
            ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            print(f"   ✅ {dep}")
        except subprocess.CalledProcessError:
            print(f"   ⚠️ Échec: {dep}")
    
    print("✅ Dépendances installées")

def create_required_directories():
    """Crée les répertoires requis."""
    print("📁 Création des répertoires...")
    
    directories = [
        "src",
        "src/api",
        "src/monitoring", 
        "src/models",
        "config",
        "models",
        "logs/api",
        "data/processed"
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
        print(f"   ✅ {directory}")

def create_init_files():
    """Crée les fichiers __init__.py manquants."""
    print("📝 Création des fichiers __init__.py...")
    
    init_dirs = [
        "src",
        "src/api", 
        "src/monitoring",
        "src/models"
    ]
    
    for directory in init_dirs:
        init_file = Path(directory) / "__init__.py"
        if not init_file.exists():
            init_file.write_text('"""Package initialization."""\n')
            print(f"   ✅ {init_file}")

def test_imports():
    """Test des imports nécessaires."""
    print("🔍 Test des imports...")
    
    imports_to_test = [
        ("fastapi", "FastAPI"),
        ("uvicorn", "Uvicorn"),
        ("numpy", "NumPy"),
        ("requests", "Requests")
    ]
    
    missing = []
    
    for module, name in imports_to_test:
        try:
            __import__(module)
            print(f"   ✅ {name}")
        except ImportError:
            print(f"   ❌ {name} manquant")
            missing.append(module)
    
    return len(missing) == 0

def start_api_server():
    """Démarre le serveur API."""
    print("🚀 Démarrage du serveur API...")
    
    try:
        # Vérifier que le fichier main.py existe
        main_file = Path("src/main.py")
        if not main_file.exists():
            print("   ❌ Fichier src/main.py non trouvé")
            return None
        
        # Démarrer l'API
        process = subprocess.Popen([
            sys.executable, "-m", "uvicorn",
            "src.main:app",
            "--host", "0.0.0.0",
            "--port", "8000",
            "--reload"
        ], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        
        # Attendre que l'API soit prête
        for i in range(30):
            try:
                response = requests.get("http://localhost:8000/health", timeout=2)
                if response.status_code == 200:
                    print("   ✅ API démarrée avec succès")
                    return process
            except:
                pass
            time.sleep(2)
            print(f"   ⏳ Attente... ({i+1}/30)")
        
        print("   ❌ Timeout - API non démarrée")
        process.terminate()
        return None
        
    except Exception as e:
        print(f"   ❌ Erreur: {e}")
        return None

def test_api_endpoints():
    """Test des endpoints de l'API."""
    print("🧪 Test des endpoints API...")
    
    base_url = "http://localhost:8000"
    results = {}
    
    # Test 1: Health check
    print("   Test 1: Health check...")
    try:
        response = requests.get(f"{base_url}/health", timeout=10)
        if response.status_code == 200:
            data = response.json()
            print(f"      ✅ Status: {data.get('status')}")
            results['health'] = True
        else:
            print(f"      ❌ Status: {response.status_code}")
            results['health'] = False
    except Exception as e:
        print(f"      ❌ Erreur: {e}")
        results['health'] = False
    
    # Test 2: Root endpoint
    print("   Test 2: Root endpoint...")
    try:
        response = requests.get(f"{base_url}/", timeout=10)
        if response.status_code == 200:
            print("      ✅ Root OK")
            results['root'] = True
        else:
            print(f"      ❌ Status: {response.status_code}")
            results['root'] = False
    except Exception as e:
        print(f"      ❌ Erreur: {e}")
        results['root'] = False
    
    # Test 3: Models list
    print("   Test 3: Liste des modèles...")
    try:
        response = requests.get(f"{base_url}/models", timeout=10)
        if response.status_code == 200:
            data = response.json()
            models = data.get('models', [])
            print(f"      ✅ {len(models)} modèle(s) trouvé(s)")
            results['models'] = True
        else:
            print(f"      ❌ Status: {response.status_code}")
            results['models'] = False
    except Exception as e:
        print(f"      ❌ Erreur: {e}")
        results['models'] = False
    
    # Test 4: Test prediction endpoint
    print("   Test 4: Endpoint de test...")
    try:
        response = requests.post(f"{base_url}/test/prediction", timeout=15)
        if response.status_code == 200:
            data = response.json()
            print(f"      ✅ Test: {data.get('status')}")
            results['test'] = True
        else:
            print(f"      ❌ Status: {response.status_code}")
            results['test'] = False
    except Exception as e:
        print(f"      ❌ Erreur: {e}")
        results['test'] = False
    
    # Test 5: Metrics endpoint
    print("   Test 5: Métriques...")
    try:
        response = requests.get(f"{base_url}/metrics", timeout=10)
        if response.status_code == 200:
            metrics_text = response.text
            if "portfolio_api_requests_total" in metrics_text:
                print("      ✅ Métriques Prometheus disponibles")
                results['metrics'] = True
            else:
                print("      ⚠️ Métriques de base seulement")
                results['metrics'] = False
        else:
            print(f"      ❌ Status: {response.status_code}")
            results['metrics'] = False
    except Exception as e:
        print(f"      ❌ Erreur: {e}")
        results['metrics'] = False
    
    # Test 6: Prediction complète
    print("   Test 6: Prédiction complète...")
    try:
        test_data = {
            "portfolio_id": "test_portfolio_final",
            "market_data": [
                # 5 features x 3 stocks x 30 periods
                [
                    [100 + i for i in range(30)],  # Stock 1
                    [200 + i for i in range(30)],  # Stock 2  
                    [150 + i for i in range(30)]   # Stock 3
                ] for _ in range(5)
            ],
            "current_weights": [0.33, 0.33, 0.34],
            "portfolio_value": 100000.0,
            "risk_tolerance": 0.5
        }
        
        response = requests.post(
            f"{base_url}/predict",
            json=test_data,
            timeout=20
        )
        
        if response.status_code == 200:
            data = response.json()
            weights = data.get('recommended_weights', [])
            if len(weights) == 3 and abs(sum(weights) - 1.0) < 0.01:
                print(f"      ✅ Prédiction: {[f'{w:.3f}' for w in weights]}")
                print(f"      🎯 Confiance: {data.get('confidence', 0):.2%}")
                results['prediction'] = True
            else:
                print(f"      ❌ Poids invalides: {weights}")
                results['prediction'] = False
        else:
            print(f"      ❌ Status: {response.status_code}")
            print(f"      📝 Réponse: {response.text[:200]}...")
            results['prediction'] = False
            
    except Exception as e:
        print(f"      ❌ Erreur: {e}")
        results['prediction'] = False
    
    return results

def generate_load_test():
    """Génère de la charge pour tester les métriques."""
    print("📈 Génération de charge pour test des métriques...")
    
    base_url = "http://localhost:8000"
    
    # Générer 20 requêtes diverses
    endpoints = [
        ("/", "GET"),
        ("/health", "GET"),
        ("/models", "GET"),
        ("/test/prediction", "POST"),
        ("/metrics", "GET")
    ]
    
    for i in range(20):
        endpoint, method = endpoints[i % len(endpoints)]
        try:
            if method == "GET":
                requests.get(f"{base_url}{endpoint}", timeout=5)
            else:
                requests.post(f"{base_url}{endpoint}", timeout=5)
            
            if i % 5 == 0:
                print(f"   📊 {i+1}/20 requêtes envoyées...")
                
        except:
            pass
        
        time.sleep(0.1)
    
    print("   ✅ Charge générée")

def test_metrics_collection():
    """Test la collecte de métriques après génération de charge."""
    print("📊 Test de collecte de métriques...")
    
    try:
        response = requests.get("http://localhost:8000/metrics", timeout=10)
        
        if response.status_code == 200:
            metrics_text = response.text
            
            # Chercher des métriques spécifiques
            expected_metrics = [
                "portfolio_api_requests_total",
                "portfolio_api_request_duration_seconds",
                "portfolio_model_predictions_total"
            ]
            
            found_metrics = []
            for metric in expected_metrics:
                if metric in metrics_text:
                    found_metrics.append(metric)
                    print(f"   ✅ {metric}")
                else:
                    print(f"   ❌ {metric} manquante")
            
            # Analyser quelques valeurs
            lines = metrics_text.split('\n')
            for line in lines:
                if 'portfolio_api_requests_total{' in line and not line.startswith('#'):
                    print(f"   📈 Exemple: {line.strip()}")
                    break
            
            return len(found_metrics) >= 2
        else:
            print(f"   ❌ Erreur métriques: {response.status_code}")
            return False
            
    except Exception as e:
        print(f"   ❌ Erreur: {e}")
        return False

def check_documentation():
    """Vérifie l'accès à la documentation."""
    print("📚 Test de la documentation...")
    
    try:
        # Test Swagger UI
        response = requests.get("http://localhost:8000/docs", timeout=10)
        if response.status_code == 200:
            print("   ✅ Documentation Swagger accessible")
            swagger_ok = True
        else:
            print(f"   ❌ Swagger: {response.status_code}")
            swagger_ok = False
        
        # Test ReDoc
        response = requests.get("http://localhost:8000/redoc", timeout=10)
        if response.status_code == 200:
            print("   ✅ Documentation ReDoc accessible")
            redoc_ok = True
        else:
            print(f"   ❌ ReDoc: {response.status_code}")
            redoc_ok = False
        
        return swagger_ok or redoc_ok
        
    except Exception as e:
        print(f"   ❌ Erreur: {e}")
        return False

def display_final_results(api_results, metrics_ok, docs_ok):
    """Affiche les résultats finaux."""
    print("\n" + "="*60)
    print("📋 RÉSULTATS FINAUX")
    print("="*60)
    
    # Résultats des endpoints
    print("\n🔗 ENDPOINTS API:")
    for endpoint, result in api_results.items():
        status = "✅ OK" if result else "❌ FAIL"
        print(f"   {status:<8} {endpoint}")
    
    # Métriques
    metrics_status = "✅ OK" if metrics_ok else "❌ FAIL"
    print(f"\n📊 MÉTRIQUES PROMETHEUS: {metrics_status}")
    
    # Documentation
    docs_status = "✅ OK" if docs_ok else "❌ FAIL"
    print(f"📚 DOCUMENTATION: {docs_status}")
    
    # Score global
    total_tests = len(api_results) + 2  # +2 pour métriques et docs
    passed_tests = sum(api_results.values()) + int(metrics_ok) + int(docs_ok)
    
    print(f"\n📊 SCORE GLOBAL: {passed_tests}/{total_tests} ({passed_tests/total_tests*100:.1f}%)")
    
    if passed_tests == total_tests:
        print("\n🎉 TOUS LES TESTS SONT PASSÉS!")
        print("💡 Votre API Portfolio RL est complètement opérationnelle!")
        
        print(f"\n🌐 ACCÈS:")
        print(f"   📍 API: http://localhost:8000")
        print(f"   📚 Documentation: http://localhost:8000/docs")
        print(f"   📊 Métriques: http://localhost:8000/metrics")
        
        print(f"\n🧪 COMMANDES DE TEST:")
        print(f"   python test_prediction.py")
        print(f"   python test_api_complete.py")
        
    elif passed_tests >= total_tests * 0.8:
        print("\n⚠️ LA PLUPART DES TESTS SONT PASSÉS")
        print("💡 Votre API fonctionne avec quelques limitations.")
        
    else:
        print("\n❌ PLUSIEURS TESTS ONT ÉCHOUÉ")
        print("💡 Vérifiez les erreurs ci-dessus.")
    
    return passed_tests == total_tests

def main():
    """Fonction principale."""
    print("🎯 TEST FINAL - API Portfolio RL avec Monitoring")
    print("="*60)
    print(f"📅 {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    api_process = None
    
    try:
        # Étape 1: Préparation
        print(f"\n{'='*20} PRÉPARATION {'='*20}")
        install_dependencies()
        create_required_directories()
        create_init_files()
        
        # Étape 2: Test des imports
        print(f"\n{'='*20} IMPORTS {'='*20}")
        if not test_imports():
            print("❌ Imports manquants - installation requise")
            return False
        
        # Étape 3: Démarrage de l'API
        print(f"\n{'='*20} DÉMARRAGE API {'='*20}")
        api_process = start_api_server()
        
        if not api_process:
            print("❌ Impossible de démarrer l'API")
            return False
        
        # Étape 4: Tests des endpoints
        print(f"\n{'='*20} TESTS ENDPOINTS {'='*20}")
        api_results = test_api_endpoints()
        
        # Étape 5: Génération de charge et test métriques
        print(f"\n{'='*20} TESTS MÉTRIQUES {'='*20}")
        generate_load_test()
        metrics_ok = test_metrics_collection()
        
        # Étape 6: Test documentation
        print(f"\n{'='*20} DOCUMENTATION {'='*20}")
        docs_ok = check_documentation()
        
        # Étape 7: Résultats finaux
        success = display_final_results(api_results, metrics_ok, docs_ok)
        
        # Demander si on garde l'API ouverte
        if success:
            try:
                response = input("\n❓ Garder l'API ouverte pour exploration ? (y/N): ")
                if response.lower() in ['y', 'yes', 'oui']:
                    print("💡 API reste ouverte. Fermez avec Ctrl+C quand terminé.")
                    print(f"   Ou utilisez: kill {api_process.pid}")
                    return True
            except KeyboardInterrupt:
                print("\n⏹️ Interruption utilisateur")
        
        return success
        
    except KeyboardInterrupt:
        print("\n⏹️ Test interrompu par l'utilisateur")
        return False
        
    except Exception as e:
        print(f"\n💥 Erreur inattendue: {e}")
        import traceback
        traceback.print_exc()
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

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)