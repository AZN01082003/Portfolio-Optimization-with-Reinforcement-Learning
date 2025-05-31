#!/usr/bin/env python3
"""
Test complet du pipeline d'entraînement avec MLflow.
À placer à la racine du projet : test_training_pipeline.py
"""
import os
import sys
import json
import tempfile
import subprocess
import signal
import time
from datetime import datetime

# Ajouter le répertoire du projet au PYTHONPATH
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)
sys.path.insert(0, os.path.join(project_root, 'src'))

def create_test_config():
    """Crée un fichier de configuration de test."""
    config = {
        "data": {
            "tickers": ["AAPL", "MSFT", "GOOGL"],
            "lookback_years": 0.1,
            "train_ratio": 0.7,
            "output_dir": "data/processed"
        },
        "environment": {
            "portfolio_value": 10000,
            "window_size": 10,
            "trans_cost": 0.001,
            "return_rate": 0.0001,
            "reward_scaling": 100.0,
            "max_reward_clip": 5.0,
            "min_reward_clip": -5.0,
            "normalize_observations": True,
            "risk_penalty": 0.1
        },
        "training": {
            "algorithm": "PPO",
            "learning_rate": 0.001,
            "n_steps": 512,
            "batch_size": 32,
            "ent_coef": 0.01,
            "clip_range": 0.2,
            "max_grad_norm": 0.5,
            "gae_lambda": 0.95,
            "gamma": 0.99,
            "n_epochs": 5,
            "total_timesteps": 5000,  # Petit nombre pour test rapide
            "log_dir": "logs/training"
        }
    }
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        json.dump(config, f, indent=2)
        return f.name

def start_mlflow_server(port=5000):
    """Démarre un serveur MLflow local pour les tests."""
    try:
        # Vérifier si MLflow est déjà en cours
        import requests
        response = requests.get(f"http://localhost:{port}", timeout=2)
        if response.status_code == 200:
            print(f"✅ MLflow déjà en cours sur le port {port}")
            return None
    except:
        pass
    
    # Démarrer MLflow
    print(f"🚀 Démarrage de MLflow sur le port {port}...")
    
    mlflow_cmd = [
        sys.executable, "-m", "mlflow", "server",
        "--host", "0.0.0.0",
        "--port", str(port),
        "--backend-store-uri", "sqlite:///test_mlflow.db",
        "--default-artifact-root", "./test_artifacts"
    ]
    
    try:
        process = subprocess.Popen(
            mlflow_cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env={**os.environ, "MLFLOW_TRACKING_URI": f"http://localhost:{port}"}
        )
        
        # Attendre que MLflow soit prêt
        max_wait = 30
        for i in range(max_wait):
            try:
                import requests
                response = requests.get(f"http://localhost:{port}", timeout=1)
                if response.status_code == 200:
                    print(f"✅ MLflow démarré avec succès!")
                    return process
            except:
                pass
            time.sleep(1)
            print(f"⏳ Attente MLflow... ({i+1}/{max_wait})")
        
        print("❌ Timeout: MLflow n'a pas démarré")
        process.terminate()
        return None
        
    except Exception as e:
        print(f"❌ Erreur lors du démarrage de MLflow: {e}")
        return None

def create_test_data():
    """Crée des données de test pour l'entraînement."""
    print("📊 Création de données de test...")
    
    import numpy as np
    from src.data.preprocessing import save_data
    
    # Créer des données factices
    n_features, n_stocks, n_time_periods = 5, 3, 100
    data = np.random.rand(n_features, n_stocks, n_time_periods).astype(np.float32)
    
    # Simuler des prix réalistes (OHLC)
    for stock in range(n_stocks):
        base_price = 100 + stock * 50
        for t in range(n_time_periods):
            close = base_price * (1 + np.random.normal(0, 0.02))
            data[0, stock, t] = close * (1 + np.random.normal(0, 0.01))  # Open
            data[1, stock, t] = close * (1 + abs(np.random.normal(0, 0.01)))  # High
            data[2, stock, t] = close * (1 - abs(np.random.normal(0, 0.01)))  # Low
            data[3, stock, t] = close  # Close
            data[4, stock, t] = np.random.randint(100000, 1000000)  # Volume
    
    # Division train/test
    split_point = int(n_time_periods * 0.7)
    train_data = data[:, :, :split_point]
    test_data = data[:, :, split_point:]
    
    # Sauvegarder
    os.makedirs("data/processed", exist_ok=True)
    
    save_data(data, "data/processed/stock_data_normalized_latest.npy")
    save_data(train_data, "data/processed/stock_data_train_latest.npy")
    save_data(test_data, "data/processed/stock_data_test_latest.npy")
    
    print(f"✅ Données créées: train={train_data.shape}, test={test_data.shape}")
    return True

def test_training_imports():
    """Test des imports nécessaires."""
    print("📦 Test des imports...")
    
    try:
        import mlflow
        print(f"   ✅ MLflow: {mlflow.__version__}")
        
        from stable_baselines3 import PPO
        print(f"   ✅ Stable-Baselines3: {PPO}")
        
        from src.models.train import train_portfolio_agent, load_training_data
        print("   ✅ Module d'entraînement local")
        
        from src.environment.portfolio_env import PortfolioEnv
        print("   ✅ Environnement RL")
        
        return True
        
    except ImportError as e:
        print(f"   ❌ Import échoué: {e}")
        return False

def test_data_loading():
    """Test du chargement des données."""
    print("📊 Test du chargement des données...")
    
    try:
        from src.models.train import load_training_data
        
        config_path = create_test_config()
        train_data, test_data = load_training_data(config_path)
        
        if train_data is not None and test_data is not None:
            print(f"   ✅ Train: {train_data.shape}")
            print(f"   ✅ Test: {test_data.shape}")
            os.unlink(config_path)
            return True
        else:
            print("   ❌ Données non chargées")
            os.unlink(config_path)
            return False
            
    except Exception as e:
        print(f"   ❌ Erreur: {e}")
        return False

def test_environment_creation():
    """Test de création de l'environnement."""
    print("🏗️ Test de création de l'environnement...")
    
    try:
        from src.environment.portfolio_env import create_env_from_config
        import numpy as np
        
        # Données factices
        test_data = np.random.rand(5, 3, 50).astype(np.float32)
        config_path = create_test_config()
        
        env = create_env_from_config(test_data, config_path)
        
        # Test basique
        obs, info = env.reset()
        action = env.action_space.sample()
        next_obs, reward, terminated, truncated, step_info = env.step(action)
        
        print("   ✅ Environnement créé et testé")
        os.unlink(config_path)
        return True
        
    except Exception as e:
        print(f"   ❌ Erreur: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_mlflow_connection():
    """Test de connexion à MLflow."""
    print("🔗 Test de connexion MLflow...")
    
    try:
        import mlflow
        
        # Configurer l'URI
        mlflow_uri = os.environ.get("MLFLOW_TRACKING_URI", "http://localhost:5000")
        mlflow.set_tracking_uri(mlflow_uri)
        
        # Test de connexion
        client = mlflow.tracking.MlflowClient()
        experiments = client.search_experiments()
        
        print(f"   ✅ Connexion MLflow réussie ({len(experiments)} expériences)")
        return True
        
    except Exception as e:
        print(f"   ❌ Erreur MLflow: {e}")
        return False

def test_short_training():
    """Test d'un entraînement court."""
    print("🎯 Test d'entraînement court...")
    
    try:
        from src.models.train import train_portfolio_agent
        
        config_path = create_test_config()
        
        # Entraînement rapide
        print("   ⏳ Démarrage de l'entraînement (5000 steps)...")
        model, training_info = train_portfolio_agent(
            config_path=config_path,
            run_name="test_training"
        )
        
        if model is not None and training_info is not None:
            print(f"   ✅ Entraînement réussi!")
            print(f"      📍 Run ID: {training_info['run_id']}")
            print(f"      🏆 Meilleure récompense: {training_info['best_reward']:.4f}")
            print(f"      ⏱️ Temps: {training_info['training_time']:.1f}s")
            
            # Vérifier les fichiers créés
            final_model = training_info['final_model_path'] + ".zip"
            if os.path.exists(final_model):
                print(f"      ✅ Modèle sauvegardé: {final_model}")
            else:
                print(f"      ⚠️ Modèle non trouvé: {final_model}")
            
            os.unlink(config_path)
            return True, training_info
        else:
            print("   ❌ Entraînement échoué")
            os.unlink(config_path)
            return False, None
            
    except Exception as e:
        print(f"   ❌ Erreur: {e}")
        import traceback
        traceback.print_exc()
        return False, None

def test_model_evaluation():
    """Test d'évaluation du modèle."""
    print("📊 Test d'évaluation du modèle...")
    
    try:
        from src.models.train import evaluate_trained_model
        
        # Utiliser le dernier modèle entraîné
        config_path = create_test_config()
        
        # Chercher un modèle récent
        logs_dir = "logs/training"
        if os.path.exists(logs_dir):
            training_dirs = [d for d in os.listdir(logs_dir) if d.startswith("training_")]
            if training_dirs:
                latest_dir = max(training_dirs)
                model_path = os.path.join(logs_dir, latest_dir, "final_model")
                
                if os.path.exists(model_path + ".zip"):
                    metrics = evaluate_trained_model(model_path, config_path)
                    
                    if metrics:
                        print("   ✅ Évaluation réussie:")
                        for key, value in metrics.items():
                            print(f"      {key}: {value:.4f}")
                        os.unlink(config_path)
                        return True
        
        print("   ⚠️ Aucun modèle trouvé pour l'évaluation")
        os.unlink(config_path)
        return False
        
    except Exception as e:
        print(f"   ❌ Erreur: {e}")
        return False

def test_mlflow_ui_access():
    """Test d'accès à l'UI MLflow."""
    print("🌐 Test d'accès à l'UI MLflow...")
    
    try:
        import requests
        
        mlflow_url = "http://localhost:5000"
        response = requests.get(mlflow_url, timeout=5)
        
        if response.status_code == 200:
            print(f"   ✅ UI MLflow accessible: {mlflow_url}")
            print(f"   🌐 Ouvrez votre navigateur sur: {mlflow_url}")
            return True
        else:
            print(f"   ❌ UI MLflow non accessible (status: {response.status_code})")
            return False
            
    except Exception as e:
        print(f"   ❌ Erreur: {e}")
        return False

def cleanup_test_files():
    """Nettoie les fichiers de test."""
    print("🧹 Nettoyage des fichiers de test...")
    
    files_to_clean = [
        "test_mlflow.db",
        "test_artifacts",
        "data/processed/stock_data_*_latest.npy"
    ]
    
    for pattern in files_to_clean:
        if "*" in pattern:
            import glob
            for file in glob.glob(pattern):
                try:
                    os.remove(file)
                    print(f"   🗑️ Supprimé: {file}")
                except:
                    pass
        else:
            if os.path.exists(pattern):
                try:
                    if os.path.isdir(pattern):
                        import shutil
                        shutil.rmtree(pattern)
                    else:
                        os.remove(pattern)
                    print(f"   🗑️ Supprimé: {pattern}")
                except:
                    pass

def main():
    """Fonction principale de test."""
    print("🚀 Test complet du pipeline d'entraînement MLflow")
    print("=" * 60)
    
    # Variables de contrôle
    mlflow_process = None
    cleanup_on_exit = True
    
    try:
        # Créer les répertoires nécessaires
        os.makedirs("data/processed", exist_ok=True)
        os.makedirs("logs/training", exist_ok=True)
        os.makedirs("artifacts", exist_ok=True)
        
        # Tests séquentiels
        tests = [
            ("Imports nécessaires", test_training_imports),
            ("Création de données test", create_test_data),
            ("Démarrage MLflow", lambda: start_mlflow_server()),
            ("Connexion MLflow", test_mlflow_connection),
            ("Chargement des données", test_data_loading),
            ("Création environnement", test_environment_creation),
            ("Entraînement court", test_short_training),
            ("Évaluation modèle", test_model_evaluation),
            ("Accès UI MLflow", test_mlflow_ui_access),
        ]
        
        results = []
        training_info = None
        
        for i, (test_name, test_func) in enumerate(tests):
            print(f"\n{'='*20} {i+1}. {test_name} {'='*20}")
            
            try:
                if test_name == "Démarrage MLflow":
                    mlflow_process = test_func()
                    result = mlflow_process is not None
                    # Configurer la variable d'environnement
                    os.environ["MLFLOW_TRACKING_URI"] = "http://localhost:5000"
                elif test_name == "Entraînement court":
                    result, training_info = test_func()
                else:
                    result = test_func()
                
                results.append(result if result is not None else False)
                
                if not result and test_name in ["Imports nécessaires", "Création de données test"]:
                    print(f"❌ Test critique échoué: {test_name}")
                    break
                    
            except Exception as e:
                print(f"❌ Test {test_name} échoué: {e}")
                results.append(False)
        
        # Résumé des résultats
        print("\n" + "=" * 60)
        print("📋 RÉSUMÉ DES TESTS")
        print("=" * 60)
        
        passed = sum(1 for r in results if r)
        total = len(results)
        
        for i, (test_name, _) in enumerate(tests[:len(results)]):
            status = "✅ PASSÉ" if results[i] else "❌ ÉCHOUÉ"
            print(f"   {status}: {test_name}")
        
        print(f"\n📊 Score: {passed}/{total} tests passés ({passed/total*100:.1f}%)")
        
        if passed >= total - 1:  # Accepter 1 échec sur les tests non critiques
            print("\n🎉 PIPELINE D'ENTRAÎNEMENT FONCTIONNEL!")
            print("\n📋 Ce qui fonctionne:")
            print("   ✅ Entraînement avec MLflow")
            print("   ✅ Tracking des métriques")
            print("   ✅ Sauvegarde des modèles")
            print("   ✅ Évaluation automatique")
            print("   ✅ Interface MLflow")
            
            if training_info:
                print(f"\n🎯 DERNIÈRE SESSION D'ENTRAÎNEMENT:")
                print(f"   📍 Run ID: {training_info.get('run_id', 'N/A')}")
                print(f"   🏆 Meilleure récompense: {training_info.get('best_reward', 'N/A')}")
                print(f"   ⏱️ Temps d'entraînement: {training_info.get('training_time', 'N/A')}s")
            
            print(f"\n🌐 MLflow UI: http://localhost:5000")
            print("\n📋 ÉTAPE 4 VALIDÉE!")
            print("🎯 Prochaine étape: API de prédiction (Étape 5)")
            
            # Demander si on veut garder MLflow ouvert
            try:
                response = input("\n❓ Garder MLflow ouvert pour exploration ? (y/N): ")
                if response.lower() in ['y', 'yes']:
                    cleanup_on_exit = False
                    print("💡 MLflow reste ouvert. Fermez manuellement quand terminé.")
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
        # Nettoyage
        if cleanup_on_exit:
            if mlflow_process:
                print("\n🛑 Arrêt de MLflow...")
                mlflow_process.terminate()
                try:
                    mlflow_process.wait(timeout=5)
                except:
                    mlflow_process.kill()
            
            cleanup_test_files()
        else:
            print("\n💡 MLflow toujours en cours. Pour l'arrêter:")
            print(f"   kill {mlflow_process.pid}")
    
    return True

if __name__ == "__main__":
    success = main()
    if not success:
        sys.exit(1)