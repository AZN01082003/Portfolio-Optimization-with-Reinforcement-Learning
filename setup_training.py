#!/usr/bin/env python3
"""
Script de setup pour préparer l'environnement d'entraînement.
À exécuter à la racine du projet : python setup_training.py
"""
import os
import sys
import subprocess
import platform

def check_python_version():
    """Vérifie la version de Python."""
    version = sys.version_info
    print(f"🐍 Python {version.major}.{version.minor}.{version.micro}")
    
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print("❌ Python 3.8+ requis")
        return False
    
    print("✅ Version Python compatible")
    return True

def install_packages():
    """Installe les packages nécessaires."""
    packages = [
        "mlflow>=2.0.0",
        "stable-baselines3[extra]>=2.0.0", 
        "gymnasium>=0.28.1",
        "torch",
        "matplotlib>=3.5.0",
        "seaborn>=0.11.2",
        "pandas>=1.3.0",
        "numpy>=1.20.0",
        "requests>=2.25.1"
    ]
    
    print("📦 Installation des packages nécessaires...")
    
    for package in packages:
        try:
            print(f"   Installing {package}...")
            subprocess.check_call([
                sys.executable, "-m", "pip", "install", package
            ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            print(f"   ✅ {package}")
        except subprocess.CalledProcessError:
            print(f"   ❌ Échec: {package}")
            return False
    
    print("✅ Tous les packages installés")
    return True

def create_directories():
    """Crée les répertoires nécessaires."""
    directories = [
        "data/raw",
        "data/processed", 
        "logs/training",
        "logs/evaluation",
        "models",
        "artifacts",
        "mlruns"
    ]
    
    print("📁 Création des répertoires...")
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        print(f"   ✅ {directory}")
    
    return True

def test_imports():
    """Teste les imports critiques."""
    print("🧪 Test des imports...")
    
    critical_imports = [
        ("mlflow", "MLflow"),
        ("stable_baselines3", "Stable-Baselines3"),
        ("gymnasium", "Gymnasium"),
        ("torch", "PyTorch"),
        ("numpy", "NumPy"),
        ("pandas", "Pandas")
    ]
    
    failed_imports = []
    
    for module, name in critical_imports:
        try:
            __import__(module)
            print(f"   ✅ {name}")
        except ImportError as e:
            print(f"   ❌ {name}: {e}")
            failed_imports.append(name)
    
    if failed_imports:
        print(f"❌ Imports échoués: {', '.join(failed_imports)}")
        return False
    
    print("✅ Tous les imports fonctionnent")
    return True

def check_gpu_support():
    """Vérifie le support GPU (optionnel)."""
    print("🖥️ Vérification du support GPU...")
    
    try:
        import torch
        if torch.cuda.is_available():
            gpu_count = torch.cuda.device_count()
            gpu_name = torch.cuda.get_device_name(0)
            print(f"   ✅ GPU détecté: {gpu_name} (x{gpu_count})")
            return True
        else:
            print("   ℹ️ Pas de GPU CUDA détecté (CPU seulement)")
            return False
    except ImportError:
        print("   ⚠️ PyTorch non installé")
        return False

def create_example_config():
    """Crée un fichier de configuration d'exemple si nécessaire."""
    config_file = "config/default.json"
    
    if os.path.exists(config_file):
        print(f"✅ Configuration existante: {config_file}")
        return True
    
    print("📝 Création d'une configuration d'exemple...")
    
    example_config = {
        "data": {
            "tickers": ["AAPL", "MSFT", "GOOGL"],
            "lookback_years": 2,
            "train_ratio": 0.7,
            "output_dir": "data/processed"
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
            "n_steps": 1024,
            "batch_size": 64,
            "ent_coef": 0.01,
            "clip_range": 0.2,
            "max_grad_norm": 0.5,
            "gae_lambda": 0.95,
            "gamma": 0.99,
            "n_epochs": 10,
            "total_timesteps": 50000,
            "log_dir": "logs/training"
        },
        "evaluation": {
            "results_dir": "logs/evaluation"
        }
    }
    
    os.makedirs("config", exist_ok=True)
    
    import json
    with open(config_file, 'w') as f:
        json.dump(example_config, f, indent=2)
    
    print(f"✅ Configuration créée: {config_file}")
    return True

def run_basic_test():
    """Exécute un test basique du système."""
    print("🧪 Test basique du système...")
    
    try:
        # Test MLflow
        import mlflow
        mlflow.set_tracking_uri("sqlite:///test.db")
        
        with mlflow.start_run(run_name="setup_test"):
            mlflow.log_param("test", "setup")
            mlflow.log_metric("test_metric", 1.0)
        
        print("   ✅ MLflow fonctionne")
        
        # Test Stable-Baselines3
        from stable_baselines3 import PPO
        from stable_baselines3.common.env_util import make_vec_env
        
        # Créer un environnement simple pour test
        env = make_vec_env("CartPole-v1", n_envs=1)
        model = PPO("MlpPolicy", env, verbose=0)
        
        print("   ✅ Stable-Baselines3 fonctionne")
        
        # Nettoyer
        if os.path.exists("test.db"):
            os.remove("test.db")
        
        print("✅ Test basique réussi")
        return True
        
    except Exception as e:
        print(f"   ❌ Erreur: {e}")
        return False

def display_next_steps():
    """Affiche les prochaines étapes."""
    print("\n" + "="*50)
    print("🎯 SETUP TERMINÉ!")
    print("="*50)
    
    print("\n📋 PROCHAINES ÉTAPES:")
    print("1. 🔧 Corriger les imports:")
    print("   python fix_imports.py")
    
    print("\n2. 📊 Préparer les données (si pas déjà fait):")
    print("   python src/data/ingestion.py")
    print("   python src/data/preprocessing.py")
    
    print("\n3. 🧪 Tester le pipeline d'entraînement:")
    print("   python test_training_pipeline.py")
    
    print("\n4. 🚀 Lancer un entraînement complet:")
    print("   python src/models/train.py")
    
    print("\n5. 🌐 Accéder à MLflow UI:")
    print("   mlflow server --host 0.0.0.0 --port 5000")
    print("   Puis: http://localhost:5000")
    
    print("\n💡 CONSEILS:")
    print("   • Utilisez 'python -m pip install --upgrade <package>' pour mettre à jour")
    print("   • Configurez votre IDE pour utiliser le bon environnement Python")
    print("   • Consultez les logs dans le dossier logs/")

def main():
    """Fonction principale de setup."""
    print("🚀 Setup de l'environnement d'entraînement MLOps")
    print("="*50)
    print(f"🖥️ Système: {platform.system()} {platform.release()}")
    
    steps = [
        ("Vérification Python", check_python_version),
        ("Installation packages", install_packages), 
        ("Création répertoires", create_directories),
        ("Test imports", test_imports),
        ("Support GPU", check_gpu_support),
        ("Configuration exemple", create_example_config),
        ("Test basique", run_basic_test),
    ]
    
    results = []
    
    for step_name, step_func in steps:
        print(f"\n{'='*20} {step_name} {'='*20}")
        
        try:
            result = step_func()
            results.append(result)
            
            if not result and step_name in ["Vérification Python", "Installation packages", "Test imports"]:
                print(f"❌ Étape critique échouée: {step_name}")
                print("🛑 Setup interrompu")
                return False
                
        except Exception as e:
            print(f"❌ Erreur lors de {step_name}: {e}")
            results.append(False)
    
    # Résumé
    passed = sum(1 for r in results if r)
    total = len(results)
    
    print(f"\n📊 Résumé: {passed}/{total} étapes réussies")
    
    if passed >= total - 2:  # Accepter 2 échecs non critiques
        display_next_steps()
        return True
    else:
        print(f"❌ Trop d'échecs ({total-passed}). Vérifiez les erreurs ci-dessus.")
        return False

if __name__ == "__main__":
    success = main()
    if not success:
        sys.exit(1)