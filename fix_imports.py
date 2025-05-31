#!/usr/bin/env python3
"""
Script pour corriger automatiquement les problèmes d'imports dans le projet.
À exécuter à la racine du projet : python fix_imports.py
"""
import os
import re

def fix_file_imports(filepath, fixes):
    """Applique les corrections d'imports à un fichier."""
    if not os.path.exists(filepath):
        print(f"⚠️ Fichier non trouvé: {filepath}")
        return False
    
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
        
        original_content = content
        
        for old_import, new_import in fixes:
            content = content.replace(old_import, new_import)
        
        if content != original_content:
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(content)
            print(f"✅ Corrigé: {filepath}")
            return True
        else:
            print(f"📝 Déjà correct: {filepath}")
            return False
            
    except Exception as e:
        print(f"❌ Erreur pour {filepath}: {e}")
        return False

def fix_portfolio_env_imports():
    """Corrige les imports dans portfolio_env.py."""
    files_to_fix = [
        "tests/portfolio_env.py",
        "tests/test_environment.py"
    ]
    
    fixes = [
        ("from portfolio_env import", "from src.environment.portfolio_env import"),
        ("import portfolio_env", "import src.environment.portfolio_env as portfolio_env"),
    ]
    
    for filepath in files_to_fix:
        fix_file_imports(filepath, fixes)

def fix_train_imports():
    """Corrige les imports dans les fichiers d'entraînement."""
    files_to_fix = [
        "tests/train.py",
        "dags/retrain_model_dag.py"
    ]
    
    fixes = [
        ("from portfolio_env import", "from src.environment.portfolio_env import"),
        ("from models.training_pipeline import", "from src.models.training_pipeline import"),
        ("from src.environment.portfolio_env import create_env_from_config", 
         "from src.environment.portfolio_env import create_env_from_config"),
    ]
    
    for filepath in files_to_fix:
        fix_file_imports(filepath, fixes)

def fix_preprocessing_imports():
    """Corrige les imports dans preprocessing."""
    files_to_fix = [
        "tests/preprocessing.py"
    ]
    
    fixes = [
        ("from ingestion import", "from src.data.ingestion import"),
    ]
    
    for filepath in files_to_fix:
        fix_file_imports(filepath, fixes)

def create_missing_init_files():
    """Crée les fichiers __init__.py manquants."""
    init_dirs = [
        "src",
        "src/data", 
        "src/models",
        "src/environment",
        "src/utils",
        "src/api",
        "tests"
    ]
    
    for dir_path in init_dirs:
        init_file = os.path.join(dir_path, "__init__.py")
        if not os.path.exists(init_file):
            os.makedirs(dir_path, exist_ok=True)
            with open(init_file, 'w') as f:
                f.write('"""Package initialization."""\n')
            print(f"✅ Créé: {init_file}")
        else:
            print(f"📝 Existe déjà: {init_file}")

def fix_dags_imports():
    """Corrige les imports dans les DAGs Airflow."""
    files_to_fix = [
        "dags/data_ingestion_dag.py",
        "dags/data_preprocessing_dag.py", 
        "dags/retrain_model_dag.py"
    ]
    
    fixes = [
        ("from models.training_pipeline import", "from src.models.training_pipeline import"),
        ("from src.data.preprocessing import main as preprocess_main", 
         "from src.data.preprocessing import main as preprocess_main"),
        ("from src.models.evaluate import evaluate_portfolio_model",
         "from src.models.evaluate import evaluate_portfolio_model"),
    ]
    
    for filepath in files_to_fix:
        fix_file_imports(filepath, fixes)

def fix_config_references():
    """Corrige les références aux fichiers de configuration."""
    files_to_fix = [
        "src/models/train.py",
        "src/data/preprocessing.py",
        "src/data/ingestion.py",
        "tests/train.py",
        "tests/preprocessing.py"
    ]
    
    # Ajouter un chemin relatif robuste pour la config
    config_fix = '''
def get_config_path(config_path="config/default.json"):
    """Obtient le chemin absolu vers le fichier de configuration."""
    if os.path.isabs(config_path):
        return config_path
    
    # Chercher depuis la racine du projet
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = current_dir
    
    # Remonter jusqu'à trouver le répertoire config
    for _ in range(5):  # Maximum 5 niveaux
        config_file = os.path.join(project_root, config_path)
        if os.path.exists(config_file):
            return config_file
        project_root = os.path.dirname(project_root)
    
    # Fallback: utiliser le chemin relatif
    return config_path
'''
    
    for filepath in files_to_fix:
        if os.path.exists(filepath):
            try:
                with open(filepath, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                # Ajouter la fonction helper si elle n'existe pas
                if "get_config_path" not in content and "load_config" in content:
                    # Insérer après les imports
                    import_end = content.find('logger = logging.getLogger(__name__)')
                    if import_end != -1:
                        insert_pos = content.find('\n', import_end) + 1
                        content = content[:insert_pos] + config_fix + content[insert_pos:]
                        
                        with open(filepath, 'w', encoding='utf-8') as f:
                            f.write(content)
                        print(f"✅ Ajouté get_config_path à: {filepath}")
            except Exception as e:
                print(f"❌ Erreur pour {filepath}: {e}")

def create_requirements_dev():
    """Crée un fichier requirements-dev.txt pour le développement."""
    dev_requirements = """# Dépendances de développement
pytest>=6.2.5
pytest-cov>=3.0.0
black>=21.6b0
isort>=5.9.2
flake8>=4.0.0
mypy>=0.910

# MLOps et monitoring  
mlflow>=2.0.0
prometheus-client>=0.15.0

# RL et ML
stable-baselines3>=2.0.0
gymnasium>=0.28.1
torch>=1.12.0

# Data et viz
pandas>=1.3.0
numpy>=1.20.0
matplotlib>=3.5.0
seaborn>=0.11.2

# API
fastapi>=0.68.0
uvicorn>=0.15.0
pydantic>=1.8.0

# Infrastructure
docker>=5.0.0
requests>=2.25.1
"""
    
    with open("requirements-dev.txt", "w") as f:
        f.write(dev_requirements)
    print("✅ Créé: requirements-dev.txt")

def main():
    """Fonction principale de correction."""
    print("🔧 Correction automatique des imports du projet")
    print("=" * 50)
    
    corrections = [
        ("Fichiers __init__.py", create_missing_init_files),
        ("Imports portfolio_env", fix_portfolio_env_imports),
        ("Imports train", fix_train_imports),
        ("Imports preprocessing", fix_preprocessing_imports),
        ("Imports DAGs", fix_dags_imports),
        ("Références config", fix_config_references),
        ("Requirements dev", create_requirements_dev),
    ]
    
    for name, fix_func in corrections:
        print(f"\n📝 {name}:")
        try:
            fix_func()
        except Exception as e:
            print(f"❌ Erreur lors de {name}: {e}")
    
    print("\n" + "=" * 50)
    print("✅ Corrections terminées!")
    print("\n📋 Actions effectuées:")
    print("   ✅ Fichiers __init__.py créés")
    print("   ✅ Imports corrigés")
    print("   ✅ Références config réparées")
    print("   ✅ Requirements dev ajoutés")
    
    print("\n💡 Prochaines étapes:")
    print("   1. Vérifiez que les corrections sont correctes")
    print("   2. Lancez: python test_training_pipeline.py")
    print("   3. Si tout fonctionne, commitez les changements")

if __name__ == "__main__":
    main()