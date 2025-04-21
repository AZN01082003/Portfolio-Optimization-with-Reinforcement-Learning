import os

# Structure du projet
project_structure = {
    ".github/": {},
    "data/": {
        "raw/": {},
        "processed/": {}
    },
    "logs/": {},
    "models/": {},
    "notebooks/": {},
    "src/": {
        "data/": {
            "__init__.py": "",
            "ingestion.py": "# Récupération des données (basé sur votre data_preprocessing.py)",
            "preprocessing.py": "# Prétraitement des données"
        },
        "environment/": {
            "__init__.py": "",
            "portfolio_env.py": "# Environnement RL (basé sur votre new_env.py)"
        },
        "models/": {
            "__init__.py": "",
            "agent.py": "# Définition de l'agent",
            "train.py": "# Entraînement (basé sur votre train.py)",
            "evaluate.py": "# Évaluation (basé sur votre testing.py)"
        },
        "api/": {
            "__init__.py": "",
            "main.py": "# API pour servir les prédictions"
        },
        "utils/": {
            "__init__.py": "",
            "visualization.py": "# Visualisations"
        }
    },
    "tests/": {},
    "config/": {
        "default.yaml": "# Configuration par défaut\ndata:\n  tickers: []\n  lookback_years: 2"
    },
    ".gitignore": "# Fichiers à ignorer par Git\n__pycache__/\ndata/\nlogs/\nmodels/\n",
    ".dvcignore": "# Fichiers à ignorer par DVC\ndata/\nlogs/\nmodels/\n",
    "requirements.txt": "# Dépendances Python\nnumpy\npandas\nyfinance\npyyaml\nmatplotlib\n",
    "setup.py": "# Installation du package\nfrom setuptools import setup, find_packages\n\nsetup(\n    name='portfolio-rl-mlops',\n    version='0.1.0',\n    packages=find_packages(),\n    install_requires=[],  # À compléter\n)\n",
    "README.md": "# Portfolio RL MLOps\n\nDocumentation principale du projet."
}


# Fonction pour créer la structure du projet
def create_project_structure(base_path, structure):
    for name, content in structure.items():
        path = os.path.join(base_path, name)
        if isinstance(content, dict):  # C'est un dossier
            os.makedirs(path, exist_ok=True)
            create_project_structure(path, content)  # Appel récursif pour les sous-dossiers
        else:  # C'est un fichier
            with open(path, "w") as f:
                f.write(content)


# Création du projet dans le dossier actif
if __name__ == "__main__":
    # Obtenir le chemin du dossier actif (où est exécuté le script)
    current_directory = os.getcwd()

    print(f"Création de la structure du projet dans : {current_directory}")
    create_project_structure(current_directory, project_structure)
    print("Structure créée avec succès !")