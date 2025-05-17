"""
Utilitaires pour DVC (Data Version Control).
"""
import os
import subprocess
import logging
import json

# Configuration du logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def load_config(config_path="config/default.json"):
    """
    Charge la configuration depuis un fichier json.
    """
    with open(config_path, 'r') as file:
        config = json.load(file)
    return config

def run_dvc_command(command, cwd=None):
    """
    Exécute une commande DVC.
    
    Args:
        command (list): Liste des arguments de la commande
        cwd (str): Répertoire de travail (optionnel)
        
    Returns:
        tuple: (stdout, stderr, return_code)
    """
    try:
        full_command = ["dvc"] + command
        logger.debug(f"Exécution de la commande DVC: {' '.join(full_command)}")
        
        process = subprocess.Popen(
            full_command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            cwd=cwd
        )
        
        stdout, stderr = process.communicate()
        return_code = process.returncode
        
        if return_code != 0:
            logger.warning(f"La commande DVC a retourné un code non nul: {return_code}")
            logger.warning(f"Stderr: {stderr}")
        
        return stdout, stderr, return_code
    
    except Exception as e:
        logger.error(f"Erreur lors de l'exécution de la commande DVC: {e}")
        return None, str(e), -1

def version_data_file(file_path, message=None, cwd=None):
    """
    Versionne un fichier de données avec DVC.
    
    Args:
        file_path (str): Chemin du fichier à versionner
        message (str): Message de commit (optionnel)
        cwd (str): Répertoire de travail (optionnel)
        
    Returns:
        bool: True si le versionnage a réussi, False sinon
    """
    try:
        # S'assurer que le fichier existe
        if not os.path.exists(file_path):
            logger.error(f"Le fichier {file_path} n'existe pas")
            return False
        
        # Ajouter le fichier à DVC
        stdout, stderr, rc = run_dvc_command(["add", file_path], cwd=cwd)
        if rc != 0:
            logger.error(f"Erreur lors de l'ajout du fichier à DVC: {stderr}")
            return False
        
        # Push des modifications
        stdout, stderr, rc = run_dvc_command(["push"], cwd=cwd)
        if rc != 0:
            logger.warning(f"Erreur lors du push DVC: {stderr}")
            # Continuer même si le push échoue
        
        # Si un message est fourni, commit les modifications Git
        if message:
            subprocess.run(
                ["git", "add", f"{file_path}.dvc"],
                check=True,
                cwd=cwd
            )
            subprocess.run(
                ["git", "commit", "-m", message],
                check=True,
                cwd=cwd
            )
        
        logger.info(f"Fichier {file_path} versionné avec succès")
        return True
    
    except Exception as e:
        logger.error(f"Erreur lors du versionnage du fichier: {e}")
        return False

def get_data_versions(file_path, cwd=None):
    """
    Récupère les versions disponibles d'un fichier de données.
    
    Args:
        file_path (str): Chemin du fichier
        cwd (str): Répertoire de travail (optionnel)
        
    Returns:
        list: Liste des versions disponibles (hashes)
    """
    try:
        stdout, stderr, rc = run_dvc_command(["list", "--dvc-only", "--recursive", file_path], cwd=cwd)
        
        if rc != 0:
            logger.error(f"Erreur lors de la récupération des versions: {stderr}")
            return []
        
        # Analyser la sortie pour extraire les hashes
        versions = []
        for line in stdout.strip().split('\n'):
            if line:
                parts = line.split()
                if len(parts) >= 1:
                    versions.append(parts[0])
        
        return versions
    
    except Exception as e:
        logger.error(f"Erreur lors de la récupération des versions: {e}")
        return []

def checkout_data_version(file_path, version=None, cwd=None):
    """
    Récupère une version spécifique d'un fichier de données.
    
    Args:
        file_path (str): Chemin du fichier
        version (str): Version à récupérer (hash) ou None pour la dernière version
        cwd (str): Répertoire de travail (optionnel)
        
    Returns:
        bool: True si le checkout a réussi, False sinon
    """
    try:
        if version:
            # Checkout d'une version spécifique
            stdout, stderr, rc = run_dvc_command(["checkout", file_path, "--rev", version], cwd=cwd)
        else:
            # Checkout de la dernière version
            stdout, stderr, rc = run_dvc_command(["checkout", file_path], cwd=cwd)
        
        if rc != 0:
            logger.error(f"Erreur lors du checkout de la version: {stderr}")
            return False
        
        logger.info(f"Checkout réussi pour {file_path}" + (f" (version {version})" if version else ""))
        return True
    
    except Exception as e:
        logger.error(f"Erreur lors du checkout de la version: {e}")
        return False

if __name__ == "__main__":
    # Exemple d'utilisation
    # version_data_file("data/raw/stock_data_20250415.csv", message="Ajout des données du 15 avril 2025")
    
    # Lister les versions d'un fichier
    versions = get_data_versions("data/processed/")
    logger.info(f"Versions disponibles: {versions}")