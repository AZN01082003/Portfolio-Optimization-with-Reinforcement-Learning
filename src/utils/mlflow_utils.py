"""
Utilitaires pour MLflow.
"""
import os
import mlflow
from mlflow.tracking import MlflowClient
import logging

# Configuration du logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def setup_mlflow(experiment_name="portfolio_optimization"):
    """
    Configure MLflow et retourne l'ID de l'expérience.
    
    Args:
        experiment_name: Nom de l'expérience
        
    Returns:
        str: ID de l'expérience
    """
    # Configurer l'URI de tracking à partir des variables d'environnement ou utiliser la valeur par défaut
    mlflow_tracking_uri = os.environ.get("MLFLOW_TRACKING_URI", "http://localhost:5000")
    mlflow.set_tracking_uri(mlflow_tracking_uri)
    
    # Vérifier si l'expérience existe déjà
    experiment = mlflow.get_experiment_by_name(experiment_name)
    if experiment:
        experiment_id = experiment.experiment_id
    else:
        # Créer une nouvelle expérience
        artifact_location = os.path.join("artifacts", experiment_name)
        experiment_id = mlflow.create_experiment(
            experiment_name,
            artifact_location=artifact_location
        )
        logger.info(f"Expérience '{experiment_name}' créée avec ID: {experiment_id}")
    
    return experiment_id

def log_model_to_registry(model, model_name, run_id=None):
    """
    Enregistre un modèle dans le registre MLflow.
    
    Args:
        model: Modèle à enregistrer
        model_name: Nom du modèle dans le registre
        run_id: ID du run MLflow (optionnel)
        
    Returns:
        str: Version du modèle
    """
    client = MlflowClient()
    
    # Vérifier si le modèle existe déjà dans le registre
    try:
        model_details = client.get_registered_model(model_name)
        logger.info(f"Modèle '{model_name}' existe déjà dans le registre")
    except:
        # Créer un nouveau modèle dans le registre
        client.create_registered_model(model_name)
        logger.info(f"Modèle '{model_name}' créé dans le registre")
    
    # Enregistrer le modèle avec MLflow
    if run_id:
        with mlflow.start_run(run_id=run_id):
            mlflow.stable_baselines3.log_model(
                sb3_model=model,
                artifact_path="model",
                registered_model_name=model_name
            )
    else:
        with mlflow.start_run() as run:
            run_id = run.info.run_id
            mlflow.stable_baselines3.log_model(
                sb3_model=model,
                artifact_path="model",
                registered_model_name=model_name
            )
    
    # Récupérer la dernière version du modèle
    versions = client.get_latest_versions(model_name)
    if versions:
        latest_version = versions[0].version
        logger.info(f"Modèle '{model_name}' version {latest_version} enregistré")
        return latest_version
    else:
        logger.warning(f"Aucune version trouvée pour le modèle '{model_name}'")
        return None

def get_best_model(model_name, metric="test_sharpe_ratio", ascending=False):
    """
    Récupère le meilleur modèle du registre selon une métrique.
    
    Args:
        model_name: Nom du modèle dans le registre
        metric: Métrique à utiliser pour la comparaison
        ascending: Si True, la valeur minimale est considérée comme la meilleure
        
    Returns:
        tuple: (run_id, version, model_uri)
    """
    client = MlflowClient()
    
    try:
        # Récupérer toutes les versions du modèle
        versions = client.get_latest_versions(model_name)
        
        best_version = None
        best_value = float('-inf') if not ascending else float('inf')
        best_run_id = None
        
        for version in versions:
            run_id = version.run_id
            run = client.get_run(run_id)
            
            if metric in run.data.metrics:
                metric_value = run.data.metrics[metric]
                
                is_better = (not ascending and metric_value > best_value) or \
                           (ascending and metric_value < best_value)
                
                if is_better:
                    best_value = metric_value
                    best_version = version.version
                    best_run_id = run_id
        
        if best_version:
            model_uri = f"models:/{model_name}/{best_version}"
            logger.info(f"Meilleur modèle trouvé: '{model_name}' version {best_version} avec {metric}={best_value}")
            return best_run_id, best_version, model_uri
        else:
            logger.warning(f"Aucun modèle trouvé avec la métrique '{metric}'")
            return None, None, None
            
    except Exception as e:
        logger.error(f"Erreur lors de la récupération du meilleur modèle: {e}")
        return None, None, None