"""
Script principal pour exécuter l'ensemble du pipeline.
"""
import os
import argparse
import logging
import json
import time
from datetime import datetime

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

def run_data_pipeline(config_path="config/default.json", force_download=False):
    """
    Exécute le pipeline de données.
    """
    from preprocessing import main as preprocess_main
    
    logger.info("=== Exécution du pipeline de données ===")
    
    try:
        # Exécuter le prétraitement (qui inclut l'ingestion)
        normalized_data, train_data, test_data = preprocess_main(config_path=config_path)
        logger.info("Pipeline de données exécuté avec succès.")
        return normalized_data, train_data, test_data
    
    except Exception as e:
        logger.error(f"Erreur lors de l'exécution du pipeline de données: {e}")
        return None, None, None

def run_training_pipeline(config_path="config/default.json", data_path=None):
    """
    Exécute le pipeline d'entraînement.
    """
    from train import train_portfolio_agent
    
    logger.info("=== Exécution du pipeline d'entraînement ===")
    
    try:
        model, training_info = train_portfolio_agent(config_path=config_path, data_path=data_path)
        
        if model is not None:
            logger.info("Pipeline d'entraînement exécuté avec succès.")
            return model, training_info
        else:
            logger.error("L'entraînement a échoué.")
            return None, None
    
    except Exception as e:
        logger.error(f"Erreur lors de l'exécution du pipeline d'entraînement: {e}")
        return None, None

def run_evaluation_pipeline(model_path, config_path="config/default.json", data_path=None):
    """
    Exécute le pipeline d'évaluation.
    """
    from evaluate import evaluate_portfolio_model
    
    logger.info("=== Exécution du pipeline d'évaluation ===")
    
    try:
        evaluation_results = evaluate_portfolio_model(
            model_path=model_path,
            config_path=config_path,
            data_path=data_path
        )
        
        if evaluation_results is not None:
            logger.info("Pipeline d'évaluation exécuté avec succès.")
            return evaluation_results
        else:
            logger.error("L'évaluation a échoué.")
            return None
    
    except Exception as e:
        logger.error(f"Erreur lors de l'exécution du pipeline d'évaluation: {e}")
        return None

def generate_visualization_report(evaluation_results):
    """
    Génère un rapport de visualisation des performances.
    """
    from visualization import plot_backtest_summary, generate_performance_report
    
    logger.info("=== Génération du rapport de visualisation ===")
    
    try:
        if evaluation_results and 'results_dir' in evaluation_results:
            results_dir = evaluation_results['results_dir']
            
            # Créer un sous-répertoire pour les rapports
            reports_dir = os.path.join(results_dir, "reports")
            os.makedirs(reports_dir, exist_ok=True)
            
            # Générer les visualisations
            plot_backtest_summary(
                results_dir=results_dir,
                save_path=os.path.join(reports_dir, "backtest_summary.png")
            )
            
            # Générer le rapport de performance
            generate_performance_report(
                results_dir=results_dir,
                save_path=os.path.join(reports_dir, "performance_report")
            )
            
            logger.info(f"Rapport de visualisation généré dans {reports_dir}")
            return True
        else:
            logger.error("Impossible de générer le rapport: résultats d'évaluation manquants.")
            return False
    
    except Exception as e:
        logger.error(f"Erreur lors de la génération du rapport de visualisation: {e}")
        return False

def main():
    """
    Fonction principale pour exécuter l'ensemble du pipeline.
    """
    parser = argparse.ArgumentParser(description="Exécute le pipeline de trading par RL")
    parser.add_argument("--config", type=str, default="config/default.json", help="Chemin du fichier de configuration")
    parser.add_argument("--skip-data", action="store_true", help="Sauter le pipeline de données")
    parser.add_argument("--skip-training", action="store_true", help="Sauter le pipeline d'entraînement")
    parser.add_argument("--skip-evaluation", action="store_true", help="Sauter le pipeline d'évaluation")
    parser.add_argument("--model-path", type=str, help="Chemin du modèle à évaluer (si --skip-training)")
    parser.add_argument("--force-download", action="store_true", help="Forcer le téléchargement des données")
    args = parser.parse_args()
    
    start_time = time.time()
    logger.info(f"Démarrage du pipeline à {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # 1. Pipeline de données
    if not args.skip_data:
        normalized_data, train_data, test_data = run_data_pipeline(
            config_path=args.config,
            force_download=args.force_download
        )
    else:
        logger.info("Pipeline de données ignoré.")
    
    # 2. Pipeline d'entraînement
    if not args.skip_training:
        model, training_info = run_training_pipeline(config_path=args.config)
        
        if model is not None and training_info is not None:
            model_path = training_info.get('best_model_path') or training_info.get('model_path')
        else:
            logger.error("Impossible de continuer: l'entraînement a échoué.")
            return
    else:
        logger.info("Pipeline d'entraînement ignoré.")
        model_path = args.model_path
        
        if not model_path:
            logger.error("Veuillez spécifier --model-path quand --skip-training est utilisé.")
            return
    
    # 3. Pipeline d'évaluation
    if not args.skip_evaluation:
        evaluation_results = run_evaluation_pipeline(
            model_path=model_path,
            config_path=args.config
        )
        
        if evaluation_results is not None:
            # 4. Génération du rapport de visualisation
            generate_visualization_report(evaluation_results)
        else:
            logger.error("Impossible de générer le rapport: l'évaluation a échoué.")
    else:
        logger.info("Pipeline d'évaluation ignoré.")
    
    # Calculer le temps total d'exécution
    total_time = time.time() - start_time
    logger.info(f"Pipeline terminé en {total_time:.2f} secondes.")

if __name__ == "__main__":
    main()