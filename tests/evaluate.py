# �valuation (bas� sur votre testing.py)

"""
Module pour l'évaluation d'un agent d'allocation de portefeuille.
"""

# Ajouter ces imports au début du fichier
import mlflow
from mlflow.tracking import MlflowClient


import os
import time
import numpy as np
import matplotlib.pyplot as plt
import json
import logging

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

#def evaluate_portfolio_model(model_path, config_path="config/default.json", data_path=None):
    """
    Évalue un modèle PPO entraîné sur un ensemble de données de test.
    
    Args:
        model_path: Chemin vers le modèle PPO sauvegardé
        config_path: Chemin du fichier de configuration
        data_path: Chemin vers les données de test (optionnel)
        
    Returns:
        dict: Statistiques d'évaluation
    """
    # Charger la configuration
    config = load_config(config_path)
    
    # Charger les données de test
    if data_path is None:
        data_path = os.path.join(config['data']['output_dir'], "stock_data_test_latest.npy")
    
    logger.info(f"Chargement des données de test depuis {data_path}...")
    test_data = np.load(data_path)
    logger.info(f"Données de test chargées avec succès. Forme: {test_data.shape}")
    
    try:
        from stable_baselines3 import PPO
        from stable_baselines3.common.monitor import Monitor
        
        from portfolio_env import create_env_from_config
        
        # Créer l'environnement de test
        logger.info("Initialisation de l'environnement de test...")
        test_env = create_env_from_config(test_data, config_path)
        logger.info(f"Environnement de test créé avec {test_env.n_stocks} actions et {test_env.n_time_periods} périodes")
        
        # Créer le dossier des résultats
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        results_dir = os.path.join(config['evaluation']['results_dir'], f"evaluation_{timestamp}")
        os.makedirs(results_dir, exist_ok=True)
        
        # Wrapper l'environnement avec Monitor pour le suivi
        test_env = Monitor(test_env, results_dir)
        
        # Charger le modèle entraîné
        logger.info(f"Chargement du modèle depuis {model_path}...")
        model = PPO.load(model_path)
        logger.info("Modèle chargé avec succès!")
        
        # Évaluer le modèle
        logger.info("\nDébut de l'évaluation sur les données de test...")
        
        # Réinitialiser l'environnement pour l'évaluation
        obs, info = test_env.reset()
        terminated = truncated = False
        cumulative_reward = 0
        
        # Accéder à l'environnement d'origine à travers le wrapper Monitor
        initial_value = test_env.unwrapped.portfolio_value
        
        # Pour le suivi pendant l'évaluation
        eval_steps = []
        eval_values = []
        eval_weights = []
        eval_rewards = []
        eval_returns = []
        eval_drawdowns = []
        eval_transactions = []
        
        step = 0
        while not (terminated or truncated):
            action, _states = model.predict(obs, deterministic=True)  # Actions déterministes pour l'évaluation
            obs, reward, terminated, truncated, info = test_env.step(action)
            cumulative_reward += reward
            
            # Suivi des métriques d'évaluation
            eval_steps.append(step)
            eval_values.append(info['portfolio_value'])
            eval_weights.append(info['weights'])
            eval_rewards.append(reward)
            eval_returns.append(info['return'])
            eval_drawdowns.append(info['drawdown'])
            eval_transactions.append(info.get('transaction_cost', 0))
            
            step += 1
            
            # Afficher la progression périodiquement
            if step % 10 == 0:
                logger.info(f"Étape {step}/{test_env.unwrapped.n_time_periods - test_env.unwrapped.window_size}: "
                          f"Valeur: {info['portfolio_value']:.2f}, "
                          f"Rendement: {info['return']*100:.2f}%, "
                          f"Récompense: {reward:.4f}")
        
        # Calculer les statistiques finales
        final_value = info['portfolio_value']
        total_return = (final_value / initial_value - 1) * 100
        
        # Calculer des statistiques supplémentaires
        returns_array = np.array(eval_returns)
        annualized_return = ((1 + returns_array.mean()) ** 252 - 1) * 100  # Supposant des données quotidiennes
        volatility = returns_array.std() * np.sqrt(252) * 100  # Volatilité annualisée
        sharpe_ratio = annualized_return / volatility if volatility > 0 else 0
        max_drawdown = max(eval_drawdowns) * 100
        avg_transaction_cost = np.mean(eval_transactions)
        
        # Tracer les résultats d'évaluation
        plt.figure(figsize=(15, 15))
        
        # Valeur du portefeuille pendant l'évaluation
        plt.subplot(3, 2, 1)
        plt.plot(eval_steps, eval_values)
        plt.title("Valeur du Portefeuille")
        plt.xlabel("Étapes")
        plt.ylabel("Valeur ($)")
        plt.grid(True)
        
        # Récompenses pendant l'évaluation
        plt.subplot(3, 2, 2)
        plt.plot(eval_steps, eval_rewards)
        plt.title("Récompenses")
        plt.xlabel("Étapes")
        plt.ylabel("Récompense")
        plt.grid(True)
        
        # Heatmap des poids du portefeuille
        plt.subplot(3, 2, 3)
        weights_array = np.array(eval_weights)
        plt.imshow(weights_array.T, aspect='auto', cmap='viridis')
        plt.title("Poids du Portefeuille au Fil du Temps")
        plt.xlabel("Étapes")
        plt.ylabel("Actif")
        plt.colorbar(label="Poids")
        
        # Rendements cumulés
        plt.subplot(3, 2, 4)
        cumulative_returns = np.cumprod(1 + np.array(eval_returns)) - 1
        plt.plot(eval_steps, cumulative_returns * 100)  # En pourcentage
        plt.title("Rendement Cumulé")
        plt.xlabel("Étapes")
        plt.ylabel("Rendement (%)")
        plt.grid(True)
        
        # Drawdowns
        plt.subplot(3, 2, 5)
        plt.plot(eval_steps, np.array(eval_drawdowns) * 100)  # En pourcentage
        plt.title("Drawdowns")
        plt.xlabel("Étapes")
        plt.ylabel("Drawdown (%)")
        plt.grid(True)
        
        # Coûts de transaction
        plt.subplot(3, 2, 6)
        plt.plot(eval_steps, eval_transactions)
        plt.title("Coûts de Transaction")
        plt.xlabel("Étapes")
        plt.ylabel("Coût ($)")
        plt.grid(True)
        
        plt.tight_layout()
        plt.savefig(f"{results_dir}/test_evaluation_results.png")
        
        # Sauvegarder les données d'évaluation pour une analyse ultérieure
        np.save(f"{results_dir}/test_evaluation_data.npy", {
            'steps': eval_steps,
            'values': eval_values,
            'weights': eval_weights,
            'rewards': eval_rewards,
            'returns': eval_returns,
            'drawdowns': eval_drawdowns,
            'transactions': eval_transactions
        })
        
        # Comparer avec une stratégie de référence (allocation égale)
        logger.info("\nComparaison avec une stratégie d'allocation égale...")
        
        # Réinitialiser l'environnement pour la stratégie de référence
        obs, info = test_env.reset()
        benchmark_values = [initial_value]
        benchmark_returns = []
        step = 0
        terminated = truncated = False
        
        while not (terminated or truncated):
            # Action d'allocation égale
            equal_action = np.ones(test_env.unwrapped.n_stocks) / test_env.unwrapped.n_stocks
            obs, reward, terminated, truncated, info = test_env.step(equal_action)
            
            benchmark_values.append(info['portfolio_value'])
            benchmark_returns.append(info['return'])
            step += 1
        
        benchmark_final_value = benchmark_values[-1]
        benchmark_total_return = (benchmark_final_value / initial_value - 1) * 100
        
        # Calculer les statistiques du benchmark
        benchmark_returns_array = np.array(benchmark_returns)
        benchmark_annual_return = ((1 + benchmark_returns_array.mean()) ** 252 - 1) * 100
        benchmark_volatility = benchmark_returns_array.std() * np.sqrt(252) * 100
        benchmark_sharpe = benchmark_annual_return / benchmark_volatility if benchmark_volatility > 0 else 0
        
        # Tracer la comparaison avec le benchmark
        plt.figure(figsize=(12, 8))
        plt.plot(eval_values, label="Modèle RL")
        plt.plot(benchmark_values, label="Allocation Égale")
        plt.title("Comparaison de la Valeur du Portefeuille")
        plt.xlabel("Étapes")
        plt.ylabel("Valeur ($)")
        plt.legend()
        plt.grid(True)
        plt.savefig(f"{results_dir}/benchmark_comparison.png")
        
        # Afficher les résultats
        logger.info("\nRésultats de l'évaluation sur les données de test:")
        logger.info(f"Nombre total d'étapes: {step}")
        logger.info(f"Valeur initiale du portefeuille: {initial_value:.2f}")
        logger.info(f"Valeur finale du portefeuille: {final_value:.2f}")
        logger.info(f"Rendement total: {total_return:.2f}%")
        logger.info(f"Rendement annualisé: {annualized_return:.2f}%")
        logger.info(f"Volatilité annualisée: {volatility:.2f}%")
        logger.info(f"Ratio de Sharpe: {sharpe_ratio:.2f}")
        logger.info(f"Drawdown maximal: {max_drawdown:.2f}%")
        logger.info(f"Coût de transaction moyen: {avg_transaction_cost:.4f}$")
        logger.info(f"Récompense cumulée: {cumulative_reward:.4f}")
        
        logger.info("\nRésultats de la stratégie d'allocation égale:")
        logger.info(f"Valeur finale du portefeuille: {benchmark_final_value:.2f}")
        logger.info(f"Rendement total: {benchmark_total_return:.2f}%")
        logger.info(f"Rendement annualisé: {benchmark_annual_return:.2f}%")
        logger.info(f"Volatilité annualisée: {benchmark_volatility:.2f}%")
        logger.info(f"Ratio de Sharpe: {benchmark_sharpe:.2f}")
        
        logger.info("\nSurperformance par rapport au benchmark:")
        logger.info(f"Différence de rendement total: {total_return - benchmark_total_return:.2f}%")
        logger.info(f"Ratio de surperformance: {final_value / benchmark_final_value:.2f}x")
        
        # Retourner les statistiques clés
        return {
            'final_value': final_value,
            'total_return': total_return,
            'annualized_return': annualized_return,
            'volatility': volatility,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'benchmark_return': benchmark_total_return,
            'outperformance': total_return - benchmark_total_return,
            'results_dir': results_dir
        }
        
    except ImportError as e:
        logger.error(f"ERREUR: {e}")
        logger.error("Veuillez installer les dépendances requises:")
        logger.error("pip install stable-baselines3[extra] gymnasium matplotlib")
        return None

def evaluate_portfolio_model(model_path, config_path="config/default.json", data_path=None, mlflow_run_id=None):
    """
    Évalue un modèle PPO entraîné sur un ensemble de données de test, avec suivi MLflow.
    
    Args:
        model_path: Chemin vers le modèle PPO sauvegardé
        config_path: Chemin du fichier de configuration
        data_path: Chemin vers les données de test (optionnel)
        mlflow_run_id: ID du run MLflow lié à l'entraînement (optionnel)
        
    Returns:
        dict: Statistiques d'évaluation
    """
    # Votre code existant pour charger la configuration et les données...
    
    try:
        from stable_baselines3 import PPO
        from stable_baselines3.common.monitor import Monitor
        
        from src.environment.portfolio_env import create_env_from_config
        
        # Configuration de MLflow
        mlflow_tracking_uri = os.environ.get("MLFLOW_TRACKING_URI", "http://localhost:5000")
        mlflow.set_tracking_uri(mlflow_tracking_uri)
        
        # Déterminer s'il faut continuer un run existant ou en créer un nouveau
        if mlflow_run_id:
            active_run = mlflow.start_run(run_id=mlflow_run_id)
        else:
            experiment_name = "portfolio_optimization"
            experiment = mlflow.get_experiment_by_name(experiment_name)
            if experiment:
                experiment_id = experiment.experiment_id
            else:
                experiment_id = mlflow.create_experiment(
                    experiment_name,
                    artifact_location=os.path.join("artifacts", experiment_name)
                )
            active_run = mlflow.start_run(experiment_id=experiment_id, run_name=f"evaluation_{time.strftime('%Y%m%d-%H%M%S')}")
        
        # Votre code existant pour créer l'environnement et charger le modèle...
        
        # Après l'évaluation, enregistrer les métriques dans MLflow
        mlflow.log_metrics({
            "test_final_value": final_value,
            "test_total_return": total_return,
            "test_annualized_return": annualized_return,
            "test_volatility": volatility,
            "test_sharpe_ratio": sharpe_ratio,
            "test_max_drawdown": max_drawdown,
            "test_benchmark_return": benchmark_total_return,
            "test_outperformance": total_return - benchmark_total_return
        })
        
        # Enregistrer les graphiques générés
        mlflow.log_artifact(f"{results_dir}/test_evaluation_results.png", "evaluation_plots")
        mlflow.log_artifact(f"{results_dir}/benchmark_comparison.png", "evaluation_plots")
        
        # Si nous avons commencé un nouveau run, terminons-le
        if not mlflow_run_id:
            mlflow.end_run()
        
        # Retourner les statistiques d'évaluation
        return {
            'final_value': final_value,
            'total_return': total_return,
            'annualized_return': annualized_return,
            'volatility': volatility,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'benchmark_return': benchmark_total_return,
            'outperformance': total_return - benchmark_total_return,
            'results_dir': results_dir,
            'mlflow_run_id': active_run.info.run_id
        }
        
    except ImportError as e:
        logger.error(f"ERREUR: {e}")
        logger.error("Veuillez installer les dépendances requises.")
        return None
    

def main(config_path="config/default.json", model_path=None):
    """
    Fonction principale pour l'évaluation d'un agent.
    
    Args:
        config_path: Chemin du fichier de configuration
        model_path: Chemin vers le modèle PPO à évaluer (si None, utilisera le dernier modèle)
    """
    # Si model_path n'est pas spécifié, chercher le modèle le plus récent
    if model_path is None:
        config = load_config(config_path)
        log_dir = config['training']['log_dir']
        
        # Trouver le dossier d'entraînement le plus récent
        if os.path.exists(log_dir):
            training_dirs = [os.path.join(log_dir, d) for d in os.listdir(log_dir) if d.startswith("portfolio_training_")]
            if training_dirs:
                latest_dir = max(training_dirs, key=os.path.getmtime)
                
                # Chercher d'abord un best_model, sinon utiliser final_model
                if os.path.exists(os.path.join(latest_dir, "best_model.zip")):
                    model_path = os.path.join(latest_dir, "best_model")
                elif os.path.exists(os.path.join(latest_dir, "final_model.zip")):
                    model_path = os.path.join(latest_dir, "final_model")
        
        if model_path is None:
            logger.error("Aucun modèle trouvé. Veuillez spécifier un chemin de modèle.")
            return None
        
        logger.info(f"Utilisation du modèle le plus récent: {model_path}")
    
    # Évaluer le modèle
    return evaluate_portfolio_model(model_path, config_path)

if __name__ == "__main__":
    main()