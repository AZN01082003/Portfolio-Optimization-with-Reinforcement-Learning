import sys
import os

# Ajouter la racine du projet au PYTHONPATH
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


"""
Pipeline d'entraînement automatisé pour le modèle d'allocation de portefeuille.
"""
#import os
import logging
import time
import numpy as np
import pandas as pd
#import yaml
import json
import mlflow
from mlflow.tracking import MlflowClient
from datetime import datetime
from typing import Dict, List, Union, Tuple, Any

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback, EvalCallback
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.monitor import Monitor

from src.data.feature_store import FeatureStore
from src.data.feature_engineering import prepare_model_features
from portfolio_env import PortfolioEnv, create_env_from_config
from src.utils.mlflow_utils import setup_mlflow, log_model_to_registry

# Configuration du logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def load_config(config_path="config/default.yaml"):
    """
    Charge la configuration depuis un fichier YAML.
    """
    with open(config_path, 'r') as file:
        config = json.load(file)
    return config

class PortfolioTrainingCallback(BaseCallback):
    """
    Callback personnalisé pour suivre la progression de l'entraînement avec MLflow.
    """
    def __init__(self, verbose=0, check_freq=1000, log_dir="./logs/", feature_store=None):
        super(PortfolioTrainingCallback, self).__init__(verbose)
        self.portfolio_values = []
        self.rewards = []
        self.timesteps = []
        self.returns = []
        self.drawdowns = []
        self.check_freq = check_freq
        self.log_dir = log_dir
        self.best_mean_reward = -np.inf
        self.no_improvement_count = 0
        self.last_improvement_step = 0
        self.feature_store = feature_store
        
        # Create log directory if it doesn't exist
        os.makedirs(log_dir, exist_ok=True)
        
        # Tracking training start time
        self.start_time = time.time()
    
    def _on_step(self):
        # Log time periodically
        if self.n_calls % 1000 == 0:
            elapsed = time.time() - self.start_time
            logger.info(f"Training step {self.n_calls}, elapsed time: {elapsed:.1f}s, FPS: {self.n_calls/max(elapsed, 1):.1f}")
        
        if self.locals.get("infos") and len(self.locals["infos"]) > 0:
            info = self.locals["infos"][0]
            reward = self.locals.get("rewards", [0])[0]
            
            self.portfolio_values.append(info.get("portfolio_value", 0))
            self.rewards.append(reward)
            self.timesteps.append(self.n_calls)
            self.returns.append(info.get("return", 0))
            self.drawdowns.append(info.get("drawdown", 0))
            
            # Check if we should save intermediate plots
            if self.n_calls % self.check_freq == 0:
                self._save_plots()
                
                # Log métriques dans MLflow
                if len(self.rewards) > 100:
                    mean_reward = np.mean(self.rewards[-100:])
                    mean_return = np.mean(self.returns[-100:]) * 100  # En pourcentage
                    mean_drawdown = np.mean(self.drawdowns[-100:]) * 100  # En pourcentage
                    
                    # Enregistrer les métriques dans MLflow
                    mlflow.log_metrics({
                        "mean_reward": mean_reward,
                        "mean_return": mean_return,
                        "mean_drawdown": mean_drawdown,
                        "portfolio_value": self.portfolio_values[-1]
                    }, step=self.n_calls)
                
                # Check for training improvement
                if len(self.rewards) > 100:
                    mean_reward = np.mean(self.rewards[-100:])
                    
                    if mean_reward > self.best_mean_reward:
                        self.best_mean_reward = mean_reward
                        self.last_improvement_step = self.n_calls
                        self.model.save(f"{self.log_dir}/best_model")
                        logger.info(f"Saving new best model with mean reward: {mean_reward:.5f}")
                        self.no_improvement_count = 0
                        
                        # Sauvegarde du modèle dans le feature store
                        if self.feature_store:
                            self.feature_store.save_feature(
                                name="best_model",
                                data=self.model,
                                entity_type="model",
                                version=f"step_{self.n_calls}",
                                description=f"Meilleur modèle à l'étape {self.n_calls}",
                                parameters={
                                    "step": self.n_calls,
                                    "mean_reward": float(mean_reward),
                                    "mean_return": float(np.mean(self.returns[-100:])),
                                    "portfolio_value": float(self.portfolio_values[-1])
                                }
                            )
                    else:
                        self.no_improvement_count += 1
                    
                    # Early stopping if no improvement for a long time
                    if self.no_improvement_count >= 5 and self.n_calls > 10000:
                        logger.info(f"No improvement for {self.no_improvement_count * self.check_freq} steps. Stopping training.")
                        return False
        
        # Check for NaN rewards
        if np.isnan(reward):
            logger.warning("NaN reward detected! Stopping training.")
            return False
            
        return True
    
    def _save_plots(self):
        """Save intermediate plots to monitor training"""
        import matplotlib.pyplot as plt
        
        plt.figure(figsize=(15, 12))
        
        # Portfolio value plot
        plt.subplot(2, 2, 1)
        plt.plot(self.timesteps, self.portfolio_values)
        plt.title("Portfolio Value")
        plt.xlabel("Steps")
        plt.ylabel("Value ($)")
        plt.grid(True)
        
        # Rewards plot
        plt.subplot(2, 2, 2)
        plt.plot(self.timesteps, self.rewards)
        plt.title("Rewards")
        plt.xlabel("Steps")
        plt.ylabel("Reward")
        plt.grid(True)
        
        # Returns plot
        plt.subplot(2, 2, 3)
        plt.plot(self.timesteps, np.array(self.returns) * 100)  # Convert to percentage
        plt.title("Returns")
        plt.xlabel("Steps")
        plt.ylabel("Return (%)")
        plt.grid(True)
        
        # Drawdowns plot
        plt.subplot(2, 2, 4)
        plt.plot(self.timesteps, np.array(self.drawdowns) * 100)  # Convert to percentage
        plt.title("Drawdowns")
        plt.xlabel("Steps")
        plt.ylabel("Drawdown (%)")
        plt.grid(True)
        
        plt.tight_layout()
        plot_path = f"{self.log_dir}/training_progress_{self.n_calls}.png"
        plt.savefig(plot_path)
        plt.close()
        
        # Log the plot in MLflow
        mlflow.log_artifact(plot_path, "plots")
        
        # Also save data for later analysis
        training_data = {
            'timesteps': self.timesteps,
            'portfolio_values': self.portfolio_values,
            'rewards': self.rewards,
            'returns': self.returns,
            'drawdowns': self.drawdowns
        }
        
        data_path = f"{self.log_dir}/training_data.npz"
        np.savez_compressed(data_path, **training_data)
        
        # Log the data in MLflow
        mlflow.log_artifact(data_path, "data")
    
    def on_training_end(self):
        # Final plots
        self._save_plots()
        
        # Calculate training statistics
        elapsed = time.time() - self.start_time
        logger.info(f"\nTraining completed in {elapsed:.1f} seconds")
        logger.info(f"Total timesteps: {self.n_calls}")
        logger.info(f"Best mean reward: {self.best_mean_reward:.5f}")
        
        if len(self.portfolio_values) > 0:
            final_value = self.portfolio_values[-1]
            initial_value = self.portfolio_values[0]
            total_return = (final_value / initial_value - 1) * 100
            logger.info(f"Final portfolio value: ${final_value:.2f}")
            logger.info(f"Total return: {total_return:.2f}%")
            
            # Log final metrics in MLflow
            mlflow.log_metrics({
                "final_portfolio_value": final_value,
                "total_return": total_return,
                "training_time": elapsed,
                "best_mean_reward": self.best_mean_reward
            })

def prepare_training_data(config_path="config/default.json", force_refresh=False):
    """
    Prépare les données d'entraînement à partir du feature store.
    
    Args:
        config_path: Chemin du fichier de configuration
        force_refresh: Si True, force la régénération des features
        
    Returns:
        tuple: (train_data, test_data, feature_store)
    """
    try:
        config = load_config(config_path)
        
        # Initialiser le feature store
        feature_store = FeatureStore(config_path=config_path)
        
        # Vérifier si les features d'entrée du modèle existent déjà
        if not force_refresh:
            model_data, metadata = feature_store.load_feature(
                name="model_input", 
                entity_type="model"
            )
            
            if model_data is not None:
                logger.info("Utilisation des features existantes du feature store")
                
                # Diviser en ensembles d'entraînement et de test
                train_ratio = config.get('data', {}).get('train_ratio', 0.7)
                n_time_periods = model_data.shape[2]
                split_point = int(n_time_periods * train_ratio)
                
                train_data = model_data[:, :, :split_point]
                test_data = model_data[:, :, split_point:]
                
                return train_data, test_data, feature_store
        
        # Si les features n'existent pas ou force_refresh est True, les générer
        logger.info("Génération de nouvelles features pour l'entraînement")
        
        # Récupérer la liste des tickers
        tickers = config.get('data', {}).get('tickers', [])
        
        # Préparer les features pour le modèle
        model_data = prepare_model_features(tickers, config_path=config_path)
        
        if model_data is None:
            logger.error("Échec de la préparation des features du modèle")
            return None, None, feature_store
        
        # Diviser en ensembles d'entraînement et de test
        train_ratio = config.get('data', {}).get('train_ratio', 0.7)
        n_time_periods = model_data.shape[2]
        split_point = int(n_time_periods * train_ratio)
        
        train_data = model_data[:, :, :split_point]
        test_data = model_data[:, :, split_point:]
        
        # Sauvegarder les ensembles d'entraînement et de test
        feature_store.save_feature(
            name="train_data",
            data=train_data,
            entity_type="model",
            version="latest",
            description="Données d'entraînement pour le modèle RL",
            parameters={"shape": train_data.shape, "train_ratio": train_ratio}
        )
        
        feature_store.save_feature(
            name="test_data",
            data=test_data,
            entity_type="model",
            version="latest",
            description="Données de test pour le modèle RL",
            parameters={"shape": test_data.shape, "train_ratio": train_ratio}
        )
        
        return train_data, test_data, feature_store
    
    except Exception as e:
        logger.error(f"Erreur lors de la préparation des données d'entraînement: {e}")
        return None, None, None

def train_portfolio_agent(config_path="config/default.json", force_refresh=False, experiment_name="portfolio_optimization"):
    """
    Entraîne un agent d'allocation de portefeuille avec suivi MLflow.
    
    Args:
        config_path: Chemin du fichier de configuration
        force_refresh: Si True, force la régénération des features
        experiment_name: Nom de l'expérience MLflow
        
    Returns:
        tuple: (model, training_info)
    """
    try:
        # Charger la configuration
        config = load_config(config_path)
        
        # Préparer les données
        train_data, test_data, feature_store = prepare_training_data(config_path, force_refresh)
        
        if train_data is None or test_data is None:
            logger.error("Données d'entraînement non disponibles")
            return None, None
        
        logger.info(f"Données d'entraînement préparées - Forme: {train_data.shape}")
        logger.info(f"Données de test préparées - Forme: {test_data.shape}")
        
        # Configurer MLflow
        mlflow_tracking_uri = os.environ.get("MLFLOW_TRACKING_URI", "http://localhost:5000")
        mlflow.set_tracking_uri(mlflow_tracking_uri)
        
        # Créer ou obtenir l'ID de l'expérience
        experiment_id = setup_mlflow(experiment_name)
        
        # Démarrer le run MLflow
        with mlflow.start_run(experiment_id=experiment_id, run_name=f"run_{time.strftime('%Y%m%d-%H%M%S')}") as run:
            run_id = run.info.run_id
            
            # Enregistrer les paramètres de configuration
            train_config = config.get('training', {})
            env_config = config.get('environment', {})
            
            # Paramètres à enregistrer
            mlflow.log_params({
                "algorithm": train_config.get('algorithm', 'PPO'),
                "learning_rate": train_config.get('learning_rate', 3e-4),
                "n_steps": train_config.get('n_steps', 1024),
                "batch_size": train_config.get('batch_size', 64),
                "gamma": train_config.get('gamma', 0.99),
                "portfolio_value": env_config.get('portfolio_value', 10000),
                "window_size": env_config.get('window_size', 30),
                "trans_cost": env_config.get('trans_cost', 0.0005),
                "data_shape": str(train_data.shape)
            })
            
            # Créer l'environnement d'entraînement
            env = create_env_from_config(train_data, config_path)
            
            # Créer le répertoire de log
            timestamp = time.strftime("%Y%m%d-%H%M%S")
            log_dir = os.path.join(config.get('training', {}).get('log_dir', "./logs/training"), f"portfolio_training_{timestamp}")
            os.makedirs(log_dir, exist_ok=True)
            
            # Wrapper l'environnement dans Monitor pour le logging
            env = Monitor(env, log_dir)
            
            # Vérifier que l'environnement est compatible avec Gymnasium
            logger.info("Vérification de l'environnement...")
            check_env(env)
            logger.info("L'environnement est valide!")
            
            # Créer un environnement de validation
            eval_env = create_env_from_config(test_data, config_path)
            eval_env = Monitor(eval_env, os.path.join(log_dir, 'eval'))
            
            # Créer le modèle avec les hyperparamètres de la configuration
            logger.info("Création du modèle PPO...")
            model = PPO(
                "MultiInputPolicy", 
                env, 
                verbose=1, 
                learning_rate=train_config.get('learning_rate', 3e-4),
                n_steps=train_config.get('n_steps', 1024),
                batch_size=train_config.get('batch_size', 64),
                ent_coef=train_config.get('ent_coef', 0.01),
                clip_range=train_config.get('clip_range', 0.2),
                max_grad_norm=train_config.get('max_grad_norm', 0.5),
                gae_lambda=train_config.get('gae_lambda', 0.95),
                gamma=train_config.get('gamma', 0.99),
                n_epochs=train_config.get('n_epochs', 10),
                tensorboard_log=log_dir
            )
            
            # Créer le callback personnalisé pour le suivi de l'entraînement
            callback = PortfolioTrainingCallback(
                verbose=1, 
                check_freq=1000, 
                log_dir=log_dir,
                feature_store=feature_store
            )
            callback.model = model  # Donner accès au modèle pour la sauvegarde
            
            # Créer le callback d'évaluation
            eval_callback = EvalCallback(
                eval_env,
                best_model_save_path=log_dir,
                log_path=log_dir,
                eval_freq=max(train_config.get('n_steps', 1024) // 5, 1000),
                deterministic=True,
                render=False,
                n_eval_episodes=5
            )
            
            # Combiner les callbacks
            callbacks = [callback, eval_callback]
            
            # Entraîner le modèle
            logger.info("Démarrage de l'entraînement...")
            try:
                model.learn(
                    total_timesteps=train_config.get('total_timesteps', 50000), 
                    callback=callbacks, 
                    progress_bar=True
                )
                logger.info("Entraînement terminé!")
            except Exception as e:
                logger.error(f"Erreur pendant l'entraînement: {e}")
                # Charger le meilleur modèle si disponible
                if os.path.exists(f"{log_dir}/best_model.zip"):
                    logger.info("Chargement du meilleur modèle sauvegardé...")
                    model = PPO.load(f"{log_dir}/best_model.zip")
            
            # Sauvegarder le modèle final
            model_path = f"{log_dir}/final_model"
            model.save(model_path)
            logger.info(f"Modèle sauvegardé dans {model_path}")
            
            # Enregistrer le modèle dans MLflow
            mlflow.stable_baselines3.log_model(
                sb3_model=model,
                artifact_path="model",
                registered_model_name="portfolio_rl_model"
            )
            
            # Enregistrer le modèle dans le registre MLflow
            model_version = log_model_to_registry(model, "portfolio_rl_model", run_id)
            
            # Sauvegarder le modèle dans le feature store
            if feature_store:
                feature_store.save_feature(
                    name="final_model",
                    data=model,
                    entity_type="model",
                    version=f"run_{timestamp}",
                    description=f"Modèle final de l'entraînement {timestamp}",
                    parameters={
                        "run_id": run_id,
                        "model_path": model_path,
                        "mlflow_model_version": model_version
                    }
                )
            
            # Informations de l'entraînement
            training_info = {
                'log_dir': log_dir,
                'model_path': model_path,
                'best_model_path': f"{log_dir}/best_model" if os.path.exists(f"{log_dir}/best_model.zip") else None,
                'mlflow_run_id': run_id,
                'mlflow_model_version': model_version,
                'timestamp': timestamp
            }
            
            # Logging des métadonnées d'entraînement
            mlflow.log_dict(training_info, "training_info.json")
            
            return model, training_info
    
    except Exception as e:
        logger.error(f"Erreur lors de l'entraînement: {e}")
        return None, None

def online_retraining(config_path="config/default.json", base_model_path=None, experiment_name="portfolio_optimization_online"):
    """
    Ré-entraîne le modèle en ligne sur de nouvelles données.
    
    Args:
        config_path: Chemin du fichier de configuration
        base_model_path: Chemin du modèle de base à ré-entraîner
        experiment_name: Nom de l'expérience MLflow
        
    Returns:
        tuple: (model, training_info)
    """
    try:
        # Charger la configuration
        config = load_config(config_path)
        
        # Préparer les données d'entraînement
        train_data, test_data, feature_store = prepare_training_data(config_path, force_refresh=True)
        
        if train_data is None or test_data is None:
            logger.error("Données d'entraînement non disponibles")
            return None, None
        
        # Si aucun modèle de base n'est spécifié, chercher le dernier modèle dans le feature store
        if base_model_path is None:
            logger.info("Recherche du dernier modèle dans le feature store...")
            model_data, metadata = feature_store.load_feature(
                name="final_model", 
                entity_type="model"
            )
            
            if model_data is not None:
                base_model = model_data
                logger.info(f"Modèle de base chargé depuis le feature store: {metadata.get('id')}")
            else:
                logger.warning("Aucun modèle trouvé dans le feature store. Création d'un nouveau modèle.")
                base_model = None
        else:
            # Charger le modèle de base spécifié
            try:
                base_model = PPO.load(base_model_path)
                logger.info(f"Modèle de base chargé depuis {base_model_path}")
            except Exception as e:
                logger.error(f"Erreur lors du chargement du modèle de base: {e}")
                base_model = None
        
        # Configurer MLflow
        mlflow_tracking_uri = os.environ.get("MLFLOW_TRACKING_URI", "http://localhost:5000")
        mlflow.set_tracking_uri(mlflow_tracking_uri)
        
        # Créer ou obtenir l'ID de l'expérience
        experiment_id = setup_mlflow(experiment_name)
        
        # Démarrer le run MLflow
        with mlflow.start_run(experiment_id=experiment_id, run_name=f"online_{time.strftime('%Y%m%d-%H%M%S')}") as run:
            run_id = run.info.run_id
            
            # Enregistrer les paramètres
            mlflow.log_params({
                "base_model": base_model_path if base_model_path else "feature_store_latest",
                "online_learning": True,
                "data_shape": str(train_data.shape),
                "timesteps": config.get('training', {}).get('online_timesteps', 10000)
            })
            
            # Créer l'environnement
            env = create_env_from_config(train_data, config_path)
            
            # Créer le répertoire de log
            timestamp = time.strftime("%Y%m%d-%H%M%S")
            log_dir = os.path.join(config.get('training', {}).get('log_dir', "./logs/training"), f"online_training_{timestamp}")
            os.makedirs(log_dir, exist_ok=True)
            
            # Wrapper l'environnement
            env = Monitor(env, log_dir)
            
            # Créer ou charger le modèle
            if base_model is not None:
                model = base_model
                # Reset l'optimizer pour le nouvel entraînement
                model.learning_rate = config.get('training', {}).get('online_learning_rate', 1e-4)
            else:
                # Créer un nouveau modèle
                model = PPO(
                    "MultiInputPolicy", 
                    env, 
                    verbose=1, 
                    learning_rate=config.get('training', {}).get('online_learning_rate', 1e-4),
                    n_steps=config.get('training', {}).get('n_steps', 1024),
                    batch_size=config.get('training', {}).get('batch_size', 64),
                    gamma=config.get('training', {}).get('gamma', 0.99)
                )
            
            # Créer le callback
            callback = PortfolioTrainingCallback(
                verbose=1, 
                check_freq=500, 
                log_dir=log_dir,
                feature_store=feature_store
            )
            callback.model = model
            
            # Entraîner le modèle en ligne
            logger.info("Démarrage du ré-entraînement en ligne...")
            try:
                model.learn(
                    total_timesteps=config.get('training', {}).get('online_timesteps', 10000),
                    callback=callback,
                    reset_num_timesteps=False  # Ne pas réinitialiser le compteur de timesteps
                )
                logger.info("Ré-entraînement terminé!")
            except Exception as e:
                logger.error(f"Erreur pendant le ré-entraînement: {e}")
            
            # Sauvegarder le modèle
            model_path = f"{log_dir}/online_model"
            model.save(model_path)
            
            # Enregistrer le modèle dans MLflow
            mlflow.stable_baselines3.log_model(
                sb3_model=model,
                artifact_path="model",
                registered_model_name="portfolio_rl_online_model"
            )
            
            # Enregistrer le modèle dans le feature store
            if feature_store:
                feature_store.save_feature(
                    name="online_model",
                    data=model,
                    entity_type="model",
                    version=f"run_{timestamp}",
                    description=f"Modèle ré-entraîné en ligne {timestamp}",
                    parameters={
                        "run_id": run_id,
                        "model_path": model_path,
                        "base_model": base_model_path
                    }
                )
            
            # Informations de l'entraînement
            training_info = {
                'log_dir': log_dir,
                'model_path': model_path,
                'mlflow_run_id': run_id,
                'timestamp': timestamp,
                'base_model': base_model_path
            }
            
            # Logging des métadonnées
            mlflow.log_dict(training_info, "training_info.json")
            
            return model, training_info
    
    except Exception as e:
        logger.error(f"Erreur lors du ré-entraînement en ligne: {e}")
        return None, None

if __name__ == "__main__":
    # Exemple d'utilisation
    config_path = "config/default.json"
    
    # Entraînement initial
    model, training_info = train_portfolio_agent(config_path)
    
    # Ré-entraînement en ligne
    if model is not None and training_info is not None:
        time.sleep(5)  # Attendre un peu pour simuler un intervalle
        
        # Simuler des nouvelles données (force_refresh=True générera de nouvelles features)
        model, online_info = online_retraining(
            config_path,
            base_model_path=training_info.get('model_path')
        )