# Entra�nement (bas� sur votre train.py)

"""
Module pour l'entraînement d'un agent d'allocation de portefeuille.
"""

# Ajouter ces imports au début du fichier
import mlflow
from mlflow.tracking import MlflowClient
from mlflow.models.signature import infer_signature
import json

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

#def train_portfolio_agent(config_path="config/default.json", data_path=None):
    """
    Entraîne un agent d'allocation de portefeuille.
    
    Args:
        config_path: Chemin du fichier de configuration
        data_path: Chemin vers les données d'entraînement (optionnel)
        
    Returns:
        tuple: (model, training_info)
    """
    # Charger la configuration
    config = load_config(config_path)
    
    # Charger les données prétraitées
    if data_path is None:
        data_path = os.path.join(config['data']['output_dir'], "stock_data_train_latest.npy")
    
    logger.info(f"Chargement des données depuis {data_path}...")
    data = np.load(data_path)
    logger.info(f"Données chargées avec succès. Forme: {data.shape}")
    
    # Importations pour l'environnement et l'entraînement
    try:
        from stable_baselines3 import PPO
        from stable_baselines3.common.env_checker import check_env
        from stable_baselines3.common.callbacks import BaseCallback
        from stable_baselines3.common.monitor import Monitor
        

        from portfolio_env import create_env_from_config
        
        
        class PortfolioCallback(BaseCallback):
            """Callback personnalisé pour suivre la progression de l'entraînement"""
            def __init__(self, verbose=0, check_freq=1000, log_dir="./logs/"):
                super(PortfolioCallback, self).__init__(verbose)
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
                        
                        # Check for training improvement
                        if len(self.rewards) > 100:
                            mean_reward = np.mean(self.rewards[-100:])
                            
                            if mean_reward > self.best_mean_reward:
                                self.best_mean_reward = mean_reward
                                self.last_improvement_step = self.n_calls
                                self.model.save(f"{self.log_dir}/best_model")
                                logger.info(f"Saving new best model with mean reward: {mean_reward:.5f}")
                                self.no_improvement_count = 0
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
                plt.plot(self.timesteps, self.returns)
                plt.title("Returns")
                plt.xlabel("Steps")
                plt.ylabel("Return (%)")
                plt.grid(True)
                
                # Drawdowns plot
                plt.subplot(2, 2, 4)
                plt.plot(self.timesteps, self.drawdowns)
                plt.title("Drawdowns")
                plt.xlabel("Steps")
                plt.ylabel("Drawdown (%)")
                plt.grid(True)
                
                plt.tight_layout()
                plt.savefig(f"{self.log_dir}/training_progress_{self.n_calls}.png")
                plt.close()
                
                # Also save data for later analysis
                np.save(f"{self.log_dir}/training_data.npy", {
                    'timesteps': self.timesteps,
                    'portfolio_values': self.portfolio_values,
                    'rewards': self.rewards,
                    'returns': self.returns,
                    'drawdowns': self.drawdowns
                })
        
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
        
        # Créer l'environnement
        env = create_env_from_config(data, config_path)
        
        # Créer le répertoire de log
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        log_dir = os.path.join(config['training']['log_dir'], f"portfolio_training_{timestamp}")
        os.makedirs(log_dir, exist_ok=True)
        
        # Wrapper l'environnement dans Monitor pour le logging
        env = Monitor(env, log_dir)
        
        # Vérifier que l'environnement est compatible avec Gym
        logger.info("Vérification de l'environnement...")
        check_env(env)
        logger.info("L'environnement est valide!")
        
        # Extraire les paramètres d'entraînement
        train_config = config['training']
        
        # Créer le modèle avec les hyperparamètres de la configuration
        logger.info("Création du modèle PPO...")
        model = PPO("MultiInputPolicy", env, verbose=1, 
                  learning_rate=train_config['learning_rate'],
                  n_steps=train_config['n_steps'],
                  batch_size=train_config['batch_size'],
                  ent_coef=train_config['ent_coef'],
                  clip_range=train_config['clip_range'],
                  max_grad_norm=train_config['max_grad_norm'],
                  gae_lambda=train_config['gae_lambda'],
                  gamma=train_config['gamma'],
                  n_epochs=train_config['n_epochs'],
                  tensorboard_log=log_dir)
        
        # Créer le callback
        callback = PortfolioCallback(verbose=1, check_freq=1000, log_dir=log_dir)
        callback.model = model  # Give callback access to model for saving
        
        # Entraîner le modèle
        logger.info("Démarrage de l'entraînement...")
        try:
            model.learn(total_timesteps=train_config['total_timesteps'], callback=callback, progress_bar=True)
            logger.info("Entraînement terminé!")
        except Exception as e:
            logger.error(f"Erreur pendant l'entraînement: {e}")
            # Load best model if available
            if os.path.exists(f"{log_dir}/best_model.zip"):
                logger.info("Chargement du meilleur modèle sauvegardé...")
                model = PPO.load(f"{log_dir}/best_model.zip")
        
        # Sauvegarder le modèle final
        model_path = f"{log_dir}/final_model"
        model.save(model_path)
        logger.info(f"Modèle sauvegardé dans {model_path}")
        
        # Retourner le modèle et les informations d'entraînement
        training_info = {
            'log_dir': log_dir,
            'model_path': model_path,
            'best_model_path': f"{log_dir}/best_model" if os.path.exists(f"{log_dir}/best_model.zip") else None
        }
        
        return model, training_info
        
    except ImportError as e:
        logger.error(f"ERREUR: {e}")
        logger.error("Veuillez installer les dépendances requises:")
        logger.error("pip install stable-baselines3[extra] gymnasium matplotlib")
        return None, None
def train_portfolio_agent(config_path="config/default.json", data_path=None):
    """
    Entraîne un agent d'allocation de portefeuille avec suivi MLflow.
    
    Args:
        config_path: Chemin du fichier de configuration
        data_path: Chemin vers les données d'entraînement (optionnel)
        
    Returns:
        tuple: (model, training_info)
    """
    # Code existant pour charger la configuration et les données...

    try:
        from stable_baselines3 import PPO
        from stable_baselines3.common.env_checker import check_env
        from stable_baselines3.common.callbacks import BaseCallback
        from stable_baselines3.common.monitor import Monitor
        
        from src.environment.portfolio_env import create_env_from_config
        
        # Configuration de MLflow
        mlflow_tracking_uri = os.environ.get("MLFLOW_TRACKING_URI", "http://localhost:5000")
        mlflow.set_tracking_uri(mlflow_tracking_uri)
        experiment_name = "portfolio_optimization"
        
        # Créer l'expérience si elle n'existe pas
        try:
            experiment_id = mlflow.create_experiment(
                experiment_name,
                artifact_location=os.path.join("artifacts", experiment_name)
            )
        except mlflow.exceptions.MlflowException:
            experiment_id = mlflow.get_experiment_by_name(experiment_name).experiment_id
        
        # Démarrer le run MLflow
        with mlflow.start_run(experiment_id=experiment_id, run_name=f"run_{time.strftime('%Y%m%d-%H%M%S')}") as run:
            # Enregistrer les paramètres de configuration
            train_config = config['training']
            env_config = config['environment']
            
            # Paramètres à enregistrer
            mlflow.log_params({
                "algorithm": train_config['algorithm'],
                "learning_rate": train_config['learning_rate'],
                "n_steps": train_config['n_steps'],
                "batch_size": train_config['batch_size'],
                "gamma": train_config['gamma'],
                "portfolio_value": env_config['portfolio_value'],
                "window_size": env_config['window_size'],
                "trans_cost": env_config['trans_cost'],
                "data_shape": str(data.shape)
            })
            
            # Intégration dans la classe PortfolioCallback
            class PortfolioCallback(BaseCallback):
                """Callback personnalisé pour suivre la progression de l'entraînement avec MLflow"""
                def __init__(self, verbose=0, check_freq=1000, log_dir="./logs/"):
                    super(PortfolioCallback, self).__init__(verbose)
                    # Votre code existant...
                    
                def _on_step(self):
                    # Votre code existant...
                    
                    # Ajout: Logging des métriques dans MLflow
                    if self.n_calls % 1000 == 0:
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
                    
                    # Reste de votre code existant...
                    return True
                
                def on_training_end(self):
                    # Votre code existant...
                    
                    # Ajout: Enregistrer les métriques finales
                    if len(self.portfolio_values) > 0:
                        final_value = self.portfolio_values[-1]
                        initial_value = self.portfolio_values[0]
                        total_return = (final_value / initial_value - 1) * 100
                        
                        mlflow.log_metrics({
                            "final_portfolio_value": final_value,
                            "total_return": total_return
                        })
            
            # Le reste de votre code pour créer l'environnement, le modèle et l'entraîner...
            
            # Enregistrer le modèle dans MLflow
            # Utilisons un fichier temporaire pour sauvegarder d'abord le modèle
            mlflow.stable_baselines3.log_model(
                sb3_model=model,
                artifact_path="model",
                registered_model_name="portfolio_rl_model"
            )
            
            # Enregistrer aussi les graphiques générés
            if os.path.exists(f"{log_dir}/training_progress_{callback.n_calls}.png"):
                mlflow.log_artifact(f"{log_dir}/training_progress_{callback.n_calls}.png", "plots")
            
            # Enregistrer l'ID du run pour référence
            run_id = run.info.run_id
            
            # Ajouter l'ID du run dans les informations d'entraînement
            training_info['mlflow_run_id'] = run_id
            
            # Retourner le modèle et les informations d'entraînement
            return model, training_info
        
    except ImportError as e:
        logger.error(f"ERREUR: {e}")
        logger.error("Veuillez installer les dépendances requises.")
        return None, None
    

def main(config_path="config/default.json"):
    """
    Fonction principale pour l'entraînement d'un agent.
    """
    train_portfolio_agent(config_path)

if __name__ == "__main__":
    main()