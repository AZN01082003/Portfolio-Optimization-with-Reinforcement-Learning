'''
Pipeline d'entraînement corrigé pour le modèle d'allocation de portefeuille.
Compatible avec l'architecture existante et MLflow.
'''
import os
import sys
import time
import json
import logging
import numpy as np
from datetime import datetime
from typing import Dict, Any, Tuple, Optional

# Configuration du logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Ajouter le répertoire du projet au PYTHONPATH
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)

# Imports avec gestion d'erreurs
try:
    import mlflow
    from mlflow.tracking import MlflowClient
    MLFLOW_AVAILABLE = True
except ImportError as e:
    print(f"⚠️ MLflow non disponible: {e}")
    MLFLOW_AVAILABLE = False

try:
    from stable_baselines3 import PPO
    from stable_baselines3.common.callbacks import BaseCallback, EvalCallback
    from stable_baselines3.common.env_checker import check_env
    from stable_baselines3.common.monitor import Monitor
    SB3_AVAILABLE = True
except ImportError as e:
    print(f"❌ Stable-Baselines3 non disponible: {e}")
    SB3_AVAILABLE = False

try:
    from src.environment.portfolio_env import create_env_from_config
    ENV_AVAILABLE = True
except ImportError as e:
    print(f"❌ Environnement non disponible: {e}")
    ENV_AVAILABLE = False

def load_config(config_path: str = "config/default.json") -> dict:
    '''Charge la configuration depuis un fichier JSON.'''
    try:
        with open(config_path, 'r', encoding='utf-8') as file:
            config = json.load(file)
        return config
    except Exception as e:
        logger.error(f"Erreur lors du chargement de la configuration: {e}")
        raise

class MLflowTrainingCallback(BaseCallback):
    '''Callback personnalisé pour intégrer l'entraînement avec MLflow.'''
    
    def __init__(self, verbose: int = 0, check_freq: int = 1000, log_dir: str = "./logs/"):
        super(MLflowTrainingCallback, self).__init__(verbose)
        self.check_freq = check_freq
        self.log_dir = log_dir
        self.best_mean_reward = -np.inf
        self.start_time = time.time()
        
        # Métriques de suivi
        self.episode_rewards = []
        self.episode_values = []
        self.episode_returns = []
        
        # Créer le répertoire de log
        os.makedirs(log_dir, exist_ok=True)
    
    def _on_step(self) -> bool:
        '''Appelé à chaque step de l'entraînement.'''
        
        # Log périodique du temps
        if self.n_calls % 1000 == 0:
            elapsed = time.time() - self.start_time
            fps = self.n_calls / max(elapsed, 1)
            logger.info(f"Training step {self.n_calls:,}, elapsed: {elapsed:.1f}s, FPS: {fps:.1f}")
        
        # Récupérer les informations de l'épisode
        if self.locals.get("infos") and len(self.locals["infos"]) > 0:
            info = self.locals["infos"][0]
            reward = self.locals.get("rewards", [0])[0]
            
            # Stocker les métriques
            self.episode_rewards.append(reward)
            self.episode_values.append(info.get("portfolio_value", 0))
            self.episode_returns.append(info.get("return", 0))
            
            # Log périodique dans MLflow
            if self.n_calls % self.check_freq == 0 and len(self.episode_rewards) > 100:
                recent_rewards = self.episode_rewards[-100:]
                recent_returns = self.episode_returns[-100:]
                recent_values = self.episode_values[-100:]
                
                mean_reward = np.mean(recent_rewards)
                mean_return = np.mean(recent_returns) * 100
                mean_portfolio_value = np.mean(recent_values)
                volatility = np.std(recent_returns) * 100
                
                # Logger dans MLflow si disponible
                if MLFLOW_AVAILABLE:
                    try:
                        mlflow.log_metrics({
                            "train/mean_reward": float(mean_reward),
                            "train/mean_return": float(mean_return),
                            "train/portfolio_value": float(mean_portfolio_value),
                            "train/volatility": float(volatility),
                        }, step=self.n_calls)
                    except Exception as e:
                        logger.warning(f"Erreur MLflow logging: {e}")
                
                # Sauvegarder le meilleur modèle
                if mean_reward > self.best_mean_reward:
                    self.best_mean_reward = mean_reward
                    best_model_path = os.path.join(self.log_dir, "best_model")
                    if hasattr(self, 'model') and self.model:
                        self.model.save(best_model_path)
                        logger.info(f"💾 Nouveau meilleur modèle sauvegardé: reward={mean_reward:.4f}")
        
        return True
    
    def on_training_end(self) -> None:
        '''Appelé à la fin de l'entraînement.'''
        total_time = time.time() - self.start_time
        logger.info(f"🎯 ENTRAÎNEMENT TERMINÉ en {total_time:.1f}s")
        logger.info(f"🎖️ Meilleure récompense: {self.best_mean_reward:.4f}")

def setup_mlflow_experiment(experiment_name: str = "portfolio_optimization") -> str:
    '''Configure MLflow et retourne l'ID de l'expérience.'''
    
    if not MLFLOW_AVAILABLE:
        logger.warning("MLflow non disponible, pas de tracking")
        return "0"
    
    try:
        # Configuration de l'URI de tracking
        mlflow_uri = os.environ.get("MLFLOW_TRACKING_URI", "http://localhost:5000")
        mlflow.set_tracking_uri(mlflow_uri)
        
        # Créer ou récupérer l'expérience
        experiment = mlflow.get_experiment_by_name(experiment_name)
        if experiment:
            experiment_id = experiment.experiment_id
            logger.info(f"📋 Utilisation de l'expérience existante: {experiment_name}")
        else:
            experiment_id = mlflow.create_experiment(
                experiment_name,
                artifact_location=f"./artifacts/{experiment_name}"
            )
            logger.info(f"📋 Nouvelle expérience créée: {experiment_name}")
        
        return experiment_id
        
    except Exception as e:
        logger.error(f"Erreur MLflow: {e}")
        return "0"

def load_training_data(config_path: str) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    '''Charge les données d'entraînement et de test.'''
    
    config = load_config(config_path)
    output_dir = config['data']['output_dir']
    
    # Chemins des fichiers de données
    train_file = os.path.join(output_dir, "stock_data_train_latest.npy")
    test_file = os.path.join(output_dir, "stock_data_test_latest.npy")
    
    try:
        # Vérifier l'existence des fichiers
        if not os.path.exists(train_file):
            logger.error(f"Fichier d'entraînement non trouvé: {train_file}")
            return None, None
        
        if not os.path.exists(test_file):
            logger.error(f"Fichier de test non trouvé: {test_file}")
            return None, None
        
        # Charger les données
        train_data = np.load(train_file)
        test_data = np.load(test_file)
        
        logger.info(f"📊 Données d'entraînement chargées: {train_data.shape}")
        logger.info(f"📊 Données de test chargées: {test_data.shape}")
        
        return train_data, test_data
        
    except Exception as e:
        logger.error(f"Erreur lors du chargement des données: {e}")
        return None, None

def train_portfolio_agent(config_path: str = "config/default.json", 
                         run_name: Optional[str] = None) -> Tuple[Optional[Any], Optional[Dict]]:
    '''Entraîne un agent d'allocation de portefeuille avec MLflow.'''
    
    # Vérifier les dépendances
    if not SB3_AVAILABLE:
        logger.error("❌ Stable-Baselines3 non disponible")
        return None, None
    
    if not ENV_AVAILABLE:
        logger.error("❌ Environnement RL non disponible")
        return None, None
    
    logger.info("🚀 DÉMARRAGE DE L'ENTRAÎNEMENT")
    logger.info("="*50)
    
    try:
        # 1. Charger la configuration
        config = load_config(config_path)
        train_config = config.get('training', {})
        env_config = config.get('environment', {})
        
        # 2. Charger les données
        train_data, test_data = load_training_data(config_path)
        if train_data is None or test_data is None:
            return None, None
        
        # 3. Setup MLflow
        experiment_id = setup_mlflow_experiment("portfolio_optimization")
        
        # 4. Démarrer le run MLflow
        if run_name is None:
            run_name = f"training_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # Context manager pour MLflow
        if MLFLOW_AVAILABLE:
            mlflow_context = mlflow.start_run(experiment_id=experiment_id, run_name=run_name)
        else:
            from contextlib import nullcontext
            mlflow_context = nullcontext()
        
        with mlflow_context as run:
            if MLFLOW_AVAILABLE and run:
                run_id = run.info.run_id
                logger.info(f"📝 MLflow Run ID: {run_id}")
            else:
                run_id = "no_mlflow"
                logger.info("📝 Entraînement sans MLflow")
            
            # 5. Logger les paramètres
            params = {
                "algorithm": train_config.get('algorithm', 'PPO'),
                "learning_rate": train_config.get('learning_rate', 3e-4),
                "n_steps": train_config.get('n_steps', 1024),
                "batch_size": train_config.get('batch_size', 64),
                "gamma": train_config.get('gamma', 0.99),
                "total_timesteps": train_config.get('total_timesteps', 50000),
                "portfolio_value": env_config.get('portfolio_value', 10000),
                "window_size": env_config.get('window_size', 30),
                "trans_cost": env_config.get('trans_cost', 0.0005),
                "train_data_shape": str(train_data.shape),
                "test_data_shape": str(test_data.shape)
            }
            
            if MLFLOW_AVAILABLE:
                try:
                    mlflow.log_params(params)
                except Exception as e:
                    logger.warning(f"Erreur log params: {e}")
            
            # 6. Créer l'environnement
            logger.info("🏗️ Création de l'environnement...")
            env = create_env_from_config(train_data, config_path)
            
            # Créer le répertoire de logs
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            log_dir = os.path.join(config.get('training', {}).get('log_dir', './logs'), f"training_{timestamp}")
            os.makedirs(log_dir, exist_ok=True)
            
            # Wrapper Monitor
            env = Monitor(env, log_dir)
            
            # Vérifier l'environnement
            logger.info("✅ Vérification de l'environnement...")
            check_env(env, warn=True)
            
            # 7. Créer le modèle
            logger.info("🤖 Création du modèle PPO...")
            model = PPO(
                "MultiInputPolicy",
                env,
                verbose=1,
                learning_rate=params["learning_rate"],
                n_steps=params["n_steps"],
                batch_size=params["batch_size"],
                gamma=params["gamma"],
                ent_coef=train_config.get('ent_coef', 0.01),
                clip_range=train_config.get('clip_range', 0.2),
                max_grad_norm=train_config.get('max_grad_norm', 0.5),
                tensorboard_log=log_dir
            )
            
            # 8. Créer les callbacks
            training_callback = MLflowTrainingCallback(
                verbose=1,
                check_freq=1000,
                log_dir=log_dir
            )
            training_callback.model = model
            
            # 9. Entraîner le modèle
            logger.info("🎯 DÉBUT DE L'ENTRAÎNEMENT...")
            logger.info(f"   Timesteps total: {params['total_timesteps']:,}")
            
            start_time = time.time()
            
            model.learn(
                total_timesteps=params["total_timesteps"],
                callback=[training_callback],
                progress_bar=True
            )
            
            training_time = time.time() - start_time
            logger.info(f"✅ Entraînement terminé en {training_time:.1f}s")
            
            # 10. Sauvegarder le modèle
            final_model_path = os.path.join(log_dir, "final_model")
            model.save(final_model_path)
            
            # 11. Informations de retour
            training_info = {
                'run_id': run_id,
                'log_dir': log_dir,
                'final_model_path': final_model_path,
                'best_model_path': os.path.join(log_dir, "best_model"),
                'training_time': training_time,
                'best_reward': training_callback.best_mean_reward,
                'total_timesteps': params["total_timesteps"]
            }
            
            logger.info("🎉 ENTRAÎNEMENT RÉUSSI!")
            logger.info(f"📍 Modèle sauvegardé: {final_model_path}")
            logger.info(f"🏆 Meilleure récompense: {training_callback.best_mean_reward:.4f}")
            
            return model, training_info
            
    except Exception as e:
        logger.error(f"❌ Erreur lors de l'entraînement: {e}")
        import traceback
        traceback.print_exc()
        return None, None

def evaluate_trained_model(model_path: str, config_path: str = "config/default.json") -> Dict[str, float]:
    '''Évalue un modèle entraîné sur les données de test.'''
    
    logger.info("📊 ÉVALUATION DU MODÈLE")
    
    try:
        # Charger le modèle
        model = PPO.load(model_path)
        logger.info(f"✅ Modèle chargé: {model_path}")
        
        # Charger les données de test
        _, test_data = load_training_data(config_path)
        if test_data is None:
            raise ValueError("Données de test non disponibles")
        
        # Créer l'environnement de test
        test_env = create_env_from_config(test_data, config_path)
        
        # Évaluation
        obs, info = test_env.reset()
        total_rewards = []
        portfolio_values = []
        episode_returns = []
        
        for step in range(min(100, test_data.shape[2] - test_env.window_size)):
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = test_env.step(action)
            
            total_rewards.append(reward)
            portfolio_values.append(info['portfolio_value'])
            episode_returns.append(info['return'])
            
            if terminated or truncated:
                break
        
        # Calculer les métriques
        metrics = {
            'mean_reward': float(np.mean(total_rewards)),
            'total_return': float((portfolio_values[-1] / portfolio_values[0] - 1) * 100),
            'final_value': float(portfolio_values[-1]),
            'volatility': float(np.std(episode_returns) * 100),
            'sharpe_ratio': float(np.mean(episode_returns) / max(np.std(episode_returns), 1e-6))
        }
        
        logger.info("📈 MÉTRIQUES D'ÉVALUATION:")
        for key, value in metrics.items():
            logger.info(f"   {key}: {value:.4f}")
        
        return metrics
        
    except Exception as e:
        logger.error(f"❌ Erreur lors de l'évaluation: {e}")
        return {}

if __name__ == "__main__":
    # Script principal
    import argparse
    
    parser = argparse.ArgumentParser(description="Entraînement du modèle de portfolio")
    parser.add_argument("--config", type=str, default="config/default.json", 
                       help="Chemin du fichier de configuration")
    parser.add_argument("--run-name", type=str, default=None,
                       help="Nom du run MLflow")
    parser.add_argument("--evaluate", type=str, default=None,
                       help="Chemin du modèle à évaluer")
    
    args = parser.parse_args()
    
    if args.evaluate:
        # Mode évaluation
        metrics = evaluate_trained_model(args.evaluate, args.config)
        print(f"✅ Évaluation terminée: {metrics}")
    else:
        # Mode entraînement
        model, info = train_portfolio_agent(args.config, args.run_name)
        
        if model is not None:
            print("✅ Entraînement réussi!")
            print(f"📍 Modèle: {info['final_model_path']}")
            print(f"🏆 MLflow Run: {info['run_id']}")
        else:
            print("❌ Entraînement échoué!")
            sys.exit(1)
