# Environnement RL (bas� sur votre new_env.py)
#Module définissant l'environnement d'apprentissage par renforcement pour l'allocation de portefeuille.

import numpy as np
import gymnasium as gym
from gymnasium import spaces
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

class PortfolioEnv(gym.Env):
    """
    Environnement compatible Gymnasium pour l'apprentissage par renforcement.
    Cet environnement simule l'allocation dynamique d'un portefeuille d'actions.
    """
    
    metadata = {"render_modes": ["human"]}
    
    def __init__(self, data, portfolio_value=10000, trans_cost=0.001, return_rate=0.0001, window_size=30,
                reward_scaling=100.0, max_reward_clip=10.0, min_reward_clip=-10.0):
        """
        Initialise l'environnement de portfolio.
        
        Args:
            data: Données financières de forme (n_features, n_stocks, n_time_periods)
            portfolio_value: Valeur initiale du portefeuille
            trans_cost: Coût de transaction (pourcentage de la valeur échangée)
            return_rate: Taux de rendement sans risque
            window_size: Taille de la fenêtre d'observation
            reward_scaling: Facteur de mise à l'échelle des récompenses
            max_reward_clip: Valeur maximale de clip pour les récompenses
            min_reward_clip: Valeur minimale de clip pour les récompenses
        """
        super(PortfolioEnv, self).__init__()
        
        # Données financières
        self.data = data
        self.n_features, self.n_stocks, self.n_time_periods = data.shape
        
        # Paramètres
        self.initial_portfolio_value = portfolio_value
        self.portfolio_value = portfolio_value
        self.trans_cost = trans_cost
        self.return_rate = return_rate
        self.window_size = window_size
        
        # Paramètres de stabilisation des récompenses
        self.reward_scaling = reward_scaling
        self.max_reward_clip = max_reward_clip
        self.min_reward_clip = min_reward_clip
        
        # État interne
        self.current_step = self.window_size
        self.done = False
        
        # Définir l'espace d'action (vecteur de poids entre 0 et 1 pour chaque action)
        self.action_space = spaces.Box(low=0, high=1, shape=(self.n_stocks,), dtype=np.float32)
        
        # Définir l'espace d'observation (fenêtre temporelle, poids actuels, valeur portefeuille)
        obs_shape = (self.n_features, self.n_stocks, self.window_size)
        weight_shape = (self.n_stocks,)
        self.observation_space = spaces.Dict({
            'market_data': spaces.Box(low=-np.inf, high=np.inf, shape=obs_shape, dtype=np.float32),
            'weights': spaces.Box(low=0, high=1, shape=weight_shape, dtype=np.float32),
            'portfolio_value': spaces.Box(low=0, high=np.inf, shape=(1,), dtype=np.float32)
        })
        
        # Initialiser l'état
        self.weights = np.ones(self.n_stocks, dtype=np.float32) / self.n_stocks  # Allocation égale
        self.portfolio_value = portfolio_value
        
        # Variables pour le suivi des performances
        self.episode_returns = []
        self.max_portfolio_value = portfolio_value
        
        logger.info(f"Environnement PortfolioEnv initialisé avec {self.n_stocks} actions et {self.n_time_periods} périodes")
    
    def get_open_values(self, step):
        """
        Obtenir les valeurs d'ouverture pour le pas de temps actuel.
        
        Args:
            step: Pas de temps actuel
            
        Returns:
            ndarray: Valeurs relatives des prix d'ouverture
        """
        # Ensure we're not going out of bounds
        if step + 1 >= self.n_time_periods:
            return np.array([1.0] * self.n_stocks, dtype=np.float32)  # Default return for final step
        
        # Calculate relative price changes (returns) for each stock
        # Add safety checks against division by zero or negative values
        returns = []
        for i in range(self.n_stocks):
            current_price = max(self.data[0, i, step], 1e-8)  # Prevent division by zero
            next_price = max(self.data[0, i, step+1], 1e-8)  # Ensure positive price
            returns.append(next_price / current_price)
            
        return np.array(returns, dtype=np.float32)
    
    def normalize_data(self, data):
        """
        Normalize data to improve training stability.
        
        Args:
            data: Données à normaliser
            
        Returns:
            ndarray: Données normalisées
        """
        if np.std(data) > 0:
            # Z-score normalization
            return (data - np.mean(data)) / np.std(data)
        return data  # Return original data if std is 0
    
    def get_window_data(self, step):
        """
        Obtenir les données pour la fenêtre temporelle actuelle.
        
        Args:
            step: Pas de temps actuel
            
        Returns:
            ndarray: Données de la fenêtre temporelle
        """
        # Ensure we don't go out of bounds
        end_step = min(step, self.n_time_periods)
        start_step = max(end_step - self.window_size, 0)
        
        # If we don't have enough data for a full window, pad with zeros
        if end_step - start_step < self.window_size:
            padding_size = self.window_size - (end_step - start_step)
            padded_data = np.zeros((self.n_features, self.n_stocks, self.window_size), dtype=np.float32)
            padded_data[:, :, padding_size:] = self.data[:, :, start_step:end_step]
            window_data = padded_data
        else:
            window_data = self.data[:, :, start_step:end_step].copy()
        
        # Normalize each feature individually for better stability
        for f in range(self.n_features):
            for s in range(self.n_stocks):
                window_data[f, s, :] = self.normalize_data(window_data[f, s, :])
        
        return window_data.astype(np.float32)
    
    def _get_observation(self):
        """
        Construire l'observation complète.
        
        Returns:
            dict: Observation complète
        """
        return {
            'market_data': self.get_window_data(self.current_step),
            'weights': self.weights.astype(np.float32),
            'portfolio_value': np.array([self.portfolio_value / self.initial_portfolio_value], dtype=np.float32)  # Normalize value
        }
    
    def reset(self, seed=None, options=None):
        """
        Réinitialiser l'environnement.
        
        Args:
            seed: Graine aléatoire
            options: Options supplémentaires
            
        Returns:
            tuple: (observation, info)
        """
        # Required for gymnasium API
        super().reset(seed=seed)
        
        self.current_step = self.window_size
        self.done = False
        self.weights = np.ones(self.n_stocks, dtype=np.float32) / self.n_stocks
        self.portfolio_value = self.initial_portfolio_value
        self.max_portfolio_value = self.initial_portfolio_value
        
        # Reset episode tracking
        self.episode_returns = []
        
        info = {'portfolio_value': float(self.portfolio_value)}
        return self._get_observation(), info
    
    def step(self, action):
        """
        Effectuer une étape dans l'environnement.
        
        Args:
            action: Action à effectuer (poids du portefeuille)
            
        Returns:
            tuple: (observation, reward, terminated, truncated, info)
        """
        # Save previous state for reward calculation
        old_portfolio_value = self.portfolio_value
        
        # Handle NaN or infinite values in action
        if np.isnan(action).any() or np.isinf(action).any():
            logger.warning("NaN or inf in action, using equal weights instead")
            action = np.ones(self.n_stocks, dtype=np.float32) / self.n_stocks
        else:
            # Ensure action values are positive (clamp if necessary)
            action = np.maximum(action, 0.0)
            
            # Normalize action to sum to 1, with safety checks
            action_sum = np.sum(action)
            if action_sum > 0:
                action = action / action_sum
            else:
                # If all actions are zero, use equal weights
                action = np.ones(self.n_stocks, dtype=np.float32) / self.n_stocks
        
        # Obtenir les valeurs d'ouverture pour le pas de temps actuel
        open_values = self.get_open_values(self.current_step)
        
        # Calculer le coût de transaction (clipped to avoid extreme costs)
        trans_cost_factor = min(np.sum(np.abs(action - self.weights)), 2.0)  # Limit to 200% turnover
        transaction_cost = self.portfolio_value * self.trans_cost * trans_cost_factor
        
        # Mettre à jour le portefeuille
        value_before_transaction = self.portfolio_value * action
        new_values = value_before_transaction * open_values
        new_values_sum = np.sum(new_values)
        
        # Soustraire le coût de transaction du portfolio total
        new_portfolio_value = new_values_sum - transaction_cost
        
        # Calculer les nouveaux poids avec validation
        if new_values_sum > 0:
            new_weights = new_values / new_values_sum
        else:
            # Fallback to equal weights if sum is zero or negative
            new_weights = np.ones(self.n_stocks, dtype=np.float32) / self.n_stocks
        
        # Update max portfolio value for drawdown calculation
        self.max_portfolio_value = max(self.max_portfolio_value, new_portfolio_value)
        
        # Calculate raw return
        portfolio_return = (new_portfolio_value / old_portfolio_value) - 1
        
        # Track returns for this episode
        self.episode_returns.append(portfolio_return)
        
        # Calculate Sharpe-like reward (return / volatility)
        if len(self.episode_returns) > 1:
            returns_std = max(np.std(self.episode_returns), 1e-6)  # Prevent division by zero
            sharpe_factor = portfolio_return / returns_std
        else:
            sharpe_factor = portfolio_return
        
        # Calculate drawdown penalty
        drawdown = (self.max_portfolio_value - new_portfolio_value) / self.max_portfolio_value
        drawdown_penalty = -drawdown * 0.5  # Penalize drawdowns
        
        # Calculate final reward with scaling and clipping for stability
        reward = portfolio_return * self.reward_scaling + drawdown_penalty
        reward = max(min(reward, self.max_reward_clip), self.min_reward_clip)  # Clip reward
        reward = float(reward)  # Ensure it's a float
        
        # Mettre à jour l'état
        self.portfolio_value = max(new_portfolio_value, 1e-8)  # Prevent zero portfolio value
        self.weights = new_weights
        self.current_step += 1
        
        # Vérifier si l'épisode est terminé
        if self.current_step >= self.n_time_periods - 1 or self.portfolio_value < self.initial_portfolio_value * 0.5:
            # End episode if we reach the end of data or lose half the initial investment
            self.done = True
            
        # Gymnasium requires five return values: obs, reward, terminated, truncated, info
        terminated = self.done
        truncated = False
        
        info = {
            'portfolio_value': float(self.portfolio_value),
            'weights': self.weights.tolist(),  # Convert to list for easier logging
            'transaction_cost': float(transaction_cost),
            'return': float(portfolio_return),
            'drawdown': float(drawdown),
            'max_portfolio_value': float(self.max_portfolio_value)
        }
        
        return self._get_observation(), reward, terminated, truncated, info
    
    def render(self, mode='human'):
        """
        Visualiser l'état actuel (facultatif).
        """
        pass

def create_env_from_config(data, config_path="config/default.json"):
    """
    Crée un environnement PortfolioEnv à partir d'un fichier de configuration.
    
    Args:
        data: Données financières
        config_path: Chemin du fichier de configuration
        
    Returns:
        PortfolioEnv: Environnement initialisé
    """
    config = load_config(config_path)
    env_config = config['environment']
    
    env = PortfolioEnv(
        data=data,
        portfolio_value=env_config['portfolio_value'],
        window_size=env_config['window_size'],
        trans_cost=env_config['trans_cost'],
        return_rate=env_config['return_rate'],
        reward_scaling=env_config['reward_scaling'],
        max_reward_clip=env_config['max_reward_clip'],
        min_reward_clip=env_config['min_reward_clip']
    )
    
    return env