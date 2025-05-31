"""
Environnement RL corrigé pour l'allocation de portefeuille.
Compatibilité Gymnasium, gestion d'erreurs robuste, et performances optimisées.
"""
import numpy as np
import gymnasium as gym
from gymnasium import spaces
import logging
import json
from typing import Dict, Any, Tuple, Optional, Union

# Configuration du logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def load_config(config_path: str = "config/default.json") -> dict:
    """
    Charge la configuration depuis un fichier json.
    
    Args:
        config_path: Chemin du fichier de configuration
        
    Returns:
        dict: Configuration chargée
    """
    try:
        with open(config_path, 'r', encoding='utf-8') as file:
            config = json.load(file)
        return config
    except Exception as e:
        logger.error(f"Erreur lors du chargement de la configuration: {e}")
        raise

class PortfolioEnv(gym.Env):
    """
    Environnement Gymnasium pour l'apprentissage par renforcement d'allocation de portefeuille.
    
    Espace d'observation:
        - market_data: Données de marché historiques (fenêtre glissante)
        - weights: Poids actuels du portefeuille
        - portfolio_value: Valeur normalisée du portefeuille
        
    Espace d'action:
        - Vecteur de poids pour chaque actif (normalisé automatiquement)
    """
    
    metadata = {"render_modes": ["human", "rgb_array"]}
    
    def __init__(self, 
                 data: np.ndarray,
                 portfolio_value: float = 10000,
                 trans_cost: float = 0.001,
                 return_rate: float = 0.0001,
                 window_size: int = 30,
                 reward_scaling: float = 100.0,
                 max_reward_clip: float = 10.0,
                 min_reward_clip: float = -10.0,
                 normalize_observations: bool = True,
                 risk_penalty: float = 0.1):
        """
        Initialise l'environnement de portfolio.
        
        Args:
            data: Données financières (n_features, n_stocks, n_time_periods)
            portfolio_value: Valeur initiale du portefeuille
            trans_cost: Coût de transaction (pourcentage)
            return_rate: Taux de rendement sans risque
            window_size: Taille de la fenêtre d'observation
            reward_scaling: Facteur d'échelle pour les récompenses
            max_reward_clip: Récompense maximale
            min_reward_clip: Récompense minimale
            normalize_observations: Normaliser les observations
            risk_penalty: Pénalité pour risque (volatilité)
        """
        super().__init__()
        
        # Validation des données d'entrée
        self._validate_data(data)
        
        # Données financières
        self.data = data.astype(np.float32)
        self.n_features, self.n_stocks, self.n_time_periods = data.shape
        
        # Paramètres d'environnement
        self.initial_portfolio_value = float(portfolio_value)
        self.trans_cost = float(trans_cost)
        self.return_rate = float(return_rate)
        self.window_size = int(window_size)
        self.normalize_observations = normalize_observations
        self.risk_penalty = float(risk_penalty)
        
        # Paramètres de récompense
        self.reward_scaling = float(reward_scaling)
        self.max_reward_clip = float(max_reward_clip)
        self.min_reward_clip = float(min_reward_clip)
        
        # Validation des paramètres
        assert self.window_size <= self.n_time_periods, f"window_size ({self.window_size}) > n_time_periods ({self.n_time_periods})"
        assert self.trans_cost >= 0, "trans_cost doit être positif"
        assert self.initial_portfolio_value > 0, "portfolio_value doit être positif"
        
        # Espaces d'action et d'observation
        self._setup_spaces()
        
        # État interne
        self.reset()
        
        logger.info(f"PortfolioEnv initialisé: {self.n_stocks} actions, {self.n_time_periods} périodes, fenêtre={self.window_size}")
    
    def _validate_data(self, data: np.ndarray) -> None:
        """Valide les données d'entrée."""
        if not isinstance(data, np.ndarray):
            raise TypeError("Les données doivent être un numpy array")
        
        if data.ndim != 3:
            raise ValueError(f"Les données doivent être 3D (n_features, n_stocks, n_time), obtenu: {data.shape}")
        
        if data.size == 0:
            raise ValueError("Les données ne peuvent pas être vides")
        
        if np.isnan(data).any():
            logger.warning("Données contiennent des NaN, remplacement par interpolation")
            # Remplacement simple des NaN
            for i in range(data.shape[0]):
                for j in range(data.shape[1]):
                    series = data[i, j, :]
                    if np.isnan(series).any():
                        # Interpolation linéaire simple
                        valid_indices = ~np.isnan(series)
                        if valid_indices.any():
                            first_valid = np.where(valid_indices)[0][0]
                            last_valid = np.where(valid_indices)[0][-1]
                            series[:first_valid] = series[first_valid]
                            series[last_valid+1:] = series[last_valid]
                            # Interpolation pour le milieu
                            for k in range(first_valid, last_valid):
                                if np.isnan(series[k]):
                                    # Trouver le prochain point valide
                                    next_valid = k + 1
                                    while next_valid <= last_valid and np.isnan(series[next_valid]):
                                        next_valid += 1
                                    if next_valid <= last_valid:
                                        # Interpolation linéaire
                                        ratio = (k - first_valid) / (next_valid - first_valid)
                                        series[k] = series[first_valid] * (1 - ratio) + series[next_valid] * ratio
                        else:
                            # Si toute la série est NaN, utiliser 1.0
                            series[:] = 1.0
                        data[i, j, :] = series
    
    def _setup_spaces(self) -> None:
        """Configure les espaces d'action et d'observation."""
        # Espace d'action: poids du portefeuille (seront normalisés)
        self.action_space = spaces.Box(
            low=0.0, 
            high=1.0, 
            shape=(self.n_stocks,), 
            dtype=np.float32
        )
        
        # Espace d'observation: dictionnaire avec données de marché, poids et valeur
        self.observation_space = spaces.Dict({
            'market_data': spaces.Box(
                low=-np.inf, 
                high=np.inf, 
                shape=(self.n_features, self.n_stocks, self.window_size), 
                dtype=np.float32
            ),
            'weights': spaces.Box(
                low=0.0, 
                high=1.0, 
                shape=(self.n_stocks,), 
                dtype=np.float32
            ),
            'portfolio_value': spaces.Box(
                low=0.0, 
                high=np.inf, 
                shape=(1,), 
                dtype=np.float32
            )
        })
    
    def reset(self, seed: Optional[int] = None, options: Optional[Dict] = None) -> Tuple[Dict[str, np.ndarray], Dict[str, Any]]:
        """
        Réinitialise l'environnement.
        
        Args:
            seed: Graine aléatoire
            options: Options supplémentaires
            
        Returns:
            Tuple[observation, info]
        """
        # Initialiser le générateur aléatoire si seed fourni
        if seed is not None:
            super().reset(seed=seed)
            np.random.seed(seed)
        
        # Réinitialiser l'état
        self.current_step = self.window_size
        self.portfolio_value = self.initial_portfolio_value
        self.weights = np.ones(self.n_stocks, dtype=np.float32) / self.n_stocks
        self.max_portfolio_value = self.initial_portfolio_value
        
        # Historique pour calculs de performance
        self.episode_returns = []
        self.episode_values = [self.initial_portfolio_value]
        self.transaction_costs = []
        
        # Informations de debug
        info = {
            'portfolio_value': float(self.portfolio_value),
            'weights': self.weights.copy(),
            'step': self.current_step
        }
        
        return self._get_observation(), info
    
    def _get_observation(self) -> Dict[str, np.ndarray]:
        """
        Construit l'observation actuelle.
        
        Returns:
            Dict contenant l'observation
        """
        # Obtenir les données de marché pour la fenêtre actuelle
        market_data = self._get_window_data(self.current_step)
        
        # Normaliser la valeur du portefeuille
        normalized_value = np.array([self.portfolio_value / self.initial_portfolio_value], dtype=np.float32)
        
        return {
            'market_data': market_data,
            'weights': self.weights.copy(),
            'portfolio_value': normalized_value
        }
    
    def _get_window_data(self, step: int) -> np.ndarray:
        """
        Obtient les données de marché pour la fenêtre temporelle.
        
        Args:
            step: Pas de temps actuel
            
        Returns:
            Données de la fenêtre (n_features, n_stocks, window_size)
        """
        # Calculer les indices de début et fin
        end_step = min(step, self.n_time_periods)
        start_step = max(end_step - self.window_size, 0)
        
        # Extraire les données
        if end_step - start_step < self.window_size:
            # Padding si nécessaire
            window_data = np.zeros((self.n_features, self.n_stocks, self.window_size), dtype=np.float32)
            actual_data = self.data[:, :, start_step:end_step]
            window_data[:, :, -actual_data.shape[2]:] = actual_data
        else:
            window_data = self.data[:, :, start_step:end_step].copy()
        
        # Normalisation optionnelle
        if self.normalize_observations:
            window_data = self._normalize_window_data(window_data)
        
        return window_data.astype(np.float32)
    
    def _normalize_window_data(self, data: np.ndarray) -> np.ndarray:
        """
        Normalise les données de la fenêtre.
        
        Args:
            data: Données à normaliser
            
        Returns:
            Données normalisées
        """
        normalized_data = data.copy()
        
        for feature_idx in range(self.n_features):
            for stock_idx in range(self.n_stocks):
                series = normalized_data[feature_idx, stock_idx, :]
                
                # Éviter la division par zéro
                if np.std(series) > 1e-8:
                    # Z-score normalization
                    normalized_data[feature_idx, stock_idx, :] = (series - np.mean(series)) / np.std(series)
                else:
                    # Si pas de variation, centrer à zéro
                    normalized_data[feature_idx, stock_idx, :] = 0.0
        
        return normalized_data
    
    def _get_price_changes(self, step: int) -> np.ndarray:
        """
        Calcule les changements de prix pour le pas de temps suivant.
        
        Args:
            step: Pas de temps actuel
            
        Returns:
            Facteurs de changement de prix (multiplicatifs)
        """
        if step + 1 >= self.n_time_periods:
            # Dernier pas: pas de changement
            return np.ones(self.n_stocks, dtype=np.float32)
        
        # Utiliser les prix de clôture (indice 3 généralement)
        close_price_idx = min(3, self.n_features - 1)
        
        current_prices = self.data[close_price_idx, :, step]
        next_prices = self.data[close_price_idx, :, step + 1]
        
        # Calculer les facteurs de changement avec protection contre division par zéro
        price_changes = np.ones(self.n_stocks, dtype=np.float32)
        
        for i in range(self.n_stocks):
            if current_prices[i] > 1e-8:  # Éviter division par zéro
                price_changes[i] = next_prices[i] / current_prices[i]
            else:
                price_changes[i] = 1.0  # Pas de changement si prix invalide
        
        # Clipper les changements extrêmes
        price_changes = np.clip(price_changes, 0.1, 10.0)
        
        return price_changes
    
    def _normalize_action(self, action: np.ndarray) -> np.ndarray:
        """
        Normalise l'action pour que la somme soit 1.
        
        Args:
            action: Action brute
            
        Returns:
            Action normalisée
        """
        # Gérer les valeurs invalides
        if np.isnan(action).any() or np.isinf(action).any():
            logger.warning("Action contient des valeurs invalides, utilisation d'allocation égale")
            return np.ones(self.n_stocks, dtype=np.float32) / self.n_stocks
        
        # S'assurer que toutes les valeurs sont positives
        action = np.maximum(action, 0.0)
        
        # Normaliser
        action_sum = np.sum(action)
        if action_sum > 1e-8:
            return action / action_sum
        else:
            # Si toutes les actions sont nulles, allocation égale
            return np.ones(self.n_stocks, dtype=np.float32) / self.n_stocks
    
    def _calculate_transaction_cost(self, old_weights: np.ndarray, new_weights: np.ndarray) -> float:
        """
        Calcule le coût de transaction.
        
        Args:
            old_weights: Anciens poids
            new_weights: Nouveaux poids
            
        Returns:
            Coût de transaction
        """
        # Turnover = somme des valeurs absolues des changements
        turnover = np.sum(np.abs(new_weights - old_weights))
        
        # Coût proportionnel au turnover et à la valeur du portefeuille
        transaction_cost = self.portfolio_value * self.trans_cost * turnover
        
        return float(transaction_cost)
    
    def _calculate_reward(self, portfolio_return: float, transaction_cost: float, volatility: float) -> float:
        """
        Calcule la récompense.
        
        Args:
            portfolio_return: Rendement du portefeuille
            transaction_cost: Coût de transaction
            volatility: Volatilité des rendements
            
        Returns:
            Récompense
        """
        # Récompense de base: rendement ajusté du risque
        base_reward = portfolio_return - self.return_rate
        
        # Pénalité de transaction
        transaction_penalty = transaction_cost / self.portfolio_value
        
        # Pénalité de risque (volatilité)
        risk_penalty = self.risk_penalty * volatility
        
        # Calcul de drawdown
        drawdown_penalty = 0.0
        if self.max_portfolio_value > 0:
            drawdown = (self.max_portfolio_value - self.portfolio_value) / self.max_portfolio_value
            drawdown_penalty = drawdown * 0.5
        
        # Récompense finale
        reward = base_reward - transaction_penalty - risk_penalty - drawdown_penalty
        
        # Mise à l'échelle et écrêtage
        reward *= self.reward_scaling
        reward = np.clip(reward, self.min_reward_clip, self.max_reward_clip)
        
        return float(reward)
    
   
    def step(self, action: np.ndarray) -> Tuple[Dict[str, np.ndarray], float, bool, bool, Dict[str, Any]]:
        """
        Exécute une étape dans l'environnement.
        
        Args:
            action: Action à exécuter (poids du portefeuille)
            
        Returns:
            Tuple[observation, reward, terminated, truncated, info]
        """
        # Sauvegarder l'état précédent
        old_portfolio_value = self.portfolio_value
        old_weights = self.weights.copy()
        
        # Normaliser l'action
        new_weights = self._normalize_action(action)
        
        # Obtenir les changements de prix
        price_changes = self._get_price_changes(self.current_step)
        
        # Calculer le coût de transaction
        transaction_cost = self._calculate_transaction_cost(old_weights, new_weights)
        
        # Mettre à jour la valeur du portefeuille
        # Valeur avant transaction (valeur actuelle répartie selon anciens poids)
        values_before = self.portfolio_value * old_weights
        
        # Appliquer les changements de prix du marché
        values_after_market = values_before * price_changes
        new_total_value = np.sum(values_after_market)
        
        # Soustraire le coût de transaction
        self.portfolio_value = max(new_total_value - transaction_cost, 1e-8)
        
        # Calculer les nouveaux poids de manière robuste
        if self.portfolio_value > 1e-8:
            # Les nouveaux poids sont simplement les poids désirés (new_weights)
            # puisque nous réallouons le portefeuille
            self.weights = new_weights.copy()
        else:
            # Si le portefeuille devient trop petit, allocation égale
            logger.warning("Portefeuille trop petit, allocation égale appliquée")
            self.weights = np.ones(self.n_stocks, dtype=np.float32) / self.n_stocks
        
        # Vérification finale des poids
        if np.isnan(self.weights).any() or np.isinf(self.weights).any():
            logger.warning("Poids invalides détectés, réinitialisation à allocation égale")
            self.weights = np.ones(self.n_stocks, dtype=np.float32) / self.n_stocks
        
        # S'assurer que les poids sont normalisés
        weights_sum = np.sum(self.weights)
        if abs(weights_sum - 1.0) > 1e-6:
            if weights_sum > 1e-8:
                self.weights = self.weights / weights_sum
            else:
                self.weights = np.ones(self.n_stocks, dtype=np.float32) / self.n_stocks
        
        # Calculer le rendement
        portfolio_return = (self.portfolio_value / old_portfolio_value) - 1 if old_portfolio_value > 0 else 0.0
        
        # Mettre à jour les historiques
        self.episode_returns.append(portfolio_return)
        self.episode_values.append(self.portfolio_value)
        self.transaction_costs.append(transaction_cost)
        self.max_portfolio_value = max(self.max_portfolio_value, self.portfolio_value)
        
        # Calculer la volatilité
        volatility = np.std(self.episode_returns) if len(self.episode_returns) > 1 else 0.0
        
        # Calculer la récompense
        reward = self._calculate_reward(portfolio_return, transaction_cost, volatility)
        
        # Mettre à jour le pas de temps
        self.current_step += 1
        
        # Vérifier les conditions de terminaison
        terminated = bool(
            self.current_step >= self.n_time_periods - 1 or  # Fin des données
            self.portfolio_value <= self.initial_portfolio_value * 0.1  # Perte de 90%
        )
        
        truncated = False  # Pas de truncation dans cette implémentation
        
        # Informations de debug
        info = {
            'portfolio_value': float(self.portfolio_value),
            'weights': self.weights.copy(),
            'transaction_cost': float(transaction_cost),
            'return': float(portfolio_return),
            'volatility': float(volatility),
            'drawdown': float((self.max_portfolio_value - self.portfolio_value) / self.max_portfolio_value) if self.max_portfolio_value > 0 else 0.0,
            'max_portfolio_value': float(self.max_portfolio_value),
            'step': self.current_step,
            'price_changes': price_changes.copy()
        }
        
        return self._get_observation(), reward, terminated, truncated, info
    

    def render(self, mode: str = "human") -> Optional[np.ndarray]:
        """
        Rendu de l'environnement (optionnel).
        
        Args:
            mode: Mode de rendu
            
        Returns:
            Image si mode="rgb_array"
        """
        if mode == "human":
            print(f"Step: {self.current_step}, Portfolio Value: ${self.portfolio_value:.2f}, Weights: {self.weights}")
        return None

def create_env_from_config(data: np.ndarray, config_path: str = "config/default.json") -> PortfolioEnv:
    """
    Crée un environnement PortfolioEnv à partir d'un fichier de configuration.
    
    Args:
        data: Données financières
        config_path: Chemin du fichier de configuration
        
    Returns:
        PortfolioEnv: Environnement initialisé
    """
    config = load_config(config_path)
    env_config = config.get('environment', {})
    
    env = PortfolioEnv(
        data=data,
        portfolio_value=env_config.get('portfolio_value', 10000),
        window_size=env_config.get('window_size', 30),
        trans_cost=env_config.get('trans_cost', 0.001),
        return_rate=env_config.get('return_rate', 0.0001),
        reward_scaling=env_config.get('reward_scaling', 100.0),
        max_reward_clip=env_config.get('max_reward_clip', 10.0),
        min_reward_clip=env_config.get('min_reward_clip', -10.0),
        normalize_observations=env_config.get('normalize_observations', True),
        risk_penalty=env_config.get('risk_penalty', 0.1)
    )
    
    return env