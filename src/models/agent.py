# D�finition de l'agent

"""
Module définissant l'agent d'allocation de portefeuille.
"""
import os
import numpy as np
import json
import logging
from stable_baselines3 import PPO

# Configuration du logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class PortfolioAgent:
    """
    Agent d'allocation de portefeuille basé sur PPO.
    """
    
    def __init__(self, model_path=None, config_path="config/default.json"):
        """
        Initialise l'agent de portefeuille.
        
        Args:
            model_path: Chemin vers un modèle pré-entraîné (optionnel)
            config_path: Chemin du fichier de configuration
        """
        self.model_path = model_path
        self.config_path = config_path
        self.model = None
        
        # Charger la configuration
        self.config = self._load_config()
        
        # Charger le modèle s'il est spécifié
        if model_path and os.path.exists(model_path + ".zip"):
            self.load_model(model_path)
    
    def _load_config(self):
        """
        Charge la configuration depuis un fichier json.
        """
        with open(self.config_path, 'r') as file:
            config = json.load(file)
        return config
    
    def load_model(self, model_path):
        """
        Charge un modèle pré-entraîné.
        
        Args:
            model_path: Chemin vers le modèle à charger
        """
        try:
            self.model = PPO.load(model_path)
            self.model_path = model_path
            logger.info(f"Modèle chargé depuis {model_path}")
            return True
        except Exception as e:
            logger.error(f"Erreur lors du chargement du modèle: {e}")
            return False
    
    def predict(self, observation, deterministic=True):
        """
        Prédit l'allocation de portefeuille pour une observation donnée.
        
        Args:
            observation: Observation de l'environnement
            deterministic: Si True, utilise une politique déterministe
            
        Returns:
            ndarray: Poids du portefeuille
        """
        if self.model is None:
            logger.error("Aucun modèle chargé. Utilisez load_model() d'abord.")
            return None
        
        try:
            action, _ = self.model.predict(observation, deterministic=deterministic)
            return action
        except Exception as e:
            logger.error(f"Erreur lors de la prédiction: {e}")
            return None
    
    def get_portfolio_allocation(self, market_data, current_weights, portfolio_value):
        """
        Obtient l'allocation de portefeuille basée sur les données du marché.
        
        Args:
            market_data: Données du marché (fenêtre temporelle)
            current_weights: Poids actuels du portefeuille
            portfolio_value: Valeur actuelle du portefeuille
            
        Returns:
            ndarray: Nouveaux poids du portefeuille
        """
        if self.model is None:
            logger.error("Aucun modèle chargé. Utilisez load_model() d'abord.")
            return None
        
        try:
            # Construire l'observation
            observation = {
                'market_data': market_data,
                'weights': current_weights,
                'portfolio_value': np.array([portfolio_value], dtype=np.float32)
            }
            
            # Prédire l'action
            action = self.predict(observation)
            
            # Normaliser les poids
            if action is not None:
                action_sum = np.sum(action)
                if action_sum > 0:
                    normalized_action = action / action_sum
                else:
                    # Allocation égale si la somme est nulle
                    normalized_action = np.ones_like(action) / len(action)
                
                return normalized_action
            
            return None
        
        except Exception as e:
            logger.error(f"Erreur lors de l'obtention de l'allocation: {e}")
            return None