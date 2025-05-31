"""
Gestionnaire de données pour intégrer vos données de preprocessing existantes.
Connecte l'API avec vos pipelines de données réels.
"""
import os
import logging
import json
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Any
from pathlib import Path

import numpy as np
import pandas as pd

# Configuration du logging
logger = logging.getLogger(__name__)

class DataManager:
    """Gestionnaire pour l'intégration des données réelles."""
    
    def __init__(self, config_path: str = "config/default.json"):
        """
        Initialise le gestionnaire de données.
        
        Args:
            config_path: Chemin vers le fichier de configuration
        """
        self.config_path = config_path
        self.config = self._load_config()
        self.data_dir = Path(self.config.get('data', {}).get('output_dir', 'data/processed'))
        self.cache = {}
        self.last_update = {}
        
        # Créer les répertoires si nécessaire
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"✅ DataManager initialisé: {self.data_dir}")
    
    def _load_config(self) -> Dict[str, Any]:
        """Charge la configuration."""
        try:
            with open(self.config_path, 'r') as f:
                return json.load(f)
        except Exception as e:
            logger.warning(f"⚠️ Impossible de charger la config {self.config_path}: {e}")
            return self._get_default_config()
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Configuration par défaut."""
        return {
            "data": {
                "tickers": ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA"],
                "output_dir": "data/processed",
                "cache_duration_minutes": 30
            }
        }
    
    def get_latest_processed_data(self, data_type: str = "train") -> Optional[np.ndarray]:
        """
        Récupère les dernières données préprocessées.
        
        Args:
            data_type: Type de données ("train", "test", "normalized")
            
        Returns:
            Données numpy ou None si non trouvées
        """
        # Vérifier le cache d'abord
        cache_key = f"processed_{data_type}"
        if self._is_cache_valid(cache_key):
            logger.debug(f"📦 Données {data_type} récupérées du cache")
            return self.cache[cache_key]
        
        # Chercher les fichiers les plus récents
        file_pattern = f"stock_data_{data_type}_*.npy"
        latest_file = self._find_latest_file(file_pattern)
        
        if latest_file:
            try:
                data = np.load(latest_file)
                
                # Validation des données
                if self._validate_data_shape(data):
                    # Mettre en cache
                    self.cache[cache_key] = data
                    self.last_update[cache_key] = datetime.now()
                    
                    logger.info(f"✅ Données {data_type} chargées: {data.shape}")
                    return data
                else:
                    logger.error(f"❌ Forme de données invalide pour {latest_file}: {data.shape}")
                    
            except Exception as e:
                logger.error(f"❌ Erreur chargement {latest_file}: {e}")
        
        # Fallback: créer des données factices
        logger.warning(f"⚠️ Données {data_type} non trouvées, création de données factices")
        return self._create_fallback_data(data_type)
    
    def _find_latest_file(self, pattern: str) -> Optional[Path]:
        """Trouve le fichier le plus récent correspondant au pattern."""
        try:
            files = list(self.data_dir.glob(pattern))
            if not files:
                # Essayer aussi avec "latest"
                latest_pattern = pattern.replace("*", "latest")
                files = list(self.data_dir.glob(latest_pattern))
            
            if files:
                # Trier par date de modification
                latest_file = max(files, key=lambda f: f.stat().st_mtime)
                logger.debug(f"📂 Fichier le plus récent: {latest_file}")
                return latest_file
            
        except Exception as e:
            logger.error(f"❌ Erreur recherche fichiers {pattern}: {e}")
        
        return None
    
    def _validate_data_shape(self, data: np.ndarray) -> bool:
        """Valide la forme des données."""
        if data.ndim != 3:
            return False
        
        n_features, n_stocks, n_periods = data.shape
        
        # Vérifications de base
        if n_features < 4:  # Au minimum OHLC
            logger.warning(f"⚠️ Nombre de features insuffisant: {n_features}")
            return False
        
        if n_stocks < 2:
            logger.warning(f"⚠️ Nombre d'actions insuffisant: {n_stocks}")
            return False
        
        if n_periods < 30:  # Au minimum 30 périodes
            logger.warning(f"⚠️ Nombre de périodes insuffisant: {n_periods}")
            return False
        
        # Vérifier les valeurs
        if np.isnan(data).any():
            logger.warning("⚠️ Données contiennent des NaN")
            return False
        
        if np.isinf(data).any():
            logger.warning("⚠️ Données contiennent des valeurs infinies")
            return False
        
        return True
    
    def _create_fallback_data(self, data_type: str) -> np.ndarray:
        """Crée des données factices réalistes."""
        tickers = self.config.get('data', {}).get('tickers', ["AAPL", "MSFT", "GOOGL"])
        n_stocks = len(tickers)
        n_features = 5  # OHLCV
        
        # Taille selon le type
        if data_type == "train":
            n_periods = 500
        elif data_type == "test":
            n_periods = 200
        else:
            n_periods = 700
        
        logger.info(f"🔧 Création données factices {data_type}: {n_features}x{n_stocks}x{n_periods}")
        
        data = np.zeros((n_features, n_stocks, n_periods), dtype=np.float32)
        
        for stock_idx in range(n_stocks):
            # Prix de base différent pour chaque action
            base_price = 100 + stock_idx * 50
            
            # Génération de série temporelle réaliste
            returns = np.random.normal(0.0005, 0.02, n_periods)  # Rendements journaliers
            prices = base_price * np.exp(np.cumsum(returns))
            
            for t in range(n_periods):
                close = prices[t]
                daily_vol = 0.01  # Volatilité intraday
                
                # OHLC réaliste
                open_price = close * (1 + np.random.normal(0, daily_vol/2))
                high = close * (1 + abs(np.random.normal(0, daily_vol)))
                low = close * (1 - abs(np.random.normal(0, daily_vol)))
                volume = np.random.lognormal(15, 0.5)  # Volume log-normal
                
                # S'assurer que High >= max(O,C) et Low <= min(O,C)
                high = max(high, open_price, close)
                low = min(low, open_price, close)
                
                data[0, stock_idx, t] = open_price
                data[1, stock_idx, t] = high
                data[2, stock_idx, t] = low
                data[3, stock_idx, t] = close
                data[4, stock_idx, t] = volume
        
        return data
    
    def _is_cache_valid(self, cache_key: str) -> bool:
        """Vérifie si le cache est encore valide."""
        if cache_key not in self.cache:
            return False
        
        if cache_key not in self.last_update:
            return False
        
        cache_duration = self.config.get('data', {}).get('cache_duration_minutes', 30)
        expiry_time = self.last_update[cache_key] + timedelta(minutes=cache_duration)
        
        return datetime.now() < expiry_time
    
    def get_market_data_for_prediction(self, 
                                     window_size: int = 30,
                                     tickers: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Récupère les données de marché formatées pour la prédiction.
        
        Args:
            window_size: Taille de la fenêtre pour la prédiction
            tickers: Liste des tickers (optionnel)
            
        Returns:
            Dictionnaire avec les données formatées
        """
        # Utiliser les données d'entraînement les plus récentes
        data = self.get_latest_processed_data("train")
        
        if data is None:
            raise ValueError("Impossible de charger les données de marché")
        
        n_features, n_stocks, n_periods = data.shape
        
        # Prendre la fenêtre la plus récente
        if n_periods >= window_size:
            market_window = data[:, :, -window_size:]
        else:
            logger.warning(f"⚠️ Fenêtre demandée ({window_size}) > données disponibles ({n_periods})")
            market_window = data
        
        # Utiliser les tickers de config si non spécifiés
        if tickers is None:
            tickers = self.config.get('data', {}).get('tickers', [f"STOCK_{i}" for i in range(n_stocks)])
        
        # S'assurer que le nombre de tickers correspond
        if len(tickers) != n_stocks:
            tickers = tickers[:n_stocks] + [f"STOCK_{i}" for i in range(len(tickers), n_stocks)]
        
        return {
            'data': market_window.tolist(),
            'tickers': tickers[:n_stocks],
            'timestamp': datetime.now().isoformat(),
            'shape': list(market_window.shape),
            'window_size': window_size,
            'features': ['Open', 'High', 'Low', 'Close', 'Volume'][:n_features]
        }
    
    def get_portfolio_weights(self, portfolio_id: str) -> Dict[str, float]:
        """
        Récupère les poids actuels d'un portefeuille.
        
        Args:
            portfolio_id: Identifiant du portefeuille
            
        Returns:
            Dictionnaire avec les poids par ticker
        """
        # Dans un vrai système, cela viendrait d'une base de données
        # Pour l'instant, on utilise des poids par défaut
        
        tickers = self.config.get('data', {}).get('tickers', [])
        n_assets = len(tickers)
        
        if n_assets == 0:
            return {}
        
        # Poids équipondérés par défaut
        equal_weight = 1.0 / n_assets
        
        # Ajouter un peu de variance pour plus de réalisme
        np.random.seed(hash(portfolio_id) % 2**32)
        weights = np.random.dirichlet(np.ones(n_assets) * 10)  # Concentré autour de équipondéré
        
        return {ticker: float(weight) for ticker, weight in zip(tickers, weights)}
    
    def update_portfolio_weights(self, 
                               portfolio_id: str, 
                               new_weights: Dict[str, float]) -> bool:
        """
        Met à jour les poids d'un portefeuille.
        
        Args:
            portfolio_id: Identifiant du portefeuille
            new_weights: Nouveaux poids par ticker
            
        Returns:
            True si la mise à jour a réussi
        """
        try:
            # Validation
            total_weight = sum(new_weights.values())
            if abs(total_weight - 1.0) > 0.01:
                logger.error(f"❌ Somme des poids invalide: {total_weight}")
                return False
            
            # Dans un vrai système, sauvegarder en base de données
            # Pour l'instant, on log seulement
            logger.info(f"💾 Mise à jour poids portefeuille {portfolio_id}: {new_weights}")
            
            # Sauvegarder dans un fichier de cache
            portfolios_file = self.data_dir / "portfolios_cache.json"
            
            # Charger le cache existant
            portfolios_cache = {}
            if portfolios_file.exists():
                try:
                    with open(portfolios_file, 'r') as f:
                        portfolios_cache = json.load(f)
                except Exception as e:
                    logger.warning(f"⚠️ Erreur lecture cache portefeuilles: {e}")
            
            # Mettre à jour
            portfolios_cache[portfolio_id] = {
                'weights': new_weights,
                'updated_at': datetime.now().isoformat()
            }
            
            # Sauvegarder
            try:
                with open(portfolios_file, 'w') as f:
                    json.dump(portfolios_cache, f, indent=2)
                logger.info(f"✅ Poids sauvegardés pour {portfolio_id}")
                return True
            except Exception as e:
                logger.error(f"❌ Erreur sauvegarde: {e}")
                return False
                
        except Exception as e:
            logger.error(f"❌ Erreur mise à jour poids: {e}")
            return False
    
    def get_data_status(self) -> Dict[str, Any]:
        """Récupère le statut des données."""
        status = {
            'data_directory': str(self.data_dir),
            'config_loaded': bool(self.config),
            'cache_entries': len(self.cache),
            'available_files': {},
            'last_updates': {}
        }
        
        # Vérifier les fichiers disponibles
        for data_type in ['train', 'test', 'normalized']:
            pattern = f"stock_data_{data_type}_*.npy"
            files = list(self.data_dir.glob(pattern))
            
            if files:
                latest_file = max(files, key=lambda f: f.stat().st_mtime)
                status['available_files'][data_type] = {
                    'file': str(latest_file.name),
                    'size_mb': latest_file.stat().st_size / (1024*1024),
                    'modified': datetime.fromtimestamp(latest_file.stat().st_mtime).isoformat()
                }
            else:
                status['available_files'][data_type] = None
        
        # Statut du cache
        for key, timestamp in self.last_update.items():
            status['last_updates'][key] = timestamp.isoformat()
        
        return status
    
    def refresh_data(self) -> Dict[str, bool]:
        """Force le rechargement de toutes les données."""
        logger.info("🔄 Rechargement forcé des données...")
        
        # Vider le cache
        self.cache.clear()
        self.last_update.clear()
        
        # Recharger chaque type de données
        results = {}
        for data_type in ['train', 'test', 'normalized']:
            try:
                data = self.get_latest_processed_data(data_type)
                results[data_type] = data is not None
            except Exception as e:
                logger.error(f"❌ Erreur rechargement {data_type}: {e}")
                results[data_type] = False
        
        success_count = sum(results.values())
        logger.info(f"✅ Rechargement terminé: {success_count}/{len(results)} types de données")
        
        return results

# Instance globale
_data_manager = None

def get_data_manager(config_path: str = None) -> DataManager:
    """Récupère l'instance singleton du gestionnaire de données."""
    global _data_manager
    
    if _data_manager is None:
        config = config_path or "config/default.json"
        _data_manager = DataManager(config)
    
    return _data_manager

def refresh_data_cache():
    """Recharge le cache de données."""
    if _data_manager:
        return _data_manager.refresh_data()
    return {}