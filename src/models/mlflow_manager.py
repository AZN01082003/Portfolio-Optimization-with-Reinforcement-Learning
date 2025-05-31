"""
Gestionnaire MLflow pour l'intégration des modèles de production.
Remplace le ModelManager factice par une version MLflow complète.
"""
import os
import logging
import json
from datetime import datetime
from typing import Dict, Any, Optional, List
from pathlib import Path

import numpy as np

# Configuration du logging
logger = logging.getLogger(__name__)

# Imports conditionnels
try:
    import mlflow
    import mlflow.pyfunc
    from mlflow.tracking import MlflowClient
    MLFLOW_AVAILABLE = True
    logger.info("✅ MLflow disponible")
except ImportError as e:
    logger.warning(f"⚠️ MLflow non disponible: {e}")
    MLFLOW_AVAILABLE = False

try:
    from stable_baselines3 import PPO
    SB3_AVAILABLE = True
except ImportError:
    SB3_AVAILABLE = False

class MLflowModelManager:
    """Gestionnaire de modèles intégré avec MLflow."""
    
    def __init__(self, mlflow_uri: str = "http://localhost:5000"):
        """
        Initialise le gestionnaire MLflow.
        
        Args:
            mlflow_uri: URI du serveur de tracking MLflow
        """
        self.mlflow_uri = mlflow_uri
        self.client = None
        self.models = {}
        self.fallback_models = {}
        
        self._setup_mlflow()
        self._load_models()
    
    def _setup_mlflow(self):
        """Configure la connexion MLflow."""
        if not MLFLOW_AVAILABLE:
            logger.warning("⚠️ MLflow non disponible - mode fallback activé")
            return
        
        try:
            # Configurer l'URI de tracking
            mlflow.set_tracking_uri(self.mlflow_uri)
            self.client = MlflowClient(self.mlflow_uri)
            
            # Vérifier la connexion
            experiments = self.client.search_experiments()
            logger.info(f"✅ Connexion MLflow réussie ({len(experiments)} expériences)")
            
        except Exception as e:
            logger.error(f"❌ Erreur connexion MLflow: {e}")
            self.client = None
    
    def _load_models(self):
        """Charge les modèles depuis MLflow et les modèles locaux."""
        # 1. Charger les modèles MLflow
        self._load_mlflow_models()
        
        # 2. Charger les modèles locaux (fallback)
        self._load_local_models()
        
        # 3. Créer des modèles factices si nécessaire
        if not self.models and not self.fallback_models:
            self._create_dummy_models()
    
    def _load_mlflow_models(self):
        """Charge les modèles depuis MLflow Model Registry."""
        if not self.client:
            return
        
        try:
            # Rechercher les modèles dans le registry
            registered_models = self.client.search_registered_models()
            
            for model in registered_models:
                model_name = model.name
                
                try:
                    # Récupérer la version en production
                    production_version = self._get_production_version(model_name)
                    
                    if production_version:
                        model_uri = f"models:/{model_name}/{production_version.version}"
                        
                        # Charger le modèle
                        loaded_model = mlflow.pyfunc.load_model(model_uri)
                        
                        self.models[model_name] = {
                            'model': loaded_model,
                            'type': 'mlflow',
                            'version': production_version.version,
                            'uri': model_uri,
                            'stage': 'Production',
                            'loaded_at': datetime.now(),
                            'metadata': self._get_model_metadata(model_name, production_version.version)
                        }
                        
                        logger.info(f"✅ Modèle MLflow chargé: {model_name} v{production_version.version}")
                    
                    # Charger aussi la version "Staging" si disponible
                    staging_version = self._get_staging_version(model_name)
                    if staging_version:
                        model_uri = f"models:/{model_name}/{staging_version.version}"
                        loaded_model = mlflow.pyfunc.load_model(model_uri)
                        
                        self.models[f"{model_name}_staging"] = {
                            'model': loaded_model,
                            'type': 'mlflow',
                            'version': staging_version.version,
                            'uri': model_uri,
                            'stage': 'Staging',
                            'loaded_at': datetime.now(),
                            'metadata': self._get_model_metadata(model_name, staging_version.version)
                        }
                        
                        logger.info(f"✅ Modèle MLflow staging chargé: {model_name} v{staging_version.version}")
                        
                except Exception as e:
                    logger.warning(f"⚠️ Impossible de charger le modèle {model_name}: {e}")
            
        except Exception as e:
            logger.error(f"❌ Erreur lors du chargement des modèles MLflow: {e}")
    
    def _get_production_version(self, model_name: str):
        """Récupère la version en production d'un modèle."""
        try:
            versions = self.client.get_latest_versions(
                model_name, 
                stages=["Production"]
            )
            return versions[0] if versions else None
        except Exception:
            return None
    
    def _get_staging_version(self, model_name: str):
        """Récupère la version en staging d'un modèle."""
        try:
            versions = self.client.get_latest_versions(
                model_name,
                stages=["Staging"]
            )
            return versions[0] if versions else None
        except Exception:
            return None
    
    def _get_model_metadata(self, model_name: str, version: str) -> Dict[str, Any]:
        """Récupère les métadonnées d'un modèle."""
        try:
            model_version = self.client.get_model_version(model_name, version)
            run = self.client.get_run(model_version.run_id)
            
            return {
                'run_id': model_version.run_id,
                'metrics': run.data.metrics,
                'params': run.data.params,
                'tags': run.data.tags,
                'creation_timestamp': model_version.creation_timestamp,
                'description': model_version.description
            }
        except Exception as e:
            logger.warning(f"⚠️ Impossible de récupérer les métadonnées: {e}")
            return {}
    
    def _load_local_models(self):
        """Charge les modèles depuis le système de fichiers local."""
        models_dir = Path("models")
        
        if not models_dir.exists():
            return
        
        # Chercher les modèles Stable-Baselines3
        for model_file in models_dir.glob("**/*.zip"):
            if not SB3_AVAILABLE:
                continue
                
            model_name = model_file.stem
            
            try:
                model = PPO.load(str(model_file))
                
                self.fallback_models[model_name] = {
                    'model': model,
                    'type': 'stable_baselines3',
                    'path': str(model_file),
                    'loaded_at': datetime.now(),
                    'version': 'local',
                    'stage': 'Local'
                }
                
                logger.info(f"✅ Modèle local chargé: {model_name}")
                
            except Exception as e:
                logger.warning(f"⚠️ Impossible de charger {model_name}: {e}")
    
    def _create_dummy_models(self):
        """Crée des modèles factices pour les tests."""
        self.fallback_models = {
            'dummy_portfolio_model': {
                'model': None,
                'type': 'dummy',
                'version': '1.0.0',
                'stage': 'Test',
                'loaded_at': datetime.now(),
                'description': 'Modèle factice pour tests et développement'
            }
        }
        logger.info("🔧 Modèles factices créés pour les tests")
    
    def list_models(self) -> List[Dict[str, Any]]:
        """Liste tous les modèles disponibles."""
        models_list = []
        
        # Modèles MLflow
        for name, info in self.models.items():
            models_list.append({
                'name': name,
                'type': info['type'],
                'version': info['version'],
                'stage': info['stage'],
                'loaded_at': info['loaded_at'].isoformat(),
                'source': 'mlflow',
                'uri': info.get('uri'),
                'metrics': info.get('metadata', {}).get('metrics', {})
            })
        
        # Modèles locaux/fallback
        for name, info in self.fallback_models.items():
            models_list.append({
                'name': name,
                'type': info['type'],
                'version': info['version'],
                'stage': info['stage'],
                'loaded_at': info['loaded_at'].isoformat(),
                'source': 'local',
                'path': info.get('path')
            })
        
        return models_list
    
    def get_model(self, model_name: str = "latest", prefer_production: bool = True):
        """
        Récupère un modèle par nom.
        
        Args:
            model_name: Nom du modèle ou "latest"
            prefer_production: Préférer les modèles en production
        """
        if model_name == "latest":
            return self._get_latest_model(prefer_production)
        
        # Chercher d'abord dans les modèles MLflow
        if model_name in self.models:
            return self.models[model_name]
        
        # Puis dans les modèles locaux
        if model_name in self.fallback_models:
            return self.fallback_models[model_name]
        
        # Modèle non trouvé
        logger.warning(f"⚠️ Modèle {model_name} non trouvé")
        return None
    
    def _get_latest_model(self, prefer_production: bool = True):
        """Récupère le modèle le plus récent."""
        all_models = {**self.models, **self.fallback_models}
        
        if not all_models:
            return None
        
        if prefer_production:
            # Préférer les modèles en production
            production_models = {
                name: info for name, info in all_models.items()
                if info.get('stage') == 'Production'
            }
            
            if production_models:
                latest_key = max(
                    production_models.keys(),
                    key=lambda k: production_models[k]['loaded_at']
                )
                return production_models[latest_key]
        
        # Sinon, prendre le plus récent
        latest_key = max(
            all_models.keys(),
            key=lambda k: all_models[k]['loaded_at']
        )
        return all_models[latest_key]
    
    def predict(self, model_name: str, market_data: np.ndarray, current_weights: np.ndarray) -> Dict[str, Any]:
        """
        Effectue une prédiction avec le modèle spécifié.
        
        Args:
            model_name: Nom du modèle
            market_data: Données de marché [features, stocks, periods]
            current_weights: Poids actuels du portefeuille
            
        Returns:
            Dictionnaire avec les résultats de la prédiction
        """
        model_info = self.get_model(model_name)
        
        if not model_info:
            raise ValueError(f"Modèle {model_name} non trouvé")
        
        start_time = datetime.now()
        
        try:
            if model_info['type'] == 'mlflow':
                result = self._predict_mlflow(model_info, market_data, current_weights)
            elif model_info['type'] == 'stable_baselines3':
                result = self._predict_sb3(model_info, market_data, current_weights)
            else:
                result = self._predict_dummy(model_info, market_data, current_weights)
            
            # Ajouter les métadonnées
            result.update({
                'model_name': model_name,
                'model_type': model_info['type'],
                'model_version': model_info['version'],
                'model_stage': model_info.get('stage', 'Unknown'),
                'prediction_time': (datetime.now() - start_time).total_seconds()
            })
            
            return result
            
        except Exception as e:
            logger.error(f"❌ Erreur lors de la prédiction avec {model_name}: {e}")
            raise
    
    def _predict_mlflow(self, model_info: Dict, market_data: np.ndarray, current_weights: np.ndarray) -> Dict[str, Any]:
        """Prédiction avec un modèle MLflow."""
        model = model_info['model']
        
        # Préparer les données d'entrée pour MLflow
        # Format: DataFrame avec les features nécessaires
        import pandas as pd
        
        # Aplatir les données de marché pour MLflow
        n_features, n_stocks, n_periods = market_data.shape
        
        # Prendre les dernières observations (fenêtre glissante)
        last_window = market_data[:, :, -30:]  # 30 dernières périodes
        
        # Créer les features pour MLflow (format tabulaire)
        features = {}
        
        # Features de marché
        for i in range(n_features):
            for j in range(n_stocks):
                # Statistiques de la fenêtre
                series = last_window[i, j, :]
                features[f'market_feature_{i}_stock_{j}_mean'] = np.mean(series)
                features[f'market_feature_{i}_stock_{j}_std'] = np.std(series)
                features[f'market_feature_{i}_stock_{j}_last'] = series[-1]
        
        # Features de portefeuille
        for i, weight in enumerate(current_weights):
            features[f'current_weight_{i}'] = weight
        
        # Créer le DataFrame
        input_df = pd.DataFrame([features])
        
        # Prédiction
        prediction = model.predict(input_df)
        
        # Post-traitement des résultats
        if isinstance(prediction, np.ndarray):
            if prediction.ndim == 2:
                weights = prediction[0]
            else:
                weights = prediction
        else:
            weights = np.array(prediction)
        
        # Normaliser les poids
        weights = np.maximum(weights, 0)  # Pas de positions courtes
        weights = weights / np.sum(weights)  # Normalisation
        
        # Calculer la confiance (entropie inverse)
        entropy = -np.sum(weights * np.log(weights + 1e-8))
        max_entropy = np.log(len(weights))
        confidence = 1.0 - (entropy / max_entropy)
        
        return {
            'weights': weights.tolist(),
            'confidence': float(confidence),
            'raw_prediction': prediction.tolist() if hasattr(prediction, 'tolist') else prediction
        }
    
    def _predict_sb3(self, model_info: Dict, market_data: np.ndarray, current_weights: np.ndarray) -> Dict[str, Any]:
        """Prédiction avec un modèle Stable-Baselines3."""
        model = model_info['model']
        
        # Préparer l'observation pour SB3
        obs = {
            'market_data': market_data.astype(np.float32),
            'weights': current_weights.astype(np.float32),
            'portfolio_value': np.array([1.0], dtype=np.float32)
        }
        
        # Prédiction
        action, _states = model.predict(obs, deterministic=True)
        
        # Post-traitement
        weights = np.array(action)
        weights = np.maximum(weights, 0)
        weights = weights / np.sum(weights)
        
        # Confiance basée sur l'entropie
        entropy = -np.sum(weights * np.log(weights + 1e-8))
        max_entropy = np.log(len(weights))
        confidence = 1.0 - (entropy / max_entropy)
        
        return {
            'weights': weights.tolist(),
            'confidence': float(confidence),
            'raw_action': action.tolist()
        }
    
    def _predict_dummy(self, model_info: Dict, market_data: np.ndarray, current_weights: np.ndarray) -> Dict[str, Any]:
        """Prédiction avec un modèle factice."""
        n_assets = len(current_weights)
        
        # Génération déterministe basée sur les données
        seed = int(np.sum(market_data[:, :, -1]) % 1000)
        np.random.seed(seed)
        
        # Poids aléatoires mais cohérents
        random_weights = np.random.dirichlet(np.ones(n_assets) * 2)
        confidence = 0.7 + np.random.random() * 0.25
        
        return {
            'weights': random_weights.tolist(),
            'confidence': confidence
        }
    
    def refresh_models(self):
        """Recharge tous les modèles depuis MLflow."""
        logger.info("🔄 Rechargement des modèles...")
        
        self.models.clear()
        self.fallback_models.clear()
        
        self._load_models()
        
        logger.info(f"✅ Rechargement terminé: {len(self.models)} MLflow + {len(self.fallback_models)} locaux")
    
    def get_model_info(self, model_name: str) -> Optional[Dict[str, Any]]:
        """Récupère les informations détaillées d'un modèle."""
        model_info = self.get_model(model_name)
        
        if not model_info:
            return None
        
        info = {
            'name': model_name,
            'type': model_info['type'],
            'version': model_info['version'],
            'stage': model_info.get('stage'),
            'loaded_at': model_info['loaded_at'].isoformat()
        }
        
        # Ajouter les métadonnées MLflow si disponibles
        if 'metadata' in model_info:
            metadata = model_info['metadata']
            info.update({
                'metrics': metadata.get('metrics', {}),
                'params': metadata.get('params', {}),
                'run_id': metadata.get('run_id'),
                'description': metadata.get('description')
            })
        
        return info

# Instance globale
_mlflow_manager = None

def get_mlflow_manager(mlflow_uri: str = None) -> MLflowModelManager:
    """Récupère l'instance singleton du gestionnaire MLflow."""
    global _mlflow_manager
    
    if _mlflow_manager is None:
        uri = mlflow_uri or os.getenv('MLFLOW_TRACKING_URI', 'http://localhost:5000')
        _mlflow_manager = MLflowModelManager(uri)
    
    return _mlflow_manager

def refresh_mlflow_models():
    """Recharge les modèles MLflow."""
    if _mlflow_manager:
        _mlflow_manager.refresh_models()