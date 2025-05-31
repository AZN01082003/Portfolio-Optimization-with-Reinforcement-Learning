"""
API FastAPI pour servir les prédictions du modèle de portfolio.
Compatible avec MLflow et l'architecture existante.
"""
import os
import sys
import json
import logging
import asyncio
from datetime import datetime
from typing import Dict, List, Optional, Any
import numpy as np

# Ajouter le répertoire du projet au PYTHONPATH
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)

# Imports FastAPI
try:
    from fastapi import FastAPI, HTTPException, Depends, BackgroundTasks
    from fastapi.middleware.cors import CORSMiddleware
    from fastapi.responses import JSONResponse
    from pydantic import BaseModel, Field, validator
    import uvicorn
    FASTAPI_AVAILABLE = True
except ImportError:
    print("❌ FastAPI non disponible. Installez avec: pip install fastapi uvicorn")
    FASTAPI_AVAILABLE = False

# Imports MLflow
try:
    import mlflow
    from mlflow.tracking import MlflowClient
    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False

# Imports modèle
try:
    from stable_baselines3 import PPO
    SB3_AVAILABLE = True
except ImportError:
    SB3_AVAILABLE = False

# Imports locaux
try:
    from src.environment.portfolio_env import create_env_from_config
    from src.models.train import load_config
    ENV_AVAILABLE = True
except ImportError:
    ENV_AVAILABLE = False

# Configuration du logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# ================================
# MODÈLES PYDANTIC
# ================================

class HealthResponse(BaseModel):
    """Réponse de santé de l'API."""
    status: str
    timestamp: str
    version: str = "1.0.0"
    dependencies: Dict[str, bool]

class MarketData(BaseModel):
    """Données de marché pour une prédiction."""
    data: List[List[List[float]]] = Field(..., description="Données de marché (features, stocks, time)")
    tickers: List[str] = Field(..., description="Liste des tickers")
    timestamp: Optional[str] = Field(None, description="Timestamp des données")
    
    @validator('data')
    def validate_data_shape(cls, v):
        if not v or len(v) == 0:
            raise ValueError("Les données ne peuvent pas être vides")
        return v
    
    @validator('tickers')
    def validate_tickers(cls, v):
        if not v or len(v) == 0:
            raise ValueError("La liste des tickers ne peut pas être vide")
        return v

class PortfolioState(BaseModel):
    """État actuel du portefeuille."""
    weights: List[float] = Field(..., description="Poids actuels du portefeuille")
    portfolio_value: float = Field(..., gt=0, description="Valeur actuelle du portefeuille")
    cash: Optional[float] = Field(0.0, description="Liquidités disponibles")
    
    @validator('weights')
    def validate_weights(cls, v):
        if not v or len(v) == 0:
            raise ValueError("Les poids ne peuvent pas être vides")
        if abs(sum(v) - 1.0) > 0.01:
            raise ValueError("La somme des poids doit être proche de 1.0")
        return v

class PredictionRequest(BaseModel):
    """Requête pour obtenir une prédiction d'allocation."""
    market_data: MarketData
    portfolio: PortfolioState
    model_version: Optional[str] = Field("latest", description="Version du modèle à utiliser")
    risk_preference: Optional[str] = Field("moderate", description="Préférence de risque: conservative, moderate, aggressive")

class PredictionResponse(BaseModel):
    """Réponse contenant la prédiction d'allocation."""
    weights: List[float] = Field(..., description="Nouveaux poids recommandés")
    expected_return: Optional[float] = Field(None, description="Rendement attendu estimé")
    expected_risk: Optional[float] = Field(None, description="Risque estimé (volatilité)")
    confidence: Optional[float] = Field(None, description="Confiance de la prédiction")
    rebalancing_needed: bool = Field(..., description="Si un rééquilibrage est nécessaire")
    transaction_cost: Optional[float] = Field(None, description="Coût de transaction estimé")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Métadonnées supplémentaires")

class BacktestRequest(BaseModel):
    """Requête pour un backtest."""
    market_data: MarketData
    initial_portfolio_value: float = Field(10000, gt=0)
    model_version: Optional[str] = "latest"
    transaction_cost: Optional[float] = 0.001

class BacktestResponse(BaseModel):
    """Réponse de backtest."""
    final_value: float
    total_return: float
    volatility: float
    sharpe_ratio: float
    max_drawdown: float
    trades_count: int
    metadata: Dict[str, Any] = Field(default_factory=dict)

# ================================
# GESTIONNAIRE DE MODÈLES
# ================================

class ModelManager:
    """Gestionnaire pour charger et utiliser les modèles."""
    
    def __init__(self):
        self.models: Dict[str, Any] = {}
        self.model_metadata: Dict[str, Dict] = {}
        self.default_config = None
        self.mlflow_client = None
        
        # Initialiser MLflow si disponible
        if MLFLOW_AVAILABLE:
            try:
                mlflow_uri = os.environ.get("MLFLOW_TRACKING_URI", "http://localhost:5000")
                mlflow.set_tracking_uri(mlflow_uri)
                self.mlflow_client = MlflowClient()
                logger.info(f"✅ MLflow connecté: {mlflow_uri}")
            except Exception as e:
                logger.warning(f"⚠️ MLflow non accessible: {e}")
        
        # Charger la configuration par défaut
        self._load_default_config()
        
        # Charger les modèles disponibles
        self._load_available_models()
    
    def _load_default_config(self):
        """Charge la configuration par défaut."""
        try:
            self.default_config = load_config("config/default.json")
            logger.info("✅ Configuration par défaut chargée")
        except Exception as e:
            logger.error(f"❌ Erreur chargement config: {e}")
            # Configuration fallback
            self.default_config = {
                "environment": {
                    "portfolio_value": 10000,
                    "window_size": 30,
                    "trans_cost": 0.001
                }
            }
    
    def _load_available_models(self):
        """Charge les modèles disponibles depuis MLflow et le système de fichiers."""
        
        # 1. Charger depuis MLflow si disponible
        if self.mlflow_client:
            try:
                self._load_from_mlflow()
            except Exception as e:
                logger.warning(f"⚠️ Erreur chargement MLflow: {e}")
        
        # 2. Charger depuis le système de fichiers
        self._load_from_filesystem()
        
        logger.info(f"📦 {len(self.models)} modèle(s) chargé(s)")
    
    def _load_from_mlflow(self):
        """Charge les modèles depuis MLflow."""
        try:
            # Chercher les modèles enregistrés
            registered_models = self.mlflow_client.search_registered_models()
            
            for model in registered_models:
                if "portfolio" in model.name.lower():
                    latest_version = self.mlflow_client.get_latest_versions(
                        model.name, stages=["None", "Staging", "Production"]
                    )
                    
                    if latest_version:
                        version = latest_version[0]
                        model_uri = f"models:/{model.name}/{version.version}"
                        
                        try:
                            # Charger le modèle
                            loaded_model = mlflow.pyfunc.load_model(model_uri)
                            self.models[f"mlflow_{model.name}"] = loaded_model
                            
                            self.model_metadata[f"mlflow_{model.name}"] = {
                                "source": "mlflow",
                                "name": model.name,
                                "version": version.version,
                                "uri": model_uri,
                                "loaded_at": datetime.now().isoformat()
                            }
                            
                            logger.info(f"✅ Modèle MLflow chargé: {model.name} v{version.version}")
                            
                        except Exception as e:
                            logger.warning(f"⚠️ Erreur chargement {model.name}: {e}")
            
        except Exception as e:
            logger.error(f"❌ Erreur recherche modèles MLflow: {e}")
    
    def _load_from_filesystem(self):
        """Charge les modèles depuis le système de fichiers."""
        
        # Chercher dans les logs d'entraînement
        logs_dir = "logs/training"
        if os.path.exists(logs_dir):
            for folder in os.listdir(logs_dir):
                folder_path = os.path.join(logs_dir, folder)
                if os.path.isdir(folder_path):
                    
                    # Chercher les modèles dans ce dossier
                    for model_name in ["best_model.zip", "final_model.zip"]:
                        model_path = os.path.join(folder_path, model_name)
                        if os.path.exists(model_path):
                            try:
                                # Charger avec SB3 si disponible
                                if SB3_AVAILABLE:
                                    model = PPO.load(model_path.replace('.zip', ''))
                                    key = f"local_{folder}_{model_name.replace('.zip', '')}"
                                    self.models[key] = model
                                    
                                    self.model_metadata[key] = {
                                        "source": "filesystem",
                                        "path": model_path,
                                        "folder": folder,
                                        "type": model_name.replace('.zip', ''),
                                        "loaded_at": datetime.now().isoformat()
                                    }
                                    
                                    logger.info(f"✅ Modèle local chargé: {key}")
                                    
                            except Exception as e:
                                logger.warning(f"⚠️ Erreur chargement {model_path}: {e}")
        
        # Ajouter un modèle par défaut si aucun trouvé
        if not self.models:
            self.models["default"] = "random"
            self.model_metadata["default"] = {
                "source": "fallback",
                "type": "random_allocation",
                "loaded_at": datetime.now().isoformat()
            }
            logger.warning("⚠️ Aucun modèle trouvé, utilisation de l'allocation aléatoire")
    
    def get_model(self, model_version: str = "latest") -> tuple[Any, Dict]:
        """Récupère un modèle et ses métadonnées."""
        
        if model_version == "latest" or model_version not in self.models:
            # Prendre le modèle le plus récent
            if self.models:
                latest_key = max(self.model_metadata.keys(), 
                               key=lambda k: self.model_metadata[k]['loaded_at'])
                model = self.models[latest_key]
                metadata = self.model_metadata[latest_key]
            else:
                raise ValueError("Aucun modèle disponible")
        else:
            model = self.models[model_version]
            metadata = self.model_metadata[model_version]
        
        return model, metadata
    
    def predict(self, market_data: np.ndarray, portfolio_state: Dict, 
                model_version: str = "latest") -> Dict:
        """Effectue une prédiction."""
        
        try:
            model, metadata = self.get_model(model_version)
            
            # Si modèle par défaut (random)
            if metadata["source"] == "fallback":
                n_assets = market_data.shape[1] if len(market_data.shape) > 1 else len(portfolio_state["weights"])
                weights = np.random.dirichlet(np.ones(n_assets))
                return {
                    "weights": weights.tolist(),
                    "confidence": 0.5,
                    "model_used": "random_allocation"
                }
            
            # Si modèle SB3
            if SB3_AVAILABLE and hasattr(model, 'predict'):
                # Créer l'observation pour le modèle
                obs = self._create_observation(market_data, portfolio_state)
                action, _ = model.predict(obs, deterministic=True)
                
                # Normaliser les poids
                if np.sum(action) > 0:
                    weights = action / np.sum(action)
                else:
                    weights = np.ones(len(action)) / len(action)
                
                return {
                    "weights": weights.tolist(),
                    "confidence": 0.8,
                    "model_used": metadata.get("type", "unknown")
                }
            
            # Fallback
            n_assets = len(portfolio_state["weights"])
            weights = np.ones(n_assets) / n_assets
            return {
                "weights": weights.tolist(),
                "confidence": 0.3,
                "model_used": "equal_weight"
            }
            
        except Exception as e:
            logger.error(f"❌ Erreur prédiction: {e}")
            # Fallback sécurisé
            n_assets = len(portfolio_state["weights"])
            weights = np.ones(n_assets) / n_assets
            return {
                "weights": weights.tolist(),
                "confidence": 0.1,
                "model_used": "fallback",
                "error": str(e)
            }
    
    def _create_observation(self, market_data: np.ndarray, portfolio_state: Dict) -> Dict:
        """Crée une observation compatible avec l'environnement RL."""
        
        try:
            # Format attendu par l'environnement
            window_size = self.default_config["environment"]["window_size"]
            
            # S'assurer que market_data a la bonne forme
            if len(market_data.shape) == 3:
                # (features, stocks, time) -> prendre les dernières observations
                if market_data.shape[2] >= window_size:
                    market_window = market_data[:, :, -window_size:]
                else:
                    # Padding si nécessaire
                    padding = np.zeros((market_data.shape[0], market_data.shape[1], 
                                      window_size - market_data.shape[2]))
                    market_window = np.concatenate([padding, market_data], axis=2)
            else:
                # Format incorrect, créer des données par défaut
                n_features, n_stocks = 5, len(portfolio_state["weights"])
                market_window = np.random.rand(n_features, n_stocks, window_size).astype(np.float32)
            
            # Observation complète
            obs = {
                'market_data': market_window.astype(np.float32),
                'weights': np.array(portfolio_state["weights"], dtype=np.float32),
                'portfolio_value': np.array([portfolio_state["portfolio_value"] / 10000], dtype=np.float32)
            }
            
            return obs
            
        except Exception as e:
            logger.error(f"❌ Erreur création observation: {e}")
            # Observation par défaut
            n_stocks = len(portfolio_state["weights"])
            window_size = 30
            return {
                'market_data': np.random.rand(5, n_stocks, window_size).astype(np.float32),
                'weights': np.array(portfolio_state["weights"], dtype=np.float32),
                'portfolio_value': np.array([1.0], dtype=np.float32)
            }

# ================================
# INSTANCE GLOBALE
# ================================

# Créer le gestionnaire de modèles global
model_manager = ModelManager() if (SB3_AVAILABLE and ENV_AVAILABLE) else None

# ================================
# APPLICATION FASTAPI
# ================================

if FASTAPI_AVAILABLE:
    app = FastAPI(
        title="Portfolio RL API",
        description="API pour les prédictions d'allocation de portefeuille par apprentissage par renforcement",
        version="1.0.0",
        docs_url="/docs",
        redoc_url="/redoc"
    )
    
    # Middleware CORS
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],  # À restreindre en production
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    # ================================
    # ENDPOINTS
    # ================================
    
    @app.get("/", response_model=Dict[str, str])
    async def root():
        """Point d'entrée de l'API."""
        return {
            "message": "Portfolio RL API",
            "version": "1.0.0",
            "docs": "/docs",
            "health": "/health"
        }
    
    @app.get("/health", response_model=HealthResponse)
    async def health():
        """Vérification de l'état de l'API."""
        return HealthResponse(
            status="healthy" if model_manager else "degraded",
            timestamp=datetime.now().isoformat(),
            dependencies={
                "fastapi": FASTAPI_AVAILABLE,
                "mlflow": MLFLOW_AVAILABLE,
                "stable_baselines3": SB3_AVAILABLE,
                "environment": ENV_AVAILABLE,
                "models_loaded": len(model_manager.models) > 0 if model_manager else False
            }
        )
    
    @app.get("/models", response_model=Dict[str, Any])
    async def list_models():
        """Liste les modèles disponibles."""
        if not model_manager:
            raise HTTPException(status_code=503, detail="Gestionnaire de modèles non disponible")
        
        return {
            "models": list(model_manager.models.keys()),
            "metadata": model_manager.model_metadata,
            "default_model": "latest"
        }
    
    @app.post("/predict", response_model=PredictionResponse)
    async def predict_allocation(request: PredictionRequest):
        """Prédit l'allocation optimale de portefeuille."""
        
        if not model_manager:
            raise HTTPException(status_code=503, detail="Service de prédiction non disponible")
        
        try:
            # Convertir les données en numpy array
            market_data_array = np.array(request.market_data.data, dtype=np.float32)
            
            # Effectuer la prédiction
            prediction = model_manager.predict(
                market_data_array,
                {
                    "weights": request.portfolio.weights,
                    "portfolio_value": request.portfolio.portfolio_value
                },
                request.model_version
            )
            
            # Calculer si un rééquilibrage est nécessaire
            current_weights = np.array(request.portfolio.weights)
            new_weights = np.array(prediction["weights"])
            weight_change = np.sum(np.abs(new_weights - current_weights))
            rebalancing_needed = weight_change > 0.05  # Seuil de 5%
            
            # Estimer le coût de transaction
            transaction_cost = weight_change * 0.001 * request.portfolio.portfolio_value
            
            return PredictionResponse(
                weights=prediction["weights"],
                expected_return=None,  # À implémenter si nécessaire
                expected_risk=None,    # À implémenter si nécessaire  
                confidence=prediction.get("confidence", 0.5),
                rebalancing_needed=rebalancing_needed,
                transaction_cost=transaction_cost,
                metadata={
                    "model_used": prediction.get("model_used", "unknown"),
                    "prediction_time": datetime.now().isoformat(),
                    "weight_change": float(weight_change),
                    "risk_preference": request.risk_preference
                }
            )
            
        except Exception as e:
            logger.error(f"❌ Erreur prédiction: {e}")
            raise HTTPException(status_code=500, detail=f"Erreur lors de la prédiction: {str(e)}")
    
    @app.post("/backtest", response_model=BacktestResponse)
    async def backtest_strategy(request: BacktestRequest):
        """Effectue un backtest de la stratégie."""
        
        if not model_manager or not ENV_AVAILABLE:
            raise HTTPException(status_code=503, detail="Service de backtest non disponible")
        
        try:
            # Convertir les données
            market_data_array = np.array(request.market_data.data, dtype=np.float32)
            
            # Créer un environnement temporaire pour le backtest
            from src.environment.portfolio_env import PortfolioEnv
            
            # Configuration pour le backtest
            test_env = PortfolioEnv(
                data=market_data_array,
                portfolio_value=request.initial_portfolio_value,
                trans_cost=request.transaction_cost,
                window_size=30
            )
            
            # Simuler la stratégie
            model, _ = model_manager.get_model(request.model_version)
            
            obs, info = test_env.reset()
            total_rewards = []
            portfolio_values = [request.initial_portfolio_value]
            trades_count = 0
            
            for step in range(min(100, market_data_array.shape[2] - 30)):
                if hasattr(model, 'predict'):
                    action, _ = model.predict(obs, deterministic=True)
                else:
                    # Modèle par défaut
                    action = np.ones(market_data_array.shape[1]) / market_data_array.shape[1]
                
                obs, reward, terminated, truncated, info = test_env.step(action)
                
                total_rewards.append(reward)
                portfolio_values.append(info['portfolio_value'])
                
                # Compter les trades significatifs
                if np.sum(np.abs(action - obs['weights'])) > 0.01:
                    trades_count += 1
                
                if terminated or truncated:
                    break
            
            # Calculer les métriques
            final_value = portfolio_values[-1]
            total_return = (final_value / request.initial_portfolio_value - 1) * 100
            
            returns = np.diff(portfolio_values) / portfolio_values[:-1]
            volatility = np.std(returns) * np.sqrt(252) * 100  # Annualisée
            
            sharpe_ratio = np.mean(returns) / max(np.std(returns), 1e-6) * np.sqrt(252)
            
            # Drawdown
            running_max = np.maximum.accumulate(portfolio_values)
            drawdowns = (portfolio_values - running_max) / running_max
            max_drawdown = abs(np.min(drawdowns)) * 100
            
            return BacktestResponse(
                final_value=final_value,
                total_return=total_return,
                volatility=volatility,
                sharpe_ratio=sharpe_ratio,
                max_drawdown=max_drawdown,
                trades_count=trades_count,
                metadata={
                    "backtest_time": datetime.now().isoformat(),
                    "periods": len(portfolio_values),
                    "model_used": request.model_version
                }
            )
            
        except Exception as e:
            logger.error(f"❌ Erreur backtest: {e}")
            raise HTTPException(status_code=500, detail=f"Erreur lors du backtest: {str(e)}")
    
    @app.on_event("startup")
    async def startup_event():
        """Événement de démarrage."""
        logger.info("🚀 Portfolio RL API démarrée")
        if model_manager:
            logger.info(f"📦 {len(model_manager.models)} modèle(s) chargé(s)")
        else:
            logger.warning("⚠️ Gestionnaire de modèles non disponible")
    
    @app.on_event("shutdown") 
    async def shutdown_event():
        """Événement d'arrêt."""
        logger.info("🛑 Portfolio RL API arrêtée")

else:
    # Fallback si FastAPI n'est pas disponible
    app = None
    logger.error("❌ FastAPI non disponible")

# ================================
# POINT D'ENTRÉE
# ================================

if __name__ == "__main__":
    if not FASTAPI_AVAILABLE:
        print("❌ FastAPI non disponible. Installez avec: pip install fastapi uvicorn")
        sys.exit(1)
    
    # Configuration du serveur
    host = os.environ.get("API_HOST", "0.0.0.0")
    port = int(os.environ.get("API_PORT", 8000))
    
    print(f"🚀 Démarrage de l'API sur http://{host}:{port}")
    print(f"📚 Documentation: http://{host}:{port}/docs")
    
    uvicorn.run(
        "src.api.main:app",
        host=host,
        port=port,
        reload=True,
        log_level="info"
    )