"""
API FastAPI corrigée pour Portfolio RL
"""
import os
import sys
import time
import logging
from datetime import datetime
from typing import Dict, Any, Optional, List
from contextlib import asynccontextmanager

# Ajouter le répertoire racine au PYTHONPATH
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

# Imports FastAPI
try:
    from fastapi import FastAPI, HTTPException, BackgroundTasks
    from fastapi.middleware.cors import CORSMiddleware
    from fastapi.responses import JSONResponse, PlainTextResponse
    from pydantic import BaseModel, Field
    import uvicorn
    FASTAPI_AVAILABLE = True
except ImportError as e:
    print(f"❌ FastAPI non disponible: {e}")
    print("💡 Installez avec: pip install fastapi uvicorn")
    sys.exit(1)

# Imports du monitoring (avec gestion d'erreurs)
try:
    from src.monitoring.simple_metrics import portfolio_metrics, PROMETHEUS_AVAILABLE
    MONITORING_AVAILABLE = True
except ImportError as e:
    print(f"⚠️ Monitoring non disponible: {e}")
    portfolio_metrics = None
    PROMETHEUS_AVAILABLE = False
    MONITORING_AVAILABLE = False

# Imports du modèle (avec gestion d'erreurs)
try:
    from stable_baselines3 import PPO
    MODEL_AVAILABLE = True
except ImportError:
    print("⚠️ Stable-Baselines3 non disponible")
    MODEL_AVAILABLE = False

# Configuration du logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# =============================================================================
# MODÈLES PYDANTIC
# =============================================================================

class HealthResponse(BaseModel):
    """Réponse de santé de l'API."""
    status: str
    timestamp: datetime
    version: str = "1.0.0"
    services: Dict[str, str]

class PredictionRequest(BaseModel):
    """Requête de prédiction d'allocation."""
    portfolio_id: str
    market_data: List[List[List[float]]] = Field(..., description="Données de marché [features][stocks][time]")
    current_weights: List[float] = Field(..., description="Poids actuels du portefeuille")
    portfolio_value: float = Field(..., gt=0, description="Valeur actuelle du portefeuille")
    risk_tolerance: Optional[float] = Field(0.5, ge=0, le=1)

class PredictionResponse(BaseModel):
    """Réponse de prédiction."""
    portfolio_id: str
    recommended_weights: List[float]
    expected_return: Optional[float] = None
    expected_risk: Optional[float] = None
    confidence: float
    model_version: str
    timestamp: datetime
    rebalancing_needed: bool

# =============================================================================
# GESTIONNAIRE DE MODÈLES SIMPLIFIÉ
# =============================================================================

class SimpleModelManager:
    """Gestionnaire de modèles simplifié."""
    
    def __init__(self):
        self.models = {}
        self.default_model = None
        self._load_models()
    
    def _load_models(self):
        """Charge les modèles disponibles."""
        try:
            # Chercher les modèles dans les logs d'entraînement
            models_dir = "logs/training"
            if os.path.exists(models_dir):
                for folder in os.listdir(models_dir):
                    folder_path = os.path.join(models_dir, folder)
                    if os.path.isdir(folder_path):
                        # Chercher best_model ou final_model
                        for model_name in ["best_model", "final_model"]:
                            model_path = os.path.join(folder_path, f"{model_name}.zip")
                            if os.path.exists(model_path):
                                try:
                                    if MODEL_AVAILABLE:
                                        model = PPO.load(model_path.replace('.zip', ''))
                                        self.models[f"{folder}_{model_name}"] = model
                                        if self.default_model is None:
                                            self.default_model = model
                                        logger.info(f"✅ Modèle chargé: {model_path}")
                                except Exception as e:
                                    logger.warning(f"⚠️ Erreur chargement {model_path}: {e}")
            
            if not self.models:
                logger.warning("⚠️ Aucun modèle trouvé, utilisation de l'allocation aléatoire")
                
        except Exception as e:
            logger.error(f"❌ Erreur lors du chargement des modèles: {e}")
    
    def predict(self, market_data, current_weights, portfolio_value):
        """Effectue une prédiction."""
        try:
            if self.default_model and MODEL_AVAILABLE:
                # Créer l'observation pour le modèle
                import numpy as np
                
                # Convertir les données en format attendu par le modèle
                market_array = np.array(market_data, dtype=np.float32)
                weights_array = np.array(current_weights, dtype=np.float32)
                value_array = np.array([portfolio_value / 10000], dtype=np.float32)  # Normaliser
                
                # Format d'observation pour l'environnement RL
                obs = {
                    'market_data': market_array,
                    'weights': weights_array,
                    'portfolio_value': value_array
                }
                
                # Prédiction
                action, _ = self.default_model.predict(obs, deterministic=True)
                
                # Normaliser les poids
                if np.sum(action) > 0:
                    weights = action / np.sum(action)
                else:
                    weights = np.ones(len(action)) / len(action)
                
                return {
                    "weights": weights.tolist(),
                    "confidence": 0.8,
                    "model_version": "rl_model_v1"
                }
            else:
                # Allocation aléatoire si pas de modèle
                import numpy as np
                n_assets = len(current_weights)
                weights = np.random.dirichlet(np.ones(n_assets))
                return {
                    "weights": weights.tolist(),
                    "confidence": 0.5,
                    "model_version": "random_allocation"
                }
                
        except Exception as e:
            logger.error(f"❌ Erreur prédiction: {e}")
            # Fallback: allocation égale
            n_assets = len(current_weights)
            weights = [1.0/n_assets] * n_assets
            return {
                "weights": weights,
                "confidence": 0.3,
                "model_version": "equal_weight_fallback"
            }

# Instance globale du gestionnaire
model_manager = SimpleModelManager()

# =============================================================================
# MIDDLEWARE DE MONITORING SIMPLIFIÉ
# =============================================================================

class SimpleMetricsMiddleware:
    """Middleware de métriques simplifié."""
    
    def __init__(self, app):
        self.app = app
    
    async def __call__(self, scope, receive, send):
        if scope["type"] == "http" and MONITORING_AVAILABLE and portfolio_metrics:
            start_time = time.time()
            
            # Wrapper pour capturer la réponse
            async def send_wrapper(message):
                if message["type"] == "http.response.start":
                    # Enregistrer la métrique
                    duration = time.time() - start_time
                    method = scope.get("method", "GET")
                    path = scope.get("path", "/")
                    status = message.get("status", 200)
                    
                    try:
                        portfolio_metrics.record_api_request(method, path, status, duration)
                    except Exception as e:
                        logger.debug(f"Erreur métrique: {e}")
                
                await send(message)
            
            await self.app(scope, receive, send_wrapper)
        else:
            await self.app(scope, receive, send)

# =============================================================================
# CRÉATION DE L'APPLICATION
# =============================================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Gestionnaire du cycle de vie de l'application."""
    logger.info("🚀 Démarrage de l'API Portfolio RL")
    yield
    logger.info("🛑 Arrêt de l'API Portfolio RL")

app = FastAPI(
    title="Portfolio RL API",
    description="API d'optimisation de portefeuille avec apprentissage par renforcement",
    version="1.0.0",
    lifespan=lifespan
)

# Middleware CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Middleware de monitoring
if MONITORING_AVAILABLE:
    app.add_middleware(SimpleMetricsMiddleware)

# =============================================================================
# ENDPOINTS
# =============================================================================

@app.get("/", response_model=Dict[str, str])
async def root():
    """Endpoint racine."""
    return {
        "message": "Portfolio RL API",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/health",
        "metrics": "/metrics" if MONITORING_AVAILABLE else "unavailable"
    }

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Vérification de santé."""
    services = {
        "api": "healthy",
        "models": "healthy" if model_manager.default_model else "no_model",
        "monitoring": "healthy" if MONITORING_AVAILABLE else "disabled"
    }
    
    return HealthResponse(
        status="healthy",
        timestamp=datetime.now(),
        services=services
    )

@app.post("/predict", response_model=PredictionResponse)
async def predict_allocation(request: PredictionRequest):
    """Prédit l'allocation optimale de portefeuille."""
    start_time = time.time()
    
    try:
        # Effectuer la prédiction
        prediction = model_manager.predict(
            request.market_data,
            request.current_weights,
            request.portfolio_value
        )
        
        # Calculer si un rééquilibrage est nécessaire
        import numpy as np
        current_weights = np.array(request.current_weights)
        new_weights = np.array(prediction["weights"])
        weight_change = np.sum(np.abs(new_weights - current_weights))
        rebalancing_needed = weight_change > 0.05  # Seuil de 5%
        
        # Enregistrer les métriques de prédiction
        if MONITORING_AVAILABLE and portfolio_metrics:
            duration = time.time() - start_time
            portfolio_metrics.record_prediction(
                prediction["model_version"], 
                "allocation", 
                duration, 
                prediction["confidence"]
            )
        
        return PredictionResponse(
            portfolio_id=request.portfolio_id,
            recommended_weights=prediction["weights"],
            confidence=prediction["confidence"],
            model_version=prediction["model_version"],
            timestamp=datetime.now(),
            rebalancing_needed=rebalancing_needed
        )
        
    except Exception as e:
        logger.error(f"❌ Erreur prédiction: {e}")
        raise HTTPException(status_code=500, detail=f"Erreur prédiction: {str(e)}")

@app.get("/models")
async def list_models():
    """Liste les modèles disponibles."""
    return {
        "models": list(model_manager.models.keys()),
        "default_model": "rl_model_v1" if model_manager.default_model else "none",
        "model_available": MODEL_AVAILABLE,
        "total_models": len(model_manager.models)
    }

# Endpoint de métriques (si monitoring disponible)
if MONITORING_AVAILABLE and portfolio_metrics:
    @app.get("/metrics", response_class=PlainTextResponse)
    async def get_metrics():
        """Endpoint pour les métriques Prometheus."""
        try:
            return portfolio_metrics.get_metrics()
        except Exception as e:
            logger.error(f"Erreur métriques: {e}")
            return f"# Erreur: {e}\n"

@app.post("/test/prediction")
async def test_prediction():
    """Endpoint de test pour vérifier l'API."""
    import numpy as np
    
    # Données de test
    test_request = PredictionRequest(
        portfolio_id="test_portfolio",
        market_data=np.random.rand(5, 5, 30).tolist(),  # 5 features, 5 stocks, 30 periods
        current_weights=[0.2, 0.2, 0.2, 0.2, 0.2],
        portfolio_value=100000.0
    )
    
    return await predict_allocation(test_request)

# =============================================================================
# POINT D'ENTRÉE
# =============================================================================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="0.0.0.0", help="Host pour l'API")
    parser.add_argument("--port", type=int, default=8000, help="Port pour l'API")
    parser.add_argument("--reload", action="store_true", help="Mode reload")
    
    args = parser.parse_args()
    
    print(f"🚀 Démarrage de l'API Portfolio RL")
    print(f"📍 URL: http://{args.host}:{args.port}")
    print(f"📚 Documentation: http://{args.host}:{args.port}/docs")
    if MONITORING_AVAILABLE:
        print(f"📊 Métriques: http://{args.host}:{args.port}/metrics")
    
    uvicorn.run(
        "main:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
        log_level="info"
    )