"""
API FastAPI pour l'optimisation de portefeuille avec RL - VERSION CORRIGÉE.
Point d'entrée principal avec métriques Prometheus intégrées.
"""
import os
import sys
import time
import logging
import traceback
from datetime import datetime
from typing import Dict, List, Optional, Any
from pathlib import Path

# Ajouter le répertoire racine au PYTHONPATH
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# Imports FastAPI
from fastapi import FastAPI, HTTPException, BackgroundTasks, Depends
from fastapi.responses import JSONResponse, PlainTextResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from pydantic import BaseModel, Field
import numpy as np

# Configuration du logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# =============================================================================
# MÉTRIQUES PROMETHEUS INTÉGRÉES
# =============================================================================

# Import et initialisation des métriques directement dans ce fichier
try:
    from prometheus_client import (
        Counter, Histogram, Gauge, generate_latest, 
        CollectorRegistry, CONTENT_TYPE_LATEST
    )
    PROMETHEUS_AVAILABLE = True
    logger.info("✅ Prometheus client disponible")
    
    # Créer un registre custom pour éviter les conflits
    CUSTOM_REGISTRY = CollectorRegistry()
    
    # Métriques principales
    API_REQUESTS = Counter(
        'portfolio_api_requests_total',
        'Total API requests',
        ['method', 'endpoint', 'status_code'],
        registry=CUSTOM_REGISTRY
    )
    
    API_DURATION = Histogram(
        'portfolio_api_request_duration_seconds',
        'API request duration',
        ['method', 'endpoint'],
        registry=CUSTOM_REGISTRY,
        buckets=(0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0)
    )
    
    MODEL_PREDICTIONS = Counter(
        'portfolio_model_predictions_total',
        'Total model predictions',
        ['model_version', 'prediction_type'],
        registry=CUSTOM_REGISTRY
    )
    
    MODEL_CONFIDENCE = Histogram(
        'portfolio_model_prediction_confidence',
        'Model prediction confidence',
        ['model_version'],
        registry=CUSTOM_REGISTRY,
        buckets=(0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 1.0)
    )
    
    PORTFOLIO_VALUE = Gauge(
        'portfolio_portfolio_value_dollars',
        'Portfolio value in dollars',
        ['portfolio_id'],
        registry=CUSTOM_REGISTRY
    )
    
    logger.info("✅ Métriques Prometheus initialisées")
    
except ImportError as e:
    logger.warning(f"⚠️ Prometheus non disponible: {e}")
    PROMETHEUS_AVAILABLE = False
    CUSTOM_REGISTRY = None
    API_REQUESTS = None
    API_DURATION = None
    MODEL_PREDICTIONS = None
    MODEL_CONFIDENCE = None
    PORTFOLIO_VALUE = None

# Fonctions helper pour les métriques
def record_api_request(method: str, endpoint: str, status_code: int, duration: float):
    """Enregistre une requête API."""
    if PROMETHEUS_AVAILABLE and API_REQUESTS and API_DURATION:
        try:
            API_REQUESTS.labels(
                method=method,
                endpoint=endpoint,
                status_code=str(status_code)
            ).inc()
            
            API_DURATION.labels(
                method=method,
                endpoint=endpoint
            ).observe(duration)
        except Exception as e:
            logger.warning(f"⚠️ Erreur métrique API: {e}")

def record_prediction(model_version: str, prediction_type: str, confidence: float):
    """Enregistre une prédiction."""
    if PROMETHEUS_AVAILABLE and MODEL_PREDICTIONS and MODEL_CONFIDENCE:
        try:
            MODEL_PREDICTIONS.labels(
                model_version=model_version,
                prediction_type=prediction_type
            ).inc()
            
            MODEL_CONFIDENCE.labels(
                model_version=model_version
            ).observe(confidence)
        except Exception as e:
            logger.warning(f"⚠️ Erreur métrique prédiction: {e}")

def update_portfolio_value(portfolio_id: str, value: float):
    """Met à jour la valeur d'un portefeuille."""
    if PROMETHEUS_AVAILABLE and PORTFOLIO_VALUE:
        try:
            PORTFOLIO_VALUE.labels(portfolio_id=portfolio_id).set(value)
        except Exception as e:
            logger.warning(f"⚠️ Erreur métrique portefeuille: {e}")

# =============================================================================
# IMPORTS CONDITIONNELS MODÈLES
# =============================================================================

try:
    from stable_baselines3 import PPO
    MODEL_AVAILABLE = True
    logger.info("✅ Stable-Baselines3 disponible")
except ImportError as e:
    logger.warning(f"⚠️ Stable-Baselines3 non disponible: {e}")
    MODEL_AVAILABLE = False

# =============================================================================
# MODÈLES PYDANTIC (inchangés)
# =============================================================================

class HealthResponse(BaseModel):
    """Réponse de santé de l'API."""
    status: str = "healthy"
    timestamp: str
    service: str = "portfolio-rl-api"
    version: str = "1.0.0"
    dependencies: Dict[str, bool]

class PredictionRequest(BaseModel):
    """Requête de prédiction d'allocation."""
    portfolio_id: str = Field(..., description="Identifiant unique du portefeuille")
    market_data: List[List[List[float]]] = Field(..., description="Données de marché [features, stocks, periods]")
    current_weights: List[float] = Field(..., description="Poids actuels du portefeuille")
    portfolio_value: float = Field(..., gt=0, description="Valeur actuelle du portefeuille")
    risk_tolerance: float = Field(0.5, ge=0, le=1, description="Tolérance au risque (0-1)")
    model_version: str = Field("latest", description="Version du modèle à utiliser")

class PredictionResponse(BaseModel):
    """Réponse de prédiction d'allocation."""
    portfolio_id: str
    recommended_weights: List[float]
    confidence: float
    model_version: str
    rebalancing_needed: bool
    expected_return: Optional[float] = None
    expected_risk: Optional[float] = None
    timestamp: str

# =============================================================================
# GESTIONNAIRE DE MODÈLES (inchangé)
# =============================================================================

class ModelManager:
    """Gestionnaire des modèles de ML."""
    
    def __init__(self):
        self.models = {}
        self.model_dir = Path("models")
        self.model_dir.mkdir(exist_ok=True)
        self._load_available_models()
    
    def _load_available_models(self):
        """Charge les modèles disponibles."""
        try:
            # Chercher les modèles PPO
            for model_file in self.model_dir.glob("**/*.zip"):
                model_name = model_file.stem
                try:
                    if MODEL_AVAILABLE:
                        model = PPO.load(str(model_file))
                        self.models[model_name] = {
                            'model': model,
                            'path': str(model_file),
                            'loaded_at': datetime.now(),
                            'type': 'PPO'
                        }
                        logger.info(f"✅ Modèle chargé: {model_name}")
                except Exception as e:
                    logger.warning(f"⚠️ Impossible de charger {model_name}: {e}")
            
            # Modèle factice si aucun modèle réel disponible
            if not self.models:
                self.models['dummy'] = {
                    'model': None,
                    'path': 'dummy',
                    'loaded_at': datetime.now(),
                    'type': 'dummy'
                }
                logger.info("🔧 Modèle factice créé pour les tests")
                
        except Exception as e:
            logger.error(f"❌ Erreur lors du chargement des modèles: {e}")
    
    def get_model(self, version: str = "latest"):
        """Récupère un modèle par version."""
        if version == "latest" and self.models:
            real_models = {k: v for k, v in self.models.items() if v['type'] != 'dummy'}
            if real_models:
                latest_key = max(real_models.keys(), key=lambda k: real_models[k]['loaded_at'])
                return real_models[latest_key]
            else:
                return list(self.models.values())[0]
        
        return self.models.get(version)
    
    def list_models(self) -> List[str]:
        """Liste les modèles disponibles."""
        return list(self.models.keys())
    
    def predict(self, model_version: str, market_data: np.ndarray, current_weights: np.ndarray) -> Dict[str, Any]:
        """Effectue une prédiction."""
        model_info = self.get_model(model_version)
        
        if not model_info:
            raise ValueError(f"Modèle {model_version} non trouvé")
        
        start_time = time.time()
        
        try:
            if model_info['type'] == 'dummy':
                # Prédiction factice pour les tests
                n_assets = len(current_weights)
                random_weights = np.random.dirichlet(np.ones(n_assets))
                confidence = 0.75 + np.random.random() * 0.2
                
                result = {
                    'weights': random_weights.tolist(),
                    'confidence': confidence,
                    'model_type': 'dummy',
                    'prediction_time': time.time() - start_time
                }
            else:
                # Prédiction réelle avec le modèle PPO
                model = model_info['model']
                
                obs = {
                    'market_data': market_data.astype(np.float32),
                    'weights': current_weights.astype(np.float32),
                    'portfolio_value': np.array([1.0], dtype=np.float32)
                }
                
                action, _states = model.predict(obs, deterministic=True)
                
                weights = np.array(action)
                weights = np.maximum(weights, 0)
                weights = weights / np.sum(weights)
                
                entropy = -np.sum(weights * np.log(weights + 1e-8))
                max_entropy = np.log(len(weights))
                confidence = 1.0 - (entropy / max_entropy)
                
                result = {
                    'weights': weights.tolist(),
                    'confidence': float(confidence),
                    'model_type': 'PPO',
                    'prediction_time': time.time() - start_time
                }
            
            # Enregistrer les métriques
            record_prediction(model_version, "allocation", result['confidence'])
            
            return result
            
        except Exception as e:
            logger.error(f"❌ Erreur lors de la prédiction: {e}")
            raise

# =============================================================================
# APPLICATION FASTAPI
# =============================================================================

# Gestionnaire de modèles
model_manager = ModelManager()

# Application FastAPI
app = FastAPI(
    title="Portfolio RL API",
    description="API pour l'optimisation de portefeuille avec Reinforcement Learning",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Middleware CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Middleware de compression
app.add_middleware(GZipMiddleware, minimum_size=1000)

# Middleware de métriques CORRIGÉ
@app.middleware("http")
async def metrics_middleware(request, call_next):
    start_time = time.time()
    response = await call_next(request)
    process_time = time.time() - start_time
    
    # Enregistrer les métriques
    record_api_request(
        request.method,
        str(request.url.path),
        response.status_code,
        process_time
    )
    
    return response

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
        "health": "/health"
    }

@app.get("/health", response_model=HealthResponse)
async def health():
    """Endpoint de santé."""
    dependencies = {
        "prometheus": PROMETHEUS_AVAILABLE,
        "ml_models": MODEL_AVAILABLE,
        "models_loaded": len(model_manager.models) > 0
    }
    
    return HealthResponse(
        timestamp=datetime.now().isoformat(),
        dependencies=dependencies
    )

@app.get("/models")
async def list_models():
    """Liste les modèles disponibles."""
    models = []
    for name, info in model_manager.models.items():
        models.append({
            "name": name,
            "type": info['type'],
            "loaded_at": info['loaded_at'].isoformat(),
            "path": info['path']
        })
    
    return {"models": models}

@app.post("/predict", response_model=PredictionResponse)
async def predict_allocation(request: PredictionRequest):
    """Prédiction d'allocation de portefeuille."""
    try:
        # Validation des données
        market_data = np.array(request.market_data, dtype=np.float32)
        current_weights = np.array(request.current_weights, dtype=np.float32)
        
        # Vérifications de base
        if len(market_data.shape) != 3:
            raise HTTPException(
                status_code=422, 
                detail=f"market_data doit avoir 3 dimensions [features, stocks, periods], reçu: {market_data.shape}"
            )
        
        if abs(np.sum(current_weights) - 1.0) > 0.01:
            raise HTTPException(
                status_code=422,
                detail=f"La somme des poids actuels doit être 1.0, reçu: {np.sum(current_weights)}"
            )
        
        n_features, n_stocks, n_periods = market_data.shape
        
        if len(current_weights) != n_stocks:
            raise HTTPException(
                status_code=422,
                detail=f"Nombre de poids ({len(current_weights)}) != nombre d'actions ({n_stocks})"
            )
        
        # Prédiction
        result = model_manager.predict(
            request.model_version,
            market_data,
            current_weights
        )
        
        # Calculer si un rééquilibrage est nécessaire
        weight_diff = np.abs(np.array(result['weights']) - current_weights)
        rebalancing_needed = np.max(weight_diff) > 0.05
        
        # Mettre à jour les métriques de portefeuille
        update_portfolio_value(request.portfolio_id, request.portfolio_value)
        
        return PredictionResponse(
            portfolio_id=request.portfolio_id,
            recommended_weights=result['weights'],
            confidence=result['confidence'],
            model_version=request.model_version,
            rebalancing_needed=rebalancing_needed,
            timestamp=datetime.now().isoformat()
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Erreur lors de la prédiction: {e}")
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Erreur interne: {str(e)}")

@app.get("/metrics", response_class=PlainTextResponse)
async def get_metrics():
    """Endpoint des métriques Prometheus."""
    if PROMETHEUS_AVAILABLE and CUSTOM_REGISTRY:
        try:
            # Générer les métriques depuis notre registre custom
            metrics_output = generate_latest(CUSTOM_REGISTRY)
            return metrics_output.decode('utf-8')
        except Exception as e:
            logger.error(f"❌ Erreur génération métriques: {e}")
            return f"# Erreur génération métriques: {e}\n"
    else:
        return "# Métriques Prometheus non disponibles\n# Installez: pip install prometheus-client\n"

@app.post("/test/prediction")
async def test_prediction():
    """Endpoint de test pour vérifier l'API."""
    try:
        # Données de test basiques
        test_data = np.random.rand(5, 3, 30).astype(np.float32)
        test_weights = np.array([0.33, 0.33, 0.34])
        
        result = model_manager.predict("latest", test_data, test_weights)
        
        # Enregistrer dans les métriques
        record_prediction("latest", "test", result['confidence'])
        update_portfolio_value("test_portfolio", 10000.0)
        
        return {
            "status": "success",
            "test_weights": test_weights.tolist(),
            "predicted_weights": result['weights'],
            "confidence": result['confidence'],
            "model_type": result['model_type'],
            "message": "Test de prédiction réussi"
        }
        
    except Exception as e:
        logger.error(f"❌ Erreur test: {e}")
        return {
            "status": "error",
            "message": str(e)
        }

# =============================================================================
# ÉVÉNEMENTS
# =============================================================================

@app.on_event("startup")
async def startup_event():
    """Événements au démarrage de l'application."""
    logger.info("🚀 Démarrage de l'API Portfolio RL")
    logger.info(f"📦 Modèles chargés: {len(model_manager.models)}")
    logger.info(f"📊 Monitoring: {'✅ Activé' if PROMETHEUS_AVAILABLE else '❌ Désactivé'}")
    
    if PROMETHEUS_AVAILABLE:
        logger.info("📈 Métriques Prometheus initialisées avec registre custom")

@app.on_event("shutdown")
async def shutdown_event():
    """Événements à l'arrêt de l'application."""
    logger.info("🛑 Arrêt de l'API Portfolio RL")

# =============================================================================
# POINT D'ENTRÉE
# =============================================================================

if __name__ == "__main__":
    import uvicorn
    
    host = os.getenv("API_HOST", "0.0.0.0")
    port = int(os.getenv("API_PORT", "8000"))
    
    logger.info(f"🌐 Démarrage du serveur sur {host}:{port}")
    
    uvicorn.run(
        "main:app",
        host=host,
        port=port,
        reload=False,
        log_level="info",
        access_log=True
    )