"""
API FastAPI Production avec intégration MLflow complète.
Remplace les modèles factices par vos vrais modèles MLflow.
"""
import os
import sys
import time
import logging
from datetime import datetime
from typing import Dict, List, Optional, Any
from pathlib import Path

# Ajouter le répertoire racine au PYTHONPATH
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.responses import PlainTextResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import numpy as np

# Import du gestionnaire MLflow
from models.mlflow_manager import get_mlflow_manager, refresh_mlflow_models

# Configuration du logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Métriques Prometheus (identique à la version précédente)
try:
    from prometheus_client import (
        Counter, Histogram, Gauge, generate_latest, 
        CollectorRegistry, CONTENT_TYPE_LATEST
    )
    PROMETHEUS_AVAILABLE = True
    
    CUSTOM_REGISTRY = CollectorRegistry()
    
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
        registry=CUSTOM_REGISTRY
    )
    
    MODEL_PREDICTIONS = Counter(
        'portfolio_model_predictions_total',
        'Total model predictions',
        ['model_name', 'model_type', 'model_stage'],
        registry=CUSTOM_REGISTRY
    )
    
    MODEL_CONFIDENCE = Histogram(
        'portfolio_model_prediction_confidence',
        'Model prediction confidence',
        ['model_name'],
        registry=CUSTOM_REGISTRY
    )
    
    PORTFOLIO_VALUES = Gauge(
        'portfolio_portfolio_value_dollars',
        'Portfolio values',
        ['portfolio_id'],
        registry=CUSTOM_REGISTRY
    )
    
    MODEL_ERRORS = Counter(
        'portfolio_model_errors_total',
        'Model prediction errors',
        ['model_name', 'error_type'],
        registry=CUSTOM_REGISTRY
    )
    
except ImportError:
    PROMETHEUS_AVAILABLE = False
    logger.warning("⚠️ Prometheus non disponible")

# Fonctions de métriques
def record_api_request(method: str, endpoint: str, status_code: int, duration: float):
    if PROMETHEUS_AVAILABLE and API_REQUESTS:
        try:
            API_REQUESTS.labels(method=method, endpoint=endpoint, status_code=str(status_code)).inc()
            API_DURATION.labels(method=method, endpoint=endpoint).observe(duration)
        except Exception as e:
            logger.warning(f"Erreur métrique API: {e}")

def record_prediction(model_name: str, model_type: str, model_stage: str, confidence: float):
    if PROMETHEUS_AVAILABLE and MODEL_PREDICTIONS:
        try:
            MODEL_PREDICTIONS.labels(
                model_name=model_name, 
                model_type=model_type,
                model_stage=model_stage
            ).inc()
            MODEL_CONFIDENCE.labels(model_name=model_name).observe(confidence)
        except Exception as e:
            logger.warning(f"Erreur métrique prédiction: {e}")

def record_model_error(model_name: str, error_type: str):
    if PROMETHEUS_AVAILABLE and MODEL_ERRORS:
        try:
            MODEL_ERRORS.labels(model_name=model_name, error_type=error_type).inc()
        except Exception as e:
            logger.warning(f"Erreur métrique erreur: {e}")

def update_portfolio_value(portfolio_id: str, value: float):
    if PROMETHEUS_AVAILABLE and PORTFOLIO_VALUES:
        try:
            PORTFOLIO_VALUES.labels(portfolio_id=portfolio_id).set(value)
        except Exception as e:
            logger.warning(f"Erreur métrique portefeuille: {e}")

# =============================================================================
# MODÈLES PYDANTIC ÉTENDUS
# =============================================================================

class HealthResponse(BaseModel):
    status: str = "healthy"
    timestamp: str
    service: str = "portfolio-rl-api-production"
    version: str = "2.0.0"
    dependencies: Dict[str, bool]
    models_info: Dict[str, Any]

class ModelInfo(BaseModel):
    name: str
    type: str
    version: str
    stage: str
    loaded_at: str
    source: str
    uri: Optional[str] = None
    metrics: Optional[Dict[str, float]] = None
    description: Optional[str] = None

class PredictionRequest(BaseModel):
    portfolio_id: str = Field(..., description="ID unique du portefeuille")
    market_data: List[List[List[float]]] = Field(..., description="Données de marché [features, stocks, periods]")
    current_weights: List[float] = Field(..., description="Poids actuels du portefeuille")
    portfolio_value: float = Field(..., gt=0, description="Valeur actuelle du portefeuille")
    risk_tolerance: float = Field(0.5, ge=0, le=1, description="Tolérance au risque (0-1)")
    model_name: str = Field("latest", description="Nom du modèle à utiliser")
    use_production: bool = Field(True, description="Utiliser les modèles en production en priorité")

class PredictionResponse(BaseModel):
    portfolio_id: str
    recommended_weights: List[float]
    confidence: float
    model_name: str
    model_type: str
    model_version: str
    model_stage: str
    rebalancing_needed: bool
    weight_changes: Dict[str, float]
    expected_return: Optional[float] = None
    expected_risk: Optional[float] = None
    prediction_metadata: Dict[str, Any]
    timestamp: str

class ModelRefreshResponse(BaseModel):
    status: str
    models_reloaded: int
    mlflow_models: int
    local_models: int
    timestamp: str

# =============================================================================
# APPLICATION FASTAPI
# =============================================================================

app = FastAPI(
    title="Portfolio RL API - Production",
    description="API de production pour l'optimisation de portefeuille avec RL et MLflow",
    version="2.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"]
)

@app.middleware("http")
async def metrics_middleware(request, call_next):
    start_time = time.time()
    response = await call_next(request)
    duration = time.time() - start_time
    
    record_api_request(request.method, str(request.url.path), response.status_code, duration)
    return response

# Instance du gestionnaire MLflow
model_manager = None

# =============================================================================
# ENDPOINTS
# =============================================================================

@app.get("/")
async def root():
    return {
        "message": "Portfolio RL API - Production avec MLflow",
        "version": "2.0.0",
        "features": [
            "MLflow model registry integration",
            "Real-time predictions",
            "Prometheus monitoring",
            "Production/Staging model support",
            "Automatic model refresh"
        ],
        "endpoints": {
            "health": "/health",
            "models": "/models",
            "predict": "/predict",
            "metrics": "/metrics",
            "docs": "/docs"
        }
    }

@app.get("/health", response_model=HealthResponse)
async def health():
    """Endpoint de santé avec informations sur les modèles."""
    global model_manager
    
    if model_manager is None:
        model_manager = get_mlflow_manager()
    
    models_list = model_manager.list_models()
    
    # Compter les modèles par type
    mlflow_models = len([m for m in models_list if m['source'] == 'mlflow'])
    local_models = len([m for m in models_list if m['source'] == 'local'])
    
    dependencies = {
        "mlflow": model_manager.client is not None,
        "prometheus": PROMETHEUS_AVAILABLE,
        "models_loaded": len(models_list) > 0,
        "production_models": len([m for m in models_list if m.get('stage') == 'Production']) > 0
    }
    
    models_info = {
        "total": len(models_list),
        "mlflow": mlflow_models,
        "local": local_models,
        "production": len([m for m in models_list if m.get('stage') == 'Production']),
        "staging": len([m for m in models_list if m.get('stage') == 'Staging'])
    }
    
    return HealthResponse(
        timestamp=datetime.now().isoformat(),
        dependencies=dependencies,
        models_info=models_info
    )

@app.get("/models", response_model=List[ModelInfo])
async def list_models():
    """Liste tous les modèles disponibles avec leurs métadonnées."""
    global model_manager
    
    if model_manager is None:
        model_manager = get_mlflow_manager()
    
    models_list = model_manager.list_models()
    
    return [
        ModelInfo(
            name=model['name'],
            type=model['type'],
            version=model['version'],
            stage=model['stage'],
            loaded_at=model['loaded_at'],
            source=model['source'],
            uri=model.get('uri'),
            metrics=model.get('metrics'),
            description=model.get('description')
        )
        for model in models_list
    ]

@app.get("/models/{model_name}")
async def get_model_details(model_name: str):
    """Récupère les détails d'un modèle spécifique."""
    global model_manager
    
    if model_manager is None:
        model_manager = get_mlflow_manager()
    
    model_info = model_manager.get_model_info(model_name)
    
    if not model_info:
        raise HTTPException(status_code=404, detail=f"Modèle {model_name} non trouvé")
    
    return model_info

@app.post("/models/refresh", response_model=ModelRefreshResponse)
async def refresh_models():
    """Recharge tous les modèles depuis MLflow."""
    global model_manager
    
    if model_manager is None:
        model_manager = get_mlflow_manager()
    
    try:
        model_manager.refresh_models()
        models_list = model_manager.list_models()
        
        mlflow_count = len([m for m in models_list if m['source'] == 'mlflow'])
        local_count = len([m for m in models_list if m['source'] == 'local'])
        
        return ModelRefreshResponse(
            status="success",
            models_reloaded=len(models_list),
            mlflow_models=mlflow_count,
            local_models=local_count,
            timestamp=datetime.now().isoformat()
        )
        
    except Exception as e:
        logger.error(f"Erreur lors du rechargement des modèles: {e}")
        raise HTTPException(status_code=500, detail=f"Erreur rechargement: {str(e)}")

@app.post("/predict", response_model=PredictionResponse)
async def predict_allocation(request: PredictionRequest):
    """Prédiction d'allocation avec modèles MLflow."""
    global model_manager
    
    if model_manager is None:
        model_manager = get_mlflow_manager()
    
    try:
        # Validation des données
        market_data = np.array(request.market_data, dtype=np.float32)
        current_weights = np.array(request.current_weights, dtype=np.float32)
        
        # Vérifications
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
        
        # Déterminer le modèle à utiliser
        model_name = request.model_name
        if model_name == "latest":
            model_name = "latest"
        
        # Effectuer la prédiction
        try:
            result = model_manager.predict(model_name, market_data, current_weights)
        except Exception as e:
            # Enregistrer l'erreur dans les métriques
            record_model_error(model_name, "prediction_error")
            logger.error(f"Erreur prédiction avec {model_name}: {e}")
            raise HTTPException(status_code=500, detail=f"Erreur prédiction: {str(e)}")
        
        # Calculer les changements de poids
        recommended_weights = np.array(result['weights'])
        weight_changes = {}
        for i, (old_w, new_w) in enumerate(zip(current_weights, recommended_weights)):
            weight_changes[f'asset_{i}'] = float(new_w - old_w)
        
        # Déterminer si un rééquilibrage est nécessaire
        max_change = max(abs(change) for change in weight_changes.values())
        rebalancing_needed = max_change > 0.05  # Seuil de 5%
        
        # Enregistrer les métriques
        record_prediction(
            result['model_name'],
            result['model_type'],
            result['model_stage'],
            result['confidence']
        )
        update_portfolio_value(request.portfolio_id, request.portfolio_value)
        
        # Métadonnées de prédiction
        prediction_metadata = {
            'prediction_time_seconds': result['prediction_time'],
            'model_uri': result.get('model_uri'),
            'max_weight_change': max_change,
            'features_shape': list(market_data.shape),
            'risk_tolerance': request.risk_tolerance
        }
        
        return PredictionResponse(
            portfolio_id=request.portfolio_id,
            recommended_weights=result['weights'],
            confidence=result['confidence'],
            model_name=result['model_name'],
            model_type=result['model_type'],
            model_version=result['model_version'],
            model_stage=result['model_stage'],
            rebalancing_needed=rebalancing_needed,
            weight_changes=weight_changes,
            prediction_metadata=prediction_metadata,
            timestamp=datetime.now().isoformat()
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Erreur inattendue lors de la prédiction: {e}")
        raise HTTPException(status_code=500, detail=f"Erreur interne: {str(e)}")

@app.post("/predict/batch")
async def batch_predictions(
    requests_list: List[PredictionRequest],
    background_tasks: BackgroundTasks
):
    """Prédictions en lot pour plusieurs portefeuilles."""
    global model_manager
    
    if model_manager is None:
        model_manager = get_mlflow_manager()
    
    if len(requests_list) > 100:
        raise HTTPException(
            status_code=422,
            detail="Maximum 100 prédictions par lot"
        )
    
    def process_batch():
        """Traite le lot en arrière-plan."""
        results = []
        for req in requests_list:
            try:
                # Effectuer la prédiction (version simplifiée)
                market_data = np.array(req.market_data, dtype=np.float32)
                current_weights = np.array(req.current_weights, dtype=np.float32)
                
                result = model_manager.predict(req.model_name, market_data, current_weights)
                
                results.append({
                    'portfolio_id': req.portfolio_id,
                    'status': 'success',
                    'weights': result['weights'],
                    'confidence': result['confidence']
                })
                
                # Métriques
                record_prediction(
                    result['model_name'],
                    result['model_type'], 
                    result['model_stage'],
                    result['confidence']
                )
                
            except Exception as e:
                results.append({
                    'portfolio_id': req.portfolio_id,
                    'status': 'error',
                    'error': str(e)
                })
        
        # Sauvegarder les résultats (optionnel)
        logger.info(f"Batch processing terminé: {len(results)} résultats")
    
    # Lancer le traitement en arrière-plan
    background_tasks.add_task(process_batch)
    
    return {
        "status": "batch_processing_started",
        "batch_size": len(requests_list),
        "message": "Le traitement en lot a commencé en arrière-plan"
    }

@app.get("/metrics", response_class=PlainTextResponse)
async def get_metrics():
    """Endpoint des métriques Prometheus."""
    if PROMETHEUS_AVAILABLE and CUSTOM_REGISTRY:
        try:
            return generate_latest(CUSTOM_REGISTRY).decode('utf-8')
        except Exception as e:
            logger.error(f"Erreur génération métriques: {e}")
            return f"# Erreur génération métriques: {e}\n"
    else:
        return "# Métriques Prometheus non disponibles\n"

@app.post("/test/prediction")
async def test_prediction():
    """Test de prédiction pour validation."""
    global model_manager
    
    if model_manager is None:
        model_manager = get_mlflow_manager()
    
    try:
        # Données de test
        test_data = np.random.rand(5, 3, 30).astype(np.float32)
        test_weights = np.array([0.33, 0.33, 0.34])
        
        # Prédiction
        result = model_manager.predict("latest", test_data, test_weights)
        
        # Métriques
        record_prediction(
            result['model_name'],
            result['model_type'],
            result['model_stage'],
            result['confidence']
        )
        
        return {
            "status": "success",
            "test_data_shape": test_data.shape,
            "predicted_weights": result['weights'],
            "confidence": result['confidence'],
            "model_info": {
                "name": result['model_name'],
                "type": result['model_type'],
                "version": result['model_version'],
                "stage": result['model_stage']
            },
            "prediction_time": result['prediction_time']
        }
        
    except Exception as e:
        logger.error(f"Erreur test prédiction: {e}")
        return {"status": "error", "message": str(e)}

# Debug endpoints
@app.get("/debug/models")
async def debug_models():
    """Debug des modèles chargés."""
    global model_manager
    
    if model_manager is None:
        model_manager = get_mlflow_manager()
    
    return {
        "mlflow_available": model_manager.client is not None,
        "mlflow_models": list(model_manager.models.keys()),
        "fallback_models": list(model_manager.fallback_models.keys()),
        "total_models": len(model_manager.models) + len(model_manager.fallback_models)
    }

@app.get("/debug/mlflow")
async def debug_mlflow():
    """Debug de la connexion MLflow."""
    global model_manager
    
    if model_manager is None:
        model_manager = get_mlflow_manager()
    
    if not model_manager.client:
        return {"status": "MLflow non connecté"}
    
    try:
        experiments = model_manager.client.search_experiments()
        registered_models = model_manager.client.search_registered_models()
        
        return {
            "status": "MLflow connecté",
            "tracking_uri": model_manager.mlflow_uri,
            "experiments_count": len(experiments),
            "registered_models_count": len(registered_models),
            "registered_models": [model.name for model in registered_models]
        }
        
    except Exception as e:
        return {"status": "Erreur MLflow", "error": str(e)}

# =============================================================================
# ÉVÉNEMENTS
# =============================================================================

@app.on_event("startup")
async def startup_event():
    """Initialisation au démarrage."""
    global model_manager
    
    logger.info("🚀 Démarrage API Portfolio RL Production")
    
    # Initialiser le gestionnaire MLflow
    try:
        model_manager = get_mlflow_manager()
        models_count = len(model_manager.list_models())
        logger.info(f"✅ Gestionnaire MLflow initialisé: {models_count} modèles")
    except Exception as e:
        logger.error(f"❌ Erreur initialisation MLflow: {e}")
    
    logger.info(f"📊 Monitoring Prometheus: {'✅ Activé' if PROMETHEUS_AVAILABLE else '❌ Désactivé'}")

@app.on_event("shutdown")
async def shutdown_event():
    """Nettoyage à l'arrêt."""
    logger.info("🛑 Arrêt API Portfolio RL Production")

# =============================================================================
# POINT D'ENTRÉE
# =============================================================================

if __name__ == "__main__":
    import uvicorn
    
    host = os.getenv("API_HOST", "0.0.0.0")
    port = int(os.getenv("API_PORT", "8000"))
    
    logger.info(f"🌐 Démarrage serveur sur {host}:{port}")
    
    uvicorn.run(
        "main_production:app",
        host=host,
        port=port,
        reload=False,
        log_level="info"
    )