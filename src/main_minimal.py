"""
API FastAPI minimale mais garantie fonctionnelle avec métriques Prometheus.
Version simplifiée pour assurer que les métriques marchent.
"""
import os
import sys
import time
import logging
from datetime import datetime
from typing import Dict, List
from pathlib import Path

# Ajouter le répertoire racine au PYTHONPATH
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from fastapi import FastAPI, HTTPException
from fastapi.responses import PlainTextResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import numpy as np

# Configuration du logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# =============================================================================
# MÉTRIQUES PROMETHEUS - VERSION ULTRA SIMPLIFIÉE
# =============================================================================

# Variables globales pour les métriques
METRICS_DATA = {
    'api_requests': {},
    'api_duration': [],
    'predictions': {},
    'startup_time': datetime.now()
}

def record_request(method: str, path: str, status: int, duration: float):
    """Enregistre une requête de manière simple."""
    key = f"{method}_{path}_{status}"
    METRICS_DATA['api_requests'][key] = METRICS_DATA['api_requests'].get(key, 0) + 1
    METRICS_DATA['api_duration'].append(duration)

def record_prediction(model: str, confidence: float):
    """Enregistre une prédiction."""
    key = f"prediction_{model}"
    METRICS_DATA['predictions'][key] = METRICS_DATA['predictions'].get(key, 0) + 1

def generate_prometheus_metrics() -> str:
    """Génère des métriques au format Prometheus."""
    metrics_lines = [
        "# HELP portfolio_api_requests_total Total API requests",
        "# TYPE portfolio_api_requests_total counter"
    ]
    
    # Requêtes API
    for key, count in METRICS_DATA['api_requests'].items():
        method, path, status = key.split('_', 2)
        metrics_lines.append(
            f'portfolio_api_requests_total{{method="{method}",endpoint="{path}",status_code="{status}"}} {count}'
        )
    
    # Durée des requêtes
    if METRICS_DATA['api_duration']:
        avg_duration = sum(METRICS_DATA['api_duration']) / len(METRICS_DATA['api_duration'])
        metrics_lines.extend([
            "# HELP portfolio_api_request_duration_seconds Average API request duration",
            "# TYPE portfolio_api_request_duration_seconds gauge",
            f'portfolio_api_request_duration_seconds {avg_duration:.6f}'
        ])
    
    # Prédictions
    metrics_lines.extend([
        "# HELP portfolio_model_predictions_total Total model predictions", 
        "# TYPE portfolio_model_predictions_total counter"
    ])
    
    for key, count in METRICS_DATA['predictions'].items():
        model = key.replace('prediction_', '')
        metrics_lines.append(
            f'portfolio_model_predictions_total{{model_version="{model}",prediction_type="allocation"}} {count}'
        )
    
    # Uptime
    uptime = (datetime.now() - METRICS_DATA['startup_time']).total_seconds()
    metrics_lines.extend([
        "# HELP portfolio_uptime_seconds Application uptime",
        "# TYPE portfolio_uptime_seconds gauge", 
        f'portfolio_uptime_seconds {uptime:.1f}'
    ])
    
    return '\n'.join(metrics_lines) + '\n'

# Tentative d'import du vrai Prometheus (optionnel)
try:
    from prometheus_client import Counter, Histogram, generate_latest, CollectorRegistry
    
    # Créer un registre et des métriques réelles
    REAL_REGISTRY = CollectorRegistry()
    
    REAL_API_REQUESTS = Counter(
        'portfolio_api_requests_total',
        'Total API requests',
        ['method', 'endpoint', 'status_code'],
        registry=REAL_REGISTRY
    )
    
    REAL_PREDICTIONS = Counter(
        'portfolio_model_predictions_total', 
        'Total predictions',
        ['model_version', 'prediction_type'],
        registry=REAL_REGISTRY
    )
    
    PROMETHEUS_AVAILABLE = True
    logger.info("✅ Prometheus client réel disponible")
    
    def record_real_request(method: str, path: str, status: int):
        REAL_API_REQUESTS.labels(method=method, endpoint=path, status_code=str(status)).inc()
    
    def record_real_prediction(model: str):
        REAL_PREDICTIONS.labels(model_version=model, prediction_type="allocation").inc()
    
    def generate_real_metrics() -> str:
        return generate_latest(REAL_REGISTRY).decode('utf-8')

except ImportError:
    PROMETHEUS_AVAILABLE = False
    logger.warning("⚠️ Prometheus client non disponible - utilisation des métriques manuelles")
    
    def record_real_request(method: str, path: str, status: int):
        pass
    
    def record_real_prediction(model: str):
        pass
    
    def generate_real_metrics() -> str:
        return ""

# =============================================================================
# MODÈLES PYDANTIC
# =============================================================================

class HealthResponse(BaseModel):
    status: str = "healthy"
    timestamp: str
    service: str = "portfolio-rl-api-minimal"
    version: str = "1.0.0"
    dependencies: Dict[str, bool]

class PredictionRequest(BaseModel):
    portfolio_id: str = Field(..., description="ID du portefeuille")
    market_data: List[List[List[float]]] = Field(..., description="Données de marché")
    current_weights: List[float] = Field(..., description="Poids actuels")
    portfolio_value: float = Field(..., gt=0, description="Valeur du portefeuille")
    risk_tolerance: float = Field(0.5, ge=0, le=1, description="Tolérance au risque")
    model_version: str = Field("latest", description="Version du modèle")

class PredictionResponse(BaseModel):
    portfolio_id: str
    recommended_weights: List[float]
    confidence: float
    model_version: str
    rebalancing_needed: bool
    timestamp: str

# =============================================================================
# GESTIONNAIRE DE MODÈLES SIMPLIFIÉ
# =============================================================================

class SimpleModelManager:
    """Gestionnaire de modèles ultra simplifié."""
    
    def __init__(self):
        self.models = {"dummy": {"type": "dummy", "loaded": True}}
    
    def predict(self, model_version: str, market_data: np.ndarray, current_weights: np.ndarray):
        """Prédiction simplifiée."""
        n_assets = len(current_weights)
        
        # Générer des poids aléatoires mais cohérents
        np.random.seed(int(time.time()) % 1000)  # Seed basé sur le temps
        random_weights = np.random.dirichlet(np.ones(n_assets) * 2)  # Plus concentré
        confidence = 0.7 + np.random.random() * 0.25  # Entre 70% et 95%
        
        return {
            'weights': random_weights.tolist(),
            'confidence': confidence,
            'model_type': 'dummy_v2'
        }

# =============================================================================
# APPLICATION FASTAPI
# =============================================================================

model_manager = SimpleModelManager()

app = FastAPI(
    title="Portfolio RL API - Minimal",
    description="API simplifiée pour test des métriques",
    version="1.0.0"
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"]
)

# Middleware de métriques SIMPLIFIÉ
@app.middleware("http")
async def simple_metrics_middleware(request, call_next):
    start_time = time.time()
    response = await call_next(request)
    duration = time.time() - start_time
    
    # Enregistrer dans les deux systèmes
    record_request(request.method, request.url.path, response.status_code, duration)
    record_real_request(request.method, request.url.path, response.status_code)
    
    return response

# =============================================================================
# ENDPOINTS
# =============================================================================

@app.get("/")
async def root():
    return {
        "message": "Portfolio RL API - Minimal Working Version",
        "version": "1.0.0",
        "metrics_available": True,
        "prometheus_real": PROMETHEUS_AVAILABLE
    }

@app.get("/health", response_model=HealthResponse)
async def health():
    return HealthResponse(
        timestamp=datetime.now().isoformat(),
        dependencies={
            "prometheus": PROMETHEUS_AVAILABLE,
            "models": True,
            "metrics": True
        }
    )

@app.get("/models")
async def list_models():
    return {
        "models": [
            {
                "name": "dummy",
                "type": "dummy_v2", 
                "status": "ready",
                "description": "Modèle de test pour validation des métriques"
            }
        ]
    }

@app.get("/metrics", response_class=PlainTextResponse)
async def get_metrics():
    """Endpoint des métriques - GARANTIE DE FONCTIONNER."""
    
    if PROMETHEUS_AVAILABLE:
        # Utiliser les vraies métriques Prometheus si disponible
        real_metrics = generate_real_metrics()
        manual_metrics = generate_prometheus_metrics()
        
        return f"""# Métriques Portfolio RL - Version Hybride
# Métriques Prometheus natives:
{real_metrics}

# Métriques manuelles (backup):
{manual_metrics}

# Informations système
portfolio_metrics_system_info{{type="hybrid",prometheus_available="{PROMETHEUS_AVAILABLE}"}} 1
"""
    else:
        # Utiliser seulement les métriques manuelles
        return f"""# Métriques Portfolio RL - Version Manuelle
{generate_prometheus_metrics()}

# Informations système  
portfolio_metrics_system_info{{type="manual",prometheus_available="false"}} 1
"""

@app.post("/predict", response_model=PredictionResponse)
async def predict_allocation(request: PredictionRequest):
    try:
        # Validation basique
        market_data = np.array(request.market_data, dtype=np.float32)
        current_weights = np.array(request.current_weights, dtype=np.float32)
        
        if len(market_data.shape) != 3:
            raise HTTPException(status_code=422, detail="Format market_data invalide")
        
        if abs(np.sum(current_weights) - 1.0) > 0.01:
            raise HTTPException(status_code=422, detail="Somme des poids doit être 1.0")
        
        # Prédiction
        result = model_manager.predict(request.model_version, market_data, current_weights)
        
        # Enregistrer la prédiction dans les métriques
        record_prediction(request.model_version, result['confidence'])
        record_real_prediction(request.model_version)
        
        # Calculer le rééquilibrage
        weight_diff = np.abs(np.array(result['weights']) - current_weights)
        rebalancing_needed = np.max(weight_diff) > 0.05
        
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
        logger.error(f"Erreur prédiction: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/test/prediction")
async def test_prediction():
    """Test de prédiction simplifié."""
    
    # Données de test
    test_data = np.random.rand(5, 3, 30).astype(np.float32)
    test_weights = np.array([0.33, 0.33, 0.34])
    
    result = model_manager.predict("latest", test_data, test_weights)
    
    # Enregistrer dans les métriques
    record_prediction("latest", result['confidence'])
    record_real_prediction("latest")
    
    return {
        "status": "success",
        "predicted_weights": result['weights'],
        "confidence": result['confidence'],
        "model_type": result['model_type'],
        "metrics_recorded": True,
        "prometheus_available": PROMETHEUS_AVAILABLE
    }

# Debug endpoint pour les métriques
@app.get("/debug/metrics")
async def debug_metrics():
    """Endpoint de debug pour voir l'état des métriques."""
    return {
        "prometheus_available": PROMETHEUS_AVAILABLE,
        "manual_metrics": METRICS_DATA,
        "metrics_count": {
            "requests": len(METRICS_DATA['api_requests']),
            "predictions": len(METRICS_DATA['predictions']),
            "duration_samples": len(METRICS_DATA['api_duration'])
        }
    }

# =============================================================================
# ÉVÉNEMENTS
# =============================================================================

@app.on_event("startup")
async def startup():
    logger.info("🚀 API Portfolio RL Minimal démarrée")
    logger.info(f"📊 Prometheus: {'✅ Disponible' if PROMETHEUS_AVAILABLE else '❌ Indisponible'}")
    logger.info("📈 Métriques manuelles activées en backup")

@app.on_event("shutdown") 
async def shutdown():
    logger.info("🛑 Arrêt API Portfolio RL Minimal")

# =============================================================================
# POINT D'ENTRÉE
# =============================================================================

if __name__ == "__main__":
    import uvicorn
    
    uvicorn.run(
        "main_minimal:app",
        host="0.0.0.0",
        port=8000,
        reload=False,
        log_level="info"
    )