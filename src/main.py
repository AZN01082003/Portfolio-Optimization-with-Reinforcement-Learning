"""
API principale du Portfolio RL avec monitoring Prometheus intégré.
Fichier: src/main.py
"""
import logging
import time
from datetime import datetime
from typing import Dict, Any, Optional, List
from contextlib import asynccontextmanager

try:
    from fastapi import FastAPI, HTTPException, BackgroundTasks
    from fastapi.middleware.cors import CORSMiddleware
    from fastapi.responses import JSONResponse
    from pydantic import BaseModel
    FASTAPI_AVAILABLE = True
except ImportError:
    print("❌ FastAPI non disponible. Installez avec: pip install fastapi uvicorn")
    exit(1)

# Imports du monitoring
from .monitoring.metrics import portfolio_metrics, PROMETHEUS_AVAILABLE
from .api.monitoring_integration import setup_monitoring, monitor_endpoint

# Configuration du logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# =============================================================================
# MODÈLES PYDANTIC
# =============================================================================

class PortfolioData(BaseModel):
    """Modèle pour les données de portefeuille."""
    portfolio_id: str
    assets: Dict[str, float]  # {symbol: weight}
    total_value: Optional[float] = None
    benchmark: Optional[str] = "SPY"

class AllocationRequest(BaseModel):
    """Modèle pour les requêtes d'allocation."""
    portfolio_id: str
    risk_tolerance: float = 0.5
    investment_horizon: int = 12  # mois
    constraints: Optional[Dict[str, Any]] = None

class AllocationResponse(BaseModel):
    """Modèle pour les réponses d'allocation."""
    portfolio_id: str
    allocation: Dict[str, float]
    expected_return: float
    expected_risk: float
    sharpe_ratio: float
    confidence: float
    model_version: str
    timestamp: datetime

class BacktestRequest(BaseModel):
    """Modèle pour les requêtes de backtest."""
    portfolio_id: str
    start_date: str
    end_date: str
    initial_value: float = 10000.0
    rebalancing_frequency: str = "monthly"

class BacktestResponse(BaseModel):
    """Modèle pour les réponses de backtest."""
    portfolio_id: str
    total_return: float
    annual_return: float
    volatility: float
    sharpe_ratio: float
    max_drawdown: float
    calmar_ratio: float
    timestamps: List[str]
    values: List[float]

class HealthResponse(BaseModel):
    """Modèle pour la réponse de santé."""
    status: str
    timestamp: datetime
    version: str
    monitoring_enabled: bool
    services: Dict[str, str]

# =============================================================================
# GESTIONNAIRE DE CYCLE DE VIE
# =============================================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Gestionnaire du cycle de vie de l'application."""
    # Démarrage
    logger.info("🚀 Démarrage de l'API Portfolio RL")
    
    # Initialiser les métriques de santé des modèles
    if portfolio_metrics and portfolio_metrics.enabled:
        portfolio_metrics.update_model_health("portfolio_rl_v1", "1.0.0", True)
        portfolio_metrics.update_model_health("risk_model_v1", "1.0.0", True)
        logger.info("✅ Santé des modèles initialisée")
    
    yield
    
    # Arrêt
    logger.info("🛑 Arrêt de l'API Portfolio RL")

# =============================================================================
# CRÉATION DE L'APPLICATION
# =============================================================================

app = FastAPI(
    title="Portfolio RL API",
    description="API de gestion de portefeuille avec apprentissage par renforcement et monitoring Prometheus",
    version="1.0.0",
    lifespan=lifespan
)

# Configuration CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # En production, spécifiez les domaines autorisés
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configuration du monitoring
setup_monitoring(app, enable_middleware=True)

# =============================================================================
# ENDPOINTS PRINCIPAUX
# =============================================================================

@app.get("/", response_model=Dict[str, str])
async def root():
    """Endpoint racine."""
    return {
        "message": "Portfolio RL API",
        "version": "1.0.0",
        "docs": "/docs",
        "metrics": "/metrics",
        "health": "/health"
    }

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Endpoint de vérification de santé détaillé."""
    
    # Vérifier les services
    services = {
        "api": "healthy",
        "monitoring": "healthy" if portfolio_metrics and portfolio_metrics.enabled else "disabled",
        "models": "healthy"  # À adapter selon vos modèles
    }
    
    return HealthResponse(
        status="healthy",
        timestamp=datetime.now(),
        version="1.0.0",
        monitoring_enabled=portfolio_metrics is not None and portfolio_metrics.enabled,
        services=services
    )

# =============================================================================
# ENDPOINTS D'ALLOCATION
# =============================================================================

@app.post("/allocation/optimize", response_model=AllocationResponse)
@monitor_endpoint("allocation_optimize")
async def optimize_allocation(request: AllocationRequest, background_tasks: BackgroundTasks):
    """
    Optimise l'allocation d'un portefeuille.
    
    Args:
        request: Paramètres d'optimisation
        background_tasks: Tâches en arrière-plan
        
    Returns:
        Allocation optimisée
    """
    start_time = time.time()
    
    try:
        # Simulation d'un modèle d'allocation (à remplacer par votre logique)
        await simulate_model_prediction(0.5)  # Simule 500ms de calcul
        
        # Allocation simulée (remplacez par votre modèle RL)
        allocation = {
            "AAPL": 0.25,
            "GOOGL": 0.20,
            "MSFT": 0.15,
            "TSLA": 0.10,
            "SPY": 0.30
        }
        
        # Métriques simulées
        expected_return = 0.12
        expected_risk = 0.15
        sharpe_ratio = (expected_return - 0.02) / expected_risk
        confidence = 0.85
        
        # Enregistrer les métriques de prédiction
        duration = time.time() - start_time
        if portfolio_metrics and portfolio_metrics.enabled:
            portfolio_metrics.record_prediction("portfolio_rl_v1", "allocation", duration, confidence)
        
        # Tâche en arrière-plan pour mettre à jour les métriques
        background_tasks.add_task(
            update_portfolio_metrics_background,
            request.portfolio_id,
            allocation,
            expected_return,
            expected_risk,
            sharpe_ratio
        )
        
        return AllocationResponse(
            portfolio_id=request.portfolio_id,
            allocation=allocation,
            expected_return=expected_return,
            expected_risk=expected_risk,
            sharpe_ratio=sharpe_ratio,
            confidence=confidence,
            model_version="portfolio_rl_v1",
            timestamp=datetime.now()
        )
        
    except Exception as e:
        logger.error(f"Erreur lors de l'optimisation: {e}")
        raise HTTPException(status_code=500, detail=f"Erreur lors de l'optimisation: {str(e)}")

@app.post("/allocation/rebalance")
@monitor_endpoint("allocation_rebalance")
async def rebalance_portfolio(portfolio_data: PortfolioData):
    """
    Rééquilibre un portefeuille existant.
    
    Args:
        portfolio_data: Données du portefeuille
        
    Returns:
        Nouvelles allocations
    """
    try:
        # Enregistrer l'événement de rééquilibrage
        if portfolio_metrics and portfolio_metrics.enabled:
            portfolio_metrics.record_rebalancing(portfolio_data.portfolio_id, "manual")
        
        # Simulation du rééquilibrage
        await simulate_model_prediction(0.3)
        
        return {
            "portfolio_id": portfolio_data.portfolio_id,
            "status": "rebalanced",
            "timestamp": datetime.now(),
            "message": "Portefeuille rééquilibré avec succès"
        }
        
    except Exception as e:
        logger.error(f"Erreur lors du rééquilibrage: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# =============================================================================
# ENDPOINTS DE BACKTEST
# =============================================================================

@app.post("/backtest/run", response_model=BacktestResponse)
@monitor_endpoint("backtest_run")
async def run_backtest(request: BacktestRequest):
    """
    Lance un backtest sur une stratégie.
    
    Args:
        request: Paramètres du backtest
        
    Returns:
        Résultats du backtest
    """
    start_time = time.time()
    
    try:
        # Simulation d'un backtest (remplacez par votre logique)
        await simulate_model_prediction(2.0)  # Simule 2s de calcul
        
        # Résultats simulés
        total_return = 0.235  # 23.5%
        annual_return = 0.118  # 11.8%
        volatility = 0.156    # 15.6%
        sharpe_ratio = (annual_return - 0.02) / volatility
        max_drawdown = 0.087  # 8.7%
        calmar_ratio = annual_return / max_drawdown
        
        # Données de série temporelle simulées
        timestamps = ["2023-01-01", "2023-06-01", "2023-12-01"]
        values = [10000, 11500, 12350]
        
        # Enregistrer les métriques
        duration = time.time() - start_time
        if portfolio_metrics and portfolio_metrics.enabled:
            portfolio_metrics.record_prediction("portfolio_rl_v1", "backtest", duration, 0.9)
            
            # Mettre à jour les métriques de portefeuille
            metrics_data = {
                "value": values[-1],
                "total_return": total_return * 100,
                "volatility": volatility * 100,
                "sharpe_ratio": sharpe_ratio,
                "max_drawdown": max_drawdown * 100
            }
            portfolio_metrics.update_portfolio_metrics(request.portfolio_id, metrics_data)
        
        return BacktestResponse(
            portfolio_id=request.portfolio_id,
            total_return=total_return,
            annual_return=annual_return,
            volatility=volatility,
            sharpe_ratio=sharpe_ratio,
            max_drawdown=max_drawdown,
            calmar_ratio=calmar_ratio,
            timestamps=timestamps,
            values=values
        )
        
    except Exception as e:
        logger.error(f"Erreur lors du backtest: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# =============================================================================
# ENDPOINTS DE DONNÉES
# =============================================================================

@app.get("/portfolio/{portfolio_id}")
@monitor_endpoint("get_portfolio")
async def get_portfolio(portfolio_id: str):
    """
    Récupère les informations d'un portefeuille.
    
    Args:
        portfolio_id: Identifiant du portefeuille
        
    Returns:
        Informations du portefeuille
    """
    try:
        # Simulation de récupération de données
        await simulate_model_prediction(0.1)
        
        return {
            "portfolio_id": portfolio_id,
            "current_value": 125000.0,
            "daily_return": 0.012,
            "assets": {
                "AAPL": {"weight": 0.25, "value": 31250},
                "GOOGL": {"weight": 0.20, "value": 25000},
                "MSFT": {"weight": 0.15, "value": 18750},
                "TSLA": {"weight": 0.10, "value": 12500},
                "SPY": {"weight": 0.30, "value": 37500}
            },
            "last_updated": datetime.now()
        }
        
    except Exception as e:
        logger.error(f"Erreur lors de la récupération du portefeuille: {e}")
        raise HTTPException(status_code=404, detail="Portefeuille non trouvé")

@app.get("/portfolios")
@monitor_endpoint("list_portfolios")
async def list_portfolios():
    """
    Liste tous les portefeuilles.
    
    Returns:
        Liste des portefeuilles
    """
    return {
        "portfolios": [
            {"id": "portfolio_1", "name": "Portefeuille Conservative", "value": 125000},
            {"id": "portfolio_2", "name": "Portefeuille Aggressif", "value": 87500},
            {"id": "portfolio_3", "name": "Portefeuille Équilibré", "value": 156000}
        ],
        "total_count": 3
    }

# =============================================================================
# ENDPOINTS DE MÉTRIQUES ET MONITORING
# =============================================================================

@app.get("/monitoring/status")
async def monitoring_status():
    """
    Statut détaillé du système de monitoring.
    
    Returns:
        Statut du monitoring
    """
    if portfolio_metrics:
        health = portfolio_metrics.health_check()
    else:
        health = {"status": "disabled", "prometheus_enabled": False}
    
    return {
        "monitoring": health,
        "prometheus_available": PROMETHEUS_AVAILABLE,
        "metrics_endpoint": "/metrics",
        "timestamp": datetime.now()
    }

@app.post("/monitoring/test")
async def test_monitoring():
    """
    Génère des métriques de test.
    
    Returns:
        Confirmation de génération
    """
    if not portfolio_metrics or not portfolio_metrics.enabled:
        raise HTTPException(status_code=503, detail="Monitoring non disponible")
    
    try:
        # Générer des métriques de test
        for i in range(5):
            portfolio_metrics.record_api_request("GET", "/test", 200, 0.1 + i * 0.05)
            portfolio_metrics.record_prediction("test_model", "test", 0.5, 0.8)
        
        # Métriques de portefeuille de test
        test_metrics = {
            "value": 100000 + i * 1000,
            "daily_return": 0.01,
            "volatility": 0.15,
            "sharpe_ratio": 1.2
        }
        portfolio_metrics.update_portfolio_metrics("test_portfolio", test_metrics)
        
        return {
            "status": "success",
            "message": "Métriques de test générées",
            "metrics_generated": 15,
            "timestamp": datetime.now()
        }
        
    except Exception as e:
        logger.error(f"Erreur lors de la génération des métriques de test: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# =============================================================================
# FONCTIONS UTILITAIRES
# =============================================================================

async def simulate_model_prediction(duration: float):
    """
    Simule une prédiction de modèle avec délai.
    
    Args:
        duration: Durée de simulation en secondes
    """
    import asyncio
    await asyncio.sleep(duration)

def update_portfolio_metrics_background(
    portfolio_id: str,
    allocation: Dict[str, float],
    expected_return: float,
    expected_risk: float,
    sharpe_ratio: float
):
    """
    Met à jour les métriques de portefeuille en arrière-plan.
    
    Args:
        portfolio_id: Identifiant du portefeuille
        allocation: Allocation des actifs
        expected_return: Rendement attendu
        expected_risk: Risque attendu
        sharpe_ratio: Ratio de Sharpe
    """
    if portfolio_metrics and portfolio_metrics.enabled:
        try:
            # Calculer la valeur simulée du portefeuille
            portfolio_value = sum(allocation.values()) * 100000  # Simulation
            
            metrics_data = {
                "value": portfolio_value,
                "total_return": expected_return * 100,
                "volatility": expected_risk * 100,
                "sharpe_ratio": sharpe_ratio
            }
            
            portfolio_metrics.update_portfolio_metrics(portfolio_id, metrics_data)
            
            # Enregistrer les poids des actifs
            portfolio_metrics.update_asset_weights(portfolio_id, allocation)
            
            logger.info(f"Métriques mises à jour pour le portefeuille {portfolio_id}")
            
        except Exception as e:
            logger.error(f"Erreur lors de la mise à jour des métriques: {e}")

# =============================================================================
# POINT D'ENTRÉE
# =============================================================================

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "src.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )