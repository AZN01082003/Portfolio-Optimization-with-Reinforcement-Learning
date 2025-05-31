"""
Module de métriques Prometheus pour le Portfolio RL.
Collecte et expose les métriques de performance et business.
"""
import os
import time
import logging
import threading
from typing import Dict, Any, Optional
from datetime import datetime

# Configuration du logging
logger = logging.getLogger(__name__)

# Import conditionnel de Prometheus
try:
    from prometheus_client import (
        Counter, Histogram, Gauge, Info, Enum,
        CollectorRegistry, generate_latest, start_http_server,
        CONTENT_TYPE_LATEST
    )
    PROMETHEUS_AVAILABLE = True
    logger.info("✅ Prometheus client disponible")
except ImportError as e:
    logger.warning(f"⚠️ Prometheus client non disponible: {e}")
    PROMETHEUS_AVAILABLE = False
    # Créer des classes factices
    class Counter:
        def __init__(self, *args, **kwargs): pass
        def inc(self, *args, **kwargs): pass
        def labels(self, *args, **kwargs): return self
    
    class Histogram:
        def __init__(self, *args, **kwargs): pass
        def observe(self, *args, **kwargs): pass
        def labels(self, *args, **kwargs): return self
    
    class Gauge:
        def __init__(self, *args, **kwargs): pass
        def set(self, *args, **kwargs): pass
        def inc(self, *args, **kwargs): pass
        def dec(self, *args, **kwargs): pass
        def labels(self, *args, **kwargs): return self
    
    class Info:
        def __init__(self, *args, **kwargs): pass
        def info(self, *args, **kwargs): pass
    
    class Enum:
        def __init__(self, *args, **kwargs): pass
        def state(self, *args, **kwargs): pass
    
    CollectorRegistry = None
    
    def generate_latest(registry=None):
        return "# Prometheus non disponible\n"
    
    def start_http_server(port, registry=None):
        logger.warning(f"⚠️ Impossible de démarrer le serveur Prometheus sur le port {port}")
        return False

try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    logger.warning("⚠️ psutil non disponible - métriques système limitées")
    PSUTIL_AVAILABLE = False

class PortfolioMetrics:
    """Collecteur de métriques pour l'application Portfolio RL."""
    
    def __init__(self, registry: Optional[CollectorRegistry] = None, prefix: str = "portfolio"):
        """
        Initialise le collecteur de métriques.
        
        Args:
            registry: Registre Prometheus (optionnel)
            prefix: Préfixe pour les noms de métriques
        """
        self.enabled = PROMETHEUS_AVAILABLE
        self.prefix = prefix
        self.registry = registry if registry and PROMETHEUS_AVAILABLE else None
        
        if not self.enabled:
            logger.warning("⚠️ Métriques désactivées (Prometheus non disponible)")
            return
        
        # Métriques générales de l'API
        self.api_requests_total = Counter(
            f'{prefix}_api_requests_total',
            'Nombre total de requêtes API',
            ['method', 'endpoint', 'status_code'],
            registry=self.registry
        )
        
        self.api_request_duration = Histogram(
            f'{prefix}_api_request_duration_seconds',
            'Durée des requêtes API en secondes',
            ['method', 'endpoint'],
            registry=self.registry,
            buckets=(0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0)
        )
        
        # Métriques des modèles ML
        self.model_predictions_total = Counter(
            f'{prefix}_model_predictions_total',
            'Nombre total de prédictions',
            ['model_version', 'prediction_type'],
            registry=self.registry
        )
        
        self.model_prediction_duration = Histogram(
            f'{prefix}_model_prediction_duration_seconds',
            'Durée des prédictions en secondes',
            ['model_version'],
            registry=self.registry,
            buckets=(0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0, 5.0)
        )
        
        self.model_prediction_confidence = Histogram(
            f'{prefix}_model_prediction_confidence',
            'Confiance des prédictions (0-1)',
            ['model_version'],
            registry=self.registry,
            buckets=(0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 1.0)
        )
        
        # Métriques business du portefeuille
        self.portfolio_value = Gauge(
            f'{prefix}_portfolio_value_dollars',
            'Valeur actuelle des portefeuilles',
            ['portfolio_id'],
            registry=self.registry
        )
        
        self.portfolio_daily_return = Gauge(
            f'{prefix}_portfolio_daily_return_percent',
            'Rendement quotidien du portefeuille (%)',
            ['portfolio_id'],
            registry=self.registry
        )
        
        self.portfolio_total_return = Gauge(
            f'{prefix}_portfolio_total_return_percent',
            'Rendement total du portefeuille (%)',
            ['portfolio_id'],
            registry=self.registry
        )
        
        self.portfolio_volatility = Gauge(
            f'{prefix}_portfolio_volatility_percent',
            'Volatilité du portefeuille (%)',
            ['portfolio_id'],
            registry=self.registry
        )
        
        self.portfolio_sharpe_ratio = Gauge(
            f'{prefix}_portfolio_sharpe_ratio',
            'Ratio de Sharpe du portefeuille',
            ['portfolio_id'],
            registry=self.registry
        )
        
        self.portfolio_max_drawdown = Gauge(
            f'{prefix}_portfolio_max_drawdown_percent',
            'Drawdown maximum du portefeuille (%)',
            ['portfolio_id'],
            registry=self.registry
        )
        
        # Métriques des transactions
        self.transaction_costs_total = Counter(
            f'{prefix}_transaction_costs_dollars_total',
            'Coûts de transaction cumulés',
            ['portfolio_id'],
            registry=self.registry
        )
        
        self.rebalancing_events_total = Counter(
            f'{prefix}_rebalancing_events_total',
            'Nombre d\'événements de rééquilibrage',
            ['portfolio_id', 'trigger'],
            registry=self.registry
        )
        
        # Métriques de santé des modèles
        self.model_health_status = Enum(
            f'{prefix}_model_health_status',
            'État de santé des modèles',
            ['model_name', 'version'],
            states=['healthy', 'degraded', 'unhealthy', 'unknown'],
            registry=self.registry
        )
        
        # Métriques système (si psutil disponible)
        if PSUTIL_AVAILABLE:
            self.system_cpu_usage = Gauge(
                f'{prefix}_system_cpu_usage_percent',
                'Utilisation CPU du système (%)',
                registry=self.registry
            )
            
            self.system_memory_usage = Gauge(
                f'{prefix}_system_memory_usage_percent',
                'Utilisation mémoire du système (%)',
                registry=self.registry
            )
            
            self.system_disk_usage = Gauge(
                f'{prefix}_system_disk_usage_percent',
                'Utilisation disque du système (%)',
                registry=self.registry
            )
        
        # Informations générales
        self.app_info = Info(
            f'{prefix}_app_info',
            'Informations sur l\'application',
            registry=self.registry
        )
        
        # Initialiser les informations de l'application
        self.app_info.info({
            'version': '1.0.0',
            'environment': os.getenv('ENVIRONMENT', 'development'),
            'started_at': datetime.now().isoformat()
        })
        
        logger.info("✅ Métriques Portfolio RL initialisées")
    
    def record_api_request(self, method: str, endpoint: str, status_code: int, duration: float):
        """Enregistre une requête API."""
        if not self.enabled:
            return
        
        try:
            self.api_requests_total.labels(
                method=method,
                endpoint=endpoint,
                status_code=str(status_code)
            ).inc()
            
            self.api_request_duration.labels(
                method=method,
                endpoint=endpoint
            ).observe(duration)
            
        except Exception as e:
            logger.warning(f"⚠️ Erreur enregistrement métrique API: {e}")
    
    def record_prediction(self, model_version: str, prediction_type: str, duration: float, confidence: float):
        """Enregistre une prédiction de modèle."""
        if not self.enabled:
            return
        
        try:
            self.model_predictions_total.labels(
                model_version=model_version,
                prediction_type=prediction_type
            ).inc()
            
            self.model_prediction_duration.labels(
                model_version=model_version
            ).observe(duration)
            
            self.model_prediction_confidence.labels(
                model_version=model_version
            ).observe(confidence)
            
        except Exception as e:
            logger.warning(f"⚠️ Erreur enregistrement métrique prédiction: {e}")
    
    def update_portfolio_metrics(self, portfolio_id: str, metrics: Dict[str, float]):
        """Met à jour les métriques d'un portefeuille."""
        if not self.enabled:
            return
        
        try:
            if 'value' in metrics:
                self.portfolio_value.labels(portfolio_id=portfolio_id).set(metrics['value'])
            
            if 'daily_return' in metrics:
                self.portfolio_daily_return.labels(portfolio_id=portfolio_id).set(metrics['daily_return'])
            
            if 'total_return' in metrics:
                self.portfolio_total_return.labels(portfolio_id=portfolio_id).set(metrics['total_return'])
            
            if 'volatility' in metrics:
                self.portfolio_volatility.labels(portfolio_id=portfolio_id).set(metrics['volatility'])
            
            if 'sharpe_ratio' in metrics:
                self.portfolio_sharpe_ratio.labels(portfolio_id=portfolio_id).set(metrics['sharpe_ratio'])
            
            if 'max_drawdown' in metrics:
                self.portfolio_max_drawdown.labels(portfolio_id=portfolio_id).set(metrics['max_drawdown'])
                
        except Exception as e:
            logger.warning(f"⚠️ Erreur mise à jour métriques portefeuille: {e}")
    
    def record_transaction_cost(self, portfolio_id: str, cost: float):
        """Enregistre un coût de transaction."""
        if not self.enabled:
            return
        
        try:
            self.transaction_costs_total.labels(portfolio_id=portfolio_id).inc(cost)
        except Exception as e:
            logger.warning(f"⚠️ Erreur enregistrement coût transaction: {e}")
    
    def record_rebalancing(self, portfolio_id: str, trigger: str):
        """Enregistre un événement de rééquilibrage."""
        if not self.enabled:
            return
        
        try:
            self.rebalancing_events_total.labels(
                portfolio_id=portfolio_id,
                trigger=trigger
            ).inc()
        except Exception as e:
            logger.warning(f"⚠️ Erreur enregistrement rééquilibrage: {e}")
    
    def update_model_health(self, model_name: str, version: str, is_healthy: bool):
        """Met à jour l'état de santé d'un modèle."""
        if not self.enabled:
            return
        
        try:
            status = 'healthy' if is_healthy else 'unhealthy'
            self.model_health_status.labels(
                model_name=model_name,
                version=version
            ).state(status)
        except Exception as e:
            logger.warning(f"⚠️ Erreur mise à jour santé modèle: {e}")
    
    def update_system_metrics(self):
        """Met à jour les métriques système."""
        if not self.enabled or not PSUTIL_AVAILABLE:
            return
        
        try:
            # CPU
            cpu_percent = psutil.cpu_percent(interval=1)
            self.system_cpu_usage.set(cpu_percent)
            
            # Mémoire
            memory = psutil.virtual_memory()
            self.system_memory_usage.set(memory.percent)
            
            # Disque
            disk = psutil.disk_usage('/')
            disk_percent = (disk.used / disk.total) * 100
            self.system_disk_usage.set(disk_percent)
            
        except Exception as e:
            logger.warning(f"⚠️ Erreur mise à jour métriques système: {e}")
    
    def get_metrics(self) -> str:
        """Retourne les métriques au format Prometheus."""
        if not self.enabled:
            return "# Métriques non disponibles (Prometheus non installé)\n"
        
        try:
            # Mettre à jour les métriques système avant l'export
            self.update_system_metrics()
            
            if self.registry:
                return generate_latest(self.registry).decode('utf-8')
            else:
                return generate_latest().decode('utf-8')
                
        except Exception as e:
            logger.error(f"❌ Erreur génération métriques: {e}")
            return f"# Erreur génération métriques: {e}\n"

# Instance globale des métriques
portfolio_metrics = PortfolioMetrics() if PROMETHEUS_AVAILABLE else None

class MetricsServer:
    """Serveur HTTP pour exposer les métriques."""
    
    def __init__(self, port: int = 8001, metrics: Optional[PortfolioMetrics] = None):
        self.port = port
        self.metrics = metrics or portfolio_metrics
        self.server_thread = None
        self.running = False
    
    def start(self) -> bool:
        """Démarre le serveur de métriques."""
        if not PROMETHEUS_AVAILABLE:
            logger.warning("⚠️ Impossible de démarrer le serveur de métriques (Prometheus non disponible)")
            return False
        
        try:
            if self.metrics and self.metrics.registry:
                start_http_server(self.port, registry=self.metrics.registry)
            else:
                start_http_server(self.port)
            
            self.running = True
            logger.info(f"📊 Serveur de métriques démarré sur le port {self.port}")
            return True
            
        except Exception as e:
            logger.error(f"❌ Erreur démarrage serveur métriques: {e}")
            return False
    
    def stop(self):
        """Arrête le serveur de métriques."""
        self.running = False
        logger.info("🛑 Serveur de métriques arrêté")

def start_metrics_server(port: int = 8001, metrics: Optional[PortfolioMetrics] = None) -> bool:
    """
    Démarre un serveur de métriques Prometheus.
    
    Args:
        port: Port d'écoute
        metrics: Instance de PortfolioMetrics (optionnel)
    
    Returns:
        bool: True si le serveur a démarré avec succès
    """
    server = MetricsServer(port, metrics)
    return server.start()

# Thread de mise à jour des métriques système
def _system_metrics_updater():
    """Thread qui met à jour périodiquement les métriques système."""
    while True:
        try:
            if portfolio_metrics and portfolio_metrics.enabled:
                portfolio_metrics.update_system_metrics()
            time.sleep(30)  # Mise à jour toutes les 30 secondes
        except Exception as e:
            logger.warning(f"⚠️ Erreur mise à jour métriques système: {e}")
            time.sleep(60)  # Attendre plus longtemps en cas d'erreur

# Démarrer le thread de mise à jour système si possible
if PROMETHEUS_AVAILABLE and PSUTIL_AVAILABLE and portfolio_metrics:
    system_thread = threading.Thread(target=_system_metrics_updater, daemon=True)
    system_thread.start()
    logger.info("🔄 Thread de métriques système démarré")

# Fonctions utilitaires pour l'intégration
def get_metrics_middleware():
    """Retourne un middleware FastAPI pour les métriques."""
    if not portfolio_metrics:
        return None
    
    async def metrics_middleware(request, call_next):
        start_time = time.time()
        response = await call_next(request)
        process_time = time.time() - start_time
        
        portfolio_metrics.record_api_request(
            request.method,
            str(request.url.path),
            response.status_code,
            process_time
        )
        
        return response
    
    return metrics_middleware

def record_business_metric(metric_type: str, **kwargs):
    """
    Enregistre une métrique business de manière générique.
    
    Args:
        metric_type: Type de métrique ('prediction', 'portfolio_update', etc.)
        **kwargs: Paramètres spécifiques à la métrique
    """
    if not portfolio_metrics:
        return
    
    try:
        if metric_type == 'prediction':
            portfolio_metrics.record_prediction(
                kwargs.get('model_version', 'unknown'),
                kwargs.get('prediction_type', 'unknown'),
                kwargs.get('duration', 0.0),
                kwargs.get('confidence', 0.0)
            )
        elif metric_type == 'portfolio_update':
            portfolio_metrics.update_portfolio_metrics(
                kwargs.get('portfolio_id', 'unknown'),
                kwargs.get('metrics', {})
            )
        elif metric_type == 'transaction':
            portfolio_metrics.record_transaction_cost(
                kwargs.get('portfolio_id', 'unknown'),
                kwargs.get('cost', 0.0)
            )
        elif metric_type == 'rebalancing':
            portfolio_metrics.record_rebalancing(
                kwargs.get('portfolio_id', 'unknown'),
                kwargs.get('trigger', 'unknown')
            )
        else:
            logger.warning(f"⚠️ Type de métrique inconnu: {metric_type}")
            
    except Exception as e:
        logger.warning(f"⚠️ Erreur enregistrement métrique business: {e}")

# Export des symboles principaux
__all__ = [
    'PortfolioMetrics',
    'portfolio_metrics',
    'MetricsServer',
    'start_metrics_server',
    'get_metrics_middleware',
    'record_business_metric',
    'PROMETHEUS_AVAILABLE'
]