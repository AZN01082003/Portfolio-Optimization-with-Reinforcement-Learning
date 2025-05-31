"""
Module de monitoring avec Prometheus pour le système Portfolio RL.
Collecte et expose les métriques de performance et de santé.
"""
import time
import logging
import threading
from datetime import datetime
from typing import Dict, Any, Optional
from functools import wraps

# Configuration du logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Imports conditionnels pour Prometheus
try:
    from prometheus_client import (
        Counter, Histogram, Gauge, Info, Summary,
        CollectorRegistry, generate_latest, CONTENT_TYPE_LATEST,
        start_http_server, REGISTRY
    )
    PROMETHEUS_AVAILABLE = True
    logger.info("✅ Prometheus client disponible")
except ImportError:
    PROMETHEUS_AVAILABLE = False
    logger.warning("⚠️ Prometheus client non disponible. Installez avec: pip install prometheus-client")
    
    # Classes factices pour éviter les erreurs
    class Counter:
        def __init__(self, *args, **kwargs): pass
        def inc(self, *args, **kwargs): pass
        def labels(self, *args, **kwargs): return self
    
    class Histogram:
        def __init__(self, *args, **kwargs): pass
        def observe(self, *args, **kwargs): pass
        def labels(self, *args, **kwargs): return self
        def time(self): return self
        def __enter__(self): return self
        def __exit__(self, *args): pass
    
    class Gauge:
        def __init__(self, *args, **kwargs): pass
        def set(self, *args, **kwargs): pass
        def inc(self, *args, **kwargs): pass
        def dec(self, *args, **kwargs): pass
        def labels(self, *args, **kwargs): return self
    
    class Info:
        def __init__(self, *args, **kwargs): pass
        def info(self, *args, **kwargs): pass
    
    class Summary:
        def __init__(self, *args, **kwargs): pass
        def observe(self, *args, **kwargs): pass
        def labels(self, *args, **kwargs): return self
    
    class CollectorRegistry:
        def __init__(self): pass
    
    REGISTRY = CollectorRegistry()
    CONTENT_TYPE_LATEST = "text/plain"
    
    def generate_latest(registry=None):
        return b"# Prometheus metrics disabled\n"
    
    def start_http_server(port, registry=None):
        logger.warning(f"Serveur métrique simulé sur port {port} (Prometheus non disponible)")
        return True

class PortfolioMetrics:
    """
    Gestionnaire principal des métriques pour le système Portfolio RL.
    """
    
    def __init__(self, registry: Optional[CollectorRegistry] = None):
        """
        Initialise le gestionnaire de métriques.
        
        Args:
            registry: Registre Prometheus personnalisé (optionnel)
        """
        self.enabled = PROMETHEUS_AVAILABLE
        
        if not self.enabled:
            logger.warning("Prometheus non disponible, métriques désactivées")
            return
        
        self.registry = registry or REGISTRY
        
        try:
            self._setup_metrics()
            self._update_system_info()
            logger.info("✅ Métriques Prometheus initialisées")
        except Exception as e:
            logger.error(f"❌ Erreur lors de l'initialisation des métriques: {e}")
            self.enabled = False
    
    def _setup_metrics(self):
        """Configure toutes les métriques Prometheus."""
        
        # === MÉTRIQUES API ===
        
        # Requêtes API
        self.api_requests_total = Counter(
            'portfolio_api_requests_total',
            'Nombre total de requêtes API',
            ['method', 'endpoint', 'status'],
            registry=self.registry
        )
        
        # Latence des requêtes API
        self.api_request_duration = Histogram(
            'portfolio_api_request_duration_seconds',
            'Durée des requêtes API en secondes',
            ['method', 'endpoint'],
            buckets=(0.005, 0.01, 0.025, 0.05, 0.075, 0.1, 0.25, 0.5, 0.75, 1.0, 2.5, 5.0, 7.5, 10.0),
            registry=self.registry
        )
        
        # Erreurs API
        self.api_errors_total = Counter(
            'portfolio_api_errors_total',
            'Nombre total d\'erreurs API',
            ['method', 'endpoint', 'error_type'],
            registry=self.registry
        )
        
        # === MÉTRIQUES MODÈLE ===
        
        # Prédictions
        self.model_predictions_total = Counter(
            'portfolio_model_predictions_total',
            'Nombre total de prédictions',
            ['model_version', 'prediction_type'],
            registry=self.registry
        )
        
        # Durée des prédictions
        self.model_prediction_duration = Histogram(
            'portfolio_model_prediction_duration_seconds',
            'Durée des prédictions en secondes',
            ['model_version'],
            buckets=(0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0),
            registry=self.registry
        )
        
        # Confiance des prédictions
        self.model_prediction_confidence = Histogram(
            'portfolio_model_prediction_confidence',
            'Confiance des prédictions (0-1)',
            ['model_version'],
            buckets=(0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0),
            registry=self.registry
        )
        
        # === MÉTRIQUES PORTFOLIO ===
        
        # Valeur du portefeuille
        self.portfolio_value = Gauge(
            'portfolio_current_value_usd',
            'Valeur actuelle du portefeuille en USD',
            ['portfolio_id'],
            registry=self.registry
        )
        
        # Rendement du portefeuille
        self.portfolio_return = Gauge(
            'portfolio_return_percentage',
            'Rendement du portefeuille en pourcentage',
            ['portfolio_id', 'period'],
            registry=self.registry
        )
        
        # Volatilité
        self.portfolio_volatility = Gauge(
            'portfolio_volatility_percentage',
            'Volatilité du portefeuille en pourcentage',
            ['portfolio_id'],
            registry=self.registry
        )
        
        # Ratio de Sharpe
        self.portfolio_sharpe_ratio = Gauge(
            'portfolio_sharpe_ratio',
            'Ratio de Sharpe du portefeuille',
            ['portfolio_id'],
            registry=self.registry
        )
        
        # Drawdown maximum
        self.portfolio_max_drawdown = Gauge(
            'portfolio_max_drawdown_percentage',
            'Drawdown maximum en pourcentage',
            ['portfolio_id'],
            registry=self.registry
        )
        
        # === MÉTRIQUES SYSTÈME ===
        
        # Santé des modèles
        self.models_health = Gauge(
            'portfolio_models_health_status',
            'État de santé des modèles (1=sain, 0=problème)',
            ['model_name', 'model_version'],
            registry=self.registry
        )
        
        # Utilisation mémoire
        self.system_memory_usage = Gauge(
            'portfolio_system_memory_usage_bytes',
            'Utilisation mémoire du système en bytes',
            registry=self.registry
        )
        
        # === MÉTRIQUES MLFLOW ===
        
        # Expériences MLflow
        self.mlflow_experiments_total = Gauge(
            'portfolio_mlflow_experiments_total',
            'Nombre total d\'expériences MLflow',
            registry=self.registry
        )
        
        # Runs MLflow
        self.mlflow_runs_total = Counter(
            'portfolio_mlflow_runs_total',
            'Nombre total de runs MLflow',
            ['experiment_name', 'status'],
            registry=self.registry
        )
        
        # === MÉTRIQUES BUSINESS ===
        
        # Coûts de transaction
        self.transaction_costs = Counter(
            'portfolio_transaction_costs_usd',
            'Coûts de transaction cumulés en USD',
            ['portfolio_id'],
            registry=self.registry
        )
        
        # Rééquilibrages
        self.rebalancing_events = Counter(
            'portfolio_rebalancing_events_total',
            'Nombre d\'événements de rééquilibrage',
            ['portfolio_id', 'trigger_type'],
            registry=self.registry
        )
        
        # Informations système
        self.system_info = Info(
            'portfolio_system_info',
            'Informations sur le système',
            registry=self.registry
        )
    
    def _update_system_info(self):
        """Met à jour les informations système."""
        if not self.enabled:
            return
            
        try:
            import platform
            import sys
            
            self.system_info.info({
                'version': '1.0.0',
                'python_version': platform.python_version(),
                'platform': platform.platform(),
                'architecture': platform.architecture()[0],
                'hostname': platform.node(),
                'startup_time': datetime.now().isoformat()
            })
        except Exception as e:
            logger.warning(f"Erreur lors de la mise à jour des infos système: {e}")
    
    def record_api_request(self, method: str, endpoint: str, status_code: int, duration: float):
        """
        Enregistre une requête API.
        
        Args:
            method: Méthode HTTP
            endpoint: Point de terminaison
            status_code: Code de statut HTTP
            duration: Durée en secondes
        """
        if not self.enabled:
            return
        
        try:
            status = str(status_code)
            self.api_requests_total.labels(method=method, endpoint=endpoint, status=status).inc()
            self.api_request_duration.labels(method=method, endpoint=endpoint).observe(duration)
            
            # Enregistrer les erreurs
            if status_code >= 400:
                error_type = 'client_error' if status_code < 500 else 'server_error'
                self.api_errors_total.labels(
                    method=method, 
                    endpoint=endpoint, 
                    error_type=error_type
                ).inc()
        except Exception as e:
            logger.error(f"Erreur lors de l'enregistrement de la requête API: {e}")
    
    def record_prediction(self, model_version: str, prediction_type: str, 
                         duration: float, confidence: float):
        """
        Enregistre une prédiction de modèle.
        
        Args:
            model_version: Version du modèle
            prediction_type: Type de prédiction (allocation, backtest, etc.)
            duration: Durée en secondes
            confidence: Confiance (0-1)
        """
        if not self.enabled:
            return
        
        try:
            self.model_predictions_total.labels(
                model_version=model_version, 
                prediction_type=prediction_type
            ).inc()
            
            self.model_prediction_duration.labels(model_version=model_version).observe(duration)
            self.model_prediction_confidence.labels(model_version=model_version).observe(confidence)
        except Exception as e:
            logger.error(f"Erreur lors de l'enregistrement de la prédiction: {e}")
    
    def update_portfolio_metrics(self, portfolio_id: str, metrics: Dict[str, float]):
        """
        Met à jour les métriques de portefeuille.
        
        Args:
            portfolio_id: Identifiant du portefeuille
            metrics: Dictionnaire des métriques
        """
        if not self.enabled:
            return
        
        try:
            if 'value' in metrics:
                self.portfolio_value.labels(portfolio_id=portfolio_id).set(metrics['value'])
            
            if 'daily_return' in metrics:
                self.portfolio_return.labels(
                    portfolio_id=portfolio_id, 
                    period='daily'
                ).set(metrics['daily_return'] * 100)
            
            if 'total_return' in metrics:
                self.portfolio_return.labels(
                    portfolio_id=portfolio_id, 
                    period='total'
                ).set(metrics['total_return'])
            
            if 'volatility' in metrics:
                self.portfolio_volatility.labels(portfolio_id=portfolio_id).set(metrics['volatility'])
            
            if 'sharpe_ratio' in metrics:
                self.portfolio_sharpe_ratio.labels(portfolio_id=portfolio_id).set(metrics['sharpe_ratio'])
            
            if 'max_drawdown' in metrics:
                self.portfolio_max_drawdown.labels(portfolio_id=portfolio_id).set(metrics['max_drawdown'])
        except Exception as e:
            logger.error(f"Erreur lors de la mise à jour des métriques de portefeuille: {e}")
    
    def record_transaction_cost(self, portfolio_id: str, cost: float):
        """
        Enregistre un coût de transaction.
        
        Args:
            portfolio_id: Identifiant du portefeuille
            cost: Coût en USD
        """
        if not self.enabled:
            return
        
        try:
            self.transaction_costs.labels(portfolio_id=portfolio_id).inc(cost)
        except Exception as e:
            logger.error(f"Erreur lors de l'enregistrement du coût de transaction: {e}")
    
    def record_rebalancing(self, portfolio_id: str, trigger_type: str):
        """
        Enregistre un événement de rééquilibrage.
        
        Args:
            portfolio_id: Identifiant du portefeuille
            trigger_type: Type de déclencheur (threshold, schedule, manual)
        """
        if not self.enabled:
            return
        
        try:
            self.rebalancing_events.labels(
                portfolio_id=portfolio_id, 
                trigger_type=trigger_type
            ).inc()
        except Exception as e:
            logger.error(f"Erreur lors de l'enregistrement du rééquilibrage: {e}")
    
    def update_model_health(self, model_name: str, model_version: str, is_healthy: bool):
        """
        Met à jour l'état de santé d'un modèle.
        
        Args:
            model_name: Nom du modèle
            model_version: Version du modèle
            is_healthy: True si le modèle est en bonne santé
        """
        if not self.enabled:
            return
        
        try:
            health_value = 1.0 if is_healthy else 0.0
            self.models_health.labels(
                model_name=model_name, 
                model_version=model_version
            ).set(health_value)
        except Exception as e:
            logger.error(f"Erreur lors de la mise à jour de la santé du modèle: {e}")
    
    def update_memory_usage(self):
        """Met à jour l'utilisation mémoire."""
        if not self.enabled:
            return
        
        try:
            import psutil
            process = psutil.Process()
            memory_bytes = process.memory_info().rss
            self.system_memory_usage.set(memory_bytes)
        except ImportError:
            # psutil non disponible, ignorer silencieusement
            pass
        except Exception as e:
            logger.warning(f"Erreur lors de la mise à jour de la mémoire: {e}")
    
    def record_mlflow_run(self, experiment_name: str, status: str):
        """
        Enregistre un run MLflow.
        
        Args:
            experiment_name: Nom de l'expérience
            status: Statut du run (FINISHED, FAILED, RUNNING)
        """
        if not self.enabled:
            return
        
        try:
            self.mlflow_runs_total.labels(
                experiment_name=experiment_name, 
                status=status
            ).inc()
        except Exception as e:
            logger.error(f"Erreur lors de l'enregistrement du run MLflow: {e}")
    
    def get_metrics(self) -> str:
        """
        Récupère toutes les métriques au format Prometheus.
        
        Returns:
            str: Métriques formatées
        """
        if not self.enabled:
            return "# Prometheus non disponible\n"
        
        try:
            # Mettre à jour les métriques dynamiques
            self.update_memory_usage()
            
            metrics_data = generate_latest(self.registry)
            if isinstance(metrics_data, bytes):
                return metrics_data.decode('utf-8')
            return metrics_data
        except Exception as e:
            logger.error(f"Erreur lors de la génération des métriques: {e}")
            return f"# Erreur: {e}\n"
    
    def health_check(self) -> Dict[str, Any]:
        """
        Effectue un contrôle de santé du système de métriques.
        
        Returns:
            Statut de santé
        """
        return {
            "prometheus_enabled": self.enabled,
            "registry_collectors": len(list(self.registry._collector_to_names.keys())) if self.enabled else 0,
            "status": "healthy" if self.enabled else "disabled"
        }


def monitor_api_request(endpoint_name: str = None):
    """
    Décorateur pour monitorer les requêtes API.
    
    Args:
        endpoint_name: Nom personnalisé pour l'endpoint
    """
    def decorator(func):
        @wraps(func)
        async def async_wrapper(request, *args, **kwargs):
            if not portfolio_metrics or not portfolio_metrics.enabled:
                return await func(request, *args, **kwargs)
            
            method = request.method
            endpoint = endpoint_name or func.__name__
            start_time = time.time()
            status_code = 200
            
            try:
                response = await func(request, *args, **kwargs)
                if hasattr(response, 'status_code'):
                    status_code = response.status_code
                return response
            except Exception as e:
                status_code = 500
                raise
            finally:
                duration = time.time() - start_time
                portfolio_metrics.record_api_request(method, endpoint, status_code, duration)
        
        @wraps(func)
        def sync_wrapper(*args, **kwargs):
            if not portfolio_metrics or not portfolio_metrics.enabled:
                return func(*args, **kwargs)
            
            endpoint = endpoint_name or func.__name__
            start_time = time.time()
            status_code = 200
            
            try:
                result = func(*args, **kwargs)
                return result
            except Exception as e:
                status_code = 500
                raise
            finally:
                duration = time.time() - start_time
                portfolio_metrics.record_api_request("FUNC", endpoint, status_code, duration)
        
        # Retourner le bon wrapper selon le type de fonction
        import asyncio
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        else:
            return sync_wrapper
    
    return decorator


def monitor_prediction(model_version: str = "unknown", prediction_type: str = "allocation"):
    """
    Décorateur pour monitorer les prédictions de modèle.
    
    Args:
        model_version: Version du modèle
        prediction_type: Type de prédiction
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            if not portfolio_metrics or not portfolio_metrics.enabled:
                return func(*args, **kwargs)
            
            start_time = time.time()
            result = func(*args, **kwargs)
            duration = time.time() - start_time
            
            # Extraire la confiance du résultat si disponible
            confidence = 0.5  # Valeur par défaut
            if isinstance(result, dict) and 'confidence' in result:
                confidence = result['confidence']
            
            portfolio_metrics.record_prediction(
                model_version, prediction_type, duration, confidence
            )
            
            return result
        return wrapper
    return decorator


class MetricsCollector:
    """
    Collecteur de métriques qui s'exécute en arrière-plan.
    """
    
    def __init__(self, metrics: PortfolioMetrics, interval: int = 30):
        """
        Initialise le collecteur.
        
        Args:
            metrics: Instance des métriques
            interval: Intervalle de collecte en secondes
        """
        self.metrics = metrics
        self.interval = interval
        self.running = False
        self.thread = None
    
    def start(self):
        """Démarre la collecte en arrière-plan."""
        if self.running or not self.metrics.enabled:
            return
        
        self.running = True
        self.thread = threading.Thread(target=self._collect_loop, daemon=True)
        self.thread.start()
        logger.info(f"🔄 Collecteur de métriques démarré (intervalle: {self.interval}s)")
    
    def stop(self):
        """Arrête la collecte."""
        self.running = False
        if self.thread:
            self.thread.join()
        logger.info("🛑 Collecteur de métriques arrêté")
    
    def _collect_loop(self):
        """Boucle principale de collecte."""
        while self.running:
            try:
                self._collect_system_metrics()
                self._collect_model_health()
                time.sleep(self.interval)
            except Exception as e:
                logger.error(f"Erreur lors de la collecte: {e}")
                time.sleep(self.interval)
    
    def _collect_system_metrics(self):
        """Collecte les métriques système."""
        self.metrics.update_memory_usage()
    
    def _collect_model_health(self):
        """Vérifie la santé des modèles."""
        # Ici, vous pouvez ajouter la logique pour vérifier la santé des modèles
        # Par exemple, vérifier si les fichiers de modèles existent, etc.
        pass


def start_metrics_server(port: int = 8001, metrics: PortfolioMetrics = None):
    """
    Démarre le serveur de métriques Prometheus.
    
    Args:
        port: Port d'écoute
        metrics: Instance des métriques
    """
    if not PROMETHEUS_AVAILABLE:
        logger.warning("Prometheus non disponible, serveur de métriques non démarré")
        return False
    
    try:
        if metrics and metrics.enabled:
            start_http_server(port, registry=metrics.registry)
        else:
            start_http_server(port)
        
        logger.info(f"🎯 Serveur de métriques démarré sur le port {port}")
        logger.info(f"📊 Métriques disponibles sur: http://localhost:{port}/metrics")
        return True
    except Exception as e:
        logger.error(f"Erreur lors du démarrage du serveur de métriques: {e}")
        return False


# Instance globale (créée seulement si Prometheus est disponible)
try:
    portfolio_metrics = PortfolioMetrics()
except Exception as e:
    logger.error(f"Erreur lors de la création de l'instance globale: {e}")
    portfolio_metrics = None

# Export des classes et fonctions principales
__all__ = [
    'PortfolioMetrics',
    'portfolio_metrics',
    'monitor_api_request',
    'monitor_prediction',
    'MetricsCollector',
    'start_metrics_server',
    'PROMETHEUS_AVAILABLE'
]