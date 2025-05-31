"""
Intégration du monitoring Prometheus avec FastAPI.
Middleware et endpoints pour exposer les métriques.
"""
import time
import logging
from typing import Optional, Dict, Any
from functools import wraps

logger = logging.getLogger(__name__)

# Imports conditionnels FastAPI
try:
    from fastapi import FastAPI, Request, Response, HTTPException
    from fastapi.middleware.base import BaseHTTPMiddleware
    from fastapi.responses import PlainTextResponse
    FASTAPI_AVAILABLE = True
    logger.info("✅ FastAPI disponible")
except ImportError:
    FASTAPI_AVAILABLE = False
    logger.warning("⚠️ FastAPI non disponible")
    
    # Classes factices
    class BaseHTTPMiddleware:
        def __init__(self, app): 
            self.app = app
        async def dispatch(self, request, call_next):
            return await call_next(request)
    
    class Request:
        def __init__(self):
            self.method = "GET"
            self.url = ""
    
    class Response:
        def __init__(self):
            self.status_code = 200
    
    class PlainTextResponse:
        def __init__(self, content, **kwargs):
            self.content = content

# Import des métriques
try:
    from .metrics import portfolio_metrics, PROMETHEUS_AVAILABLE
except ImportError:
    try:
        from src.monitoring.simple_metrics import portfolio_metrics, PROMETHEUS_AVAILABLE
    except ImportError:
        logger.warning("⚠️ Impossible d'importer les métriques")
        portfolio_metrics = None
        PROMETHEUS_AVAILABLE = False


class MetricsMiddleware(BaseHTTPMiddleware):
    """
    Middleware FastAPI pour collecter automatiquement les métriques des requêtes.
    """
    
    def __init__(self, app, metrics_instance=None):
        """
        Initialise le middleware.
        
        Args:
            app: Application FastAPI
            metrics_instance: Instance des métriques (optionnel)
        """
        super().__init__(app)
        self.metrics = metrics_instance or portfolio_metrics
        self.enabled = (
            FASTAPI_AVAILABLE and 
            self.metrics is not None and 
            getattr(self.metrics, 'enabled', False)
        )
        
        if self.enabled:
            logger.info("✅ Middleware de métriques activé")
        else:
            logger.warning("⚠️ Middleware de métriques désactivé")
    
    async def dispatch(self, request: Request, call_next):
        """
        Traite une requête et collecte les métriques.
        
        Args:
            request: Requête HTTP
            call_next: Fonction suivante dans la chaîne
            
        Returns:
            Response: Réponse HTTP
        """
        if not self.enabled:
            return await call_next(request)
        
        # Extraire les informations de la requête
        method = request.method
        path = str(request.url.path)
        start_time = time.time()
        
        # Incrémenter les requêtes actives
        if hasattr(self.metrics, 'increment_active_requests'):
            self.metrics.increment_active_requests()
        
        try:
            # Traiter la requête
            response = await call_next(request)
            status_code = response.status_code
            
        except Exception as e:
            # En cas d'erreur, définir le code de statut
            status_code = 500
            logger.error(f"Erreur lors du traitement de la requête {method} {path}: {e}")
            raise
        
        finally:
            # Calculer la durée et enregistrer les métriques
            duration = time.time() - start_time
            
            try:
                # Enregistrer les métriques
                self.metrics.record_api_request(method, path, status_code, duration)
                
                # Décrémenter les requêtes actives
                if hasattr(self.metrics, 'decrement_active_requests'):
                    self.metrics.decrement_active_requests()
                    
            except Exception as e:
                logger.error(f"Erreur lors de l'enregistrement des métriques: {e}")
        
        return response


class HealthMiddleware(BaseHTTPMiddleware):
    """
    Middleware pour ajouter des headers de santé.
    """
    
    def __init__(self, app, service_name: str = "portfolio-api"):
        super().__init__(app)
        self.service_name = service_name
    
    async def dispatch(self, request: Request, call_next):
        response = await call_next(request)
        
        # Ajouter des headers de santé
        if hasattr(response, 'headers'):
            response.headers["X-Service-Name"] = self.service_name
            response.headers["X-Timestamp"] = str(int(time.time()))
        
        return response


def setup_monitoring(app: FastAPI, metrics_instance=None, enable_middleware: bool = True):
    """
    Configure le monitoring pour une application FastAPI.
    
    Args:
        app: Application FastAPI
        metrics_instance: Instance des métriques personnalisée
        enable_middleware: Activer le middleware automatique
    """
    if not FASTAPI_AVAILABLE:
        logger.warning("FastAPI non disponible, monitoring non configuré")
        return False
    
    metrics = metrics_instance or portfolio_metrics
    
    if not metrics or not getattr(metrics, 'enabled', False):
        logger.warning("Métriques non disponibles, monitoring limité")
        metrics = None
    
    # Ajouter le middleware de métriques
    if enable_middleware and metrics:
        app.add_middleware(MetricsMiddleware, metrics_instance=metrics)
        logger.info("✅ Middleware de métriques ajouté")
    
    # Ajouter le middleware de santé
    app.add_middleware(HealthMiddleware)
    logger.info("✅ Middleware de santé ajouté")
    
    # Ajouter les endpoints de monitoring
    setup_metrics_endpoints(app, metrics)
    
    logger.info("✅ Monitoring configuré pour FastAPI")
    return True


def setup_metrics_endpoints(app: FastAPI, metrics_instance=None):
    """
    Ajoute les endpoints de métriques à l'application FastAPI.
    
    Args:
        app: Application FastAPI
        metrics_instance: Instance des métriques
    """
    metrics = metrics_instance or portfolio_metrics
    
    @app.get("/metrics", response_class=PlainTextResponse, include_in_schema=False)
    async def get_metrics():
        """
        Endpoint pour exposer les métriques Prometheus.
        
        Returns:
            Métriques au format Prometheus
        """
        if not metrics or not getattr(metrics, 'enabled', False):
            return PlainTextResponse(
                "# Métriques non disponibles\n# Prometheus client requis\n",
                media_type="text/plain"
            )
        
        try:
            metrics_data = metrics.get_metrics()
            return PlainTextResponse(metrics_data, media_type="text/plain")
        except Exception as e:
            logger.error(f"Erreur lors de la récupération des métriques: {e}")
            return PlainTextResponse(
                f"# Erreur lors de la récupération des métriques: {e}\n",
                media_type="text/plain",
                status_code=500
            )
    
    @app.get("/health", include_in_schema=False)
    async def health_check():
        """
        Endpoint de vérification de santé.
        
        Returns:
            Statut de santé de l'application
        """
        health_status = {
            "status": "healthy",
            "timestamp": int(time.time()),
            "service": "portfolio-api",
            "version": "1.0.0"
        }
        
        # Ajouter les informations sur les métriques si disponibles
        if metrics and hasattr(metrics, 'health_check'):
            try:
                metrics_health = metrics.health_check()
                health_status["metrics"] = metrics_health
            except Exception as e:
                health_status["metrics"] = {"error": str(e)}
        else:
            health_status["metrics"] = {"status": "disabled"}
        
        return health_status
    
    @app.get("/ready", include_in_schema=False)
    async def readiness_check():
        """
        Endpoint de vérification de disponibilité.
        
        Returns:
            Statut de disponibilité
        """
        return {
            "ready": True,
            "timestamp": int(time.time()),
            "dependencies": {
                "metrics": metrics is not None and getattr(metrics, 'enabled', False),
                "prometheus": PROMETHEUS_AVAILABLE
            }
        }
    
    @app.get("/metrics/summary", include_in_schema=False)
    async def metrics_summary():
        """
        Endpoint pour un résumé des métriques principales.
        
        Returns:
            Résumé des métriques
        """
        if not metrics or not getattr(metrics, 'enabled', False):
            raise HTTPException(
                status_code=503, 
                detail="Métriques non disponibles"
            )
        
        try:
            # Ici vous pouvez ajouter la logique pour extraire un résumé
            # des métriques principales depuis l'instance
            summary = {
                "prometheus_enabled": getattr(metrics, 'enabled', False),
                "total_collectors": "N/A",
                "last_updated": int(time.time())
            }
            
            if hasattr(metrics, 'health_check'):
                health = metrics.health_check()
                summary.update(health)
            
            return summary
            
        except Exception as e:
            logger.error(f"Erreur lors de la génération du résumé: {e}")
            raise HTTPException(
                status_code=500,
                detail=f"Erreur lors de la génération du résumé: {e}"
            )


def monitor_endpoint(endpoint_name: str = None, record_body_size: bool = False):
    """
    Décorateur pour monitorer un endpoint FastAPI spécifique.
    
    Args:
        endpoint_name: Nom personnalisé pour l'endpoint
        record_body_size: Enregistrer la taille du body de réponse
    """
    def decorator(func):
        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            if not portfolio_metrics or not getattr(portfolio_metrics, 'enabled', False):
                return await func(*args, **kwargs)
            
            endpoint = endpoint_name or func.__name__
            start_time = time.time()
            
            try:
                result = await func(*args, **kwargs)
                
                # Enregistrer des métriques spécifiques si nécessaire
                if record_body_size and hasattr(result, '__len__'):
                    # Vous pouvez ajouter ici la logique pour enregistrer la taille
                    pass
                
                return result
                
            except Exception as e:
                # Enregistrer l'erreur
                if hasattr(portfolio_metrics, 'record_error'):
                    portfolio_metrics.record_error("endpoint_error", endpoint)
                raise
            
            finally:
                duration = time.time() - start_time
                # Les métriques seront automatiquement enregistrées par le middleware
        
        @wraps(func)
        def sync_wrapper(*args, **kwargs):
            return func(*args, **kwargs)
        
        # Retourner le bon wrapper
        import asyncio
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        else:
            return sync_wrapper
    
    return decorator


def create_monitored_app(title: str = "Portfolio API", version: str = "1.0.0") -> Optional['FastAPI']:
    """
    Crée une application FastAPI avec monitoring pré-configuré.
    
    Args:
        title: Titre de l'application
        version: Version de l'application
        
    Returns:
        Application FastAPI configurée ou None si FastAPI n'est pas disponible
    """
    if not FASTAPI_AVAILABLE:
        logger.error("FastAPI non disponible, impossible de créer l'application")
        return None
    
    # Importer FastAPI ici pour éviter les erreurs
    from fastapi import FastAPI
    
    app = FastAPI(
        title=title,
        version=version,
        description="API Portfolio avec monitoring Prometheus intégré"
    )
    
    # Configurer le monitoring
    setup_monitoring(app)
    
    logger.info(f"✅ Application FastAPI '{title}' créée avec monitoring")
    return app


# Fonctions utilitaires
def get_metrics_middleware_instance():
    """Retourne l'instance du middleware de métriques."""
    return MetricsMiddleware


def is_monitoring_available() -> bool:
    """Vérifie si le monitoring est disponible."""
    return (
        FASTAPI_AVAILABLE and 
        PROMETHEUS_AVAILABLE and 
        portfolio_metrics is not None and 
        getattr(portfolio_metrics, 'enabled', False)
    )


# Export des classes et fonctions principales
__all__ = [
    'MetricsMiddleware',
    'HealthMiddleware', 
    'setup_monitoring',
    'setup_metrics_endpoints',
    'monitor_endpoint',
    'create_monitored_app',
    'is_monitoring_available',
    'FASTAPI_AVAILABLE'
]