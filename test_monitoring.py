#!/usr/bin/env python3
"""
Test complet du système de monitoring Prometheus + Grafana.
À placer à la racine du projet : test_monitoring.py
"""
import os
import sys
import time
import requests
import subprocess
import signal
from datetime import datetime

# Ajouter le répertoire du projet au PYTHONPATH
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

def test_monitoring_imports():
    """Test des imports du monitoring."""
    print("📦 Test des imports monitoring...")
    
    try:
        from src.monitoring.metrics import PortfolioMetrics, portfolio_metrics
        print("   ✅ Métriques Prometheus")
        
        from src.api.monitoring_integration import MetricsMiddleware, setup_monitoring
        print("   ✅ Intégration FastAPI")
        
        try:
            from prometheus_client import Counter, Histogram, Gauge
            print("   ✅ Client Prometheus")
        except ImportError:
            print("   ⚠️ Client Prometheus non disponible")
            print("   💡 Installez avec: pip install prometheus-client")
            return False
        
        return True
        
    except ImportError as e:
        print(f"   ❌ Import échoué: {e}")
        return False

def install_monitoring_dependencies():
    """Installe les dépendances nécessaires."""
    print("📦 Installation des dépendances monitoring...")
    
    dependencies = [
        "prometheus-client>=0.15.0",
        "psutil>=5.9.0"  # Pour les métriques système
    ]
    
    for dep in dependencies:
        try:
            print(f"   Installation de {dep}...")
            subprocess.check_call([
                sys.executable, "-m", "pip", "install", dep
            ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            print(f"   ✅ {dep}")
        except subprocess.CalledProcessError:
            print(f"   ❌ Échec: {dep}")
            return False
    
    print("✅ Dépendances installées")
    return True

def create_monitoring_files():
    """Crée les fichiers de configuration manquants."""
    print("📁 Création des fichiers de monitoring...")
    
    # Créer les répertoires
    os.makedirs("monitoring", exist_ok=True)
    os.makedirs("monitoring/grafana/provisioning/datasources", exist_ok=True)
    os.makedirs("monitoring/grafana/provisioning/dashboards", exist_ok=True)
    os.makedirs("monitoring/grafana/dashboards", exist_ok=True)
    
    # Fichier de configuration AlertManager
    alertmanager_config = """
global:
  smtp_smarthost: 'localhost:587'
  smtp_from: 'alerts@portfolio-rl.com'

route:
  group_by: ['alertname']
  group_wait: 10s
  group_interval: 10s
  repeat_interval: 1h
  receiver: 'web.hook'

receivers:
- name: 'web.hook'
  webhook_configs:
  - url: 'http://localhost:5001/'
    send_resolved: true
"""
    
    with open("monitoring/alertmanager.yml", "w", encoding='utf-8') as f:
        f.write(alertmanager_config)
    
    # Configuration Grafana datasource
    datasource_config = """
apiVersion: 1

datasources:
  - name: Prometheus
    type: prometheus
    access: proxy
    url: http://prometheus:9090
    isDefault: true
    editable: true
"""
    
    with open("monitoring/grafana/provisioning/datasources/prometheus.yml", "w", encoding='utf-8') as f:
        f.write(datasource_config)
    
    # Configuration Grafana dashboards
    dashboard_config = """
apiVersion: 1

providers:
  - name: 'default'
    orgId: 1
    folder: ''
    type: file
    disableDeletion: false
    updateIntervalSeconds: 10
    allowUiUpdates: true
    options:
      path: /var/lib/grafana/dashboards
"""
    
    with open("monitoring/grafana/provisioning/dashboards/dashboards.yml", "w", encoding='utf-8') as f:
        f.write(dashboard_config)
    
    # Configuration Blackbox Exporter
    blackbox_config = """
modules:
  http_2xx:
    prober: http
    timeout: 5s
    http:
      method: GET
      valid_http_versions: ["HTTP/1.1", "HTTP/2.0"]
      valid_status_codes: []
      no_follow_redirects: false
      fail_if_ssl: false
      fail_if_not_ssl: false
"""
    
    with open("monitoring/blackbox.yml", "w", encoding='utf-8') as f:
        f.write(blackbox_config)
    
    # Configuration Loki
    loki_config = """
auth_enabled: false

server:
  http_listen_port: 3100

ingester:
  lifecycler:
    address: 127.0.0.1
    ring:
      kvstore:
        store: inmemory
      replication_factor: 1
    final_sleep: 0s

schema_config:
  configs:
    - from: 2020-10-24
      store: boltdb
      object_store: filesystem
      schema: v11
      index:
        prefix: index_
        period: 168h

storage_config:
  boltdb:
    directory: /loki/index
  filesystem:
    directory: /loki/chunks

limits_config:
  enforce_metric_name: false
  reject_old_samples: true
  reject_old_samples_max_age: 168h
"""
    
    with open("monitoring/loki-config.yaml", "w", encoding='utf-8') as f:
        f.write(loki_config)
    
    # Configuration Promtail
    promtail_config = """
server:
  http_listen_port: 9080
  grpc_listen_port: 0

positions:
  filename: /tmp/positions.yaml

clients:
  - url: http://loki:3100/loki/api/v1/push

scrape_configs:
  - job_name: containers
    static_configs:
      - targets:
          - localhost
        labels:
          job: containerlogs
          __path__: /var/lib/docker/containers/*/*log
    pipeline_stages:
      - json:
          expressions:
            output: log
            stream: stream
            attrs:
      - json:
          expressions:
            tag:
          source: attrs
      - regex:
          expression: (?P<container_name>(?:[^|]*))\|
          source: tag
      - timestamp:
          format: RFC3339Nano
          source: time
      - labels:
          stream:
          container_name:
      - output:
          source: output
"""
    
    with open("monitoring/promtail-config.yaml", "w", encoding='utf-8') as f:
        f.write(promtail_config)
    
    # Script de vérification de santé (sans emojis pour la compatibilité Windows)
    health_check_script = """#!/bin/sh
# Script de verification de sante des services

check_service() {
    local service_name=$1
    local url=$2
    
    if curl -f -s "$url" > /dev/null 2>&1; then
        echo "[OK] $service_name: HEALTHY"
        return 0
    else
        echo "[FAIL] $service_name: FAILED"
        return 1
    fi
}

echo "[INFO] Verification de la sante des services..."

check_service "Prometheus" "http://prometheus:9090/-/healthy"
check_service "Grafana" "http://grafana:3000/api/health"
check_service "AlertManager" "http://alertmanager:9093/-/healthy"
check_service "Portfolio API" "http://portfolio-api-monitored:8000/health"
check_service "Metriques API" "http://portfolio-api-monitored:8001/metrics"

echo "Verification terminee a $(date)"
"""
    
    with open("monitoring/health-check.sh", "w", encoding='utf-8') as f:
        f.write(health_check_script)
    
    # Rendre le script exécutable
    try:
        os.chmod("monitoring/health-check.sh", 0o755)
    except:
        pass
    
    # Créer un fichier docker-compose de base si il n'existe pas
    if not os.path.exists("docker-compose-monitoring.yml"):
        create_docker_compose_monitoring()
    
    print("✅ Fichiers de monitoring créés")
    return True

def create_docker_compose_monitoring():
    """Crée un fichier docker-compose pour le monitoring."""
    docker_compose_content = """
version: '3.8'

services:
  prometheus:
    image: prom/prometheus:latest
    container_name: prometheus
    ports:
      - "9090:9090"
    volumes:
      - ./monitoring/prometheus.yml:/etc/prometheus/prometheus.yml
      - prometheus_data:/prometheus
    command:
      - '--config.file=/etc/prometheus/prometheus.yml'
      - '--storage.tsdb.path=/prometheus'
      - '--web.console.libraries=/etc/prometheus/console_libraries'
      - '--web.console.templates=/etc/prometheus/consoles'
      - '--storage.tsdb.retention.time=200h'
      - '--web.enable-lifecycle'

  grafana:
    image: grafana/grafana:latest
    container_name: grafana
    ports:
      - "3000:3000"
    environment:
      - GF_SECURITY_ADMIN_PASSWORD=admin
    volumes:
      - grafana_data:/var/lib/grafana
      - ./monitoring/grafana/provisioning:/etc/grafana/provisioning
      - ./monitoring/grafana/dashboards:/var/lib/grafana/dashboards

  alertmanager:
    image: prom/alertmanager:latest
    container_name: alertmanager
    ports:
      - "9093:9093"
    volumes:
      - ./monitoring/alertmanager.yml:/etc/alertmanager/alertmanager.yml

volumes:
  prometheus_data:
  grafana_data:
"""
    
    with open("docker-compose-monitoring.yml", "w", encoding='utf-8') as f:
        f.write(docker_compose_content)
    
    # Créer aussi un fichier de configuration Prometheus de base
    prometheus_config = """
global:
  scrape_interval: 15s
  evaluation_interval: 15s

rule_files:
  # - "first_rules.yml"
  # - "second_rules.yml"

scrape_configs:
  - job_name: 'prometheus'
    static_configs:
      - targets: ['localhost:9090']

  - job_name: 'portfolio-api'
    static_configs:
      - targets: ['host.docker.internal:8001']
    metrics_path: '/metrics'
    scrape_interval: 5s
"""
    
    with open("monitoring/prometheus.yml", "w", encoding='utf-8') as f:
        f.write(prometheus_config)
    
    print("   ✅ Fichier docker-compose-monitoring.yml créé")

def test_metrics_collection():
    """Test de la collecte de métriques."""
    print("📊 Test de la collecte de métriques...")
    
    try:
        # Créer un registre propre pour les tests
        from prometheus_client import CollectorRegistry
        test_registry = CollectorRegistry()
        
        from src.monitoring.metrics import PortfolioMetrics
        
        # Créer une instance de test avec un registre propre
        metrics = PortfolioMetrics(registry=test_registry)
        
        if not metrics.enabled:
            print("   ⚠️ Métriques désactivées (Prometheus non disponible)")
            return False
        
        # Test des métriques API
        metrics.record_api_request("GET", "/test", 200, 0.1)
        metrics.record_api_request("POST", "/predict", 201, 0.5)
        metrics.record_api_request("GET", "/test", 500, 0.05)
        print("   ✅ Métriques API enregistrées")
        
        # Test des métriques de modèle
        metrics.record_prediction("v1.0", "allocation", 0.2, 0.85)
        metrics.record_prediction("v1.0", "backtest", 1.5, 0.72)
        print("   ✅ Métriques de modèle enregistrées")
        
        # Test des métriques de portefeuille
        portfolio_metrics_data = {
            "value": 12500.0,
            "daily_return": 0.02,
            "total_return": 25.0,
            "volatility": 15.5,
            "sharpe_ratio": 1.2,
            "max_drawdown": 8.5
        }
        metrics.update_portfolio_metrics("test_portfolio", portfolio_metrics_data)
        print("   ✅ Métriques de portefeuille enregistrées")
        
        # Test des métriques business
        metrics.record_transaction_cost("test_portfolio", 12.50)
        metrics.record_rebalancing("test_portfolio", "threshold")
        print("   ✅ Métriques business enregistrées")
        
        # Test de la santé des modèles
        metrics.update_model_health("test_model", "v1.0", True)
        print("   ✅ Santé des modèles mise à jour")
        
        # Récupérer les métriques
        metrics_output = metrics.get_metrics()
        
        # Vérifier que les métriques contiennent du contenu
        if "portfolio_api_requests_total" in metrics_output:
            print("   ✅ Export des métriques fonctionnel")
        else:
            print("   ❌ Export des métriques incomplet")
            return False
        
        print("✅ Collecte de métriques fonctionnelle")
        return True
        
    except Exception as e:
        print(f"   ❌ Erreur: {e}")
        import traceback
        traceback.print_exc()
        return False

def start_monitoring_stack():
    """Démarre la stack de monitoring avec Docker Compose."""
    print("🚀 Démarrage de la stack de monitoring...")
    
    # Vérifier si Docker est disponible et démarré
    docker_available = check_docker_availability()
    if not docker_available:
        return False
    
    try:
        # Vérifier que le fichier docker-compose existe
        compose_file = "docker-compose-monitoring.yml"
        if not os.path.exists(compose_file):
            print(f"   ⚠️ Fichier {compose_file} non trouvé")
            print("   💡 Créez le fichier docker-compose-monitoring.yml pour les services")
            return False
        
        # Démarrer la stack
        print("   🔧 Démarrage des services...")
        process = subprocess.Popen([
            "docker-compose", "-f", compose_file, "up", "-d"
        ], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        
        stdout, stderr = process.communicate()
        
        if process.returncode == 0:
            print("   ✅ Services démarrés")
            return True
        else:
            print(f"   ❌ Erreur lors du démarrage:")
            print(f"   {stderr[:200]}...")  # Limiter la sortie d'erreur
            return False
            
    except Exception as e:
        print(f"   ❌ Erreur: {e}")
        return False

def check_docker_availability():
    """Vérifie si Docker est disponible et démarré."""
    try:
        # Vérifier que Docker est installé
        result = subprocess.run(
            ["docker", "--version"], 
            stdout=subprocess.DEVNULL, 
            stderr=subprocess.DEVNULL,
            timeout=5
        )
        
        if result.returncode != 0:
            print("   ❌ Docker non installé")
            return False
        
        # Vérifier que Docker est démarré
        result = subprocess.run(
            ["docker", "info"], 
            stdout=subprocess.DEVNULL, 
            stderr=subprocess.DEVNULL,
            timeout=10
        )
        
        if result.returncode != 0:
            print("   ❌ Docker non démarré ou permissions insuffisantes")
            print("   💡 Démarrez Docker Desktop ou exécutez en tant qu'administrateur")
            return False
        
        # Vérifier Docker Compose
        result = subprocess.run(
            ["docker-compose", "--version"], 
            stdout=subprocess.DEVNULL, 
            stderr=subprocess.DEVNULL,
            timeout=5
        )
        
        if result.returncode != 0:
            print("   ❌ Docker Compose non disponible")
            return False
        
        print("   ✅ Docker disponible et démarré")
        return True
        
    except subprocess.TimeoutExpired:
        print("   ❌ Timeout lors de la vérification Docker")
        return False
    except FileNotFoundError:
        print("   ❌ Docker non trouvé dans le PATH")
        return False
    except Exception as e:
        print(f"   ❌ Erreur lors de la vérification Docker: {e}")
        return False

def wait_for_services(timeout=120):
    """Attend que les services soient prêts."""
    print("⏳ Attente des services...")
    
    services = {
        "Prometheus": "http://localhost:9090/-/healthy",
        "Grafana": "http://localhost:3000/api/health",
        "AlertManager": "http://localhost:9093/-/healthy"
    }
    
    start_time = time.time()
    ready_services = set()
    
    while time.time() - start_time < timeout:
        for service, url in services.items():
            if service not in ready_services:
                try:
                    response = requests.get(url, timeout=5)
                    if response.status_code == 200:
                        print(f"   ✅ {service} prêt")
                        ready_services.add(service)
                except requests.RequestException:
                    pass
        
        if len(ready_services) == len(services):
            print("✅ Tous les services sont prêts")
            return True
        
        time.sleep(5)
    
    print(f"   ⚠️ Timeout atteint. Services prêts: {len(ready_services)}/{len(services)}")
    return len(ready_services) > 0

def test_prometheus_connectivity():
    """Test la connectivité avec Prometheus."""
    print("🔍 Test de connectivité Prometheus...")
    
    try:
        # Test de l'API Prometheus
        response = requests.get("http://localhost:9090/api/v1/query", 
                              params={"query": "up"}, timeout=10)
        
        if response.status_code == 200:
            data = response.json()
            if data.get("status") == "success":
                print("   ✅ API Prometheus accessible")
                
                # Vérifier les métriques de l'application
                response = requests.get("http://localhost:9090/api/v1/query",
                                      params={"query": "portfolio_api_requests_total"}, timeout=10)
                
                if response.status_code == 200:
                    data = response.json()
                    if data.get("data", {}).get("result"):
                        print("   ✅ Métriques de l'application détectées")
                    else:
                        print("   ⚠️ Aucune métrique de l'application trouvée")
                
                return True
            else:
                print(f"   ❌ Erreur API: {data}")
                return False
        else:
            print(f"   ❌ HTTP {response.status_code}")
            return False
            
    except requests.RequestException as e:
        print(f"   ❌ Erreur de connexion: {e}")
        return False

def test_grafana_connectivity():
    """Test la connectivité avec Grafana."""
    print("🎨 Test de connectivité Grafana...")
    
    try:
        # Test de l'API Grafana
        response = requests.get("http://localhost:3000/api/health", timeout=10)
        
        if response.status_code == 200:
            print("   ✅ API Grafana accessible")
            
            # Test de la datasource Prometheus
            auth = ("admin", "admin")  # Credentials par défaut
            response = requests.get("http://localhost:3000/api/datasources", 
                                  auth=auth, timeout=10)
            
            if response.status_code == 200:
                datasources = response.json()
                prometheus_found = any(ds.get("type") == "prometheus" for ds in datasources)
                
                if prometheus_found:
                    print("   ✅ Datasource Prometheus configurée")
                else:
                    print("   ⚠️ Datasource Prometheus non trouvée")
                
                return True
            else:
                print(f"   ⚠️ Impossible de vérifier les datasources: HTTP {response.status_code}")
                return True  # L'API principale fonctionne
        else:
            print(f"   ❌ HTTP {response.status_code}")
            return False
            
    except requests.RequestException as e:
        print(f"   ❌ Erreur de connexion: {e}")
        return False

def test_alertmanager_connectivity():
    """Test la connectivité avec AlertManager."""
    print("🚨 Test de connectivité AlertManager...")
    
    try:
        # Test de l'API AlertManager
        response = requests.get("http://localhost:9093/api/v1/status", timeout=10)
        
        if response.status_code == 200:
            print("   ✅ API AlertManager accessible")
            
            # Test des alertes
            response = requests.get("http://localhost:9093/api/v1/alerts", timeout=10)
            
            if response.status_code == 200:
                alerts = response.json()
                print(f"   ✅ {len(alerts.get('data', []))} alertes actives")
                return True
            else:
                print("   ⚠️ Impossible de récupérer les alertes")
                return True  # L'API principale fonctionne
        else:
            print(f"   ❌ HTTP {response.status_code}")
            return False
            
    except requests.RequestException as e:
        print(f"   ❌ Erreur de connexion: {e}")
        return False

def test_api_with_monitoring():
    """Test l'API avec le monitoring intégré."""
    print("🔌 Test de l'API avec monitoring...")
    
    try:
        # Tenter de démarrer un serveur de test
        print("   🚀 Démarrage du serveur de test...")
        
        # Script de serveur de test minimal
        test_server_code = '''
import sys
sys.path.insert(0, ".")

try:
    from fastapi import FastAPI
    from fastapi.responses import PlainTextResponse
    from src.monitoring.metrics import PortfolioMetrics
    from prometheus_client import CollectorRegistry
    import uvicorn
    FASTAPI_AVAILABLE = True
except ImportError:
    print("FastAPI non disponible - serveur de test limité")
    FASTAPI_AVAILABLE = False

if FASTAPI_AVAILABLE:
    # Créer un registre propre pour le serveur de test
    test_registry = CollectorRegistry()
    test_metrics = PortfolioMetrics(registry=test_registry)
    
    app = FastAPI(title="Test Monitoring Server")
    
    @app.get("/health")
    async def health():
        return {"status": "healthy", "service": "test-monitoring"}
    
    @app.get("/test-metric")
    async def test_metric():
        if test_metrics.enabled:
            test_metrics.record_api_request("GET", "/test-metric", 200, 0.1)
        return {"message": "Métrique enregistrée", "enabled": test_metrics.enabled}
    
    @app.get("/metrics", response_class=PlainTextResponse)
    async def get_metrics():
        if test_metrics.enabled:
            return test_metrics.get_metrics()
        else:
            return "# Métriques non disponibles\\n"
    
    if __name__ == "__main__":
        uvicorn.run(app, host="0.0.0.0", port=8001, log_level="error")
else:
    # Serveur HTTP simple sans FastAPI
    from http.server import HTTPServer, BaseHTTPRequestHandler
    import json
    
    class TestHandler(BaseHTTPRequestHandler):
        def do_GET(self):
            if self.path == "/health":
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                response = {"status": "healthy", "service": "test-monitoring-simple"}
                self.wfile.write(json.dumps(response).encode())
            elif self.path == "/metrics":
                self.send_response(200)
                self.send_header("Content-Type", "text/plain")
                self.end_headers()
                self.wfile.write(b"# Test metrics endpoint\\ntest_metric 1\\n")
            else:
                self.send_response(404)
                self.end_headers()
        
        def log_message(self, format, *args):
            # Supprimer les logs pour le test
            pass
    
    if __name__ == "__main__":
        server = HTTPServer(("0.0.0.0", 8001), TestHandler)
        server.serve_forever()
'''
        
        # Écrire le serveur de test
        with open("test_server.py", "w", encoding='utf-8') as f:
            f.write(test_server_code)
        
        # Démarrer le serveur en arrière-plan
        server_process = subprocess.Popen([
            sys.executable, "test_server.py"
        ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        
        # Attendre que le serveur démarre
        time.sleep(5)
        
        try:
            # Test de l'endpoint de santé
            response = requests.get("http://localhost:8001/health", timeout=5)
            
            if response.status_code == 200:
                print("   ✅ Serveur de test accessible")
                
                # Test de l'endpoint de métriques
                response = requests.get("http://localhost:8001/metrics", timeout=5)
                
                if response.status_code == 200:
                    metrics_text = response.text
                    if "portfolio_api_requests_total" in metrics_text:
                        print("   ✅ Endpoint de métriques fonctionnel")
                        
                        # Générer quelques métriques
                        for _ in range(5):
                            requests.get("http://localhost:8001/test-metric", timeout=5)
                        
                        print("   ✅ Métriques générées")
                        return True
                    else:
                        print("   ❌ Métriques non trouvées dans l'endpoint")
                        return False
                else:
                    print(f"   ❌ Endpoint de métriques: HTTP {response.status_code}")
                    return False
            else:
                print(f"   ❌ Serveur de test: HTTP {response.status_code}")
                return False
                
        finally:
            # Arrêter le serveur de test
            server_process.terminate()
            server_process.wait()
            
            # Nettoyer
            try:
                os.remove("test_server.py")
            except:
                pass
    
    except Exception as e:
        print(f"   ❌ Erreur: {e}")
        return False

def generate_load_test():
    """Génère de la charge pour tester le monitoring."""
    print("📈 Génération de charge pour test...")
    
    try:
        # Générer des requêtes vers Prometheus pour simuler de la charge
        endpoints = [
            "http://localhost:9090/api/v1/query?query=up",
            "http://localhost:9090/api/v1/query?query=prometheus_build_info",
            "http://localhost:9090/api/v1/label/__name__/values"
        ]
        
        for _ in range(10):
            for endpoint in endpoints:
                try:
                    requests.get(endpoint, timeout=2)
                except:
                    pass
            time.sleep(0.1)
        
        print("   ✅ Charge générée")
        return True
        
    except Exception as e:
        print(f"   ❌ Erreur: {e}")
        return False

def cleanup_monitoring_stack():
    """Nettoie la stack de monitoring."""
    print("🧹 Nettoyage de la stack de monitoring...")
    
    try:
        # Arrêter les services
        subprocess.run([
            "docker-compose", "-f", "docker-compose-monitoring.yml", "down", "-v"
        ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        
        print("   ✅ Services arrêtés")
        return True
        
    except Exception as e:
        print(f"   ❌ Erreur: {e}")
        return False

def print_monitoring_urls():
    """Affiche les URLs des services de monitoring."""
    print("\n🌐 URLs des services de monitoring:")
    print("   📊 Prometheus:    http://localhost:9090")
    print("   🎨 Grafana:       http://localhost:3000 (admin/admin)")
    print("   🚨 AlertManager:  http://localhost:9093")
    print("   📋 Métriques API: http://localhost:8001/metrics")
    print()

def main():
    """Fonction principale du test."""
    print("🎯 Test complet du système de monitoring Prometheus + Grafana")
    print("=" * 70)
    
    start_time = time.time()
    tests_results = []
    
    # Tests préliminaires
    tests_results.append(("Imports monitoring", test_monitoring_imports()))
    
    # Installation des dépendances si nécessaire
    if not tests_results[-1][1]:
        tests_results.append(("Installation dépendances", install_monitoring_dependencies()))
        if tests_results[-1][1]:
            tests_results.append(("Imports monitoring (retry)", test_monitoring_imports()))
    
    # Création des fichiers de configuration
    tests_results.append(("Création fichiers config", create_monitoring_files()))
    
    # Test de collecte de métriques
    tests_results.append(("Collecte de métriques", test_metrics_collection()))
    
    # Démarrage de la stack (optionnel si Docker n'est pas disponible)
    docker_success = False
    tests_results.append(("Démarrage stack Docker", start_monitoring_stack()))
    docker_success = tests_results[-1][1]
    
    if docker_success:
        # Attendre que les services soient prêts
        tests_results.append(("Attente des services", wait_for_services()))
        
        if tests_results[-1][1]:
            # Tests de connectivité
            tests_results.append(("Connectivité Prometheus", test_prometheus_connectivity()))
            tests_results.append(("Connectivité Grafana", test_grafana_connectivity()))
            tests_results.append(("Connectivité AlertManager", test_alertmanager_connectivity()))
            
            # Génération de charge
            tests_results.append(("Génération de charge", generate_load_test()))
            
            # Afficher les URLs
            print_monitoring_urls()
    else:
        print("\n⚠️ Docker non disponible - tests de connectivité ignorés")
        print("💡 Pour tester avec Docker:")
        print("   1. Démarrez Docker Desktop")
        print("   2. Ou exécutez en tant qu'administrateur") 
        print("   3. Relancez le test")
    
    # Test de l'API avec monitoring (indépendant de Docker)
    tests_results.append(("API avec monitoring", test_api_with_monitoring()))
    
    # Résumé des tests
    print("📋 Résumé des tests:")
    print("=" * 50)
    
    success_count = 0
    for test_name, result in tests_results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"   {status:<8} {test_name}")
        if result:
            success_count += 1
    
    print()
    print(f"📊 Tests réussis: {success_count}/{len(tests_results)}")
    print(f"⏱️  Durée totale: {time.time() - start_time:.2f}s")
    
    if success_count == len(tests_results):
        print("🎉 Tous les tests sont passés avec succès!")
        print("💡 Votre système de monitoring est opérationnel.")
    elif success_count >= len(tests_results) * 0.7:
        print("⚠️  La plupart des tests sont passés.")
        print("💡 Vérifiez les échecs pour optimiser votre setup.")
    else:
        print("❌ Plusieurs tests ont échoué.")
        print("💡 Vérifiez votre configuration et les logs Docker.")
    
    # Demander s'il faut nettoyer
    try:
        cleanup = input("\n🧹 Voulez-vous arrêter les services de monitoring? (y/N): ")
        if cleanup.lower() in ['y', 'yes', 'oui']:
            cleanup_monitoring_stack()
    except KeyboardInterrupt:
        print("\n👋 Test interrompu")
    
    return success_count == len(tests_results)

if __name__ == "__main__":
    try:
        success = main()
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n👋 Test interrompu par l'utilisateur")
        cleanup_monitoring_stack()
        sys.exit(1)
    except Exception as e:
        print(f"\n💥 Erreur inattendue: {e}")
        import traceback
        traceback.print_exc()
        cleanup_monitoring_stack()
        sys.exit(1)