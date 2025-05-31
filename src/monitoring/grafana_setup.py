"""
Configuration automatique de Grafana pour le monitoring Portfolio RL.
Crée les dashboards, alertes et datasources automatiquement.
"""
import os
import json
import requests
import time
import logging
from pathlib import Path
from typing import Dict, List, Any

logger = logging.getLogger(__name__)

class GrafanaSetup:
    """Configuration automatique de Grafana."""
    
    def __init__(self, grafana_url: str = "http://localhost:3000", 
                 admin_user: str = "admin", admin_password: str = "admin"):
        """
        Initialise la configuration Grafana.
        
        Args:
            grafana_url: URL de l'instance Grafana
            admin_user: Utilisateur administrateur
            admin_password: Mot de passe administrateur
        """
        self.grafana_url = grafana_url.rstrip('/')
        self.auth = (admin_user, admin_password)
        self.headers = {"Content-Type": "application/json"}
        
        # Créer les répertoires
        self.monitoring_dir = Path("monitoring")
        self.grafana_dir = self.monitoring_dir / "grafana"
        self.dashboards_dir = self.grafana_dir / "dashboards"
        self.provisioning_dir = self.grafana_dir / "provisioning"
        
        self._create_directories()
    
    def _create_directories(self):
        """Crée la structure de répertoires Grafana."""
        dirs = [
            self.monitoring_dir,
            self.grafana_dir,
            self.dashboards_dir,
            self.provisioning_dir,
            self.provisioning_dir / "datasources",
            self.provisioning_dir / "dashboards"
        ]
        
        for directory in dirs:
            directory.mkdir(parents=True, exist_ok=True)
        
        logger.info("📁 Structure Grafana créée")
    
    def wait_for_grafana(self, timeout: int = 60) -> bool:
        """Attend que Grafana soit disponible."""
        logger.info("⏳ Attente de Grafana...")
        
        start_time = time.time()
        while time.time() - start_time < timeout:
            try:
                response = requests.get(f"{self.grafana_url}/api/health", timeout=5)
                if response.status_code == 200:
                    logger.info("✅ Grafana accessible")
                    return True
            except requests.RequestException:
                pass
            
            time.sleep(2)
        
        logger.error("❌ Timeout: Grafana non accessible")
        return False
    
    def setup_datasource(self) -> bool:
        """Configure la datasource Prometheus."""
        logger.info("🔗 Configuration datasource Prometheus...")
        
        datasource_config = {
            "name": "Prometheus",
            "type": "prometheus",
            "url": "http://prometheus:9090",
            "access": "proxy",
            "isDefault": True,
            "jsonData": {
                "timeInterval": "5s",
                "httpMethod": "POST"
            }
        }
        
        try:
            # Vérifier si la datasource existe déjà
            response = requests.get(
                f"{self.grafana_url}/api/datasources/name/Prometheus",
                auth=self.auth
            )
            
            if response.status_code == 200:
                logger.info("✅ Datasource Prometheus existe déjà")
                return True
            
            # Créer la datasource
            response = requests.post(
                f"{self.grafana_url}/api/datasources",
                auth=self.auth,
                headers=self.headers,
                data=json.dumps(datasource_config)
            )
            
            if response.status_code == 200:
                logger.info("✅ Datasource Prometheus créée")
                return True
            else:
                logger.error(f"❌ Erreur création datasource: {response.status_code}")
                return False
                
        except Exception as e:
            logger.error(f"❌ Erreur configuration datasource: {e}")
            return False
    
    def create_portfolio_dashboard(self) -> Dict[str, Any]:
        """Crée le dashboard principal Portfolio RL."""
        
        dashboard = {
            "dashboard": {
                "id": None,
                "title": "Portfolio RL - Dashboard Principal",
                "tags": ["portfolio", "ml", "trading"],
                "timezone": "browser",
                "refresh": "30s",
                "time": {
                    "from": "now-1h",
                    "to": "now"
                },
                "panels": [
                    # Panel 1: API Requests Rate
                    {
                        "id": 1,
                        "title": "Taux de requêtes API",
                        "type": "stat",
                        "gridPos": {"h": 8, "w": 6, "x": 0, "y": 0},
                        "targets": [
                            {
                                "expr": "rate(portfolio_api_requests_total[5m]) * 60",
                                "legendFormat": "Requêtes/min",
                                "refId": "A"
                            }
                        ],
                        "fieldConfig": {
                            "defaults": {
                                "color": {"mode": "thresholds"},
                                "thresholds": {
                                    "steps": [
                                        {"color": "green", "value": None},
                                        {"color": "red", "value": 100}
                                    ]
                                },
                                "unit": "reqps"
                            }
                        }
                    },
                    
                    # Panel 2: Response Time
                    {
                        "id": 2,
                        "title": "Temps de réponse API",
                        "type": "stat",
                        "gridPos": {"h": 8, "w": 6, "x": 6, "y": 0},
                        "targets": [
                            {
                                "expr": "histogram_quantile(0.95, rate(portfolio_api_request_duration_seconds_bucket[5m]))",
                                "legendFormat": "P95",
                                "refId": "A"
                            }
                        ],
                        "fieldConfig": {
                            "defaults": {
                                "color": {"mode": "thresholds"},
                                "thresholds": {
                                    "steps": [
                                        {"color": "green", "value": None},
                                        {"color": "yellow", "value": 0.1},
                                        {"color": "red", "value": 0.5}
                                    ]
                                },
                                "unit": "s"
                            }
                        }
                    },
                    
                    # Panel 3: Model Predictions
                    {
                        "id": 3,
                        "title": "Prédictions par modèle",
                        "type": "stat",
                        "gridPos": {"h": 8, "w": 6, "x": 12, "y": 0},
                        "targets": [
                            {
                                "expr": "rate(portfolio_model_predictions_total[5m]) * 60",
                                "legendFormat": "{{model_name}}",
                                "refId": "A"
                            }
                        ],
                        "fieldConfig": {
                            "defaults": {
                                "color": {"mode": "palette-classic"},
                                "unit": "ops"
                            }
                        }
                    },
                    
                    # Panel 4: Model Confidence
                    {
                        "id": 4,
                        "title": "Confiance moyenne des modèles",
                        "type": "gauge",
                        "gridPos": {"h": 8, "w": 6, "x": 18, "y": 0},
                        "targets": [
                            {
                                "expr": "avg(portfolio_model_prediction_confidence)",
                                "legendFormat": "Confiance",
                                "refId": "A"
                            }
                        ],
                        "fieldConfig": {
                            "defaults": {
                                "color": {"mode": "thresholds"},
                                "thresholds": {
                                    "steps": [
                                        {"color": "red", "value": None},
                                        {"color": "yellow", "value": 0.6},
                                        {"color": "green", "value": 0.8}
                                    ]
                                },
                                "min": 0,
                                "max": 1,
                                "unit": "percentunit"
                            }
                        }
                    },
                    
                    # Panel 5: API Requests Timeline
                    {
                        "id": 5,
                        "title": "Timeline des requêtes API",
                        "type": "graph",
                        "gridPos": {"h": 8, "w": 12, "x": 0, "y": 8},
                        "targets": [
                            {
                                "expr": "rate(portfolio_api_requests_total[1m])",
                                "legendFormat": "{{method}} {{endpoint}}",
                                "refId": "A"
                            }
                        ],
                        "yAxes": [
                            {"label": "Requêtes/sec", "min": 0},
                            {"show": False}
                        ],
                        "legend": {"show": True, "alignAsTable": True, "rightSide": True}
                    },
                    
                    # Panel 6: Error Rate
                    {
                        "id": 6,
                        "title": "Taux d'erreurs",
                        "type": "graph", 
                        "gridPos": {"h": 8, "w": 12, "x": 12, "y": 8},
                        "targets": [
                            {
                                "expr": "rate(portfolio_api_requests_total{status_code=~\"4..|5..\"}[5m])",
                                "legendFormat": "Erreurs {{status_code}}",
                                "refId": "A"
                            }
                        ],
                        "yAxes": [
                            {"label": "Erreurs/sec", "min": 0},
                            {"show": False}
                        ],
                        "alert": {
                            "conditions": [
                                {
                                    "evaluator": {"params": [0.1], "type": "gt"},
                                    "operator": {"type": "and"},
                                    "query": {"params": ["A", "5m", "now"]},
                                    "reducer": {"params": [], "type": "avg"},
                                    "type": "query"
                                }
                            ],
                            "executionErrorState": "alerting",
                            "for": "5m",
                            "frequency": "10s",
                            "handler": 1,
                            "name": "Taux d'erreur élevé",
                            "noDataState": "no_data",
                            "notifications": []
                        }
                    },
                    
                    # Panel 7: Portfolio Values
                    {
                        "id": 7,
                        "title": "Valeurs des portefeuilles",
                        "type": "graph",
                        "gridPos": {"h": 8, "w": 24, "x": 0, "y": 16},
                        "targets": [
                            {
                                "expr": "portfolio_portfolio_value_dollars",
                                "legendFormat": "{{portfolio_id}}",
                                "refId": "A"
                            }
                        ],
                        "yAxes": [
                            {"label": "Valeur ($)", "min": 0},
                            {"show": False}
                        ],
                        "legend": {"show": True, "alignAsTable": True, "rightSide": False}
                    }
                ]
            },
            "overwrite": True
        }
        
        return dashboard
    
    def create_models_dashboard(self) -> Dict[str, Any]:
        """Crée le dashboard de monitoring des modèles."""
        
        dashboard = {
            "dashboard": {
                "id": None,
                "title": "Portfolio RL - Monitoring des Modèles",
                "tags": ["portfolio", "models", "ml"],
                "timezone": "browser",
                "refresh": "1m",
                "time": {
                    "from": "now-6h",
                    "to": "now"
                },
                "panels": [
                    # Prédictions par modèle
                    {
                        "id": 1,
                        "title": "Prédictions par modèle",
                        "type": "graph",
                        "gridPos": {"h": 8, "w": 12, "x": 0, "y": 0},
                        "targets": [
                            {
                                "expr": "rate(portfolio_model_predictions_total[5m])",
                                "legendFormat": "{{model_name}} ({{model_stage}})",
                                "refId": "A"
                            }
                        ]
                    },
                    
                    # Distribution de confiance
                    {
                        "id": 2,
                        "title": "Distribution de confiance",
                        "type": "heatmap",
                        "gridPos": {"h": 8, "w": 12, "x": 12, "y": 0},
                        "targets": [
                            {
                                "expr": "rate(portfolio_model_prediction_confidence_bucket[5m])",
                                "legendFormat": "{{le}}",
                                "refId": "A"
                            }
                        ]
                    },
                    
                    # Erreurs des modèles
                    {
                        "id": 3,
                        "title": "Erreurs des modèles",
                        "type": "graph",
                        "gridPos": {"h": 8, "w": 24, "x": 0, "y": 8},
                        "targets": [
                            {
                                "expr": "rate(portfolio_model_errors_total[5m])",
                                "legendFormat": "{{model_name}} - {{error_type}}",
                                "refId": "A"
                            }
                        ]
                    }
                ]
            },
            "overwrite": True
        }
        
        return dashboard
    
    def upload_dashboard(self, dashboard: Dict[str, Any]) -> bool:
        """Upload un dashboard vers Grafana."""
        try:
            response = requests.post(
                f"{self.grafana_url}/api/dashboards/db",
                auth=self.auth,
                headers=self.headers,
                data=json.dumps(dashboard)
            )
            
            if response.status_code == 200:
                result = response.json()
                title = dashboard["dashboard"]["title"]
                logger.info(f"✅ Dashboard '{title}' créé/mis à jour")
                return True
            else:
                logger.error(f"❌ Erreur upload dashboard: {response.status_code}")
                logger.error(response.text)
                return False
                
        except Exception as e:
            logger.error(f"❌ Erreur upload dashboard: {e}")
            return False
    
    def create_provisioning_files(self):
        """Crée les fichiers de provisioning pour Grafana."""
        
        # Datasource provisioning
        datasource_config = {
            "apiVersion": 1,
            "datasources": [
                {
                    "name": "Prometheus",
                    "type": "prometheus",
                    "access": "proxy",
                    "url": "http://prometheus:9090",
                    "isDefault": True,
                    "editable": True
                }
            ]
        }
        
        datasource_file = self.provisioning_dir / "datasources" / "prometheus.yml"
        with open(datasource_file, 'w') as f:
            json.dump(datasource_config, f, indent=2)
        
        # Dashboard provisioning
        dashboard_config = {
            "apiVersion": 1,
            "providers": [
                {
                    "name": "default",
                    "orgId": 1,
                    "folder": "",
                    "type": "file",
                    "disableDeletion": False,
                    "updateIntervalSeconds": 10,
                    "allowUiUpdates": True,
                    "options": {
                        "path": "/var/lib/grafana/dashboards"
                    }
                }
            ]
        }
        
        dashboard_file = self.provisioning_dir / "dashboards" / "dashboards.yml"
        with open(dashboard_file, 'w') as f:
            json.dump(dashboard_config, f, indent=2)
        
        logger.info("✅ Fichiers de provisioning créés")
    
    def save_dashboards_to_files(self):
        """Sauvegarde les dashboards dans des fichiers JSON."""
        
        # Dashboard principal
        main_dashboard = self.create_portfolio_dashboard()
        main_file = self.dashboards_dir / "portfolio_main_dashboard.json"
        with open(main_file, 'w') as f:
            json.dump(main_dashboard["dashboard"], f, indent=2)
        
        # Dashboard modèles
        models_dashboard = self.create_models_dashboard()
        models_file = self.dashboards_dir / "portfolio_models_dashboard.json"
        with open(models_file, 'w') as f:
            json.dump(models_dashboard["dashboard"], f, indent=2)
        
        logger.info("✅ Dashboards sauvegardés dans les fichiers")
    
    def setup_alerts(self) -> bool:
        """Configure les alertes Grafana."""
        logger.info("🚨 Configuration des alertes...")
        
        # Configuration basique des notifications (optionnel)
        notification_channel = {
            "name": "portfolio-alerts",
            "type": "slack",  # ou "email", "webhook", etc.
            "settings": {
                "url": "",  # À configurer selon vos besoins
                "channel": "#alerts",
                "title": "Portfolio RL Alert",
                "text": "Alert: {{range .Alerts}}{{.Annotations.summary}}{{end}}"
            }
        }
        
        try:
            # Vérifier si le channel existe
            response = requests.get(
                f"{self.grafana_url}/api/alert-notifications",
                auth=self.auth
            )
            
            if response.status_code == 200:
                existing_channels = response.json()
                exists = any(ch["name"] == "portfolio-alerts" for ch in existing_channels)
                
                if not exists:
                    # Créer le channel
                    response = requests.post(
                        f"{self.grafana_url}/api/alert-notifications",
                        auth=self.auth,
                        headers=self.headers,
                        data=json.dumps(notification_channel)
                    )
                    
                    if response.status_code == 200:
                        logger.info("✅ Canal de notification créé")
                    else:
                        logger.warning("⚠️ Impossible de créer le canal de notification")
                
                return True
                
        except Exception as e:
            logger.warning(f"⚠️ Erreur configuration alertes: {e}")
            return False
        
        return True
    
    def setup_complete_monitoring(self) -> Dict[str, bool]:
        """Configuration complète du monitoring Grafana."""
        logger.info("🚀 Setup complet Grafana...")
        
        results = {}
        
        # 1. Attendre Grafana
        results["grafana_ready"] = self.wait_for_grafana()
        if not results["grafana_ready"]:
            return results
        
        # 2. Créer les fichiers de provisioning
        try:
            self.create_provisioning_files()
            results["provisioning_files"] = True
        except Exception as e:
            logger.error(f"❌ Erreur provisioning: {e}")
            results["provisioning_files"] = False
        
        # 3. Configurer la datasource
        results["datasource"] = self.setup_datasource()
        
        # 4. Sauvegarder les dashboards
        try:
            self.save_dashboards_to_files()
            results["dashboard_files"] = True
        except Exception as e:
            logger.error(f"❌ Erreur sauvegarde dashboards: {e}")
            results["dashboard_files"] = False
        
        # 5. Upload des dashboards
        try:
            main_dashboard = self.create_portfolio_dashboard()
            models_dashboard = self.create_models_dashboard()
            
            results["main_dashboard"] = self.upload_dashboard(main_dashboard)
            results["models_dashboard"] = self.upload_dashboard(models_dashboard)
        except Exception as e:
            logger.error(f"❌ Erreur upload dashboards: {e}")
            results["main_dashboard"] = False
            results["models_dashboard"] = False
        
        # 6. Configuration des alertes
        results["alerts"] = self.setup_alerts()
        
        # Résumé
        success_count = sum(results.values())
        total_count = len(results)
        
        logger.info(f"📊 Setup Grafana terminé: {success_count}/{total_count} réussi")
        
        if success_count == total_count:
            logger.info("🎉 Grafana entièrement configuré!")
        elif success_count >= total_count * 0.8:
            logger.info("✅ Grafana configuré avec succès (quelques échecs mineurs)")
        else:
            logger.warning("⚠️ Configuration Grafana partielle")
        
        return results

def setup_grafana_monitoring(grafana_url: str = "http://localhost:3000") -> Dict[str, bool]:
    """
    Point d'entrée principal pour configurer Grafana.
    
    Args:
        grafana_url: URL de l'instance Grafana
        
    Returns:
        Dictionnaire avec le statut de chaque étape
    """
    setup = GrafanaSetup(grafana_url)
    return setup.setup_complete_monitoring()

if __name__ == "__main__":
    # Configuration standalone
    logging.basicConfig(level=logging.INFO)
    
    results = setup_grafana_monitoring()
    
    print("\n🎯 RÉSULTATS SETUP GRAFANA:")
    for step, success in results.items():
        status = "✅" if success else "❌"
        print(f"   {status} {step}")
    
    if all(results.values()):
        print("\n🎉 Grafana entièrement configuré!")
        print("🌐 Accès: http://localhost:3000 (admin/admin)")
    else:
        print("\n⚠️ Configuration partielle")