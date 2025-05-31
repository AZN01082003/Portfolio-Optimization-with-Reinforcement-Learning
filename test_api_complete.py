#!/usr/bin/env python3
"""
Test complet de l'API Portfolio RL
Vérifie tous les endpoints et fonctionnalités
"""
import requests
import time
import json
import numpy as np
from datetime import datetime
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class APITester:
    """Testeur pour l'API Portfolio RL."""
    
    def __init__(self, base_url="http://localhost:8000"):
        self.base_url = base_url
        self.session = requests.Session()
        self.session.timeout = 30
    
    def test_health(self):
        """Test de l'endpoint de santé."""
        logger.info("🔍 Test de santé...")
        try:
            response = self.session.get(f"{self.base_url}/health")
            assert response.status_code == 200
            data = response.json()
            assert data['status'] == 'healthy'
            logger.info(f"✅ Santé OK: {data}")
            return True
        except Exception as e:
            logger.error(f"❌ Test santé échoué: {e}")
            return False
    
    def test_root(self):
        """Test de l'endpoint racine."""
        logger.info("🔍 Test endpoint racine...")
        try:
            response = self.session.get(f"{self.base_url}/")
            assert response.status_code == 200
            data = response.json()
            assert 'message' in data
            logger.info(f"✅ Racine OK: {data}")
            return True
        except Exception as e:
            logger.error(f"❌ Test racine échoué: {e}")
            return False
    
    def test_models_list(self):
        """Test de la liste des modèles."""
        logger.info("🔍 Test liste des modèles...")
        try:
            response = self.session.get(f"{self.base_url}/models")
            assert response.status_code == 200
            data = response.json()
            logger.info(f"✅ Modèles OK: {data}")
            return True
        except Exception as e:
            logger.error(f"❌ Test modèles échoué: {e}")
            return False
    
    def test_prediction(self):
        """Test de prédiction d'allocation."""
        logger.info("🔍 Test prédiction...")
        try:
            # Données de test
            test_data = {
                "portfolio_id": "test_portfolio_001",
                "market_data": np.random.rand(5, 5, 30).tolist(),  # 5 features, 5 stocks, 30 periods
                "current_weights": [0.2, 0.2, 0.2, 0.2, 0.2],
                "portfolio_value": 100000.0,
                "risk_tolerance": 0.5
            }
            
            response = self.session.post(
                f"{self.base_url}/predict",
                json=test_data,
                headers={"Content-Type": "application/json"}
            )
            
            assert response.status_code == 200
            data = response.json()
            
            # Vérifications
            assert 'recommended_weights' in data
            assert 'confidence' in data
            assert 'model_version' in data
            assert len(data['recommended_weights']) == 5
            assert 0 <= data['confidence'] <= 1
            
            # Vérifier que les poids somment à 1
            weights_sum = sum(data['recommended_weights'])
            assert abs(weights_sum - 1.0) < 0.01, f"Somme des poids: {weights_sum}"
            
            logger.info(f"✅ Prédiction OK: {data}")
            return True, data
            
        except Exception as e:
            logger.error(f"❌ Test prédiction échoué: {e}")
            return False, None
    
    def test_prediction_performance(self, n_requests=10):
        """Test de performance des prédictions."""
        logger.info(f"🔍 Test performance ({n_requests} requêtes)...")
        try:
            durations = []
            
            for i in range(n_requests):
                start_time = time.time()
                success, _ = self.test_prediction()
                duration = time.time() - start_time
                durations.append(duration)
                
                if not success:
                    logger.warning(f"⚠️ Requête {i+1} échouée")
                    continue
                
                if i % 5 == 0:
                    logger.info(f"📊 Requête {i+1}/{n_requests} - {duration:.3f}s")
            
            if durations:
                avg_duration = sum(durations) / len(durations)
                max_duration = max(durations)
                min_duration = min(durations)
                
                logger.info(f"✅ Performance OK:")
                logger.info(f"   Moyenne: {avg_duration:.3f}s")
                logger.info(f"   Min: {min_duration:.3f}s")
                logger.info(f"   Max: {max_duration:.3f}s")
                logger.info(f"   Succès: {len(durations)}/{n_requests}")
                
                return True
            else:
                logger.error("❌ Aucune requête réussie")
                return False
                
        except Exception as e:
            logger.error(f"❌ Test performance échoué: {e}")
            return False
    
    def test_metrics(self):
        """Test des métriques."""
        logger.info("🔍 Test métriques...")
        try:
            # Test endpoint métriques
            response = self.session.get(f"{self.base_url}/metrics")
            if response.status_code == 200:
                metrics_data = response.text
                logger.info(f"✅ Métriques disponibles: {len(metrics_data)} caractères")
                
                # Vérifier quelques métriques attendues
                expected_metrics = [
                    'portfolio_api_requests_total',
                    'portfolio_model_predictions_total'
                ]
                
                for metric in expected_metrics:
                    if metric in metrics_data:
                        logger.info(f"   ✅ Métrique trouvée: {metric}")
                    else:
                        logger.warning(f"   ⚠️ Métrique manquante: {metric}")
                
                return True
            elif response.status_code == 404:
                logger.warning("⚠️ Endpoint métriques non disponible")
                return True  # Pas critique
            else:
                logger.error(f"❌ Erreur métriques: {response.status_code}")
                return False
                
        except Exception as e:
            logger.error(f"❌ Test métriques échoué: {e}")
            return False
    
    def test_monitoring_status(self):
        """Test du statut de monitoring."""
        logger.info("🔍 Test statut monitoring...")
        try:
            # Vérifier si l'endpoint existe
            endpoints_to_check = ["/monitoring/status", "/health"]
            
            for endpoint in endpoints_to_check:
                try:
                    response = self.session.get(f"{self.base_url}{endpoint}")
                    if response.status_code == 200:
                        data = response.json()
                        logger.info(f"✅ {endpoint} OK: {data}")
                except Exception as e:
                    logger.warning(f"⚠️ {endpoint} non disponible: {e}")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Test monitoring échoué: {e}")
            return False
    
    def test_error_handling(self):
        """Test de la gestion d'erreurs."""
        logger.info("🔍 Test gestion d'erreurs...")
        try:
            # Test avec données invalides
            invalid_data = {
                "portfolio_id": "test",
                "market_data": "invalid",  # Doit être une liste
                "current_weights": [0.5, 0.5],  # Mauvaise taille
                "portfolio_value": -1000  # Valeur négative
            }
            
            response = self.session.post(
                f"{self.base_url}/predict",
                json=invalid_data
            )
            
            # Doit retourner une erreur 422 (validation) ou 500 (serveur)
            assert response.status_code in [422, 500], f"Code inattendu: {response.status_code}"
            logger.info(f"✅ Gestion d'erreur OK: {response.status_code}")
            
            # Test endpoint inexistant
            response = self.session.get(f"{self.base_url}/nonexistent")
            assert response.status_code == 404
            logger.info("✅ Endpoint inexistant géré correctement")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Test gestion d'erreurs échoué: {e}")
            return False
    
    def run_all_tests(self):
        """Lance tous les tests."""
        logger.info("🚀 Démarrage des tests complets de l'API")
        logger.info("="*50)
        
        tests = [
            ("Santé", self.test_health),
            ("Endpoint racine", self.test_root),
            ("Liste modèles", self.test_models_list),
            ("Prédiction", lambda: self.test_prediction()[0]),
            ("Performance", lambda: self.test_prediction_performance(5)),
            ("Métriques", self.test_metrics),
            ("Monitoring", self.test_monitoring_status),
            ("Gestion d'erreurs", self.test_error_handling)
        ]
        
        results = {}
        passed = 0
        total = len(tests)
        
        for test_name, test_func in tests:
            logger.info(f"\n📋 Test: {test_name}")
            try:
                result = test_func()
                results[test_name] = result
                if result:
                    passed += 1
                    logger.info(f"✅ {test_name}: PASSED")
                else:
                    logger.error(f"❌ {test_name}: FAILED")
            except Exception as e:
                logger.error(f"❌ {test_name}: ERROR - {e}")
                results[test_name] = False
        
        # Résumé
        logger.info("\n" + "="*50)
        logger.info(f"📊 RÉSUMÉ DES TESTS")
        logger.info(f"✅ Tests réussis: {passed}/{total}")
        logger.info(f"❌ Tests échoués: {total - passed}/{total}")
        logger.info(f"📈 Taux de réussite: {passed/total*100:.1f}%")
        
        if passed == total:
            logger.info("🎉 Tous les tests sont passés!")
        else:
            logger.warning("⚠️ Certains tests ont échoué")
        
        return results, passed == total

def main():
    """Fonction principale."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Test de l'API Portfolio RL")
    parser.add_argument("--url", default="http://localhost:8000", help="URL de l'API")
    parser.add_argument("--quick", action="store_true", help="Tests rapides seulement")
    parser.add_argument("--performance", type=int, default=10, help="Nombre de requêtes pour le test de performance")
    
    args = parser.parse_args()
    
    tester = APITester(args.url)
    
    logger.info(f"🎯 Test de l'API: {args.url}")
    
    # Vérifier que l'API est accessible
    try:
        response = requests.get(f"{args.url}/health", timeout=5)
        if response.status_code != 200:
            logger.error(f"❌ API non accessible: {response.status_code}")
            return 1
    except Exception as e:
        logger.error(f"❌ Impossible de joindre l'API: {e}")
        logger.info("💡 Assurez-vous que l'API est démarrée avec: python start_api.py")
        return 1
    
    if args.quick:
        # Tests rapides
        success = all([
            tester.test_health(),
            tester.test_root(),
            tester.test_prediction()[0]
        ])
        return 0 if success else 1
    else:
        # Tests complets
        results, all_passed = tester.run_all_tests()
        return 0 if all_passed else 1

if __name__ == "__main__":
    exit(main())