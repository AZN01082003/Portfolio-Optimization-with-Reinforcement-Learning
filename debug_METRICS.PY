#!/usr/bin/env python3
"""
Script de diagnostic pour déboguer le problème des métriques Prometheus.
"""
import requests
import time

def test_metrics_endpoint_detailed():
    """Test détaillé de l'endpoint des métriques."""
    print("🔍 DIAGNOSTIC DÉTAILLÉ DES MÉTRIQUES")
    print("="*50)
    
    base_url = "http://localhost:8000"
    
    # Étape 1: Vérifier que l'API répond
    print("1. Test de connectivité de base...")
    try:
        response = requests.get(f"{base_url}/health", timeout=5)
        print(f"   ✅ API accessible (status: {response.status_code})")
    except Exception as e:
        print(f"   ❌ API non accessible: {e}")
        return False
    
    # Étape 2: Générer quelques requêtes pour créer des métriques
    print("\n2. Génération de requêtes pour créer des métriques...")
    endpoints_to_test = [
        ("/", "GET"),
        ("/health", "GET"),
        ("/models", "GET"),
        ("/test/prediction", "POST")
    ]
    
    for endpoint, method in endpoints_to_test:
        try:
            if method == "GET":
                resp = requests.get(f"{base_url}{endpoint}", timeout=5)
            else:
                resp = requests.post(f"{base_url}{endpoint}", timeout=5)
            print(f"   ✅ {method} {endpoint} -> {resp.status_code}")
        except Exception as e:
            print(f"   ❌ {method} {endpoint} -> {e}")
    
    # Étape 3: Récupérer et analyser les métriques
    print("\n3. Récupération des métriques...")
    try:
        response = requests.get(f"{base_url}/metrics", timeout=10)
        print(f"   📊 Status code: {response.status_code}")
        print(f"   📊 Content-Type: {response.headers.get('content-type')}")
        
        if response.status_code == 200:
            metrics_text = response.text
            print(f"   📊 Taille de la réponse: {len(metrics_text)} caractères")
            
            # Afficher les premiers caractères
            print(f"\n4. Contenu des métriques (premiers 500 caractères):")
            print("-" * 50)
            print(metrics_text[:500])
            print("-" * 50)
            
            # Chercher des métriques spécifiques
            print(f"\n5. Recherche de métriques spécifiques:")
            metrics_to_find = [
                "portfolio_api_requests_total",
                "portfolio_api_request_duration_seconds", 
                "portfolio_model_predictions_total",
                "portfolio_model_prediction_confidence",
                "portfolio_portfolio_value_dollars"
            ]
            
            found_metrics = []
            for metric in metrics_to_find:
                if metric in metrics_text:
                    found_metrics.append(metric)
                    print(f"   ✅ {metric}")
                    
                    # Afficher les lignes contenant cette métrique
                    lines = [line for line in metrics_text.split('\n') if metric in line and not line.startswith('#')]
                    for line in lines[:3]:  # Afficher les 3 premières occurrences
                        print(f"      → {line.strip()}")
                else:
                    print(f"   ❌ {metric}")
            
            # Afficher toutes les métriques trouvées
            print(f"\n6. Toutes les métriques détectées:")
            all_lines = metrics_text.split('\n')
            metric_lines = [line for line in all_lines if line and not line.startswith('#') and '=' in line]
            
            if metric_lines:
                print(f"   📊 {len(metric_lines)} lignes de métriques trouvées:")
                for line in metric_lines[:10]:  # Afficher les 10 premières
                    print(f"      → {line.strip()}")
                if len(metric_lines) > 10:
                    print(f"      ... et {len(metric_lines) - 10} autres")
            else:
                print("   ❌ Aucune ligne de métrique trouvée")
            
            return len(found_metrics) > 0
        else:
            print(f"   ❌ Erreur HTTP: {response.status_code}")
            print(f"   📝 Réponse: {response.text}")
            return False
            
    except Exception as e:
        print(f"   ❌ Erreur: {e}")
        return False

def test_prometheus_import():
    """Test l'import de Prometheus dans l'API."""
    print("\n7. Test de l'import Prometheus dans l'API...")
    
    try:
        # Importer directement ce que l'API utilise
        from prometheus_client import Counter, Histogram, generate_latest, CollectorRegistry
        print("   ✅ prometheus_client importé avec succès")
        
        # Créer un registre de test
        test_registry = CollectorRegistry()
        test_counter = Counter('test_counter', 'Test counter', registry=test_registry)
        test_counter.inc()
        
        # Générer les métriques
        metrics_output = generate_latest(test_registry)
        print(f"   ✅ Génération de métriques test réussie ({len(metrics_output)} bytes)")
        
        if b'test_counter' in metrics_output:
            print("   ✅ Métrique test trouvée dans la sortie")
            return True
        else:
            print("   ❌ Métrique test non trouvée")
            return False
            
    except ImportError as e:
        print(f"   ❌ Import prometheus_client échoué: {e}")
        return False
    except Exception as e:
        print(f"   ❌ Erreur test Prometheus: {e}")
        return False

def main():
    """Fonction principale de diagnostic."""
    print("🚀 DIAGNOSTIC COMPLET DES MÉTRIQUES PROMETHEUS")
    print("="*60)
    
    # Test 1: Import Prometheus
    prometheus_ok = test_prometheus_import()
    
    # Test 2: Métriques API
    metrics_ok = test_metrics_endpoint_detailed()
    
    # Résumé
    print("\n" + "="*60)
    print("📋 RÉSUMÉ DU DIAGNOSTIC")
    print("="*60)
    print(f"🔧 Import Prometheus: {'✅ OK' if prometheus_ok else '❌ FAIL'}")
    print(f"📊 Métriques API: {'✅ OK' if metrics_ok else '❌ FAIL'}")
    
    if prometheus_ok and not metrics_ok:
        print("\n💡 DIAGNOSTIC:")
        print("   - Prometheus fonctionne localement")
        print("   - Problème dans l'intégration API")
        print("   - Vérifiez que le middleware de métriques s'exécute")
        print("   - Vérifiez que les métriques sont bien enregistrées")
        
        print("\n🔧 SOLUTIONS SUGGÉRÉES:")
        print("   1. Redémarrer l'API")
        print("   2. Vérifier les logs de l'API")
        print("   3. Tester manuellement avec curl")
        
    elif not prometheus_ok:
        print("\n💡 DIAGNOSTIC:")
        print("   - Problème avec l'installation de prometheus_client")
        print("   - Réinstallez: pip install prometheus-client")
        
    else:
        print("\n🎉 Métriques fonctionnelles!")
    
    return prometheus_ok and metrics_ok

if __name__ == "__main__":
    success = main()
    if not success:
        print("\n🔗 ÉTAPES DE DÉBOGAGE SUPPLÉMENTAIRES:")
        print("1. Vérifiez les logs de l'API:")
        print("   - Recherchez des erreurs dans la console où l'API tourne")
        print("2. Testez manuellement:")
        print("   curl http://localhost:8000/metrics")
        print("3. Redémarrez l'API et retestez")