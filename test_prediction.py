#!/usr/bin/env python3
"""
Test manuel de l'endpoint de prédiction
"""
import requests
import json
import numpy as np

def test_prediction_endpoint():
    """Test détaillé de l'endpoint de prédiction."""
    base_url = "http://localhost:8000"
    
    print("🧪 Test détaillé de l'endpoint de prédiction")
    print("="*50)
    
    # Données de test simples
    test_data = {
        "portfolio_id": "test_portfolio_001",
        "market_data": [
            [  # Feature 1 (ex: Close prices)
                [100, 101, 102, 103, 104] * 6,  # Stock 1 - 30 valeurs
                [200, 201, 202, 203, 204] * 6,  # Stock 2 - 30 valeurs
                [150, 151, 152, 153, 154] * 6,  # Stock 3 - 30 valeurs
                [300, 301, 302, 303, 304] * 6,  # Stock 4 - 30 valeurs
                [250, 251, 252, 253, 254] * 6   # Stock 5 - 30 valeurs
            ],
            [  # Feature 2 (ex: Volume)
                [1000, 1100, 1200, 1300, 1400] * 6,  # Stock 1
                [2000, 2100, 2200, 2300, 2400] * 6,  # Stock 2
                [1500, 1600, 1700, 1800, 1900] * 6,  # Stock 3
                [3000, 3100, 3200, 3300, 3400] * 6,  # Stock 4
                [2500, 2600, 2700, 2800, 2900] * 6   # Stock 5
            ],
            [  # Feature 3 (ex: High prices)
                [105, 106, 107, 108, 109] * 6,  # Stock 1
                [205, 206, 207, 208, 209] * 6,  # Stock 2
                [155, 156, 157, 158, 159] * 6,  # Stock 3
                [305, 306, 307, 308, 309] * 6,  # Stock 4
                [255, 256, 257, 258, 259] * 6   # Stock 5
            ],
            [  # Feature 4 (ex: Low prices)
                [95, 96, 97, 98, 99] * 6,     # Stock 1
                [195, 196, 197, 198, 199] * 6,  # Stock 2
                [145, 146, 147, 148, 149] * 6,  # Stock 3
                [295, 296, 297, 298, 299] * 6,  # Stock 4
                [245, 246, 247, 248, 249] * 6   # Stock 5
            ],
            [  # Feature 5 (ex: Open prices)
                [98, 99, 100, 101, 102] * 6,    # Stock 1
                [198, 199, 200, 201, 202] * 6,  # Stock 2
                [148, 149, 150, 151, 152] * 6,  # Stock 3
                [298, 299, 300, 301, 302] * 6,  # Stock 4
                [248, 249, 250, 251, 252] * 6   # Stock 5
            ]
        ],
        "current_weights": [0.2, 0.2, 0.2, 0.2, 0.2],
        "portfolio_value": 100000.0,
        "risk_tolerance": 0.5
    }
    
    print(f"📊 Format des données:")
    print(f"   - Portfolio ID: {test_data['portfolio_id']}")
    print(f"   - Market data shape: {len(test_data['market_data'])} features x {len(test_data['market_data'][0])} stocks x {len(test_data['market_data'][0][0])} periods")
    print(f"   - Current weights: {test_data['current_weights']}")
    print(f"   - Portfolio value: ${test_data['portfolio_value']:,.2f}")
    
    try:
        print("\n🚀 Envoi de la requête...")
        response = requests.post(
            f"{base_url}/predict",
            json=test_data,
            headers={"Content-Type": "application/json"},
            timeout=30
        )
        
        print(f"📡 Status code: {response.status_code}")
        
        if response.status_code == 200:
            data = response.json()
            print("✅ Prédiction réussie !")
            print(f"\n📈 Résultats:")
            print(f"   - Portfolio ID: {data.get('portfolio_id', 'N/A')}")
            print(f"   - Poids recommandés: {data.get('recommended_weights', [])}")
            print(f"   - Confiance: {data.get('confidence', 0):.2%}")
            print(f"   - Modèle utilisé: {data.get('model_version', 'N/A')}")
            print(f"   - Rééquilibrage nécessaire: {data.get('rebalancing_needed', False)}")
            print(f"   - Timestamp: {data.get('timestamp', 'N/A')}")
            
            # Vérifications
            weights = data.get('recommended_weights', [])
            if weights:
                weights_sum = sum(weights)
                print(f"\n🔍 Vérifications:")
                print(f"   - Somme des poids: {weights_sum:.6f}")
                print(f"   - Nombre d'actifs: {len(weights)}")
                print(f"   - Poids min: {min(weights):.4f}")
                print(f"   - Poids max: {max(weights):.4f}")
                
                if abs(weights_sum - 1.0) < 0.001:
                    print("   ✅ Poids correctement normalisés")
                else:
                    print("   ⚠️ Poids non normalisés")
            
            return True
            
        elif response.status_code == 422:
            print("❌ Erreur de validation des données (422)")
            try:
                error_detail = response.json()
                print(f"   Détail: {json.dumps(error_detail, indent=2)}")
            except:
                print(f"   Réponse brute: {response.text}")
                
        elif response.status_code == 500:
            print("❌ Erreur serveur interne (500)")
            try:
                error_detail = response.json()
                print(f"   Détail: {error_detail}")
            except:
                print(f"   Réponse brute: {response.text}")
        else:
            print(f"❌ Erreur inattendue: {response.status_code}")
            print(f"   Réponse: {response.text}")
            
        return False
        
    except requests.exceptions.Timeout:
        print("❌ Timeout - La requête a pris trop de temps")
        return False
    except requests.exceptions.ConnectionError:
        print("❌ Erreur de connexion - L'API n'est pas accessible")
        return False
    except Exception as e:
        print(f"❌ Erreur inattendue: {e}")
        return False

def test_simple_endpoint():
    """Test l'endpoint de test simple."""
    base_url = "http://localhost:8000"
    
    print("\n🧪 Test de l'endpoint de test simple")
    print("-" * 40)
    
    try:
        response = requests.post(f"{base_url}/test/prediction", timeout=10)
        
        if response.status_code == 200:
            data = response.json()
            print("✅ Endpoint de test OK")
            print(f"   Résultat: {json.dumps(data, indent=2)}")
            return True
        else:
            print(f"❌ Erreur endpoint de test: {response.status_code}")
            print(f"   Réponse: {response.text}")
            return False
            
    except Exception as e:
        print(f"❌ Erreur: {e}")
        return False

def main():
    """Fonction principale."""
    print("🎯 Test de l'API Portfolio RL - Endpoint Prédiction")
    print("=" * 60)
    
    # Test 1: Endpoint simple
    success1 = test_simple_endpoint()
    
    # Test 2: Endpoint complet
    success2 = test_prediction_endpoint()
    
    print("\n" + "=" * 60)
    print("📊 Résumé des tests:")
    print(f"   Test simple: {'✅ OK' if success1 else '❌ FAIL'}")
    print(f"   Test complet: {'✅ OK' if success2 else '❌ FAIL'}")
    
    if success1 and success2:
        print("\n🎉 Tous les tests de prédiction sont OK !")
    elif success1:
        print("\n⚠️ L'endpoint simple fonctionne, mais il y a un problème avec les données complexes")
    else:
        print("\n❌ Problèmes détectés dans les endpoints de prédiction")

if __name__ == "__main__":
    main()