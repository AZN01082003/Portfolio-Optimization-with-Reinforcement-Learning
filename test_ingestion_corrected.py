#!/usr/bin/env python3
"""
Script de test pour valider l'ingestion de données corrigée.
"""
import os
import sys
import json
import tempfile
from datetime import datetime, timedelta

# Ajouter le répertoire src au PYTHONPATH
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

def create_test_config():
    """Crée un fichier de configuration de test."""
    config = {
        "data": {
            "tickers": ["AAPL", "MSFT"],  # Seulement 2 tickers pour un test rapide
            "lookback_years": 0.1,  # Environ 1 mois de données
            "train_ratio": 0.7,
            "output_dir": "data/processed"
        },
        "environment": {
            "portfolio_value": 10000,
            "window_size": 30,
            "trans_cost": 0.0005,
            "return_rate": 0.0001,
            "reward_scaling": 100.0,
            "max_reward_clip": 5.0,
            "min_reward_clip": -5.0
        }
    }
    
    # Créer un fichier temporaire
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        json.dump(config, f, indent=2)
        return f.name

def test_ingestion():
    """Test principal de l'ingestion."""
    print("🧪 Début des tests d'ingestion...")
    
    # Créer la configuration de test
    config_path = create_test_config()
    print(f"📝 Configuration de test créée: {config_path}")
    
    try:
        # Importer le module corrigé
        from src.data.ingestion import main, load_config, validate_tickers
        
        # Test 1: Validation des tickers
        print("\n1️⃣ Test de validation des tickers...")
        valid_tickers = validate_tickers(["AAPL", "msft", " GOOGL "])
        assert valid_tickers == ["AAPL", "MSFT", "GOOGL"], f"Attendu: ['AAPL', 'MSFT', 'GOOGL'], obtenu: {valid_tickers}"
        print("✅ Validation des tickers réussie")
        
        # Test 2: Chargement de configuration
        print("\n2️⃣ Test de chargement de configuration...")
        config = load_config(config_path)
        assert 'data' in config, "Configuration manquante: 'data'"
        assert 'tickers' in config['data'], "Configuration manquante: 'data.tickers'"
        print("✅ Chargement de configuration réussi")
        
        # Test 3: Ingestion de données
        print("\n3️⃣ Test d'ingestion de données...")
        print("⏳ Téléchargement en cours (peut prendre quelques minutes)...")
        
        # Utiliser des dates récentes pour un test rapide
        end_date = datetime.now()
        start_date = end_date - timedelta(days=30)  # 30 jours de données
        
        data, tickers, loaded_config = main(
            config_path=config_path,
            start_date=start_date,
            end_date=end_date
        )
        
        # Vérifications
        assert data is not None, "Aucune donnée téléchargée"
        assert not data.empty, "DataFrame vide"
        assert len(tickers) > 0, "Aucun ticker traité"
        
        print(f"✅ Ingestion réussie!")
        print(f"   📊 Forme des données: {data.shape}")
        print(f"   🏢 Tickers: {tickers}")
        print(f"   📅 Période: {start_date.date()} - {end_date.date()}")
        
        # Test 4: Vérification des fichiers créés
        print("\n4️⃣ Test de vérification des fichiers...")
        expected_file = f"data/raw/stock_data_{end_date.strftime('%Y%m%d')}.csv"
        expected_metadata = expected_file.replace('.csv', '_metadata.json')
        
        assert os.path.exists(expected_file), f"Fichier de données non créé: {expected_file}"
        assert os.path.exists(expected_metadata), f"Fichier de métadonnées non créé: {expected_metadata}"
        
        print("✅ Fichiers créés avec succès")
        print(f"   📄 Données: {expected_file}")
        print(f"   📋 Métadonnées: {expected_metadata}")
        
        # Afficher les métadonnées
        with open(expected_metadata, 'r') as f:
            metadata = json.load(f)
        print(f"   📊 Métadonnées: {json.dumps(metadata, indent=2)}")
        
        return True
        
    except Exception as e:
        print(f"❌ Erreur durant les tests: {e}")
        import traceback
        traceback.print_exc()
        return False
        
    finally:
        # Nettoyer le fichier de configuration temporaire
        try:
            os.unlink(config_path)
        except:
            pass

if __name__ == "__main__":
    print("🚀 Démarrage des tests d'ingestion corrigée")
    print("=" * 50)
    
    # Créer les répertoires nécessaires
    os.makedirs("data/raw", exist_ok=True)
    os.makedirs("data/processed", exist_ok=True)
    os.makedirs("logs", exist_ok=True)
    
    success = test_ingestion()
    
    print("\n" + "=" * 50)
    if success:
        print("🎉 Tous les tests sont passés avec succès!")
        print("\n📋 Prochaines étapes:")
        print("   1. Intégrer ce module corrigé dans votre projet")
        print("   2. Tester avec vos vrais paramètres de configuration")
        print("   3. Passer à l'étape suivante: correction de l'environnement RL")
    else:
        print("💥 Certains tests ont échoué. Vérifiez les erreurs ci-dessus.")
        sys.exit(1)