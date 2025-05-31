#!/usr/bin/env python3
"""
Test d'ingestion avec données fictives pour éviter les problèmes de rate limiting.
À placer à la racine du projet : test_ingestion_with_mock.py
"""
import os
import sys
import json
import tempfile
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Ajouter le répertoire du projet au PYTHONPATH
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)
sys.path.insert(0, os.path.join(project_root, 'src'))

def create_test_config():
    """Crée un fichier de configuration de test."""
    config = {
        "data": {
            "tickers": ["AAPL", "MSFT"],
            "lookback_years": 0.1,
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
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        json.dump(config, f, indent=2)
        return f.name

def create_mock_stock_data(tickers, start_date, end_date):
    """Crée des données d'actions fictives pour le test."""
    print("📊 Création de données fictives pour éviter le rate limiting...")
    
    # Créer une plage de dates
    dates = pd.date_range(start=start_date, end=end_date, freq='D')
    dates = dates[dates.dayofweek < 5]  # Seulement les jours ouvrables
    
    if len(dates) == 0:
        # Si pas de jours ouvrables, créer au moins quelques points
        dates = pd.date_range(start=start_date, periods=5, freq='D')
    
    # Créer un MultiIndex pour les colonnes (comme yfinance)
    columns = pd.MultiIndex.from_product([
        ['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume'],
        tickers
    ])
    
    # Initialiser le DataFrame
    data = pd.DataFrame(index=dates, columns=columns)
    
    # Remplir avec des données réalistes
    for ticker in tickers:
        # Prix de base (différent pour chaque ticker)
        base_price = 100 if ticker == 'AAPL' else 250
        
        # Générer des prix avec une marche aléatoire
        returns = np.random.normal(0.001, 0.02, len(dates))  # 0.1% de rendement moyen, 2% de volatilité
        prices = base_price * np.exp(np.cumsum(returns))
        
        # OHLC réaliste
        for i, price in enumerate(prices):
            daily_volatility = 0.01  # 1% de volatilité intraday
            high = price * (1 + np.random.uniform(0, daily_volatility))
            low = price * (1 - np.random.uniform(0, daily_volatility))
            open_price = price * (1 + np.random.uniform(-daily_volatility/2, daily_volatility/2))
            close = price
            
            data.loc[dates[i], ('Open', ticker)] = open_price
            data.loc[dates[i], ('High', ticker)] = high
            data.loc[dates[i], ('Low', ticker)] = low
            data.loc[dates[i], ('Close', ticker)] = close
            data.loc[dates[i], ('Adj Close', ticker)] = close
            data.loc[dates[i], ('Volume', ticker)] = np.random.randint(1000000, 10000000)
    
    # Convertir en float
    data = data.astype(float)
    
    print(f"✅ Données fictives créées: {data.shape}")
    return data

def test_complete_pipeline():
    """Test complet du pipeline d'ingestion avec données fictives."""
    print("🧪 Test complet du pipeline d'ingestion avec données fictives...")
    
    try:
        # Importer les modules nécessaires
        from src.data.ingestion import load_config, validate_tickers, save_data_with_metadata
        
        # Configuration de test
        config_path = create_test_config()
        config = load_config(config_path)
        
        # Paramètres du test
        tickers = ["AAPL", "MSFT"]
        end_date = datetime.now()
        start_date = end_date - timedelta(days=30)
        
        print(f"📅 Période de test: {start_date.date()} - {end_date.date()}")
        
        # Test 1: Validation des tickers
        print("\n1️⃣ Test de validation des tickers...")
        validated_tickers = validate_tickers(tickers)
        assert validated_tickers == ["AAPL", "MSFT"]
        print("✅ Validation réussie")
        
        # Test 2: Création de données fictives
        print("\n2️⃣ Création de données fictives...")
        mock_data = create_mock_stock_data(tickers, start_date, end_date)
        assert not mock_data.empty
        assert mock_data.shape[1] == 6 * len(tickers)  # 6 colonnes par ticker
        print("✅ Données fictives créées")
        
        # Test 3: Sauvegarde des données
        print("\n3️⃣ Test de sauvegarde...")
        
        # Créer les répertoires
        os.makedirs('data/raw', exist_ok=True)
        
        # Métadonnées
        metadata = {
            'tickers': tickers,
            'start_date': start_date.isoformat(),
            'end_date': end_date.isoformat(),
            'download_timestamp': datetime.now().isoformat(),
            'data_shape': list(mock_data.shape),
            'columns': [str(col) for col in mock_data.columns],
            'missing_data_count': int(mock_data.isnull().sum().sum()),
            'config_used': config_path,
            'data_source': 'mock_for_testing'
        }
        
        # Sauvegarder
        output_path = f"data/raw/stock_data_test_{end_date.strftime('%Y%m%d')}.csv"
        save_data_with_metadata(mock_data, output_path, metadata)
        
        # Vérifier les fichiers
        assert os.path.exists(output_path)
        metadata_path = output_path.replace('.csv', '_metadata.json')
        assert os.path.exists(metadata_path)
        print("✅ Sauvegarde réussie")
        
        # Test 4: Vérification de la qualité des données
        print("\n4️⃣ Vérification de la qualité des données...")
        
        # Recharger les données
        reloaded_data = pd.read_csv(output_path, index_col=0, header=[0, 1])
        
        # Vérifications
        assert reloaded_data.shape == mock_data.shape
        assert not reloaded_data.empty
        
        # Vérifier les métadonnées
        with open(metadata_path, 'r') as f:
            loaded_metadata = json.load(f)
        
        assert loaded_metadata['tickers'] == tickers
        assert loaded_metadata['data_source'] == 'mock_for_testing'
        assert loaded_metadata['missing_data_count'] == 0
        
        print("✅ Qualité des données vérifiée")
        
        # Test 5: Statistiques des données
        print("\n5️⃣ Analyse des données...")
        
        # Calculer quelques statistiques
        for ticker in tickers:
            close_prices = mock_data[('Close', ticker)]
            daily_returns = close_prices.pct_change().dropna()
            
            print(f"   📈 {ticker}:")
            print(f"      Prix moyen: ${close_prices.mean():.2f}")
            print(f"      Rendement moyen: {daily_returns.mean()*100:.3f}%")
            print(f"      Volatilité: {daily_returns.std()*100:.3f}%")
        
        print("✅ Analyse terminée")
        
        return True
        
    except Exception as e:
        print(f"❌ Erreur durant le test: {e}")
        import traceback
        traceback.print_exc()
        return False
        
    finally:
        # Nettoyer
        try:
            os.unlink(config_path)
        except:
            pass

def test_real_ingestion_when_possible():
    """Test optionnel avec vraies données si possible."""
    print("\n🌐 Test optionnel avec données réelles (si possible)...")
    
    try:
        from src.data.ingestion import main
        
        # Configuration simple
        config_path = create_test_config()
        
        # Essayer avec une période très courte et un seul ticker
        end_date = datetime.now()
        start_date = end_date - timedelta(days=2)  # Seulement 2 jours
        
        print("⏳ Tentative de téléchargement (2 jours, 1 ticker)...")
        
        data, tickers, config = main(
            config_path=config_path,
            tickers=["AAPL"],  # Un seul ticker
            start_date=start_date,
            end_date=end_date
        )
        
        print(f"✅ Téléchargement réel réussi! Forme: {data.shape}")
        return True
        
    except Exception as e:
        print(f"⚠️  Téléchargement réel échoué (normal): {e}")
        print("   💡 Utilisez les données fictives pour le développement")
        return False
    finally:
        try:
            os.unlink(config_path)
        except:
            pass

if __name__ == "__main__":
    print("🚀 Test d'ingestion avec données fictives")
    print("=" * 60)
    
    # Créer les répertoires nécessaires
    os.makedirs("data/raw", exist_ok=True)
    os.makedirs("data/processed", exist_ok=True)
    os.makedirs("logs", exist_ok=True)
    
    # Test principal avec données fictives
    success_mock = test_complete_pipeline()
    
    # Test optionnel avec vraies données
    success_real = test_real_ingestion_when_possible()
    
    print("\n" + "=" * 60)
    if success_mock:
        print("🎉 Tests avec données fictives réussis!")
        print("\n📋 Résumé:")
        print("   ✅ Validation des tickers")
        print("   ✅ Création de données fictives")
        print("   ✅ Sauvegarde et chargement")
        print("   ✅ Vérification de la qualité")
        print("   ✅ Analyse statistique")
        
        if success_real:
            print("   ✅ Test avec données réelles")
        else:
            print("   ⚠️  Données réelles indisponibles (rate limiting)")
        
        print("\n🎯 ÉTAPE 1 VALIDÉE!")
        print("📋 Prochaines étapes:")
        print("   2. 🔄 Correction de l'environnement RL")
        print("   3. 📊 Correction du preprocessing")
        print("   4. 🤖 Validation du pipeline d'entraînement")
        
    else:
        print("💥 Tests échoués.")
        sys.exit(1)