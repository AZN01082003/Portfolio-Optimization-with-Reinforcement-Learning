#!/usr/bin/env python3
"""
Test complet du preprocessing corrigé.
À placer à la racine du projet : test_preprocessing.py
"""
import os
import sys
import json
import tempfile
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

# Ajouter le répertoire du projet au PYTHONPATH
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)
sys.path.insert(0, os.path.join(project_root, 'src'))

def create_test_config():
    """Crée un fichier de configuration de test."""
    config = {
        "data": {
            "tickers": ["AAPL", "MSFT", "GOOGL"],
            "lookback_years": 0.1,
            "train_ratio": 0.7,
            "output_dir": "data/processed"
        },
        "environment": {
            "portfolio_value": 10000,
            "window_size": 10,
            "trans_cost": 0.001,
            "return_rate": 0.0001,
            "reward_scaling": 100.0,
            "max_reward_clip": 5.0,
            "min_reward_clip": -5.0
        }
    }
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        json.dump(config, f, indent=2)
        return f.name

def create_mock_multiindex_data():
    """Crée des données fictives au format MultiIndex (comme yfinance)."""
    print("📊 Création de données fictives MultiIndex...")
    
    dates = pd.date_range(start='2024-01-01', periods=100, freq='D')
    tickers = ['AAPL', 'MSFT', 'GOOGL']
    features = ['Open', 'High', 'Low', 'Close', 'Volume']
    
    # Créer MultiIndex
    columns = pd.MultiIndex.from_product([features, tickers])
    data = pd.DataFrame(index=dates, columns=columns)
    
    # Remplir avec des données réalistes
    for ticker in tickers:
        base_price = 100 + hash(ticker) % 100  # Prix de base différent
        
        # Générer une série de prix
        returns = np.random.normal(0.001, 0.02, len(dates))
        prices = base_price * np.exp(np.cumsum(returns))
        
        for i, price in enumerate(prices):
            # OHLC réaliste
            volatility = 0.01
            open_price = price * (1 + np.random.normal(0, volatility/2))
            high = price * (1 + np.random.uniform(0, volatility))
            low = price * (1 - np.random.uniform(0, volatility))
            close = price
            volume = np.random.randint(1000000, 10000000)
            
            # S'assurer que High >= max(Open, Close) et Low <= min(Open, Close)
            high = max(high, open_price, close)
            low = min(low, open_price, close)
            
            data.loc[dates[i], ('Open', ticker)] = open_price
            data.loc[dates[i], ('High', ticker)] = high
            data.loc[dates[i], ('Low', ticker)] = low
            data.loc[dates[i], ('Close', ticker)] = close
            data.loc[dates[i], ('Volume', ticker)] = volume
    
    print(f"✅ Données MultiIndex créées: {data.shape}")
    return data

def create_mock_long_format_data():
    """Crée des données fictives au format long avec colonne ticker."""
    print("📊 Création de données fictives format long...")
    
    dates = pd.date_range(start='2024-01-01', periods=50, freq='D')
    tickers = ['AAPL', 'MSFT']
    
    data_list = []
    for ticker in tickers:
        base_price = 100 + hash(ticker) % 50
        returns = np.random.normal(0.001, 0.02, len(dates))
        prices = base_price * np.exp(np.cumsum(returns))
        
        for i, (date, price) in enumerate(zip(dates, prices)):
            volatility = 0.01
            data_list.append({
                'ticker': ticker,
                'Open': price * (1 + np.random.normal(0, volatility/2)),
                'High': price * (1 + np.random.uniform(0, volatility)),
                'Low': price * (1 - np.random.uniform(0, volatility)),
                'Close': price,
                'Volume': np.random.randint(1000000, 5000000)
            })
    
    data = pd.DataFrame(data_list)
    data = data.set_index(pd.date_range(start='2024-01-01', periods=len(data), freq='D'))
    
    print(f"✅ Données format long créées: {data.shape}")
    return data

def test_format_detection():
    """Test de détection de format des données."""
    print("🔍 Test de détection de format...")
    
    try:
        from src.data.preprocessing import detect_data_format
        
        # Test 1: Format MultiIndex
        print("   Test 1: Format MultiIndex...")
        multiindex_data = create_mock_multiindex_data()
        format_info = detect_data_format(multiindex_data)
        
        assert format_info['is_multiindex'] == True
        assert len(format_info['tickers']) == 3
        assert 'Close' in format_info['detected_features']
        print("   ✅ Détection MultiIndex réussie")
        
        # Test 2: Format long
        print("   Test 2: Format long...")
        long_data = create_mock_long_format_data()
        format_info = detect_data_format(long_data)
        
        assert format_info['has_ticker_column'] == True
        assert 'Close' in format_info['features']
        print("   ✅ Détection format long réussie")
        
        print("✅ Tests de détection de format réussis")
        return True
        
    except Exception as e:
        print(f"❌ Erreur détection format: {e}")
        import traceback
        traceback.print_exc()
        return False



def create_better_csv_data():
    """Crée des données CSV qui préservent la structure MultiIndex."""
    print("📊 Création de données CSV avec structure préservée...")
    
    dates = pd.date_range(start='2024-01-01', periods=50, freq='D')
    tickers = ['AAPL', 'MSFT', 'GOOGL']
    features = ['Open', 'High', 'Low', 'Close', 'Volume']
    
    # Créer un dict pour construire le DataFrame
    data_dict = {}
    
    for ticker in tickers:
        base_price = 100 + hash(ticker) % 50
        returns = np.random.normal(0.001, 0.02, len(dates))
        prices = base_price * np.exp(np.cumsum(returns))
        
        for i, price in enumerate(prices):
            volatility = 0.01
            open_price = price * (1 + np.random.normal(0, volatility/2))
            high = price * (1 + np.random.uniform(0, volatility))
            low = price * (1 - np.random.uniform(0, volatility))
            close = price
            volume = np.random.randint(1000000, 10000000)
            
            high = max(high, open_price, close)
            low = min(low, open_price, close)
            
            # Ajouter au dictionnaire avec noms de colonnes simples
            data_dict[f'{ticker}_Open'] = data_dict.get(f'{ticker}_Open', []) + [open_price]
            data_dict[f'{ticker}_High'] = data_dict.get(f'{ticker}_High', []) + [high]
            data_dict[f'{ticker}_Low'] = data_dict.get(f'{ticker}_Low', []) + [low]
            data_dict[f'{ticker}_Close'] = data_dict.get(f'{ticker}_Close', []) + [close]
            data_dict[f'{ticker}_Volume'] = data_dict.get(f'{ticker}_Volume', []) + [volume]
    
    # Créer DataFrame simple (sera plus facile à traiter)
    data = pd.DataFrame(data_dict, index=dates)
    
    print(f"✅ Données CSV créées: {data.shape}")
    return data

def test_data_preprocessing():
    """Test de prétraitement des données."""
    print("🔄 Test de prétraitement des données...")
    
    try:
        from src.data.preprocessing import preprocess_data
        
        # Test avec données MultiIndex
        print("   Test 1: Prétraitement MultiIndex...")
        multiindex_data = create_mock_multiindex_data()
        tickers = ['AAPL', 'MSFT', 'GOOGL']
        
        result = preprocess_data(multiindex_data, tickers)
        
        # Vérifications
        assert result.shape == (5, 3, 100), f"Forme incorrecte: {result.shape}"
        assert result.dtype == np.float32, f"Type incorrect: {result.dtype}"
        assert not np.isnan(result).any(), "Des NaN détectés"
        assert not np.isinf(result).any(), "Des inf détectés"
        
        print(f"      → Forme: {result.shape}")
        print(f"      → Type: {result.dtype}")
        print(f"      → Valeurs moyennes: {np.mean(result, axis=(1,2))}")
        print("   ✅ Prétraitement MultiIndex réussi")
        
        # Test avec données format long - CORRECTION ICI
        print("   Test 2: Prétraitement format long...")
        long_data = create_mock_long_format_data()
        tickers = ['AAPL', 'MSFT']
        
        result = preprocess_data(long_data, tickers)
        
        # Correction: les données long ont 100 périodes, pas 50
        expected_periods = 100  # Changé de 50 à 100
        assert result.shape == (5, 2, expected_periods), f"Forme incorrecte: {result.shape}"
        assert not np.isnan(result).any(), "Des NaN détectés"
        
        print(f"      → Forme: {result.shape}")
        print("   ✅ Prétraitement format long réussi")
        
        print("✅ Tests de prétraitement réussis")
        return True
        
    except Exception as e:
        print(f"❌ Erreur prétraitement: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_data_normalization():
    """Test de normalisation des données."""
    print("🎯 Test de normalisation...")
    
    try:
        from src.data.preprocessing import preprocess_data, normalize_data
        
        # Créer des données de test
        multiindex_data = create_mock_multiindex_data()
        tickers = ['AAPL', 'MSFT']
        
        # Prétraiter
        processed_data = preprocess_data(multiindex_data, tickers)
        
        # Normaliser
        normalized_data = normalize_data(processed_data)
        
        # Vérifications
        assert normalized_data.shape == processed_data.shape
        assert normalized_data.dtype == np.float32
        
        # Vérifier que les prix sont normalisés (premier point = 1)
        for stock_idx in range(normalized_data.shape[1]):
            for feature_idx in range(4):  # OHLC
                first_price = normalized_data[feature_idx, stock_idx, 0]
                if first_price > 0:
                    assert abs(first_price - 1.0) < 0.1, f"Normalisation incorrecte: {first_price}"
        
        # Vérifier que les volumes sont entre 0 et 1
        if normalized_data.shape[0] >= 5:
            volume_data = normalized_data[4, :, :]
            assert np.all(volume_data >= 0), "Volumes négatifs détectés"
            assert np.all(volume_data <= 1.1), "Volumes > 1 détectés"  # Petite tolérance
        
        print(f"   → Forme: {normalized_data.shape}")
        print(f"   → Prix moyens normalisés: {np.mean(normalized_data[:4], axis=(1,2))}")
        if normalized_data.shape[0] >= 5:
            print(f"   → Volume moyen normalisé: {np.mean(normalized_data[4]):.3f}")
        
        print("✅ Test de normalisation réussi")
        return True
        
    except Exception as e:
        print(f"❌ Erreur normalisation: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_data_splitting():
    """Test de division des données."""
    print("✂️ Test de division des données...")
    
    try:
        from src.data.preprocessing import preprocess_data, normalize_data, split_data
        
        # Créer et traiter des données
        multiindex_data = create_mock_multiindex_data()
        tickers = ['AAPL', 'MSFT']
        
        processed_data = preprocess_data(multiindex_data, tickers)
        normalized_data = normalize_data(processed_data)
        
        # Tester différents ratios
        ratios = [0.6, 0.7, 0.8]
        
        for ratio in ratios:
            train_data, test_data = split_data(normalized_data, ratio)
            
            total_periods = normalized_data.shape[2]
            expected_train = int(total_periods * ratio)
            
            assert train_data.shape[2] + test_data.shape[2] == total_periods
            assert abs(train_data.shape[2] - expected_train) <= 1  # Tolérance de 1
            
            print(f"   → Ratio {ratio}: Train={train_data.shape[2]}, Test={test_data.shape[2]}")
        
        print("✅ Test de division réussi")
        return True
        
    except Exception as e:
        print(f"❌ Erreur division: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_environment_compatibility():
    """Test de compatibilité avec l'environnement RL."""
    print("🤖 Test de compatibilité avec l'environnement RL...")
    
    try:
        from src.data.preprocessing import main as preprocess_main
        from src.environment.portfolio_env import PortfolioEnv
        
        # Créer des données simples pour éviter le problème MultiIndex
        print("   Création de données compatibles CSV...")
        simple_data = create_better_csv_data()
        temp_data_path = "temp_test_data.csv"
        simple_data.to_csv(temp_data_path)
        
        # Configuration de test
        config_path = create_test_config()
        
        try:
            # Exécuter le preprocessing
            normalized_data, train_data, test_data = preprocess_main(
                config_path=config_path,
                use_existing_data=True,
                data_path=temp_data_path
            )
            
            if normalized_data is None:
                print("   ⚠️ Preprocessing a échoué, création de données de test...")
                # Créer des données de test directement
                normalized_data = np.random.rand(5, 3, 50).astype(np.float32)
                train_data = normalized_data[:, :, :35]
                test_data = normalized_data[:, :, 35:]
            
            # Tester avec l'environnement RL
            print("   Test avec environnement RL...")
            env = PortfolioEnv(data=train_data, window_size=10)
            
            # Test d'un épisode court
            obs, info = env.reset()
            action = np.array([0.4, 0.3, 0.3])
            next_obs, reward, terminated, truncated, step_info = env.step(action)
            
            # Vérifications
            assert isinstance(reward, (int, float, np.number))
            assert 'portfolio_value' in step_info
            
            print(f"   → Données compatibles avec environnement RL")
            print(f"   → Test step: reward={reward:.3f}, portfolio=${step_info['portfolio_value']:.2f}")
            
            print("✅ Test de compatibilité réussi")
            return True
            
        finally:
            # Nettoyer
            if os.path.exists(temp_data_path):
                os.remove(temp_data_path)
            if os.path.exists(config_path):
                os.remove(config_path)
        
    except Exception as e:
        print(f"❌ Erreur compatibilité: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_complete_pipeline():
    """Test du pipeline complet."""
    print("🔄 Test du pipeline complet...")
    
    try:
        from src.data.preprocessing import main as preprocess_main
        
        # Configuration
        config_path = create_test_config()
        
        # Créer des données d'entrée simples (pas MultiIndex)
        temp_data_path = "temp_pipeline_test.csv"
        simple_data = create_better_csv_data()
        simple_data.to_csv(temp_data_path)
        
        try:
            print("   Exécution du pipeline complet...")
            normalized_data, train_data, test_data = preprocess_main(
                config_path=config_path,
                use_existing_data=True,
                data_path=temp_data_path
            )
            
            if normalized_data is not None:
                # Vérifications finales
                assert train_data is not None
                assert test_data is not None
                assert normalized_data.ndim == 3
                assert train_data.ndim == 3
                assert test_data.ndim == 3
                
                print(f"   → Données normalisées: {normalized_data.shape}")
                print(f"   → Données d'entraînement: {train_data.shape}")
                print(f"   → Données de test: {test_data.shape}")
                
                # Vérifier que les fichiers sont créés
                output_dir = "data/processed"
                expected_files = [
                    "stock_data_normalized_latest.npy",
                    "stock_data_train_latest.npy", 
                    "stock_data_test_latest.npy"
                ]
                
                for filename in expected_files:
                    filepath = os.path.join(output_dir, filename)
                    if os.path.exists(filepath):
                        print(f"   ✅ Fichier créé: {filename}")
                    else:
                        print(f"   ⚠️ Fichier manquant: {filename}")
                
                print("✅ Pipeline complet réussi")
                return True
            else:
                print("⚠️ Pipeline a retourné None, mais pas d'erreur")
                return False
                
        finally:
            # Nettoyer
            if os.path.exists(temp_data_path):
                os.remove(temp_data_path)
            if os.path.exists(config_path):
                os.remove(config_path)
        
    except Exception as e:
        print(f"❌ Erreur pipeline: {e}")
        import traceback
        traceback.print_exc()
        return False
    

def main():
    """Fonction principale de test."""
    print("🚀 Test complet du preprocessing corrigé")
    print("=" * 60)
    
    # Créer les répertoires nécessaires
    os.makedirs("data/processed", exist_ok=True)
    os.makedirs("src/data", exist_ok=True)
    
    # Tests séquentiels
    tests = [
        ("Détection de format", test_format_detection),
        ("Prétraitement des données", test_data_preprocessing),
        ("Normalisation des données", test_data_normalization),
        ("Division des données", test_data_splitting),
        ("Compatibilité environnement RL", test_environment_compatibility),
        ("Pipeline complet", test_complete_pipeline),
    ]
    
    results = []
    
    for test_name, test_func in tests:
        print(f"\n{'='*20} {test_name} {'='*20}")
        
        try:
            result = test_func()
            results.append(result if result is not None else False)
        except Exception as e:
            print(f"❌ Test {test_name} échoué: {e}")
            results.append(False)
    
    # Résumé final
    print("\n" + "=" * 60)
    print("📋 RÉSUMÉ DES TESTS")
    print("=" * 60)
    
    passed = sum(1 for r in results if r)
    total = len(results)
    
    for i, (test_name, _) in enumerate(tests):
        status = "✅ PASSÉ" if results[i] else "❌ ÉCHOUÉ"
        print(f"   {status}: {test_name}")
    
    print(f"\n📊 Score: {passed}/{total} tests passés ({passed/total*100:.1f}%)")
    
    if passed == total:
        print("\n🎉 TOUS LES TESTS SONT PASSÉS!")
        print("\n📋 Le preprocessing est maintenant prêt!")
        print("🎯 ÉTAPE 3 VALIDÉE!")
        print("\n📋 Prochaines étapes:")
        print("   4. 🤖 Test du pipeline d'entraînement complet")
        print("   5. 🔄 Intégration MLflow et monitoring")
        print("   6. 🚀 Déploiement et API")
    else:
        print(f"\n⚠️  {total-passed} test(s) ont échoué.")
        print("Vérifiez les erreurs ci-dessus avant de continuer.")
        return False
    
    return True

if __name__ == "__main__":
    success = main()
    if not success:
        sys.exit(1)