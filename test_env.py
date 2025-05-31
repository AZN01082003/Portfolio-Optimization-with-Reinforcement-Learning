#!/usr/bin/env python3
"""
Test complet de l'environnement RL corrigé - VERSION FIXÉE.
À placer à la racine du projet : test_portfolio_env_fixed.py
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
            "min_reward_clip": -5.0,
            "normalize_observations": True,
            "risk_penalty": 0.1
        }
    }
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        json.dump(config, f, indent=2)
        return f.name

def create_test_data(n_features=5, n_stocks=3, n_time_periods=50):
    """Crée des données de test réalistes."""
    print(f"📊 Création de données de test: {n_features} features, {n_stocks} stocks, {n_time_periods} périodes")
    
    data = np.zeros((n_features, n_stocks, n_time_periods), dtype=np.float32)
    
    for stock_idx in range(n_stocks):
        base_price = 100 + stock_idx * 50
        returns = np.random.normal(0.0005, 0.02, n_time_periods)
        prices = base_price * np.exp(np.cumsum(returns))
        
        for t in range(n_time_periods):
            close = prices[t]
            daily_vol = 0.01
            
            if t == 0:
                open_price = close * (1 + np.random.normal(0, daily_vol/2))
            else:
                open_price = prices[t-1] * (1 + np.random.normal(0, daily_vol/2))
            
            high = close * (1 + np.random.uniform(0, daily_vol))
            low = close * (1 - np.random.uniform(0, daily_vol))
            
            high = max(high, open_price, close)
            low = min(low, open_price, close)
            
            volume = np.random.randint(100000, 1000000)
            
            if n_features >= 5:
                data[0, stock_idx, t] = open_price
                data[1, stock_idx, t] = high
                data[2, stock_idx, t] = low
                data[3, stock_idx, t] = close
                data[4, stock_idx, t] = volume
            else:
                data[min(3, n_features-1), stock_idx, t] = close
    
    print(f"✅ Données créées. Forme: {data.shape}")
    return data

def test_environment_creation():
    """Test de création et validation de l'environnement."""
    print("🏗️  Test de création de l'environnement...")
    
    try:
        from src.environment.portfolio_env import PortfolioEnv, create_env_from_config
        
        test_data = create_test_data(n_features=5, n_stocks=3, n_time_periods=50)
        
        print("\n1️⃣ Test de création directe...")
        env = PortfolioEnv(
            data=test_data,
            portfolio_value=10000,
            window_size=10,
            trans_cost=0.001
        )
        print("✅ Création directe réussie")
        
        print("\n2️⃣ Test de création via configuration...")
        config_path = create_test_config()
        env_config = create_env_from_config(test_data, config_path)
        print("✅ Création via configuration réussie")
        
        print("\n3️⃣ Test des espaces d'action et d'observation...")
        assert env.action_space.shape == (3,), f"Action space incorrect: {env.action_space.shape}"
        assert env.observation_space['market_data'].shape == (5, 3, 10)
        assert env.observation_space['weights'].shape == (3,)
        assert env.observation_space['portfolio_value'].shape == (1,)
        print("✅ Espaces validés")
        
        return True, config_path
        
    except Exception as e:
        print(f"❌ Erreur lors de la création: {e}")
        import traceback
        traceback.print_exc()
        return False, None

def test_environment_reset():
    """Test de réinitialisation de l'environnement."""
    print("\n🔄 Test de réinitialisation...")
    
    try:
        from src.environment.portfolio_env import PortfolioEnv
        
        test_data = create_test_data(n_features=5, n_stocks=3, n_time_periods=50)
        env = PortfolioEnv(data=test_data, window_size=10)
        
        obs, info = env.reset(seed=42)
        
        assert isinstance(obs, dict), "L'observation doit être un dictionnaire"
        assert 'market_data' in obs
        assert 'weights' in obs
        assert 'portfolio_value' in obs
        
        assert obs['market_data'].shape == (5, 3, 10)
        assert obs['weights'].shape == (3,)
        assert obs['portfolio_value'].shape == (1,)
        
        assert np.allclose(obs['weights'], 1/3)
        assert np.allclose(obs['portfolio_value'], [1.0])
        
        assert 'portfolio_value' in info
        assert info['portfolio_value'] == 10000
        
        print("✅ Réinitialisation validée")
        return True
        
    except Exception as e:
        print(f"❌ Erreur lors du reset: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_environment_step():
    """Test d'exécution de pas dans l'environnement."""
    print("\n👣 Test d'exécution de pas...")
    
    try:
        from src.environment.portfolio_env import PortfolioEnv
        
        test_data = create_test_data(n_features=5, n_stocks=3, n_time_periods=50)
        env = PortfolioEnv(data=test_data, window_size=10)
        obs, info = env.reset(seed=42)
        
        print(f"   État initial: portfolio={info['portfolio_value']}, weights={obs['weights']}")
        
        test_actions = [
            np.array([1.0, 0.0, 0.0]),
            np.array([0.0, 1.0, 0.0]),
            np.array([0.5, 0.3, 0.2]),
            np.array([0.33, 0.33, 0.34]),
        ]
        
        for i, action in enumerate(test_actions):
            print(f"\n   Test action {i+1}: {action}")
            
            next_obs, reward, terminated, truncated, step_info = env.step(action)
            
            assert isinstance(next_obs, dict)
            assert isinstance(reward, (int, float, np.number))
            assert isinstance(terminated, (bool, np.bool_))  # Accepter les deux types
            assert isinstance(truncated, (bool, np.bool_))
            assert isinstance(step_info, dict)
            
            weights_sum = np.sum(next_obs['weights'])
            assert abs(weights_sum - 1.0) < 1e-6, f"Poids non normalisés: somme={weights_sum}"
            
            required_info_keys = ['portfolio_value', 'weights', 'transaction_cost', 'return', 'volatility']
            for key in required_info_keys:
                assert key in step_info, f"Clé manquante dans step_info: {key}"
            
            print(f"      → Portfolio: ${step_info['portfolio_value']:.2f}")
            print(f"      → Récompense: {reward:.4f}")
            print(f"      → Rendement: {step_info['return']*100:.3f}%")
            print(f"      → Nouveaux poids: {next_obs['weights']}")
            
            if terminated:
                print("      → Épisode terminé")
                break
        
        print("✅ Exécution de pas validée")
        return True
        
    except Exception as e:
        print(f"❌ Erreur lors de l'exécution de pas: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_environment_episode():
    """Test d'un épisode complet."""
    print("\n🎬 Test d'épisode complet...")
    
    try:
        from src.environment.portfolio_env import PortfolioEnv
        
        test_data = create_test_data(n_features=5, n_stocks=3, n_time_periods=30)
        env = PortfolioEnv(data=test_data, window_size=10)
        
        obs, info = env.reset(seed=42)
        
        episode_rewards = []
        episode_values = []
        episode_returns = []
        step_count = 0
        
        print(f"   Démarrage épisode: ${info['portfolio_value']}")
        
        while True:
            action = np.random.dirichlet([1, 1, 1])
            obs, reward, terminated, truncated, step_info = env.step(action)
            
            episode_rewards.append(reward)
            episode_values.append(step_info['portfolio_value'])
            episode_returns.append(step_info['return'])
            step_count += 1
            
            if step_count % 5 == 0 or terminated:
                print(f"   Pas {step_count}: ${step_info['portfolio_value']:.2f}, "
                      f"R={reward:.3f}, ret={step_info['return']*100:.2f}%")
            
            if terminated or truncated:
                break
            
            if step_count > 50:
                print("   ⚠️  Protection boucle infinie activée")
                break
        
        print(f"\n📊 Analyse de l'épisode ({step_count} pas):")
        print(f"   💰 Valeur finale: ${episode_values[-1]:.2f}")
        print(f"   📈 Rendement total: {((episode_values[-1]/episode_values[0])-1)*100:.2f}%")
        print(f"   🎯 Récompense moyenne: {np.mean(episode_rewards):.3f}")
        print(f"   📊 Récompense totale: {np.sum(episode_rewards):.3f}")
        print(f"   📉 Volatilité des rendements: {np.std(episode_returns)*100:.3f}%")
        
        assert step_count > 0
        assert len(episode_rewards) == step_count
        assert all(isinstance(r, (int, float, np.number)) for r in episode_rewards)
        
        print("✅ Épisode complet validé")
        return True
        
    except Exception as e:
        print(f"❌ Erreur lors de l'épisode: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_gymnasium_compatibility():
    """Test de compatibilité avec Gymnasium."""
    print("\n🏋️  Test de compatibilité Gymnasium...")
    
    try:
        import gymnasium as gym
        from gymnasium.utils.env_checker import check_env
        from src.environment.portfolio_env import PortfolioEnv
        
        test_data = create_test_data(n_features=5, n_stocks=3, n_time_periods=30)
        env = PortfolioEnv(data=test_data, window_size=10)
        
        print("   Vérification avec check_env de Gymnasium...")
        check_env(env, warn=True, skip_render_check=True)
        
        print("✅ Compatibilité Gymnasium validée")
        return True
        
    except Exception as e:
        print(f"❌ Erreur de compatibilité Gymnasium: {e}")
        print("⚠️  Continuons malgré l'erreur de compatibilité")
        return False

def test_edge_cases():
    """Test des cas limites."""
    print("\n🎯 Test des cas limites...")
    
    try:
        from src.environment.portfolio_env import PortfolioEnv
        
        print("   Test 1: Données avec NaN...")
        test_data = create_test_data(n_features=5, n_stocks=3, n_time_periods=30)
        test_data[0, 0, 5] = np.nan
        env = PortfolioEnv(data=test_data, window_size=10)
        obs, info = env.reset()
        print("   ✅ Gestion des NaN réussie")
        
        print("   Test 2: Actions invalides...")
        
        action_nan = np.array([np.nan, 0.5, 0.5])
        obs, reward, terminated, truncated, info = env.step(action_nan)
        assert np.allclose(np.sum(obs['weights']), 1.0)
        print("   ✅ Gestion NaN dans action réussie")
        
        action_neg = np.array([-0.5, 1.0, 0.5])
        obs, reward, terminated, truncated, info = env.step(action_neg)
        assert np.all(obs['weights'] >= 0)
        print("   ✅ Gestion valeurs négatives réussie")
        
        action_zero = np.array([0.0, 0.0, 0.0])
        obs, reward, terminated, truncated, info = env.step(action_zero)
        assert np.allclose(np.sum(obs['weights']), 1.0)
        print("   ✅ Gestion somme nulle réussie")
        
        print("✅ Cas limites validés")
        return True
        
    except Exception as e:
        print(f"❌ Erreur dans les cas limites: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Fonction principale de test."""
    print("🚀 Test complet de l'environnement RL corrigé")
    print("=" * 60)
    
    os.makedirs("src/environment", exist_ok=True)
    os.makedirs("logs", exist_ok=True)
    
    tests = [
        ("Création de l'environnement", test_environment_creation),
        ("Réinitialisation", test_environment_reset),
        ("Exécution de pas", test_environment_step),
        ("Épisode complet", test_environment_episode),
        ("Compatibilité Gymnasium", test_gymnasium_compatibility),
        ("Cas limites", test_edge_cases),
    ]
    
    results = []
    config_path = None
    
    for test_name, test_func in tests:
        print(f"\n{'='*20} {test_name} {'='*20}")
        
        try:
            if test_name == "Création de l'environnement":
                success, config_path = test_func()
                results.append(success)
            else:
                result = test_func()
                results.append(result if result is not None else False)
        except Exception as e:
            print(f"❌ Test {test_name} échoué: {e}")
            results.append(False)
    
    if config_path:
        try:
            os.unlink(config_path)
        except:
            pass
    
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
        print("\n📋 L'environnement RL est maintenant prêt!")
        print("🎯 ÉTAPE 2 VALIDÉE!")
        print("\n📋 Prochaines étapes:")
        print("   3. 📊 Correction et test du preprocessing")
        print("   4. 🤖 Test du pipeline d'entraînement complet")
        print("   5. 🔄 Intégration MLflow et monitoring")
    else:
        print(f"\n⚠️  {total-passed} test(s) ont échoué.")
        print("Vérifiez les erreurs ci-dessus avant de continuer.")
        return False
    
    return True

if __name__ == "__main__":
    success = main()
    if not success:
        sys.exit(1)