"""
Tests unitaires pour l'environnement de portfolio.
"""
import numpy as np
import pytest
from portfolio_env import PortfolioEnv

@pytest.fixture
def sample_data():
    """Crée des données factices pour les tests."""
    n_features, n_stocks, n_time = 5, 3, 100
    data = np.ones((n_features, n_stocks, n_time))
    
    # Ajouter une tendance haussière pour les prix
    for t in range(n_time):
        factor = 1.0 + 0.001 * t
        # OHLC (indices 0-3)
        for i in range(4):
            data[i, :, t] = factor
    
    # Volume aléatoire (indice 4)
    data[4, :, :] = np.random.rand(n_stocks, n_time) * 1000
    
    return data

def test_environment_initialization(sample_data):
    """Teste l'initialisation de l'environnement."""
    env = PortfolioEnv(data=sample_data)
    
    # Vérifier les dimensions
    assert env.n_features == 5
    assert env.n_stocks == 3
    assert env.n_time_periods == 100
    
    # Vérifier les espaces
    assert env.action_space.shape == (3,)
    assert env.observation_space['market_data'].shape == (5, 3, 30)
    assert env.observation_space['weights'].shape == (3,)
    assert env.observation_space['portfolio_value'].shape == (1,)

def test_environment_reset(sample_data):
    """Teste la réinitialisation de l'environnement."""
    env = PortfolioEnv(data=sample_data)
    obs, info = env.reset()
    
    # Vérifier la forme de l'observation
    assert 'market_data' in obs
    assert 'weights' in obs
    assert 'portfolio_value' in obs
    
    # Vérifier les valeurs initiales
    assert np.allclose(obs['weights'], np.array([1/3, 1/3, 1/3]))
    assert np.allclose(obs['portfolio_value'], np.array([1.0]))
    assert info['portfolio_value'] == 10000

def test_environment_step(sample_data):
    """Teste un pas dans l'environnement."""
    env = PortfolioEnv(data=sample_data)
    obs, info = env.reset()
    
    # Action: allouer tout à la première action
    action = np.array([1.0, 0.0, 0.0])
    
    next_obs, reward, terminated, truncated, info = env.step(action)
    
    # Vérifier la forme de l'observation
    assert 'market_data' in next_obs
    assert 'weights' in next_obs
    assert 'portfolio_value' in next_obs
    
    # Vérifier que les infos sont présentes
    assert 'portfolio_value' in info
    assert 'weights' in info
    assert 'transaction_cost' in info
    assert 'return' in info
    assert 'drawdown' in info
    
    # Comme nous avons créé des données avec une tendance haussière,
    # la récompense devrait être positive
    assert reward > 0