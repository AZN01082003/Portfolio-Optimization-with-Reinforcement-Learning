"""
Tests unitaires pour le prétraitement des données.
"""
import numpy as np
import pandas as pd
import pytest
import os
from src.data.preprocessing import preprocess_data, normalize_data, split_data

@pytest.fixture
def sample_dataframe():
    """Crée un DataFrame factice pour les tests."""
    # Créer des données multi-index pour plusieurs actions
    dates = pd.date_range(start='2023-01-01', periods=100)
    tickers = ['AAPL', 'MSFT', 'GOOGL']
    
    # Créer un DataFrame multi-index
    columns = pd.MultiIndex.from_product([['Open', 'High', 'Low', 'Close', 'Volume'], tickers])
    df = pd.DataFrame(columns=columns, index=dates)
    
    # Remplir avec des données factices
    for ticker in tickers:
        for feature in ['Open', 'High', 'Low', 'Close']:
            # Prix commençant à 100 avec légère tendance haussière
            prices = 100 + np.cumsum(np.random.normal(0.05, 0.5, 100))
            df[(feature, ticker)] = prices
        
        # Volume aléatoire
        df[('Volume', ticker)] = np.random.randint(1000, 100000, 100)
    
    return df

def test_preprocess_data(sample_dataframe):
    """Teste la fonction de prétraitement."""
    tickers = ['AAPL', 'MSFT', 'GOOGL']
    result = preprocess_data(sample_dataframe, tickers)
    
    # Vérifier la forme
    assert result.shape == (5, 3, 100)
    
    # Vérifier que les données ne sont pas toutes nulles
    assert np.any(result != 0)

def test_normalize_data():
    """Teste la fonction de normalisation."""
    # Créer des données factices
    data = np.ones((5, 3, 100))
    
    # Ajouter une tendance pour les prix (indices 0-3)
    for t in range(100):
        data[0:4, :, t] = 100 + t
    
    # Volume aléatoire (indice 4)
    data[4, :, :] = np.random.randint(1000, 100000, (3, 100))
    
    normalized = normalize_data(data)
    
    # Vérifier la forme
    assert normalized.shape == (5, 3, 100)
    
    # Vérifier que les prix sont normalisés par rapport au premier jour
    for i in range(4):  # OHLC
        for j in range(3):  # Stocks
            assert np.isclose(normalized[i, j, 0], 1.0)
    
    # Vérifier que le volume est normalisé
    for j in range(3):  # Stocks
        assert np.max(normalized[4, j, :]) <= 1.0

def test_split_data():
    """Teste la fonction de division des données."""
    # Créer des données factices
    data = np.ones((5, 3, 100))
    
    # Diviser avec un ratio de 0.7
    train_data, test_data = split_data(data, train_ratio=0.7)
    
    # Vérifier les formes
    assert train_data.shape == (5, 3, 70)
    assert test_data.shape == (5, 3, 30)