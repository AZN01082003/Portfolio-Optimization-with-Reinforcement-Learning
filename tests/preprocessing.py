# Pr�traitement des donn�es

"""
Module pour le prétraitement des données financières.
"""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import logging
from datetime import datetime
import json

from ingestion import load_config, main as ingest_data


# Configuration du logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)



def preprocess_data(data, tickers):
    """
    Prétraite les données téléchargées pour les rendre compatibles avec l'environnement RL.
    
    Args:
        data (pandas.DataFrame): Données financières
        tickers (list): Liste des symboles d'actions
    
    Returns:
        numpy.ndarray: Données prétraitées de forme (n_features, n_stocks, n_time_periods)
    """
    # Vérification de la structure des données
    if isinstance(data.columns, pd.MultiIndex):
        # Données multi-index (plusieurs tickers)
        logger.info("Données multi-index détectées")
        features = ['Open', 'High', 'Low', 'Close', 'Volume']
        n_features = len(features)
        n_stocks = len(tickers)
        n_time_periods = len(data)
        
        # Initialiser le tableau résultat
        result = np.zeros((n_features, n_stocks, n_time_periods))
        
        for i, feature in enumerate(features):
            for j, ticker in enumerate(tickers):
                # Extraire les données pour cette action et cette caractéristique
                if (feature, ticker) in data.columns:
                    values = data[(feature, ticker)].values
                    result[i, j, :] = values
                else:
                    logger.warning(f"Avertissement: ({feature}, {ticker}) non trouvé dans les données")
    else:
        # Données pour un seul ticker
        logger.info("Données pour un seul ticker détectées")
        features = ['Open', 'High', 'Low', 'Close', 'Volume']
        n_features = len(features)
        n_stocks = 1
        n_time_periods = len(data)
        
        # Initialiser le tableau résultat
        result = np.zeros((n_features, n_stocks, n_time_periods))
        
        for i, feature in enumerate(features):
            if feature in data.columns:
                values = data[feature].values
                result[i, 0, :] = values
            else:
                logger.warning(f"Avertissement: {feature} non trouvé dans les données")
    
    logger.info(f"Forme des données prétraitées: {result.shape}")
    return result

def normalize_data(data):
    """
    Normalise les données pour améliorer l'apprentissage.
    
    Pour les prix (Open, High, Low, Close): normalisation relative au premier jour
    Pour le volume: normalisation par rapport au volume maximum
    
    Args:
        data (numpy.ndarray): Données prétraitées
    
    Returns:
        numpy.ndarray: Données normalisées
    """
    normalized_data = np.copy(data)
    
    # Normaliser les prix (OHLC)
    for i in range(4):  # Indices 0-3 sont Open, High, Low, Close
        for j in range(normalized_data.shape[1]):  # Pour chaque action
            first_valid_idx = 0
            while first_valid_idx < normalized_data.shape[2] and normalized_data[i, j, first_valid_idx] == 0:
                first_valid_idx += 1
                
            if first_valid_idx < normalized_data.shape[2]:
                reference_price = normalized_data[i, j, first_valid_idx]
                if reference_price > 0:  # Éviter division par zéro
                    normalized_data[i, j, :] = normalized_data[i, j, :] / reference_price
    
    # Normaliser le volume (indice 4)
    for j in range(normalized_data.shape[1]):  # Pour chaque action
        max_volume = np.max(normalized_data[4, j, :])
        if max_volume > 0:
            normalized_data[4, j, :] = normalized_data[4, j, :] / max_volume
    
    logger.info("Données normalisées avec succès")
    return normalized_data

def visualize_data(data, tickers, feature_idx=3, save_path=None):
    """
    Visualise les données pour une caractéristique spécifique (par défaut: Close).
    
    Args:
        data (numpy.ndarray): Données normalisées
        tickers (list): Liste des symboles d'actions
        feature_idx (int): Index de la caractéristique à visualiser (0=Open, 1=High, 2=Low, 3=Close, 4=Volume)
        save_path (str, optional): Chemin pour sauvegarder le graphique
    """
    features = ['Open', 'High', 'Low', 'Close', 'Volume']
    feature = features[feature_idx]
    
    plt.figure(figsize=(14, 7))
    for j, ticker in enumerate(tickers):
        plt.plot(data[feature_idx, j, :], label=ticker)
    
    plt.title(f"Évolution de {feature} pour les actions sélectionnées")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
        logger.info(f"Graphique sauvegardé dans {save_path}")
    else:
        plt.show()
    
    plt.close()

def split_data(data, train_ratio=0.7):
    """
    Sépare les données en ensembles d'entraînement et de test.
    
    Args:
        data (numpy.ndarray): Données normalisées
        train_ratio (float): Ratio pour la division entraînement/test
    
    Returns:
        tuple: (train_data, test_data)
    """
    n_time_periods = data.shape[2]
    split_point = int(n_time_periods * train_ratio)
    
    train_data = data[:, :, :split_point]
    test_data = data[:, :, split_point:]
    
    logger.info(f"Division des données - Entraînement: {train_data.shape}, Test: {test_data.shape}")
    return train_data, test_data

def save_data(data, filepath):
    """
    Enregistre les données prétraitées.
    
    Args:
        data (numpy.ndarray): Données à sauvegarder
        filepath (str): Chemin du fichier de sortie
    """
    # S'assurer que le répertoire existe
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    
    np.save(filepath, data)
    logger.info(f"Données enregistrées dans {filepath}")
    logger.info(f"Forme des données: {data.shape}")

def main(config_path="config/default.json", use_existing_data=False, data_path=None):
    """
    Fonction principale pour préparer les données des actions.
    
    Args:
        config_path (str): Chemin du fichier de configuration
        use_existing_data (bool): Si True, utilise des données existantes au lieu de télécharger
        data_path (str): Chemin vers des données existantes si use_existing_data=True
    
    Returns:
        tuple: (normalized_data, train_data, test_data)
    """
    # Charger la configuration
    config = load_config(config_path)
    
    # Extraire les paramètres
    output_dir = config['data']['output_dir']
    train_ratio = config['data']['train_ratio']
    
    # Créer le répertoire de sortie
    os.makedirs(output_dir, exist_ok=True)
    
    # Obtenir les données
    if use_existing_data and data_path:
        logger.info(f"Utilisation des données existantes depuis {data_path}")
        data = pd.read_csv(data_path, index_col=0, parse_dates=True)
        tickers = config['data']['tickers']
    else:
        # Télécharger de nouvelles données
        data, tickers, _ = ingest_data(config_path)
    
    # Prétraiter les données
    processed_data = preprocess_data(data, tickers)
    
    # Normaliser les données
    normalized_data = normalize_data(processed_data)
    
    # Visualiser les données
    viz_path = os.path.join(output_dir, "price_evolution.png")
    visualize_data(normalized_data, tickers, save_path=viz_path)
    
    # Diviser en ensembles d'entraînement et de test
    train_data, test_data = split_data(normalized_data, train_ratio)
    
    # Sauvegarder les données
    timestamp = datetime.now().strftime("%Y%m%d")
    save_data(normalized_data, os.path.join(output_dir, f'stock_data_normalized_{timestamp}.npy'))
    save_data(train_data, os.path.join(output_dir, f'stock_data_train_{timestamp}.npy'))
    save_data(test_data, os.path.join(output_dir, f'stock_data_test_{timestamp}.npy'))
    
    # Créer des liens symboliques vers les dernières données
    for src, dst in [
        (f'stock_data_normalized_{timestamp}.npy', 'stock_data_normalized_latest.npy'),
        (f'stock_data_train_{timestamp}.npy', 'stock_data_train_latest.npy'),
        (f'stock_data_test_{timestamp}.npy', 'stock_data_test_latest.npy')
    ]:
        src_path = os.path.join(output_dir, src)
        dst_path = os.path.join(output_dir, dst)
        
        # Supprimer le lien existant s'il existe
        if os.path.exists(dst_path):
            os.remove(dst_path)
        
        # Créer le nouveau lien symbolique
        if os.name == 'nt':  # Windows
            import shutil
            shutil.copy2(src_path, dst_path)
        else:  # Linux/Mac
            os.symlink(src, dst_path)
    
    logger.info("\nTraitement terminé!")
    logger.info(f"Données d'entraînement: {train_data.shape}")
    logger.info(f"Données de test: {test_data.shape}")
    
    return normalized_data, train_data, test_data

if __name__ == "__main__":
    main()