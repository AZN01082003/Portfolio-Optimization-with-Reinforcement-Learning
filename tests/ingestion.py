"""
Module pour l'ingestion de données financières depuis Yahoo Finance.
"""
import os
import json
import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
import logging

# Configuration du logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def load_config(config_path="config/default.json"):
    """
    Charge la configuration depuis un fichier JSON.
    """
    # Chemin absolu basé sur le répertoire du script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    config_abs_path = os.path.join(script_dir, '..', '..', config_path)
    
    with open(config_abs_path, 'r') as file:
        config = json.load(file)
    return config

def download_stock_data(tickers, start_date, end_date):
    """
    Télécharge les données des actions depuis Yahoo Finance.
    
    Args:
        tickers (list): Liste des symboles d'actions à télécharger
        start_date (datetime): Date de début
        end_date (datetime): Date de fin
    
    Returns:
        pandas.DataFrame: Données des actions
    """
    logger.info(f"Téléchargement des données pour {len(tickers)} actions...")
    data = yf.download(tickers, start=start_date, end=end_date)
    logger.info("Téléchargement terminé!")
    return data

def check_and_create_dirs(dirs):
    """
    Crée les répertoires s'ils n'existent pas.
    
    Args:
        dirs (list): Liste des chemins de répertoires à créer
    """
    for dir_path in dirs:
        os.makedirs(dir_path, exist_ok=True)
        logger.info(f"Répertoire vérifié/créé: {dir_path}")

def main(config_path="config/default.json"):
    """
    Fonction principale pour l'ingestion des données.
    """
    # Charger la configuration
    config = load_config(config_path)
    
    # Extraire les paramètres de configuration
    tickers = config['data']['tickers']
    lookback_years = config['data']['lookback_years']
    output_dir = config['data']['output_dir']
    
    # Préparer les dates
    end_date = datetime.now()
    start_date = end_date - timedelta(days=365 * lookback_years)
    
    logger.info(f"Période de téléchargement: {start_date.date()} - {end_date.date()}")
    
    # Créer les répertoires nécessaires
    check_and_create_dirs([
        'data/raw',
        output_dir
    ])
    
    # Télécharger les données
    stock_data = download_stock_data(tickers, start_date, end_date)
    
    # Sauvegarder les données brutes
    raw_data_path = os.path.join('data/raw', f'stock_data_{end_date.strftime("%Y%m%d")}.csv')
    stock_data.to_csv(raw_data_path)
    logger.info(f"Données brutes sauvegardées dans {raw_data_path}")
    
    # Retourner les données pour le prétraitement
    return stock_data, tickers, config

if __name__ == "__main__":
    main()