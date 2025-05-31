"""
Module pour l'ingestion de données financières depuis Yahoo Finance.
Version corrigée compatible avec toutes les versions de yfinance.
"""
import os
import json
import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
import logging
from typing import List, Tuple, Optional

# Configuration du logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def load_config(config_path: str = "config/default.json") -> dict:
    """
    Charge la configuration depuis un fichier JSON.
    
    Args:
        config_path: Chemin vers le fichier de configuration
        
    Returns:
        dict: Configuration chargée
        
    Raises:
        FileNotFoundError: Si le fichier de configuration n'existe pas
        json.JSONDecodeError: Si le fichier JSON est malformé
    """
    try:
        # Gérer les chemins relatifs et absolus
        if not os.path.isabs(config_path):
            script_dir = os.path.dirname(os.path.abspath(__file__))
            config_abs_path = os.path.join(script_dir, '..', '..', config_path)
        else:
            config_abs_path = config_path
            
        if not os.path.exists(config_abs_path):
            raise FileNotFoundError(f"Fichier de configuration non trouvé: {config_abs_path}")
            
        with open(config_abs_path, 'r', encoding='utf-8') as file:
            config = json.load(file)
            
        logger.info(f"Configuration chargée depuis: {config_abs_path}")
        return config
        
    except Exception as e:
        logger.error(f"Erreur lors du chargement de la configuration: {e}")
        raise

def validate_tickers(tickers: List[str]) -> List[str]:
    """
    Valide et nettoie la liste des tickers.
    
    Args:
        tickers: Liste des symboles d'actions
        
    Returns:
        List[str]: Liste des tickers validés
    """
    if not tickers or not isinstance(tickers, list):
        raise ValueError("La liste des tickers ne peut pas être vide")
        
    # Nettoyer et valider les tickers
    clean_tickers = []
    for ticker in tickers:
        if isinstance(ticker, str) and ticker.strip():
            clean_tickers.append(ticker.strip().upper())
        else:
            logger.warning(f"Ticker invalide ignoré: {ticker}")
            
    if not clean_tickers:
        raise ValueError("Aucun ticker valide trouvé")
        
    logger.info(f"Tickers validés: {clean_tickers}")
    return clean_tickers

def download_stock_data(tickers: List[str], start_date: datetime, end_date: datetime, 
                       retry_count: int = 3) -> pd.DataFrame:
    """
    Télécharge les données des actions depuis Yahoo Finance avec retry.
    Compatible avec toutes les versions de yfinance.
    
    Args:
        tickers: Liste des symboles d'actions à télécharger
        start_date: Date de début
        end_date: Date de fin
        retry_count: Nombre de tentatives en cas d'échec
    
    Returns:
        pandas.DataFrame: Données des actions
        
    Raises:
        Exception: Si le téléchargement échoue après toutes les tentatives
    """
    tickers = validate_tickers(tickers)
    
    for attempt in range(retry_count):
        try:
            logger.info(f"Tentative {attempt + 1}/{retry_count} - Téléchargement des données pour {len(tickers)} actions...")
            logger.info(f"Période: {start_date.date()} - {end_date.date()}")
            
            # Télécharger les données avec paramètres compatibles
            # Utiliser seulement les paramètres supportés par toutes les versions
            data = yf.download(
                tickers, 
                start=start_date, 
                end=end_date,
                progress=False,  # Désactiver la barre de progression
                threads=True
            )
            
            # Vérifier si des données ont été téléchargées
            if data.empty:
                raise ValueError("Aucune donnée téléchargée")
                
            # Vérifier la qualité des données
            total_cells = data.shape[0] * data.shape[1]
            missing_cells = data.isnull().sum().sum()
            missing_data_ratio = missing_cells / total_cells if total_cells > 0 else 0
            
            if missing_data_ratio > 0.5:
                logger.warning(f"Beaucoup de données manquantes: {missing_data_ratio:.2%}")
                
            logger.info(f"Téléchargement réussi! Forme des données: {data.shape}")
            logger.info(f"Données manquantes: {missing_data_ratio:.2%}")
            
            return data
            
        except Exception as e:
            logger.error(f"Tentative {attempt + 1} échouée: {e}")
            if attempt == retry_count - 1:
                logger.error("Toutes les tentatives de téléchargement ont échoué")
                raise
            else:
                logger.info(f"Nouvelle tentative dans 3 secondes...")
                import time
                time.sleep(3)

def check_and_create_dirs(dirs: List[str]) -> None:
    """
    Crée les répertoires s'ils n'existent pas.
    
    Args:
        dirs: Liste des chemins de répertoires à créer
    """
    for dir_path in dirs:
        try:
            os.makedirs(dir_path, exist_ok=True)
            logger.debug(f"Répertoire vérifié/créé: {dir_path}")
        except Exception as e:
            logger.error(f"Erreur lors de la création du répertoire {dir_path}: {e}")
            raise

def save_data_with_metadata(data: pd.DataFrame, output_path: str, metadata: dict) -> None:
    """
    Sauvegarde les données avec des métadonnées.
    
    Args:
        data: DataFrame à sauvegarder
        output_path: Chemin de sauvegarde
        metadata: Métadonnées à inclure
    """
    try:
        # Sauvegarder les données
        data.to_csv(output_path)
        logger.info(f"Données sauvegardées dans {output_path}")
        
        # Sauvegarder les métadonnées
        metadata_path = output_path.replace('.csv', '_metadata.json')
        with open(metadata_path, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2, default=str)
        logger.info(f"Métadonnées sauvegardées dans {metadata_path}")
        
    except Exception as e:
        logger.error(f"Erreur lors de la sauvegarde: {e}")
        raise

def main(config_path: str = "config/default.json", 
         tickers: Optional[List[str]] = None,
         start_date: Optional[datetime] = None,
         end_date: Optional[datetime] = None) -> Tuple[pd.DataFrame, List[str], dict]:
    """
    Fonction principale pour l'ingestion des données.
    
    Args:
        config_path: Chemin du fichier de configuration
        tickers: Liste des tickers (optionnel, sinon lu depuis config)
        start_date: Date de début (optionnel, sinon calculée depuis config)
        end_date: Date de fin (optionnel, sinon datetime.now())
    
    Returns:
        Tuple[pd.DataFrame, List[str], dict]: (données, tickers, config)
    """
    try:
        # Charger la configuration
        config = load_config(config_path)
        
        # Utiliser les paramètres fournis ou ceux de la configuration
        if tickers is None:
            tickers = config['data']['tickers']
        if end_date is None:
            end_date = datetime.now()
        if start_date is None:
            lookback_years = config['data']['lookback_years']
            start_date = end_date - timedelta(days=365 * lookback_years)
        
        # Valider les paramètres
        tickers = validate_tickers(tickers)
        output_dir = config['data']['output_dir']
        
        logger.info(f"Démarrage de l'ingestion pour {len(tickers)} tickers")
        logger.info(f"Période: {start_date.date()} - {end_date.date()}")
        
        # Créer les répertoires nécessaires
        check_and_create_dirs([
            'data/raw',
            output_dir,
            'logs'
        ])
        
        # Télécharger les données
        stock_data = download_stock_data(tickers, start_date, end_date)
        
        # Préparer les métadonnées
        metadata = {
            'tickers': tickers,
            'start_date': start_date.isoformat(),
            'end_date': end_date.isoformat(),
            'download_timestamp': datetime.now().isoformat(),
            'data_shape': list(stock_data.shape),
            'columns': list(stock_data.columns) if hasattr(stock_data.columns, 'tolist') else str(stock_data.columns),
            'missing_data_count': int(stock_data.isnull().sum().sum()),
            'config_used': config_path
        }
        
        # Sauvegarder les données brutes avec métadonnées
        raw_data_path = os.path.join('data/raw', f'stock_data_{end_date.strftime("%Y%m%d")}.csv')
        save_data_with_metadata(stock_data, raw_data_path, metadata)
        
        logger.info("Ingestion des données terminée avec succès")
        return stock_data, tickers, config
        
    except Exception as e:
        logger.error(f"Erreur lors de l'ingestion des données: {e}")
        raise

if __name__ == "__main__":
    try:
        data, tickers, config = main()
        print(f"✅ Ingestion réussie pour {len(tickers)} tickers")
        print(f"📊 Forme des données: {data.shape}")
    except Exception as e:
        print(f"❌ Erreur: {e}")
        exit(1)