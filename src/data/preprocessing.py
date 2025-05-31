"""
Module pour le prétraitement des données financières.
Version corrigée avec compatibilité environnement RL et gestion d'erreurs robuste.
"""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import logging
from datetime import datetime
import json
from typing import Tuple, Optional, List, Dict, Any

# Configuration du logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def load_config(config_path: str = "config/default.json") -> dict:
    """
    Charge la configuration depuis un fichier json.
    
    Args:
        config_path: Chemin du fichier de configuration
        
    Returns:
        dict: Configuration chargée
    """
    try:
        with open(config_path, 'r', encoding='utf-8') as file:
            config = json.load(file)
        logger.info(f"Configuration chargée depuis: {config_path}")
        return config
    except Exception as e:
        logger.error(f"Erreur lors du chargement de la configuration: {e}")
        raise



# Dans src/data/preprocessing.py, ajoutez cette fonction avant detect_data_format :

def load_csv_data(filepath: str) -> pd.DataFrame:
    """
    Charge un fichier CSV en détectant automatiquement s'il s'agit d'un MultiIndex.
    
    Args:
        filepath: Chemin vers le fichier CSV
        
    Returns:
        pd.DataFrame: DataFrame chargé avec la structure appropriée
    """
    try:
        # Essayer de charger avec MultiIndex
        df_multiindex = pd.read_csv(filepath, index_col=0, header=[0, 1], parse_dates=True)
        
        # Vérifier si c'est vraiment un MultiIndex valide
        if (isinstance(df_multiindex.columns, pd.MultiIndex) and 
            not any('Unnamed:' in str(col) for col in df_multiindex.columns.get_level_values(0))):
            logger.info("Fichier CSV chargé comme MultiIndex")
            return df_multiindex
    except:
        pass
    
    try:
        # Essayer de charger normalement
        df_normal = pd.read_csv(filepath, index_col=0, parse_dates=True)
        logger.info("Fichier CSV chargé comme DataFrame normal")
        return df_normal
    except Exception as e:
        logger.error(f"Erreur lors du chargement du CSV: {e}")
        raise



def detect_data_format(data: pd.DataFrame) -> Dict[str, Any]:
    """
    Détecte le format des données d'entrée.
    
    Args:
        data: DataFrame à analyser
        
    Returns:
        Dict contenant les informations sur le format
    """
    format_info = {
        'is_multiindex': isinstance(data.columns, pd.MultiIndex),
        'n_columns': len(data.columns),
        'n_rows': len(data),
        'has_ticker_column': 'ticker' in data.columns,
        'column_names': list(data.columns),
        'index_type': type(data.index).__name__
    }
    
    # Détection des features financières standard
    standard_features = ['Open', 'High', 'Low', 'Close', 'Volume', 'Adj Close']
    format_info['detected_features'] = []
    
    if format_info['is_multiindex']:
        # Format MultiIndex (feature, ticker)
        level_0_names = data.columns.get_level_values(0).unique().tolist()
        level_1_names = data.columns.get_level_values(1).unique().tolist()
        format_info['features'] = level_0_names
        format_info['tickers'] = level_1_names
        format_info['detected_features'] = [f for f in standard_features if f in level_0_names]
    else:
        # Format simple ou avec colonne ticker
        if format_info['has_ticker_column']:
            # Format long avec colonne ticker
            format_info['features'] = [col for col in data.columns if col != 'ticker']
            format_info['tickers'] = data['ticker'].unique().tolist() if 'ticker' in data.columns else []
        else:
            # Format simple (un seul ticker)
            format_info['features'] = list(data.columns)
            format_info['tickers'] = ['SINGLE_TICKER']
        
        format_info['detected_features'] = [f for f in standard_features if f in format_info['features']]
    
    logger.info(f"Format détecté: {format_info}")
    return format_info

def preprocess_data(data: pd.DataFrame, tickers: List[str]) -> np.ndarray:
    """
    Prétraite les données téléchargées pour les rendre compatibles avec l'environnement RL.
    
    Args:
        data: DataFrame des données financières
        tickers: Liste des symboles d'actions
    
    Returns:
        np.ndarray: Données prétraitées de forme (n_features, n_stocks, n_time_periods)
    """
    try:
        # Analyser le format des données
        format_info = detect_data_format(data)
        
        # Définir les features à extraire (dans l'ordre pour l'environnement RL)
        target_features = ['Open', 'High', 'Low', 'Close', 'Volume']
        
        # Initialiser le tableau résultat
        n_features = len(target_features)
        n_stocks = len(tickers)
        n_time_periods = len(data)
        
        result = np.zeros((n_features, n_stocks, n_time_periods), dtype=np.float32)
        
        logger.info(f"Prétraitement: {n_features} features, {n_stocks} stocks, {n_time_periods} périodes")
        
        if format_info['is_multiindex']:
            # Traitement données MultiIndex
            logger.info("Traitement des données MultiIndex")
            
            for i, feature in enumerate(target_features):
                for j, ticker in enumerate(tickers):
                    if (feature, ticker) in data.columns:
                        values = data[(feature, ticker)].values
                        result[i, j, :] = values
                    elif feature == 'Volume' and ('Vol', ticker) in data.columns:
                        # Gestion alternative pour Volume
                        values = data[('Vol', ticker)].values
                        result[i, j, :] = values
                    elif feature == 'Close' and ('Adj Close', ticker) in data.columns:
                        # Utiliser Adj Close si Close n'est pas disponible
                        values = data[('Adj Close', ticker)].values
                        result[i, j, :] = values
                    else:
                        # Utiliser une valeur par défaut ou interpoler
                        if i < 4:  # Prix (OHLC)
                            # Essayer d'utiliser Close comme fallback
                            if ('Close', ticker) in data.columns:
                                values = data[('Close', ticker)].values
                                result[i, j, :] = values
                            else:
                                logger.warning(f"Feature {feature} non trouvée pour {ticker}, utilisation de valeurs par défaut")
                                result[i, j, :] = 100.0  # Prix par défaut
                        else:  # Volume
                            logger.warning(f"Volume non trouvé pour {ticker}, utilisation de valeurs par défaut")
                            result[i, j, :] = 1000000  # Volume par défaut
        
        elif format_info['has_ticker_column']:
            # Traitement données format long avec colonne ticker
            logger.info("Traitement des données format long")
            
            for j, ticker in enumerate(tickers):
                ticker_data = data[data['ticker'] == ticker]
                if len(ticker_data) == 0:
                    logger.warning(f"Aucune donnée trouvée pour {ticker}")
                    # Remplir avec des valeurs par défaut
                    for i in range(n_features):
                        if i < 4:
                            result[i, j, :] = 100.0 + i * 10  # Prix différents OHLC
                        else:
                            result[i, j, :] = 1000000  # Volume
                    continue
                
                # Réindexer pour s'assurer d'avoir toutes les périodes
                ticker_data = ticker_data.set_index(ticker_data.index).reindex(data.index)
                
                for i, feature in enumerate(target_features):
                    if feature in ticker_data.columns:
                        values = ticker_data[feature].values
                        result[i, j, :] = values
                    else:
                        logger.warning(f"Feature {feature} non trouvée pour {ticker}")
                        if i < 4:
                            result[i, j, :] = 100.0
                        else:
                            result[i, j, :] = 1000000
        
        else:
            # Traitement données simple (un seul ticker)
            logger.info("Traitement des données simple ticker")
            
            if len(tickers) > 1:
                logger.warning("Plusieurs tickers demandés mais données pour un seul ticker")
                # Dupliquer les données pour tous les tickers
                for j, ticker in enumerate(tickers):
                    for i, feature in enumerate(target_features):
                        if feature in data.columns:
                            values = data[feature].values
                            result[i, j, :] = values
                        else:
                            if i < 4:
                                result[i, j, :] = 100.0
                            else:
                                result[i, j, :] = 1000000
            else:
                # Un seul ticker
                for i, feature in enumerate(target_features):
                    if feature in data.columns:
                        values = data[feature].values
                        result[i, 0, :] = values
                    else:
                        if i < 4:
                            result[i, 0, :] = 100.0
                        else:
                            result[i, 0, :] = 1000000
        
        # Vérifications et nettoyage
        result = clean_data(result)
        
        logger.info(f"Prétraitement terminé. Forme finale: {result.shape}")
        return result
        
    except Exception as e:
        logger.error(f"Erreur lors du prétraitement: {e}")
        raise

def clean_data(data: np.ndarray) -> np.ndarray:
    """
    Nettoie les données (NaN, valeurs aberrantes, etc.).
    
    Args:
        data: Données à nettoyer
        
    Returns:
        np.ndarray: Données nettoyées
    """
    logger.info("Nettoyage des données...")
    
    # Remplacer les NaN et les inf
    data = np.nan_to_num(data, nan=0.0, posinf=0.0, neginf=0.0)
    
    # Remplacer les zéros dans les prix par interpolation
    for feature_idx in range(min(4, data.shape[0])):  # OHLC seulement
        for stock_idx in range(data.shape[1]):
            series = data[feature_idx, stock_idx, :]
            
            # Trouver les zéros et les remplacer
            zero_mask = (series <= 0)
            if np.any(zero_mask):
                # Utiliser la valeur précédente ou suivante non nulle
                non_zero_indices = np.where(~zero_mask)[0]
                if len(non_zero_indices) > 0:
                    # Interpolation simple
                    for idx in np.where(zero_mask)[0]:
                        # Trouver la valeur précédente non nulle
                        prev_idx = non_zero_indices[non_zero_indices < idx]
                        next_idx = non_zero_indices[non_zero_indices > idx]
                        
                        if len(prev_idx) > 0 and len(next_idx) > 0:
                            # Interpolation linéaire
                            prev_val = series[prev_idx[-1]]
                            next_val = series[next_idx[0]]
                            series[idx] = (prev_val + next_val) / 2
                        elif len(prev_idx) > 0:
                            series[idx] = series[prev_idx[-1]]
                        elif len(next_idx) > 0:
                            series[idx] = series[next_idx[0]]
                        else:
                            series[idx] = 100.0  # Valeur par défaut
                else:
                    # Aucune valeur non nulle, utiliser valeur par défaut
                    series[:] = 100.0
                
                data[feature_idx, stock_idx, :] = series
    
    # S'assurer que les volumes sont positifs
    if data.shape[0] >= 5:  # Si on a un index Volume
        data[4, :, :] = np.maximum(data[4, :, :], 1000)  # Volume minimum de 1000
    
    logger.info("Nettoyage terminé")
    return data

def normalize_data(data: np.ndarray) -> np.ndarray:
    """
    Normalise les données pour améliorer l'apprentissage.
    
    Args:
        data: Données prétraitées
    
    Returns:
        np.ndarray: Données normalisées
    """
    logger.info("Normalisation des données...")
    
    normalized_data = np.copy(data).astype(np.float32)
    
    # Normaliser les prix (OHLC) - indices 0-3
    for i in range(min(4, data.shape[0])):
        for j in range(data.shape[1]):
            series = normalized_data[i, j, :]
            
            # Trouver le premier prix valide
            first_valid_idx = 0
            while (first_valid_idx < len(series) and 
                   (series[first_valid_idx] <= 0 or np.isnan(series[first_valid_idx]))):
                first_valid_idx += 1
            
            if first_valid_idx < len(series):
                reference_price = series[first_valid_idx]
                if reference_price > 0:
                    # Normalisation relative au premier prix valide
                    normalized_data[i, j, :] = series / reference_price
                else:
                    normalized_data[i, j, :] = 1.0
            else:
                # Aucun prix valide trouvé
                normalized_data[i, j, :] = 1.0
    
    # Normaliser le volume (indice 4) si présent
    if data.shape[0] >= 5:
        for j in range(data.shape[1]):
            volume_series = normalized_data[4, j, :]
            max_volume = np.max(volume_series)
            if max_volume > 0:
                normalized_data[4, j, :] = volume_series / max_volume
            else:
                normalized_data[4, j, :] = 0.1  # Volume normalisé par défaut
    
    logger.info("Normalisation terminée")
    return normalized_data

def visualize_data(data: np.ndarray, tickers: List[str], feature_idx: int = 3, save_path: Optional[str] = None):
    """
    Visualise les données pour une caractéristique spécifique.
    
    Args:
        data: Données normalisées
        tickers: Liste des symboles d'actions
        feature_idx: Index de la caractéristique (0=Open, 1=High, 2=Low, 3=Close, 4=Volume)
        save_path: Chemin pour sauvegarder le graphique (optionnel)
    """
    try:
        features = ['Open', 'High', 'Low', 'Close', 'Volume']
        feature = features[min(feature_idx, len(features) - 1)]
        
        plt.figure(figsize=(14, 8))
        
        for j, ticker in enumerate(tickers):
            if j < data.shape[1]:
                plt.plot(data[feature_idx, j, :], label=ticker, linewidth=2)
        
        plt.title(f"Évolution de {feature} pour les actions sélectionnées", fontsize=14)
        plt.xlabel("Période de temps", fontsize=12)
        plt.ylabel(f"{feature} (normalisé)", fontsize=12)
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Graphique sauvegardé dans {save_path}")
        else:
            plt.show()
        
        plt.close()
        
    except Exception as e:
        logger.error(f"Erreur lors de la visualisation: {e}")

def split_data(data: np.ndarray, train_ratio: float = 0.7) -> Tuple[np.ndarray, np.ndarray]:
    """
    Sépare les données en ensembles d'entraînement et de test.
    
    Args:
        data: Données normalisées
        train_ratio: Ratio pour la division entraînement/test
    
    Returns:
        Tuple[np.ndarray, np.ndarray]: (train_data, test_data)
    """
    if not 0 < train_ratio < 1:
        raise ValueError(f"train_ratio doit être entre 0 et 1, obtenu: {train_ratio}")
    
    n_time_periods = data.shape[2]
    split_point = int(n_time_periods * train_ratio)
    
    # S'assurer qu'on a au moins quelques points dans chaque ensemble
    split_point = max(10, min(split_point, n_time_periods - 10))
    
    train_data = data[:, :, :split_point].copy()
    test_data = data[:, :, split_point:].copy()
    
    logger.info(f"Division des données - Entraînement: {train_data.shape}, Test: {test_data.shape}")
    return train_data, test_data

def save_data(data: np.ndarray, filepath: str, metadata: Optional[Dict] = None):
    """
    Enregistre les données prétraitées avec métadonnées.
    
    Args:
        data: Données à sauvegarder
        filepath: Chemin du fichier de sortie
        metadata: Métadonnées optionnelles
    """
    try:
        # S'assurer que le répertoire existe
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        # Sauvegarder les données
        np.save(filepath, data)
        logger.info(f"Données enregistrées dans {filepath}")
        logger.info(f"Forme des données: {data.shape}")
        
        # Sauvegarder les métadonnées si fournies
        if metadata:
            metadata_path = filepath.replace('.npy', '_metadata.json')
            metadata_to_save = {
                'shape': list(data.shape),
                'dtype': str(data.dtype),
                'created_at': datetime.now().isoformat(),
                **metadata
            }
            
            with open(metadata_path, 'w', encoding='utf-8') as f:
                json.dump(metadata_to_save, f, indent=2, default=str)
            logger.info(f"Métadonnées sauvegardées dans {metadata_path}")
            
    except Exception as e:
        logger.error(f"Erreur lors de la sauvegarde: {e}")
        raise


# Et modifiez la fonction main pour utiliser cette nouvelle fonction :

def main(config_path: str = "config/default.json", 
         use_existing_data: bool = False, 
         data_path: Optional[str] = None) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray]]:
    """
    Fonction principale pour préparer les données.
    
    Args:
        config_path: Chemin du fichier de configuration
        use_existing_data: Si True, utilise des données existantes
        data_path: Chemin vers les données existantes
    
    Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray]: (normalized_data, train_data, test_data)
    """
    try:
        # Charger la configuration
        config = load_config(config_path)
        
        # Extraire les paramètres
        output_dir = config['data']['output_dir']
        train_ratio = config['data']['train_ratio']
        tickers = config['data']['tickers']
        
        # Créer le répertoire de sortie
        os.makedirs(output_dir, exist_ok=True)
        
        # Obtenir les données
        if use_existing_data and data_path and os.path.exists(data_path):
            logger.info(f"Utilisation des données existantes depuis {data_path}")
            data = load_csv_data(data_path)  # Utiliser la nouvelle fonction
        else:
            # Importer et utiliser l'ingestion
            from src.data.ingestion import main as ingest_data
            data, actual_tickers, _ = ingest_data(config_path)
            tickers = actual_tickers  # Utiliser les tickers effectivement récupérés
        
        if data is None or data.empty:
            logger.error("Aucune donnée disponible pour le prétraitement")
            return None, None, None
        
        # Prétraiter les données
        logger.info("Début du prétraitement...")
        processed_data = preprocess_data(data, tickers)
        
        # Normaliser les données
        logger.info("Début de la normalisation...")
        normalized_data = normalize_data(processed_data)
        
        # Visualiser les données
        viz_path = os.path.join(output_dir, "price_evolution.png")
        visualize_data(normalized_data, tickers, save_path=viz_path)
        
        # Diviser en ensembles d'entraînement et de test
        logger.info("Division des données...")
        train_data, test_data = split_data(normalized_data, train_ratio)
        
        # Préparer les métadonnées
        metadata = {
            'tickers': tickers,
            'n_features': normalized_data.shape[0],
            'n_stocks': normalized_data.shape[1],
            'n_time_periods': normalized_data.shape[2],
            'train_ratio': train_ratio,
            'features': ['Open', 'High', 'Low', 'Close', 'Volume'][:normalized_data.shape[0]]
        }
        
        # Sauvegarder les données
        timestamp = datetime.now().strftime("%Y%m%d")
        
        save_data(normalized_data, 
                 os.path.join(output_dir, f'stock_data_normalized_{timestamp}.npy'),
                 metadata)
        
        save_data(train_data, 
                 os.path.join(output_dir, f'stock_data_train_{timestamp}.npy'),
                 {**metadata, 'data_type': 'train'})
        
        save_data(test_data, 
                 os.path.join(output_dir, f'stock_data_test_{timestamp}.npy'),
                 {**metadata, 'data_type': 'test'})
        
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
            
            # Créer le nouveau lien (copie sur Windows)
            if os.name == 'nt':  # Windows
                import shutil
                shutil.copy2(src_path, dst_path)
            else:  # Linux/Mac
                os.symlink(src, dst_path)
        
        logger.info("\n🎉 Traitement terminé avec succès!")
        logger.info(f"📊 Données d'entraînement: {train_data.shape}")
        logger.info(f"📊 Données de test: {test_data.shape}")
        logger.info(f"📈 Tickers traités: {tickers}")
        
        return normalized_data, train_data, test_data
        
    except Exception as e:
        logger.error(f"Erreur lors du traitement principal: {e}")
        return None, None, None
    


if __name__ == "__main__":
    result = main()
    if result[0] is not None:
        print("✅ Preprocessing réussi!")
    else:
        print("❌ Preprocessing échoué!")
        exit(1)