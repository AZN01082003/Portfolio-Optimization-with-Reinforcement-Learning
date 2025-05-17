"""
Module pour la génération de features financières à partir des données brutes.
"""
import os
import logging
import numpy as np
import pandas as pd
import json
from datetime import datetime, timedelta
import ta  # Bibliothèque pour les indicateurs techniques
from typing import Dict, List, Union

from src.data.feature_store import FeatureStore

# Configuration du logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def load_config(config_path="config/default.json"):
    """
    Charge la configuration depuis un fichier json.
    """
    with open(config_path, 'r') as file:
        config = json.load(file)
    return config

def load_stock_data(ticker=None, start_date=None, end_date=None, config_path="config/default.json"):
    """
    Charge les données d'actions depuis les fichiers bruts.
    
    Args:
        ticker: Symbole de l'action (optionnel, si None, charge toutes les actions)
        start_date: Date de début (str ou datetime)
        end_date: Date de fin (str ou datetime)
        config_path: Chemin du fichier de configuration
        
    Returns:
        pd.DataFrame: DataFrame contenant les données d'actions
    """
    try:
        config = load_config(config_path)
        data_dir = config.get('data', {}).get('output_dir', 'data/processed')
        
        # Trouver le fichier de données le plus récent
        raw_data_dir = "data/raw"
        csv_files = [f for f in os.listdir(raw_data_dir) if f.startswith('stock_data_') and f.endswith('.csv')]
        
        if not csv_files:
            logger.error("Aucun fichier de données brutes trouvé")
            return None
        
        # Trier par date décroissante
        csv_files.sort(reverse=True)
        latest_file = os.path.join(raw_data_dir, csv_files[0])
        
        # Charger les données
        df = pd.read_csv(latest_file, index_col=0, parse_dates=True)
        
        # S'il s'agit d'un MultiIndex DataFrame, le restructurer
        if isinstance(df.columns, pd.MultiIndex):
            # Créer un DataFrame par action
            dfs = {}
            for feature in ['Open', 'High', 'Low', 'Close', 'Volume']:
                feature_df = df[feature]
                for tick in feature_df.columns:
                    if tick not in dfs:
                        dfs[tick] = pd.DataFrame(index=df.index)
                    dfs[tick][feature] = feature_df[tick]
            
            # Si un ticker spécifique est demandé
            if ticker and ticker in dfs:
                df = dfs[ticker]
            elif ticker:
                logger.warning(f"Ticker {ticker} non trouvé dans les données")
                return None
            else:
                # Convertir en DataFrame avec colonne 'ticker'
                combined_df = pd.concat([df.assign(ticker=tick) for tick, df in dfs.items()])
                df = combined_df.reset_index()
        
        # Filtrer par date si spécifié
        if start_date:
            if isinstance(start_date, str):
                start_date = pd.to_datetime(start_date)
            df = df[df.index >= start_date]
        
        if end_date:
            if isinstance(end_date, str):
                end_date = pd.to_datetime(end_date)
            df = df[df.index <= end_date]
        
        return df
    
    except Exception as e:
        logger.error(f"Erreur lors du chargement des données d'actions: {e}")
        return None

def add_technical_indicators(df, window_size=14, volatility_window=30):
    """
    Ajoute des indicateurs techniques au DataFrame.
    
    Args:
        df: DataFrame contenant les données OHLCV
        window_size: Taille de la fenêtre pour certains indicateurs
        volatility_window: Taille de la fenêtre pour les calculs de volatilité
        
    Returns:
        pd.DataFrame: DataFrame avec les indicateurs ajoutés
    """
    try:
        # S'assurer que les colonnes nécessaires sont présentes
        required_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
        
        if not all(col in df.columns for col in required_cols):
            logger.error(f"Colonnes manquantes dans le DataFrame. Colonnes requises: {required_cols}")
            return df
        
        # Copier le DataFrame pour éviter les modifications en place
        result_df = df.copy()
        
        # Tendance
        result_df['SMA'] = ta.trend.sma_indicator(result_df['Close'], window=window_size)
        result_df['EMA'] = ta.trend.ema_indicator(result_df['Close'], window=window_size)
        result_df['MACD'] = ta.trend.macd(result_df['Close'], window_fast=12, window_slow=26)
        result_df['MACD_Signal'] = ta.trend.macd_signal(result_df['Close'], window_fast=12, window_slow=26, window_sign=9)
        result_df['MACD_Diff'] = ta.trend.macd_diff(result_df['Close'], window_fast=12, window_slow=26, window_sign=9)
        
        # Momentum
        result_df['RSI'] = ta.momentum.rsi(result_df['Close'], window=window_size)
        result_df['Stoch'] = ta.momentum.stoch(result_df['High'], result_df['Low'], result_df['Close'], window=window_size)
        result_df['Stoch_Signal'] = ta.momentum.stoch_signal(result_df['High'], result_df['Low'], result_df['Close'], window=window_size)
        
        # Volatilité
        result_df['BB_High'] = ta.volatility.bollinger_hband(result_df['Close'], window=window_size)
        result_df['BB_Low'] = ta.volatility.bollinger_lband(result_df['Close'], window=window_size)
        result_df['BB_Width'] = (result_df['BB_High'] - result_df['BB_Low']) / result_df['Close']
        
        # Volatilité historique
        result_df['Returns'] = result_df['Close'].pct_change()
        result_df['Volatility'] = result_df['Returns'].rolling(window=volatility_window).std() * np.sqrt(252)  # Annualisée
        
        # Volume
        result_df['Volume_SMA'] = ta.trend.sma_indicator(result_df['Volume'], window=window_size)
        result_df['Volume_Change'] = result_df['Volume'].pct_change()
        
        # Remplir les NaN résultant des calculs basés sur des fenêtres
        result_df = result_df.fillna(method='bfill')
        
        return result_df
    
    except Exception as e:
        logger.error(f"Erreur lors de l'ajout des indicateurs techniques: {e}")
        return df

def add_price_features(df, windows=[5, 10, 20, 60]):
    """
    Ajoute des features basées sur les prix.
    
    Args:
        df: DataFrame contenant les données OHLCV
        windows: Liste des tailles de fenêtres pour les calculs
        
    Returns:
        pd.DataFrame: DataFrame avec les features ajoutées
    """
    try:
        result_df = df.copy()
        
        # Calcul des rendements
        result_df['Daily_Return'] = result_df['Close'].pct_change()
        
        # Rendements cumulés sur différentes périodes
        for window in windows:
            result_df[f'Return_{window}d'] = result_df['Close'].pct_change(periods=window)
            result_df[f'MA_{window}d'] = result_df['Close'].rolling(window=window).mean()
            result_df[f'Std_{window}d'] = result_df['Close'].rolling(window=window).std()
            result_df[f'Max_{window}d'] = result_df['Close'].rolling(window=window).max()
            result_df[f'Min_{window}d'] = result_df['Close'].rolling(window=window).min()
        
        # Ratio entre le prix actuel et les moyennes mobiles
        for window in windows:
            result_df[f'Price_to_MA_{window}d'] = result_df['Close'] / result_df[f'MA_{window}d']
        
        # Écart entre le prix et les moyennes mobiles (en pourcentage)
        for window in windows:
            result_df[f'Price_MA_{window}d_Gap'] = (result_df['Close'] - result_df[f'MA_{window}d']) / result_df[f'MA_{window}d'] * 100
        
        # True Range et ATR
        result_df['High_Low'] = result_df['High'] - result_df['Low']
        result_df['High_PrevClose'] = abs(result_df['High'] - result_df['Close'].shift(1))
        result_df['Low_PrevClose'] = abs(result_df['Low'] - result_df['Close'].shift(1))
        result_df['True_Range'] = result_df[['High_Low', 'High_PrevClose', 'Low_PrevClose']].max(axis=1)
        
        for window in windows:
            result_df[f'ATR_{window}d'] = result_df['True_Range'].rolling(window=window).mean()
        
        # Remplir les NaN
        result_df = result_df.fillna(method='bfill')
        
        return result_df
    
    except Exception as e:
        logger.error(f"Erreur lors de l'ajout des features de prix: {e}")
        return df

def add_volume_features(df, windows=[5, 10, 20, 60]):
    """
    Ajoute des features basées sur le volume.
    
    Args:
        df: DataFrame contenant les données OHLCV
        windows: Liste des tailles de fenêtres pour les calculs
        
    Returns:
        pd.DataFrame: DataFrame avec les features ajoutées
    """
    try:
        result_df = df.copy()
        
        # Variation quotidienne du volume
        result_df['Volume_Change'] = result_df['Volume'].pct_change()
        
        # Volume moyen sur différentes périodes
        for window in windows:
            result_df[f'Volume_MA_{window}d'] = result_df['Volume'].rolling(window=window).mean()
        
        # Ratio entre le volume actuel et les moyennes mobiles
        for window in windows:
            result_df[f'Volume_Ratio_{window}d'] = result_df['Volume'] / result_df[f'Volume_MA_{window}d']
        
        # On-Balance Volume (OBV)
        result_df['OBV'] = (np.sign(result_df['Close'].diff()) * result_df['Volume']).fillna(0).cumsum()
        
        # Chaikin Money Flow
        for window in windows:
            money_flow = ((result_df['Close'] - result_df['Low']) - (result_df['High'] - result_df['Close'])) / (result_df['High'] - result_df['Low'])
            money_flow = money_flow * result_df['Volume']
            result_df[f'CMF_{window}d'] = money_flow.rolling(window=window).sum() / result_df['Volume'].rolling(window=window).sum()
        
        # Remplir les NaN
        result_df = result_df.fillna(method='bfill')
        
        return result_df
    
    except Exception as e:
        logger.error(f"Erreur lors de l'ajout des features de volume: {e}")
        return df

def add_correlation_features(df_dict, windows=[10, 20, 60]):
    """
    Ajoute des features basées sur les corrélations entre actions.
    
    Args:
        df_dict: Dictionnaire de DataFrames par ticker
        windows: Liste des tailles de fenêtres pour les calculs
        
    Returns:
        dict: Dictionnaire mis à jour avec les features de corrélation
    """
    try:
        result_dict = {ticker: df.copy() for ticker, df in df_dict.items()}
        
        # Extraire les prix de clôture de toutes les actions
        close_prices = pd.DataFrame({ticker: df['Close'] for ticker, df in df_dict.items()})
        
        # Calculer les corrélations glissantes pour chaque action
        for ticker, df in result_dict.items():
            for window in windows:
                # Pour chaque action, calculer sa corrélation avec toutes les autres
                for other_ticker in df_dict.keys():
                    if ticker != other_ticker:
                        corr = close_prices[ticker].rolling(window=window).corr(close_prices[other_ticker])
                        result_dict[ticker][f'Corr_{other_ticker}_{window}d'] = corr
        
        # Remplir les NaN
        for ticker in result_dict:
            result_dict[ticker] = result_dict[ticker].fillna(method='bfill')
        
        return result_dict
    
    except Exception as e:
        logger.error(f"Erreur lors de l'ajout des features de corrélation: {e}")
        return df_dict

def add_market_features(df, market_df, windows=[5, 10, 20, 60]):
    """
    Ajoute des features basées sur la comparaison avec le marché.
    
    Args:
        df: DataFrame de l'action
        market_df: DataFrame de l'indice de marché (ex: S&P 500)
        windows: Liste des tailles de fenêtres pour les calculs
        
    Returns:
        pd.DataFrame: DataFrame avec les features ajoutées
    """
    try:
        result_df = df.copy()
        
        # S'assurer que les index sont alignés
        market_df = market_df.reindex(result_df.index, method='ffill')
        
        # Beta et Alpha
        for window in windows:
            # Rendements
            stock_returns = result_df['Close'].pct_change(periods=1)
            market_returns = market_df['Close'].pct_change(periods=1)
            
            # Beta (régression linéaire glissante)
            cov = stock_returns.rolling(window=window).cov(market_returns)
            market_var = market_returns.rolling(window=window).var()
            result_df[f'Beta_{window}d'] = cov / market_var
            
            # Alpha (Jensen)
            risk_free_rate = 0.01 / 252  # Taux sans risque quotidien (1% annuel)
            result_df[f'Alpha_{window}d'] = stock_returns.rolling(window=window).mean() - risk_free_rate - \
                                           result_df[f'Beta_{window}d'] * (market_returns.rolling(window=window).mean() - risk_free_rate)
            
            # Corrélation avec le marché
            result_df[f'Market_Corr_{window}d'] = stock_returns.rolling(window=window).corr(market_returns)
            
            # Performance relative
            result_df[f'Rel_Strength_{window}d'] = (result_df['Close'] / result_df['Close'].shift(window)) / \
                                                   (market_df['Close'] / market_df['Close'].shift(window))
        
        # Remplir les NaN
        result_df = result_df.fillna(method='bfill')
        
        return result_df
    
    except Exception as e:
        logger.error(f"Erreur lors de l'ajout des features de marché: {e}")
        return df

def generate_stock_features(ticker=None, config_path="config/default.json"):
    """
    Génère l'ensemble des features pour une ou plusieurs actions.
    
    Args:
        ticker: Symbole de l'action (optionnel, si None, traite toutes les actions)
        config_path: Chemin du fichier de configuration
        
    Returns:
        dict: Dictionnaire de DataFrames avec les features générées
    """
    try:
        # Créer ou récupérer le feature store
        feature_store = FeatureStore(config_path=config_path)
        
        # Charger les données
        df = load_stock_data(ticker=ticker, config_path=config_path)
        if df is None:
            logger.error("Échec du chargement des données")
            return None
        
        # Si les données contiennent plusieurs actions
        if 'ticker' in df.columns:
            # Séparer en DataFrames par ticker
            tickers = df['ticker'].unique()
            df_dict = {tick: df[df['ticker'] == tick].drop('ticker', axis=1) for tick in tickers}
        else:
            # Une seule action
            df_dict = {ticker: df}
        
        # Générer les features pour chaque action
        result_dict = {}
        for tick, tick_df in df_dict.items():
            logger.info(f"Génération des features pour {tick}...")
            
            # Ajouter les indicateurs techniques
            tech_df = add_technical_indicators(tick_df)
            
            # Ajouter les features de prix
            price_df = add_price_features(tech_df)
            
            # Ajouter les features de volume
            volume_df = add_volume_features(price_df)
            
            # Stocker le résultat
            result_dict[tick] = volume_df
        
        # Ajouter les features de corrélation
        result_dict = add_correlation_features(result_dict)
        
        # Sauvegarder les features dans le store
        saved_features = {}
        for tick, tick_df in result_dict.items():
            # Sauvegarder le DataFrame complet
            feature_id = feature_store.save_feature(
                name=f"{tick}_full",
                data=tick_df,
                entity_type="stock",
                version="latest",
                description=f"Features complètes pour {tick}",
                parameters={"ticker": tick}
            )
            
            # Sauvegarder également les catégories de features séparément
            # Features de prix
            price_cols = [col for col in tick_df.columns if any(term in col for term in 
                                                                ['Open', 'High', 'Low', 'Close', 'Return', 'MA', 'Price'])]
            price_df = tick_df[price_cols]
            price_id = feature_store.save_feature(
                name=f"{tick}_price",
                data=price_df,
                entity_type="stock",
                version="latest",
                description=f"Features de prix pour {tick}",
                parameters={"ticker": tick}
            )
            
            # Features de volume
            volume_cols = [col for col in tick_df.columns if any(term in col for term in 
                                                                 ['Volume', 'OBV', 'CMF'])]
            volume_df = tick_df[volume_cols]
            volume_id = feature_store.save_feature(
                name=f"{tick}_volume",
                data=volume_df,
                entity_type="stock",
                version="latest",
                description=f"Features de volume pour {tick}",
                parameters={"ticker": tick}
            )
            
            # Features techniques
            tech_cols = [col for col in tick_df.columns if any(term in col for term in 
                                                               ['SMA', 'EMA', 'MACD', 'RSI', 'Stoch', 'BB', 'Volatility'])]
            tech_df = tick_df[tech_cols]
            tech_id = feature_store.save_feature(
                name=f"{tick}_technical",
                data=tech_df,
                entity_type="stock",
                version="latest",
                description=f"Indicateurs techniques pour {tick}",
                parameters={"ticker": tick}
            )
            
            # Stocker les IDs des features
            saved_features[tick] = {
                "full": feature_id,
                "price": price_id,
                "volume": volume_id,
                "technical": tech_id
            }
        
        # Créer un feature set regroupant toutes les actions
        if len(saved_features) > 1:
            feature_list = []
            for tick, ids in saved_features.items():
                for category, feature_id in ids.items():
                    feature_list.append({
                        "id": feature_id,
                        "name": f"{tick}_{category}",
                        "ticker": tick,
                        "category": category
                    })
            
            feature_set_id = feature_store.save_feature_set(
                name="all_stocks",
                features=feature_list,
                version="latest",
                description="Ensemble des features pour toutes les actions"
            )
            
            logger.info(f"Feature set sauvegardé avec ID: {feature_set_id}")
        
        logger.info("Génération des features terminée")
        return result_dict
    
    except Exception as e:
        logger.error(f"Erreur lors de la génération des features: {e}")
        return None

def prepare_model_features(tickers, lookback_window=30, config_path="config/default.json"):
    """
    Prépare les features pour l'entraînement du modèle RL.
    
    Args:
        tickers: Liste des tickers à inclure
        lookback_window: Nombre de jours d'historique à inclure
        config_path: Chemin du fichier de configuration
        
    Returns:
        np.ndarray: Tableau de features au format attendu par l'environnement RL
    """
    try:
        # Créer ou récupérer le feature store
        feature_store = FeatureStore(config_path=config_path)
        
        # Vérifier si les features existent, sinon les générer
        for ticker in tickers:
            data, metadata = feature_store.load_feature(name=f"{ticker}_full", entity_type="stock")
            if data is None:
                logger.info(f"Features pour {ticker} non trouvées. Génération en cours...")
                generate_stock_features(ticker=ticker, config_path=config_path)
        
        # Charger les features pour tous les tickers
        features_dict = {}
        for ticker in tickers:
            data, metadata = feature_store.load_feature(name=f"{ticker}_full", entity_type="stock")
            if data is not None:
                features_dict[ticker] = data
        
        if not features_dict:
            logger.error("Aucune feature trouvée ou générée")
            return None
        
        # Identifier les colonnes d'intérêt pour le modèle RL
        # Nous allons nous concentrer sur les caractéristiques principales: OHLCV, quelques indicateurs techniques et de volume
        selected_features = ['Open', 'High', 'Low', 'Close', 'Volume',
                             'SMA', 'EMA', 'RSI', 'BB_Width', 'Volatility',
                             'Volume_Change', 'Daily_Return']
        
        # Créer le tableau 3D pour l'environnement RL: (n_features, n_stocks, n_time_periods)
        # D'abord, déterminer la période commune à tous les tickers
        common_dates = None
        for ticker, df in features_dict.items():
            dates = df.index
            if common_dates is None:
                common_dates = set(dates)
            else:
                common_dates = common_dates.intersection(set(dates))
        
        common_dates = sorted(common_dates)
        
        # Limiter à la fenêtre temporelle spécifiée
        if len(common_dates) > lookback_window:
            common_dates = common_dates[-lookback_window:]
        
        # Initialiser le tableau 3D
        n_features = len(selected_features)
        n_stocks = len(tickers)
        n_time_periods = len(common_dates)
        
        feature_array = np.zeros((n_features, n_stocks, n_time_periods))
        
        # Remplir le tableau
        for i, feature in enumerate(selected_features):
            for j, ticker in enumerate(tickers):
                if ticker in features_dict:
                    df = features_dict[ticker]
                    # Filtrer par dates communes et sélectionner la feature
                    if feature in df.columns:
                        values = df.loc[common_dates, feature].values
                        if len(values) == n_time_periods:
                            feature_array[i, j, :] = values
        
        logger.info(f"Features préparées avec forme: {feature_array.shape}")
        
        # Sauvegarder les features préparées pour le modèle
        feature_store.save_feature(
            name="model_input",
            data=feature_array,
            entity_type="model",
            version="latest",
            description="Features d'entrée pour le modèle RL",
            parameters={
                "tickers": tickers,
                "lookback_window": lookback_window,
                "features": selected_features
            }
        )
        
        return feature_array
    
    except Exception as e:
        logger.error(f"Erreur lors de la préparation des features pour le modèle: {e}")
        return None

if __name__ == "__main__":
    # Exemple d'utilisation
    config_path = "config/default.json"
    
    # Générer les features pour toutes les actions
    features_dict = generate_stock_features(config_path=config_path)
    
    # Préparer les features pour le modèle
    tickers = ["AAPL", "MSFT", "GOOGL", "AMZN", "META"]
    model_features = prepare_model_features(tickers, config_path=config_path)
    
    logger.info("Traitement terminé")