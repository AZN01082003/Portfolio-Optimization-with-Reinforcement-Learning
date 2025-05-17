"""
Module pour le traitement par lots des données stockées dans Kafka.
"""
import os
import json
import logging
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from confluent_kafka import Consumer, KafkaError

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

def create_kafka_consumer(topic, group_id, config_path="config/default.json"):
    """
    Crée et configure un consommateur Kafka.
    
    Args:
        topic (str): Topic à écouter
        group_id (str): ID du groupe de consommateurs
        config_path (str): Chemin du fichier de configuration
        
    Returns:
        Consumer: Instance du consommateur Kafka
    """
    config = load_config(config_path)
    kafka_config = config.get('kafka', {})
    
    # Configuration du consommateur
    consumer_config = {
        'bootstrap.servers': kafka_config.get('bootstrap_servers', 'localhost:9092'),
        'group.id': group_id,
        'auto.offset.reset': 'earliest',
        'enable.auto.commit': False  # Désactiver le commit automatique pour le traitement par lots
    }
    
    consumer = Consumer(consumer_config)
    consumer.subscribe([topic])
    
    return consumer

def process_batch_data(topic, group_id, batch_size=1000, timeout=60, config_path="config/default.json"):
    """
    Traite un lot de messages du topic Kafka.
    
    Args:
        topic (str): Topic à consommer
        group_id (str): ID du groupe de consommateurs
        batch_size (int): Taille du lot à traiter
        timeout (int): Timeout en secondes pour chaque poll
        config_path (str): Chemin du fichier de configuration
        
    Returns:
        pandas.DataFrame: DataFrame des messages traités
    """
    consumer = create_kafka_consumer(topic, group_id, config_path)
    messages = []
    message_count = 0
    
    try:
        logger.info(f"Démarrage du traitement par lots pour {topic}, taille du lot: {batch_size}")
        
        start_time = datetime.now()
        time_limit = start_time + timedelta(seconds=timeout)
        
        while message_count < batch_size and datetime.now() < time_limit:
            msg = consumer.poll(1.0)  # timeout de 1 seconde par poll
            
            if msg is None:
                continue
            
            if msg.error():
                if msg.error().code() == KafkaError._PARTITION_EOF:
                    logger.debug(f"Atteinte de la fin d'une partition pour {topic}")
                else:
                    logger.error(f"Erreur Kafka: {msg.error()}")
            else:
                try:
                    # Décoder et parser le message
                    data = json.loads(msg.value().decode('utf-8'))
                    messages.append(data)
                    message_count += 1
                    
                    # Log périodique
                    if message_count % 100 == 0:
                        logger.info(f"Traité {message_count}/{batch_size} messages")
                
                except json.JSONDecodeError:
                    logger.error(f"Erreur de décodage JSON: {msg.value()}")
                except Exception as e:
                    logger.error(f"Erreur lors du traitement du message: {e}")
        
        # Commit manuel des offsets
        consumer.commit()
        
        logger.info(f"Traitement par lots terminé. {message_count} messages traités en {(datetime.now() - start_time).total_seconds():.2f} secondes")
        
        # Convertir en DataFrame
        if messages:
            df = pd.DataFrame(messages)
            return df
        else:
            logger.warning(f"Aucun message traité pour {topic}")
            return pd.DataFrame()
    
    except Exception as e:
        logger.error(f"Erreur lors du traitement par lots: {e}")
        return pd.DataFrame()
    finally:
        consumer.close()

def aggregate_stock_data(df):
    """
    Agrège les données d'actions par ticker et période.
    
    Args:
        df (pandas.DataFrame): DataFrame des données brutes
        
    Returns:
        pandas.DataFrame: DataFrame des données agrégées
    """
    if df.empty:
        return df
    
    try:
        # Assurons-nous que les colonnes requises sont présentes
        required_columns = ['ticker', 'timestamp', 'open', 'high', 'low', 'close', 'volume']
        for col in required_columns:
            if col not in df.columns:
                logger.error(f"Colonne requise manquante: {col}")
                return df
        
        # Convertir timestamp en datetime
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        # Extraire la date et l'heure
        df['date'] = df['timestamp'].dt.date
        df['hour'] = df['timestamp'].dt.hour
        
        # Agréger par ticker, date et heure
        aggregated = df.groupby(['ticker', 'date', 'hour']).agg({
            'open': 'first',
            'high': 'max',
            'low': 'min',
            'close': 'last',
            'volume': 'sum'
        }).reset_index()
        
        logger.info(f"Données agrégées: {len(df)} entrées réduites à {len(aggregated)} entrées")
        
        return aggregated
    
    except Exception as e:
        logger.error(f"Erreur lors de l'agrégation des données: {e}")
        return df

def save_processed_batch(df, output_dir="data/processed", prefix="batch_data", config_path="config/default.json"):
    """
    Sauvegarde un lot de données traitées.
    
    Args:
        df (pandas.DataFrame): DataFrame des données traitées
        output_dir (str): Répertoire de sortie
        prefix (str): Préfixe pour le nom du fichier
        config_path (str): Chemin du fichier de configuration
        
    Returns:
        str: Chemin du fichier sauvegardé ou None en cas d'erreur
    """
    if df.empty:
        logger.warning("DataFrame vide, aucune sauvegarde effectuée")
        return None
    
    try:
        # Créer le répertoire si nécessaire
        os.makedirs(output_dir, exist_ok=True)
        
        # Générer un nom de fichier avec timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{prefix}_{timestamp}.csv"
        filepath = os.path.join(output_dir, filename)
        
        # Sauvegarder en CSV
        df.to_csv(filepath, index=False)
        logger.info(f"Données sauvegardées dans {filepath}")
        
        # Versionner avec DVC si configuré
        config = load_config(config_path)
        use_dvc = config.get('data_versioning', {}).get('enabled', False)
        
        if use_dvc:
            from src.utils.dvc_utils import version_data_file
            version_data_file(filepath, message=f"Ajout du lot de données {timestamp}")
        
        return filepath
    
    except Exception as e:
        logger.error(f"Erreur lors de la sauvegarde des données: {e}")
        return None

if __name__ == "__main__":
    # Exemple d'utilisation
    config_path = "config/default.json"
    topic = "real_time_stock_data"
    group_id = "batch_processor_group"
    
    # Traiter un lot de données
    df = process_batch_data(topic, group_id, batch_size=500, timeout=120, config_path=config_path)
    
    if not df.empty:
        # Agréger les données
        aggregated_df = aggregate_stock_data(df)
        
        # Sauvegarder les données
        save_processed_batch(aggregated_df, prefix="stock_data_aggregated", config_path=config_path)