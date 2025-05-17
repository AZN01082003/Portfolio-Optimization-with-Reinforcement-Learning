"""
Module pour produire des messages Kafka liés aux données financières.
"""
import os
import json
import time
import logging
import json
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
from confluent_kafka import Producer
import socket
import yfinance as yf

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

def delivery_report(err, msg):
    """
    Fonction de rappel pour les accusés de réception des messages.
    """
    if err is not None:
        logger.error(f"Échec de livraison du message: {err}")
    else:
        logger.debug(f"Message livré à {msg.topic()} [{msg.partition()}] à l'offset {msg.offset()}")

def create_kafka_producer(config_path="config/default.json"):
    """
    Crée et configure un producteur Kafka.
    
    Returns:
        Producer: Instance du producteur Kafka
    """
    config = load_config(config_path)
    kafka_config = config.get('kafka', {})
    
    # Configuration du producteur
    producer_config = {
        'bootstrap.servers': kafka_config.get('bootstrap_servers', 'localhost:9092'),
        'client.id': socket.gethostname(),
        'acks': 'all',  # Attendre confirmation de tous les réplicas
        'retries': 3,    # Nombre de tentatives en cas d'échec
        'retry.backoff.ms': 500  # Délai entre les tentatives
    }
    
    return Producer(producer_config)

def fetch_real_time_data(tickers, interval='1m', config_path="config/default.json"):
    """
    Récupère les données en temps réel de Yahoo Finance et les envoie à Kafka.
    
    Args:
        tickers (list): Liste des tickers à surveiller
        interval (str): Intervalle de temps pour les données ('1m', '5m', etc.)
        config_path (str): Chemin du fichier de configuration
    """
    # Créer le producteur Kafka
    producer = create_kafka_producer(config_path)
    
    # Récupérer la configuration
    config = load_config(config_path)
    topic = config.get('kafka', {}).get('topics', {}).get('real_time_data', 'real_time_stock_data')
    
    logger.info(f"Démarrage du flux de données en temps réel pour {tickers} avec intervalle {interval}")
    
    try:
        # Récupérer les données historiques récentes pour initialiser
        end_time = datetime.now()
        start_time = end_time - timedelta(days=1)  # Données des dernières 24h
        
        # Pour les données de démonstration, utilisons des données historiques
        historical_data = yf.download(tickers, start=start_time, end=end_time, interval=interval)
        
        # S'il s'agit d'un seul ticker, assurons-nous que les données ont le bon format
        if len(tickers) == 1:
            historical_data = historical_data.reset_index()
            historical_data['Ticker'] = tickers[0]
        else:
            # Convertir MultiIndex en format plat
            historical_data = historical_data.stack(level=1).reset_index()
            historical_data.rename(columns={'level_1': 'Ticker'}, inplace=True)
        
        # Envoyer les données historiques une par une pour simuler un flux en temps réel
        for idx, row in historical_data.iterrows():
            # Créer le message
            timestamp = row.get('Datetime', row.get('Date', datetime.now())).isoformat()
            
            message = {
                'ticker': row['Ticker'],
                'timestamp': timestamp,
                'open': float(row['Open']),
                'high': float(row['High']),
                'low': float(row['Low']),
                'close': float(row['Close']),
                'volume': float(row['Volume']),
                'event_time': datetime.now().isoformat()
            }
            
            # Convertir en JSON et envoyer
            message_json = json.dumps(message)
            producer.produce(topic, key=row['Ticker'], value=message_json, callback=delivery_report)
            
            # Forcer l'envoi des messages
            producer.flush()
            
            # Attendre un peu pour simuler un flux de données
            time.sleep(0.5)
            
            # Log périodique
            if idx % 10 == 0:
                logger.info(f"Envoyé {idx+1}/{len(historical_data)} points de données à Kafka")
        
        logger.info("Flux de données terminé. Toutes les données ont été envoyées à Kafka.")
        
    except Exception as e:
        logger.error(f"Erreur lors de la récupération ou de l'envoi des données: {e}")
    finally:
        # Assurer que tous les messages sont envoyés
        producer.flush()

def send_batch_data_to_kafka(data, topic, config_path="config/default.json"):
    """
    Envoie un lot de données à un topic Kafka.
    
    Args:
        data: Données à envoyer (DataFrame ou dict)
        topic (str): Nom du topic Kafka
        config_path (str): Chemin du fichier de configuration
    """
    producer = create_kafka_producer(config_path)
    
    try:
        if isinstance(data, pd.DataFrame):
            # Pour un DataFrame, envoyer chaque ligne comme un message
            for idx, row in data.iterrows():
                # Convertir la ligne en dictionnaire
                message = row.to_dict()
                
                # Ajouter un timestamp si non présent
                if 'timestamp' not in message:
                    message['timestamp'] = datetime.now().isoformat()
                
                message_json = json.dumps(message)
                producer.produce(topic, value=message_json, callback=delivery_report)
                
                # Forcer un envoi périodique
                if idx % 100 == 0:
                    producer.flush()
                    
            logger.info(f"Envoyé {len(data)} messages à {topic}")
        
        elif isinstance(data, dict) or isinstance(data, list):
            # Pour un dictionnaire ou une liste, envoyer directement
            message_json = json.dumps(data)
            producer.produce(topic, value=message_json, callback=delivery_report)
            logger.info(f"Envoyé message à {topic}")
        
        else:
            logger.error(f"Type de données non supporté: {type(data)}")
        
        # S'assurer que tous les messages sont envoyés
        producer.flush()
        
    except Exception as e:
        logger.error(f"Erreur lors de l'envoi des données à Kafka: {e}")
    finally:
        producer.flush()

if __name__ == "__main__":
    # Exemple d'utilisation
    config_path = "config/default.json"
    tickers = ["AAPL", "MSFT", "GOOGL", "AMZN", "META"]
    
    # Simuler un flux de données en temps réel
    fetch_real_time_data(tickers, interval='1m', config_path=config_path)