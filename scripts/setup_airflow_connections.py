"""
Script pour configurer les connexions Airflow.
"""
import os
import sys
import json
import logging
from airflow.models import Connection
from airflow.utils.db import create_session

# Configuration du logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def load_config(config_path):
    """
    Charge la configuration depuis un fichier json.
    """
    with open(config_path, 'r') as file:
        config = json.load(file)
    return config

def setup_kafka_connection():
    """
    Configure la connexion Kafka dans Airflow.
    """
    try:
        config_path = os.path.join(os.path.dirname(__file__), "..", "config", "default.json")
        config = load_config(config_path)
        
        # Récupérer les paramètres Kafka
        kafka_config = config.get('kafka', {})
        bootstrap_servers = kafka_config.get('bootstrap_servers', 'kafka:9092')
        
        # Créer la connexion
        with create_session() as session:
            # Vérifier si la connexion existe déjà
            existing_conn = session.query(Connection).filter(Connection.conn_id == 'kafka_default').first()
            
            if existing_conn:
                logger.info("La connexion Kafka existe déjà, mise à jour...")
                existing_conn.host = bootstrap_servers
                existing_conn.conn_type = 'kafka'
                existing_conn.extra = '{"bootstrap.servers": "' + bootstrap_servers + '"}'
            else:
                logger.info("Création d'une nouvelle connexion Kafka...")
                new_conn = Connection(
                    conn_id='kafka_default',
                    conn_type='kafka',
                    host=bootstrap_servers,
                    extra='{"bootstrap.servers": "' + bootstrap_servers + '"}'
                )
                session.add(new_conn)
            
            session.commit()
            logger.info("Connexion Kafka configurée avec succès")
    
    except Exception as e:
        logger.error(f"Erreur lors de la configuration de la connexion Kafka: {e}")

def setup_yahoo_finance_connection():
    """
    Configure la connexion Yahoo Finance dans Airflow.
    """
    try:
        # Créer la connexion
        with create_session() as session:
            # Vérifier si la connexion existe déjà
            existing_conn = session.query(Connection).filter(Connection.conn_id == 'yahoo_finance_api').first()
            
            if existing_conn:
                logger.info("La connexion Yahoo Finance existe déjà, mise à jour...")
                existing_conn.host = 'query1.finance.yahoo.com'
                existing_conn.conn_type = 'http'
                existing_conn.schema = 'https'
            else:
                logger.info("Création d'une nouvelle connexion Yahoo Finance...")
                new_conn = Connection(
                    conn_id='yahoo_finance_api',
                    conn_type='http',
                    host='query1.finance.yahoo.com',
                    schema='https'
                )
                session.add(new_conn)
            
            session.commit()
            logger.info("Connexion Yahoo Finance configurée avec succès")
    
    except Exception as e:
        logger.error(f"Erreur lors de la configuration de la connexion Yahoo Finance: {e}")

if __name__ == "__main__":
    setup_kafka_connection()
    setup_yahoo_finance_connection()
    logger.info("Configuration des connexions Airflow terminée")