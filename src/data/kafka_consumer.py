"""
Module pour consommer les messages Kafka liés aux données financières.
"""
import os
import json
import logging
import threading
import signal
import sys
from datetime import datetime
from confluent_kafka import Consumer, KafkaError, KafkaException

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
    return json

def create_kafka_consumer(topics, group_id, config_path="config/default.json"):
    """
    Crée et configure un consommateur Kafka.
    
    Args:
        topics (list): Liste des topics à écouter
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
        'auto.offset.reset': 'earliest',  # Commencer au début du topic
        'enable.auto.commit': True,       # Commit automatique des offsets
        'auto.commit.interval.ms': 5000,  # Intervalle de commit (5 secondes)
        'max.poll.interval.ms': 300000    # 5 minutes max entre les appels à poll()
    }
    
    consumer = Consumer(consumer_config)
    consumer.subscribe(topics)
    
    return consumer

def process_message(msg_value, topic):
    """
    Traite un message consommé de Kafka.
    
    Args:
        msg_value: Valeur du message (JSON)
        topic: Topic d'origine du message
        
    Returns:
        dict: Données traitées
    """
    try:
        # Convertir le message JSON en dictionnaire
        data = json.loads(msg_value)
        
        # Ajouter des métadonnées
        data['processed_time'] = datetime.now().isoformat()
        data['source_topic'] = topic
        
        # Log informatif
        logger.debug(f"Message traité depuis {topic}: {json.dumps(data)[:100]}...")
        
        return data
    
    except json.JSONDecodeError:
        logger.error(f"Erreur de décodage JSON: {msg_value}")
        return None
    except Exception as e:
        logger.error(f"Erreur lors du traitement du message: {e}")
        return None

def consume_messages(topics, group_id, config_path="config/default.json", callback=None, timeout=300):
    """
    Consomme les messages de Kafka en continu.
    
    Args:
        topics (list): Liste des topics à écouter
        group_id (str): ID du groupe de consommateurs
        config_path (str): Chemin du fichier de configuration
        callback (function): Fonction à appeler pour chaque message (optionnel)
        timeout (int): Timeout en secondes pour chaque poll
    """
    consumer = create_kafka_consumer(topics, group_id, config_path)
    
    # Gérer les signaux pour une fermeture propre
    running = True
    
    def handle_shutdown(signum, frame):
        nonlocal running
        logger.info("Signal d'arrêt reçu, fermeture en cours...")
        running = False
    
    signal.signal(signal.SIGINT, handle_shutdown)
    signal.signal(signal.SIGTERM, handle_shutdown)
    
    try:
        logger.info(f"Démarrage de la consommation des topics: {', '.join(topics)}")
        
        while running:
            msg = consumer.poll(timeout=timeout)
            
            if msg is None:
                continue
            
            if msg.error():
                if msg.error().code() == KafkaError._PARTITION_EOF:
                    # Fin d'une partition, pas une erreur
                    logger.debug(f"Atteinte de la fin d'une partition pour {msg.topic()}")
                elif msg.error().code() == KafkaError.UNKNOWN_TOPIC_OR_PART:
                    logger.warning(f"Topic ou partition inconnu: {msg.topic()}")
                else:
                    logger.error(f"Erreur Kafka: {msg.error()}")
            else:
                # Traiter le message
                processed_data = process_message(msg.value().decode('utf-8'), msg.topic())
                
                # Appeler la fonction de callback si fournie
                if callback and processed_data:
                    callback(processed_data, msg.topic())
    
    except KafkaException as e:
        logger.error(f"Erreur Kafka: {e}")
    except Exception as e:
        logger.error(f"Erreur inattendue: {e}")
    finally:
        # Fermer proprement le consommateur
        consumer.close()
        logger.info("Consommateur Kafka fermé")

def start_consumer_thread(topics, group_id, config_path="config/default.json", callback=None):
    """
    Démarre la consommation Kafka dans un thread séparé.
    
    Args:
        topics (list): Liste des topics à écouter
        group_id (str): ID du groupe de consommateurs
        config_path (str): Chemin du fichier de configuration
        callback (function): Fonction à appeler pour chaque message (optionnel)
        
    Returns:
        Thread: Objet thread pour la consommation
    """
    consumer_thread = threading.Thread(
        target=consume_messages,
        args=(topics, group_id, config_path, callback),
        daemon=True  # Le thread se terminera lorsque le thread principal se terminera
    )
    consumer_thread.start()
    
    logger.info(f"Thread de consommation démarré pour les topics: {', '.join(topics)}")
    
    return consumer_thread

def example_callback(data, topic):
    """
    Exemple de fonction de callback pour le traitement des messages.
    """
    logger.info(f"Nouveau message du topic {topic}: {json.dumps(data)[:100]}...")
    
    # Ici, vous pouvez ajouter votre logique de traitement spécifique
    # Par exemple, stocker les données, déclencher un calcul, etc.

if __name__ == "__main__":
    # Exemple d'utilisation
    config_path = "config/default.json"
    topics = ["real_time_stock_data", "stock_tickers", "model_evaluation_metrics"]
    group_id = "portfolio_consumer_group"
    
    # Commencer à consommer
    consume_messages(topics, group_id, config_path, callback=example_callback)