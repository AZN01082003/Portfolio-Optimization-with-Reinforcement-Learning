"""
DAG pour le prétraitement des données financières.
"""
from datetime import datetime, timedelta
import os
import json
import pendulum
from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.operators.bash import BashOperator
from airflow.sensors.filesystem import FileSensor
from airflow.providers.kafka.operators.produce import ProduceToTopicOperator
from airflow.models import Variable

# Définition des arguments par défaut
default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

# Fonction pour prétraiter les données
def preprocess_stock_data(**kwargs):
    """
    Exécute le prétraitement des données financières.
    """
    import sys
    sys.path.append('/opt/airflow')
    
    from src.data.preprocessing import main as preprocess_main
    
    # Exécuter le prétraitement
    normalized_data, train_data, test_data = preprocess_main(
        config_path='/opt/airflow/config/default.json',
        use_existing_data=True
    )
    
    # Récupérer le timestamp actuel pour les noms de fichiers
    timestamp = datetime.now().strftime("%Y%m%d")
    
    # Chemins des fichiers générés
    processed_data_paths = {
        'normalized_data': f'/opt/airflow/data/processed/stock_data_normalized_{timestamp}.npy',
        'train_data': f'/opt/airflow/data/processed/stock_data_train_{timestamp}.npy',
        'test_data': f'/opt/airflow/data/processed/stock_data_test_{timestamp}.npy',
        'timestamp': datetime.now().isoformat(),
        'normalized_shape': normalized_data.shape if normalized_data is not None else None,
        'train_shape': train_data.shape if train_data is not None else None,
        'test_shape': test_data.shape if test_data is not None else None
    }
    
    return processed_data_paths

# Définition du DAG
with DAG(
    'data_preprocessing_daily',
    default_args=default_args,
    description='Prétraitement quotidien des données financières',
    schedule_interval='30 1 * * 1-5',  # Jours ouvrables à 1h30 du matin (après l'ingestion)
    start_date=pendulum.datetime(2025, 4, 1, tz="UTC"),
    catchup=False,
    tags=['portfolio', 'data', 'financial', 'preprocessing'],
) as dag:
    
    # Tâche pour vérifier l'existence des données brutes
    check_raw_data = FileSensor(
        task_id='check_raw_data_exists',
        filepath='/opt/airflow/data/raw/stock_data_*.csv',
        poke_interval=60,  # Vérifier toutes les 60 secondes
        timeout=600,  # Timeout après 10 minutes
        mode='poke'
    )
    
    # Tâche pour prétraiter les données
    preprocess_data = PythonOperator(
        task_id='preprocess_stock_data',
        python_callable=preprocess_stock_data,
    )
    
    # Tâche pour envoyer les métadonnées à Kafka
    send_metadata_to_kafka = ProduceToTopicOperator(
        task_id='send_processed_metadata_to_kafka',
        kafka_config_id='kafka_default',
        topic='processed_data_metadata',
        producer_function=lambda: json.dumps("{{ ti.xcom_pull(task_ids='preprocess_stock_data') }}"),
        poll_timeout=10,
    )
    
    # Tâche pour versionner les données avec DVC
    version_data = BashOperator(
        task_id='version_processed_data_with_dvc',
        bash_command='''
        cd /opt/airflow && \
        dvc add data/processed/stock_data_*.npy && \
        dvc push
        ''',
    )
    
    # Définition du flux de tâches
    check_raw_data >> preprocess_data >> send_metadata_to_kafka >> version_data