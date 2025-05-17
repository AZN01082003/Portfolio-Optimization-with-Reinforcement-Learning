"""
DAG pour l'ingestion quotidienne des données financières.
"""
from datetime import datetime, timedelta
import os
import json
import pendulum
from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.operators.bash import BashOperator
from airflow.providers.http.sensors.http import HttpSensor
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

# Fonction pour générer la liste des tickers à récupérer
def get_tickers(**kwargs):
    """
    Récupère la liste des tickers depuis la configuration ou les paramètres Airflow.
    """
    try:
        # Essayer de lire depuis les variables Airflow
        tickers_json = Variable.get("stock_tickers", default_var=None)
        if tickers_json:
            tickers = json.loads(tickers_json)
        else:
            # Sinon, lire depuis le fichier de configuration
            import json
            with open('/opt/airflow/config/default.json', 'r') as file:
                config = json.load(file)
            tickers = config['data']['tickers']
        
        # Convertir en format JSON pour Kafka
        tickers_data = {
            'tickers': tickers,
            'timestamp': datetime.now().isoformat(),
            'source': 'airflow_dag'
        }
        
        return json.dumps(tickers_data)
    except Exception as e:
        print(f"Erreur lors de la récupération des tickers: {e}")
        # Ticker par défaut en cas d'échec
        return json.dumps({
            'tickers': ["AAPL", "MSFT", "GOOGL", "AMZN", "META"],
            'timestamp': datetime.now().isoformat(),
            'source': 'airflow_dag_default'
        })

# Fonction pour exécuter le script de téléchargement
def download_stock_data(**kwargs):
    """
    Exécute le script Python pour télécharger les données des actions.
    """
    import sys
    sys.path.append('/opt/airflow')
    
    from src.data.ingestion import main as ingest_data
    
    # Récupérer les tickers depuis XCom
    tickers_data = json.loads(kwargs['ti'].xcom_pull(task_ids='get_stock_tickers'))
    tickers = tickers_data['tickers']
    
    # Configuration des dates
    end_date = datetime.now()
    start_date = end_date - timedelta(days=5)  # 5 jours de données
    
    # Exécuter l'ingestion
    data, _, _ = ingest_data(
        tickers=tickers,
        start_date=start_date,
        end_date=end_date,
        config_path='/opt/airflow/config/default.json'
    )
    
    # Enregistrer le chemin du fichier pour les étapes suivantes
    raw_data_path = f'/opt/airflow/data/raw/stock_data_{end_date.strftime("%Y%m%d")}.csv'
    
    # Retourner des informations sur les données téléchargées
    return {
        'raw_data_path': raw_data_path,
        'num_tickers': len(tickers),
        'start_date': start_date.isoformat(),
        'end_date': end_date.isoformat(),
        'timestamp': datetime.now().isoformat()
    }

# Définition du DAG
with DAG(
    'data_ingestion_daily',
    default_args=default_args,
    description='Ingestion quotidienne des données financières',
    schedule_interval='0 1 * * 1-5',  # Jours ouvrables à 1h du matin
    start_date=pendulum.datetime(2025, 4, 1, tz="UTC"),
    catchup=False,
    tags=['portfolio', 'data', 'financial'],
) as dag:
    
    # Tâche pour vérifier que le service Yahoo Finance est disponible
    check_api = HttpSensor(
        task_id='check_yahoo_finance_api',
        http_conn_id='yahoo_finance_api',
        endpoint='',
        request_params={},
        response_check=lambda response: response.status_code == 200,
        poke_interval=60,  # Vérifier toutes les 60 secondes
        timeout=300,  # Timeout après 5 minutes
    )
    
    # Tâche pour récupérer la liste des tickers
    get_tickers_task = PythonOperator(
        task_id='get_stock_tickers',
        python_callable=get_tickers,
    )
    
    # Tâche pour envoyer la liste des tickers à Kafka
    send_tickers_to_kafka = ProduceToTopicOperator(
        task_id='send_tickers_to_kafka',
        kafka_config_id='kafka_default',
        topic='stock_tickers',
        producer_function=lambda: "{{ ti.xcom_pull(task_ids='get_stock_tickers') }}",
        poll_timeout=10,
    )
    
    # Tâche pour télécharger les données
    download_data = PythonOperator(
        task_id='download_stock_data',
        python_callable=download_stock_data,
    )
    
    # Tâche pour envoyer les métadonnées à Kafka
    send_metadata_to_kafka = ProduceToTopicOperator(
        task_id='send_metadata_to_kafka',
        kafka_config_id='kafka_default',
        topic='stock_data_metadata',
        producer_function=lambda: json.dumps("{{ ti.xcom_pull(task_ids='download_stock_data') }}"),
        poll_timeout=10,
    )
    
    # Tâche pour versionner les données avec DVC
    version_data = BashOperator(
        task_id='version_data_with_dvc',
        bash_command='''
        cd /opt/airflow && \
        dvc add data/raw/stock_data_*.csv && \
        dvc push
        ''',
    )
    
    # Définition du flux de tâches
    check_api >> get_tickers_task >> send_tickers_to_kafka >> download_data >> send_metadata_to_kafka >> version_data