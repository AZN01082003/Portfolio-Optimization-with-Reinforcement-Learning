"""
DAG pour le ré-entraînement périodique du modèle.
"""
from datetime import datetime, timedelta
import os
import json
import pendulum
from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.sensors.filesystem import FileSensor
from airflow.providers.kafka.operators.produce import ProduceToTopicOperator
from airflow.models import Variable
from airflow.operators.bash import BashOperator

# Définition des arguments par défaut
default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'email_on_failure': True,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

# Fonction pour entraîner le modèle
def train_model(**kwargs):
    """
    Exécute l'entraînement du modèle avec MLflow.
    """
    import sys
    sys.path.append('/opt/airflow')
    
    from models.training_pipeline import train_portfolio_agent
    
    # Exécuter l'entraînement
    model, training_info = train_portfolio_agent(
        config_path='/opt/airflow/config/default.json',
        data_path=None  # Utiliser les dernières données disponibles
    )
    
    if model is not None and training_info is not None:
        return {
            'success': True,
            'model_path': training_info.get('model_path'),
            'best_model_path': training_info.get('best_model_path'),
            'mlflow_run_id': training_info.get('mlflow_run_id'),
            'log_dir': training_info.get('log_dir'),
            'timestamp': datetime.now().isoformat()
        }
    else:
        return {
            'success': False,
            'error': "L'entraînement a échoué",
            'timestamp': datetime.now().isoformat()
        }

# Fonction pour évaluer le modèle
def evaluate_model(**kwargs):
    """
    Évalue le modèle entraîné.
    """
    import sys
    sys.path.append('/opt/airflow')
    
    from src.models.evaluate import evaluate_portfolio_model
    
    # Récupérer les informations d'entraînement
    training_info = kwargs['ti'].xcom_pull(task_ids='train_model')
    
    if not training_info or not training_info.get('success', False):
        return {
            'success': False,
            'error': "L'entraînement a échoué, impossible d'évaluer le modèle",
            'timestamp': datetime.now().isoformat()
        }
    
    # Sélectionner le meilleur modèle si disponible
    model_path = training_info.get('best_model_path') or training_info.get('model_path')
    
    if not model_path:
        return {
            'success': False,
            'error': "Chemin du modèle non disponible",
            'timestamp': datetime.now().isoformat()
        }
    
    # Exécuter l'évaluation
    evaluation_results = evaluate_portfolio_model(
        model_path=model_path,
        config_path='/opt/airflow/config/default.json',
        data_path=None,  # Utiliser les dernières données de test disponibles
        mlflow_run_id=training_info.get('mlflow_run_id')
    )
    
    if evaluation_results is not None:
        return {
            'success': True,
            'model_path': model_path,
            'final_value': float(evaluation_results.get('final_value', 0)),
            'total_return': float(evaluation_results.get('total_return', 0)),
            'sharpe_ratio': float(evaluation_results.get('sharpe_ratio', 0)),
            'max_drawdown': float(evaluation_results.get('max_drawdown', 0)),
            'benchmark_return': float(evaluation_results.get('benchmark_return', 0)),
            'outperformance': float(evaluation_results.get('outperformance', 0)),
            'results_dir': evaluation_results.get('results_dir'),
            'mlflow_run_id': evaluation_results.get('mlflow_run_id'),
            'timestamp': datetime.now().isoformat()
        }
    else:
        return {
            'success': False,
            'error': "L'évaluation a échoué",
            'timestamp': datetime.now().isoformat()
        }

# Définition du DAG
with DAG(
    'retrain_model_weekly',
    default_args=default_args,
    description='Ré-entraînement hebdomadaire du modèle',
    schedule_interval='0 2 * * 1',  # Tous les lundis à 2h du matin
    start_date=pendulum.datetime(2025, 4, 1, tz="UTC"),
    catchup=False,
    tags=['portfolio', 'model', 'training'],
) as dag:
    
    # Tâche pour vérifier l'existence des données prétraitées
    check_processed_data = FileSensor(
        task_id='check_processed_data_exists',
        filepath='/opt/airflow/data/processed/stock_data_train_latest.npy',
        poke_interval=60,  # Vérifier toutes les 60 secondes
        timeout=600,  # Timeout après 10 minutes
        mode='poke'
    )
    
    # Tâche pour entraîner le modèle
    train_model_task = PythonOperator(
        task_id='train_model',
        python_callable=train_model,
    )
    
    # Tâche pour évaluer le modèle
    evaluate_model_task = PythonOperator(
        task_id='evaluate_model',
        python_callable=evaluate_model,
    )
    
    # Tâche pour envoyer les métriques d'évaluation à Kafka
    send_metrics_to_kafka = ProduceToTopicOperator(
        task_id='send_evaluation_metrics_to_kafka',
        kafka_config_id='kafka_default',
        topic='model_evaluation_metrics',
        producer_function=lambda: json.dumps("{{ ti.xcom_pull(task_ids='evaluate_model') }}"),
        poll_timeout=10,
    )
    
    # Tâche pour versionner le modèle avec DVC
    version_model = BashOperator(
        task_id='version_model_with_dvc',
        bash_command='''
        cd /opt/airflow && \
        dvc add models/*.zip && \
        dvc push
        ''',
    )
    
    # Définition du flux de tâches
    check_processed_data >> train_model_task >> evaluate_model_task >> send_metrics_to_kafka >> version_model