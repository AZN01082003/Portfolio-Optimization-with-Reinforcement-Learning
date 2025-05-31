"""
Pipeline MLOps complet pour Portfolio RL avec Airflow.
Orchestre l'ingestion, preprocessing, entraînement, évaluation et déploiement.
"""
import os
import json
import logging
import subprocess
import sys
from datetime import datetime, timedelta
from typing import Dict, Any

from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.operators.http import SimpleHttpOperator
from airflow.operators.email import EmailOperator
from airflow.operators.bash import BashOperator
from airflow.sensors.http_sensor import HttpSensor
from airflow.sensors.filesystem import FileSensor
from airflow.utils.trigger_rule import TriggerRule

# Configuration
logger = logging.getLogger(__name__)

# Arguments par défaut
default_args = {
    'owner': 'mlops-team',
    'depends_on_past': False,
    'start_date': datetime(2025, 1, 1),
    'email_on_failure': True,
    'email_on_retry': False,
    'retries': 2,
    'retry_delay': timedelta(minutes=5),
    'max_active_runs': 1,
    'email': ['admin@portfolio-rl.com']  # À adapter
}

# =============================================================================
# FONCTIONS DES TÂCHES
# =============================================================================

def check_data_quality(**context) -> bool:
    """Vérifie la qualité des données avant traitement."""
    import numpy as np
    import pandas as pd
    from pathlib import Path
    
    logger.info("🔍 Vérification qualité des données")
    
    # Vérifier les fichiers de données brutes
    data_dir = Path("/opt/airflow/data/raw")
    
    # Trouver le fichier le plus récent
    data_files = list(data_dir.glob("stock_data_*.csv"))
    
    if not data_files:
        logger.error("❌ Aucun fichier de données trouvé")
        return False
    
    latest_file = max(data_files, key=lambda f: f.stat().st_mtime)
    logger.info(f"📂 Fichier le plus récent: {latest_file}")
    
    try:
        # Charger et vérifier
        df = pd.read_csv(latest_file, header=[0, 1], index_col=0)
        
        # Vérifications de base
        if df.empty:
            logger.error("❌ DataFrame vide")
            return False
        
        if df.isnull().sum().sum() > len(df) * 0.1:  # Plus de 10% de NaN
            logger.error("❌ Trop de valeurs manquantes")
            return False
        
        # Vérifier les colonnes attendues
        expected_features = ['Open', 'High', 'Low', 'Close', 'Volume']
        available_features = [col[0] for col in df.columns.get_level_values(0).unique()]
        
        missing_features = set(expected_features) - set(available_features)
        if missing_features:
            logger.error(f"❌ Features manquantes: {missing_features}")
            return False
        
        logger.info(f"✅ Qualité données OK: {df.shape}")
        
        # Passer les métadonnées au contexte
        context['task_instance'].xcom_push(
            key='data_info',
            value={
                'file_path': str(latest_file),
                'shape': list(df.shape),
                'features': available_features,
                'date_range': [str(df.index.min()), str(df.index.max())],
                'quality_score': 1.0 - (df.isnull().sum().sum() / df.size)
            }
        )
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Erreur vérification: {e}")
        return False

def run_data_ingestion(**context) -> str:
    """Exécute l'ingestion de données."""
    logger.info("📥 Démarrage ingestion de données")
    
    try:
        # Exécuter le script d'ingestion
        result = subprocess.run([
            sys.executable, "/opt/airflow/src/data/ingestion.py",
            "--config", "/opt/airflow/config/default.json"
        ], capture_output=True, text=True, timeout=600)
        
        if result.returncode == 0:
            logger.info("✅ Ingestion réussie")
            
            # Extraire les informations du stdout
            output_lines = result.stdout.split('\n')
            data_info = {}
            
            for line in output_lines:
                if "Données téléchargées:" in line:
                    data_info['downloaded'] = True
                elif "Fichier sauvegardé:" in line:
                    data_info['file_path'] = line.split(":")[-1].strip()
            
            # Passer au contexte
            context['task_instance'].xcom_push(key='ingestion_result', value=data_info)
            
            return "success"
        else:
            logger.error(f"❌ Erreur ingestion: {result.stderr}")
            raise Exception(f"Ingestion failed: {result.stderr}")
            
    except subprocess.TimeoutExpired:
        logger.error("❌ Timeout ingestion")
        raise Exception("Ingestion timeout")
    except Exception as e:
        logger.error(f"❌ Erreur ingestion: {e}")
        raise

def run_preprocessing(**context) -> str:
    """Exécute le preprocessing des données."""
    logger.info("🔄 Démarrage preprocessing")
    
    # Récupérer les infos de l'ingestion
    ingestion_info = context['task_instance'].xcom_pull(
        task_ids='data_ingestion', 
        key='ingestion_result'
    )
    
    logger.info(f"📊 Infos ingestion: {ingestion_info}")
    
    try:
        result = subprocess.run([
            sys.executable, "/opt/airflow/src/data/preprocessing.py",
            "--config", "/opt/airflow/config/default.json"
        ], capture_output=True, text=True, timeout=900)
        
        if result.returncode == 0:
            logger.info("✅ Preprocessing réussi")
            
            # Extraire les informations de sortie
            preprocessing_info = {
                'status': 'success',
                'processed_file': 'data/processed/features.npy',
                'labels_file': 'data/processed/labels.npy'
            }
            
            context['task_instance'].xcom_push(
                key='preprocessing_result', 
                value=preprocessing_info
            )
            
            return "success"
        else:
            logger.error(f"❌ Erreur preprocessing: {result.stderr}")
            raise Exception(f"Preprocessing failed: {result.stderr}")
            
    except subprocess.TimeoutExpired:
        logger.error("❌ Timeout preprocessing")
        raise Exception("Preprocessing timeout")
    except Exception as e:
        logger.error(f"❌ Erreur preprocessing: {e}")
        raise

def run_training(**context) -> str:
    """Exécute l'entraînement du modèle RL."""
    logger.info("🤖 Démarrage entraînement du modèle RL")
    
    # Récupérer les infos du preprocessing
    preprocessing_info = context['task_instance'].xcom_pull(
        task_ids='data_preprocessing',
        key='preprocessing_result'
    )
    
    logger.info(f"📊 Infos preprocessing: {preprocessing_info}")
    
    try:
        # Paramètres d'entraînement
        training_params = [
            "--config", "/opt/airflow/config/default.json",
            "--experiment-name", f"training_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            "--log-mlflow", "true"
        ]
        
        result = subprocess.run([
            sys.executable, "/opt/airflow/src/models/train.py"
        ] + training_params, capture_output=True, text=True, timeout=3600)
        
        if result.returncode == 0:
            logger.info("✅ Entraînement réussi")
            
            # Extraire les métriques d'entraînement
            training_info = {
                'status': 'success',
                'model_path': f"models/portfolio_model_{datetime.now().strftime('%Y%m%d_%H%M%S')}.zip",
                'experiment_name': f"training_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                'training_duration': 'extracted_from_logs'
            }
            
            # Parser les métriques du stdout
            output_lines = result.stdout.split('\n')
            for line in output_lines:
                if "Final reward:" in line:
                    training_info['final_reward'] = float(line.split(":")[-1].strip())
                elif "Model saved:" in line:
                    training_info['model_path'] = line.split(":")[-1].strip()
                elif "Training time:" in line:
                    training_info['training_duration'] = line.split(":")[-1].strip()
            
            context['task_instance'].xcom_push(
                key='training_result',
                value=training_info
            )
            
            return "success"
        else:
            logger.error(f"❌ Erreur entraînement: {result.stderr}")
            raise Exception(f"Training failed: {result.stderr}")
            
    except subprocess.TimeoutExpired:
        logger.error("❌ Timeout entraînement (1h)")
        raise Exception("Training timeout")
    except Exception as e:
        logger.error(f"❌ Erreur entraînement: {e}")
        raise

def run_evaluation(**context) -> str:
    """Exécute l'évaluation du modèle entraîné."""
    logger.info("📊 Démarrage évaluation du modèle")
    
    # Récupérer les infos d'entraînement
    training_info = context['task_instance'].xcom_pull(
        task_ids='model_training',
        key='training_result'
    )
    
    logger.info(f"🤖 Infos entraînement: {training_info}")
    
    try:
        model_path = training_info.get('model_path', 'models/latest_model.zip')
        
        evaluation_params = [
            "--config", "/opt/airflow/config/default.json",
            "--model-path", model_path,
            "--evaluation-episodes", "100",
            "--save-results", "true"
        ]
        
        result = subprocess.run([
            sys.executable, "/opt/airflow/src/models/evaluate.py"
        ] + evaluation_params, capture_output=True, text=True, timeout=1800)
        
        if result.returncode == 0:
            logger.info("✅ Évaluation réussie")
            
            # Parser les métriques d'évaluation
            evaluation_metrics = {
                'status': 'success',
                'mean_reward': 0.0,
                'std_reward': 0.0,
                'sharpe_ratio': 0.0,
                'max_drawdown': 0.0,
                'total_return': 0.0,
                'win_rate': 0.0
            }
            
            output_lines = result.stdout.split('\n')
            for line in output_lines:
                if "Mean reward:" in line:
                    evaluation_metrics['mean_reward'] = float(line.split(":")[-1].strip())
                elif "Std reward:" in line:
                    evaluation_metrics['std_reward'] = float(line.split(":")[-1].strip())
                elif "Sharpe ratio:" in line:
                    evaluation_metrics['sharpe_ratio'] = float(line.split(":")[-1].strip())
                elif "Max drawdown:" in line:
                    evaluation_metrics['max_drawdown'] = float(line.split(":")[-1].strip())
                elif "Total return:" in line:
                    evaluation_metrics['total_return'] = float(line.split(":")[-1].strip())
                elif "Win rate:" in line:
                    evaluation_metrics['win_rate'] = float(line.split(":")[-1].strip())
            
            # Vérifier les seuils de qualité
            quality_check = {
                'reward_threshold': evaluation_metrics['mean_reward'] > 0.1,
                'return_threshold': evaluation_metrics['total_return'] > 5.0,
                'sharpe_threshold': evaluation_metrics['sharpe_ratio'] > 0.5,
                'drawdown_threshold': evaluation_metrics['max_drawdown'] < 0.2
            }
            
            evaluation_metrics['quality_passed'] = all(quality_check.values())
            evaluation_metrics['quality_checks'] = quality_check
            
            context['task_instance'].xcom_push(
                key='evaluation_result',
                value=evaluation_metrics
            )
            
            # Décider si le modèle est bon pour le déploiement
            if evaluation_metrics['quality_passed']:
                logger.info("✅ Modèle validé pour déploiement")
                return "deploy"
            else:
                logger.warning("⚠️ Modèle ne passe pas les critères de qualité")
                return "retrain"
            
        else:
            logger.error(f"❌ Erreur évaluation: {result.stderr}")
            raise Exception(f"Evaluation failed: {result.stderr}")
            
    except subprocess.TimeoutExpired:
        logger.error("❌ Timeout évaluation (30min)")
        raise Exception("Evaluation timeout")
    except Exception as e:
        logger.error(f"❌ Erreur évaluation: {e}")
        raise

def run_model_deployment(**context) -> str:
    """Déploie le modèle validé."""
    logger.info("🚀 Démarrage déploiement du modèle")
    
    # Récupérer les infos d'évaluation
    evaluation_info = context['task_instance'].xcom_pull(
        task_ids='model_evaluation',
        key='evaluation_result'
    )
    
    training_info = context['task_instance'].xcom_pull(
        task_ids='model_training',
        key='training_result'
    )
    
    logger.info(f"📊 Métriques d'évaluation: {evaluation_info}")
    
    try:
        if not evaluation_info.get('quality_passed', False):
            logger.warning("⚠️ Modèle non validé - arrêt du déploiement")
            return "skipped"
        
        model_path = training_info.get('model_path', 'models/latest_model.zip')
        
        # 1. Sauvegarder le modèle dans le registre MLflow
        deployment_params = [
            "--model-path", model_path,
            "--model-name", "portfolio_rl_model",
            "--stage", "Production",
            "--mlflow-uri", "http://mlflow:5000"
        ]
        
        result = subprocess.run([
            sys.executable, "/opt/airflow/src/models/mlflow_manager.py",
            "register-model"
        ] + deployment_params, capture_output=True, text=True, timeout=300)
        
        if result.returncode != 0:
            logger.error(f"❌ Erreur enregistrement MLflow: {result.stderr}")
            raise Exception("MLflow registration failed")
        
        # 2. Déployer sur l'API
        api_deployment_params = [
            "--model-name", "portfolio_rl_model",
            "--api-endpoint", "http://portfolio-api:8000/models/deploy",
            "--wait-ready", "true"
        ]
        
        result = subprocess.run([
            sys.executable, "/opt/airflow/src/api/deploy_model.py"
        ] + api_deployment_params, capture_output=True, text=True, timeout=300)
        
        if result.returncode == 0:
            logger.info("✅ Déploiement réussi")
            
            deployment_info = {
                'status': 'success',
                'model_name': 'portfolio_rl_model',
                'deployment_time': datetime.now().isoformat(),
                'model_version': training_info.get('experiment_name', 'unknown'),
                'api_endpoint': 'http://portfolio-api:8000',
                'mlflow_model_uri': f"models:/portfolio_rl_model/Production"
            }
            
            context['task_instance'].xcom_push(
                key='deployment_result',
                value=deployment_info
            )
            
            return "success"
        else:
            logger.error(f"❌ Erreur déploiement API: {result.stderr}")
            raise Exception("API deployment failed")
            
    except subprocess.TimeoutExpired:
        logger.error("❌ Timeout déploiement")
        raise Exception("Deployment timeout")
    except Exception as e:
        logger.error(f"❌ Erreur déploiement: {e}")
        raise

def send_success_notification(**context) -> str:
    """Envoie une notification de succès."""
    logger.info("📧 Envoi notification de succès")
    
    # Récupérer toutes les informations du pipeline
    deployment_info = context['task_instance'].xcom_pull(
        task_ids='model_deployment',
        key='deployment_result'
    )
    
    evaluation_info = context['task_instance'].xcom_pull(
        task_ids='model_evaluation', 
        key='evaluation_result'
    )
    
    training_info = context['task_instance'].xcom_pull(
        task_ids='model_training',
        key='training_result'
    )
    
    # Créer le rapport de succès
    report = {
        'pipeline_status': 'SUCCESS',
        'execution_date': context['execution_date'].isoformat(),
        'training_metrics': {
            'final_reward': training_info.get('final_reward', 'N/A'),
            'training_duration': training_info.get('training_duration', 'N/A')
        },
        'evaluation_metrics': {
            'mean_reward': evaluation_info.get('mean_reward', 0),
            'sharpe_ratio': evaluation_info.get('sharpe_ratio', 0),
            'total_return': evaluation_info.get('total_return', 0),
            'quality_passed': evaluation_info.get('quality_passed', False)
        },
        'deployment_info': {
            'model_version': deployment_info.get('model_version', 'unknown'),
            'api_endpoint': deployment_info.get('api_endpoint', 'N/A')
        }
    }
    
    # Sauvegarder le rapport
    import json
    report_file = f"/opt/airflow/reports/pipeline_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    os.makedirs(os.path.dirname(report_file), exist_ok=True)
    
    with open(report_file, 'w') as f:
        json.dump(report, f, indent=2)
    
    logger.info(f"📊 Rapport sauvegardé: {report_file}")
    
    # Log des métriques importantes
    logger.info("🎉 PIPELINE TERMINÉ AVEC SUCCÈS!")
    logger.info(f"   📈 Reward moyen: {evaluation_info.get('mean_reward', 'N/A')}")
    logger.info(f"   💰 Retour total: {evaluation_info.get('total_return', 'N/A')}%")
    logger.info(f"   📊 Sharpe ratio: {evaluation_info.get('sharpe_ratio', 'N/A')}")
    logger.info(f"   🚀 Modèle déployé: {deployment_info.get('model_version', 'N/A')}")
    
    return "success"

def send_failure_notification(**context) -> str:
    """Envoie une notification d'échec."""
    logger.error("📧 Envoi notification d'échec")
    
    # Récupérer les informations d'erreur
    failed_task = context.get('task_instance')
    exception = context.get('exception')
    
    failure_report = {
        'pipeline_status': 'FAILED',
        'execution_date': context['execution_date'].isoformat(),
        'failed_task': failed_task.task_id if failed_task else 'unknown',
        'error_message': str(exception) if exception else 'Unknown error',
        'dag_run_id': context.get('dag_run').run_id if context.get('dag_run') else 'unknown'
    }
    
    # Sauvegarder le rapport d'échec
    failure_file = f"/opt/airflow/reports/failure_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    os.makedirs(os.path.dirname(failure_file), exist_ok=True)
    
    with open(failure_file, 'w') as f:
        json.dump(failure_report, f, indent=2)
    
    logger.error(f"💥 Rapport d'échec sauvegardé: {failure_file}")
    logger.error(f"💥 PIPELINE ÉCHOUÉ - Tâche: {failure_report['failed_task']}")
    logger.error(f"💥 Erreur: {failure_report['error_message']}")
    
    return "failed"

def cleanup_old_models(**context) -> str:
    """Nettoie les anciens modèles et artifacts."""
    logger.info("🧹 Nettoyage des anciens modèles")
    
    try:
        # Nettoyer les modèles locaux (garder les 5 plus récents)
        cleanup_script = """
import os
import glob
from pathlib import Path

models_dir = Path("/opt/airflow/models")
model_files = list(models_dir.glob("*.zip"))

# Trier par date de modification
model_files.sort(key=lambda x: x.stat().st_mtime, reverse=True)

# Supprimer les anciens (garder les 5 plus récents)
for old_model in model_files[5:]:
    try:
        old_model.unlink()
        print(f"Supprimé: {old_model}")
    except Exception as e:
        print(f"Erreur suppression {old_model}: {e}")

print(f"Nettoyage terminé. {len(model_files[5:])} modèles supprimés.")
"""
        
        result = subprocess.run([
            sys.executable, "-c", cleanup_script
        ], capture_output=True, text=True, timeout=60)
        
        if result.returncode == 0:
            logger.info("✅ Nettoyage réussi")
            logger.info(result.stdout)
        else:
            logger.warning(f"⚠️ Erreur nettoyage: {result.stderr}")
        
        return "success"
        
    except Exception as e:
        logger.warning(f"⚠️ Erreur nettoyage: {e}")
        return "partial"

def run_model_validation_tests(**context) -> str:
    """Exécute des tests de validation sur le modèle déployé."""
    logger.info("🧪 Tests de validation du modèle déployé")
    
    try:
        # Tests de l'API
        test_params = [
            "--api-endpoint", "http://portfolio-api:8000",
            "--test-suite", "full",
            "--timeout", "300"
        ]
        
        result = subprocess.run([
            sys.executable, "/opt/airflow/src/tests/test_deployed_model.py"
        ] + test_params, capture_output=True, text=True, timeout=600)
        
        if result.returncode == 0:
            logger.info("✅ Tests de validation réussis")
            
            # Parser les résultats des tests
            test_results = {
                'api_health': True,
                'model_prediction': True,
                'performance_baseline': True,
                'latency_check': True
            }
            
            context['task_instance'].xcom_push(
                key='validation_tests',
                value=test_results
            )
            
            return "success"
        else:
            logger.error(f"❌ Tests de validation échoués: {result.stderr}")
            return "failed"
            
    except subprocess.TimeoutExpired:
        logger.error("❌ Timeout tests de validation")
        return "timeout"
    except Exception as e:
        logger.error(f"❌ Erreur tests: {e}")
        return "error"

# =============================================================================
# DÉFINITION DU DAG
# =============================================================================

# Création du DAG principal
dag = DAG(
    'portfolio_ml_pipeline',
    default_args=default_args,
    description='Pipeline MLOps complet pour Portfolio RL',
    schedule_interval='@daily',  # Exécution quotidienne
    catchup=False,
    max_active_runs=1,
    tags=['mlops', 'portfolio', 'reinforcement-learning']
)

# =============================================================================
# DÉFINITION DES TÂCHES
# =============================================================================

# 1. Vérification de la qualité des données
data_quality_check = PythonOperator(
    task_id='data_quality_check',
    python_callable=check_data_quality,
    dag=dag,
    retries=1
)

# 2. Ingestion des données
data_ingestion = PythonOperator(
    task_id='data_ingestion',
    python_callable=run_data_ingestion,
    dag=dag
)

# 3. Preprocessing des données
data_preprocessing = PythonOperator(
    task_id='data_preprocessing',
    python_callable=run_preprocessing,
    dag=dag
)

# 4. Entraînement du modèle
model_training = PythonOperator(
    task_id='model_training',
    python_callable=run_training,
    dag=dag
)

# 5. Évaluation du modèle
model_evaluation = PythonOperator(
    task_id='model_evaluation',
    python_callable=run_evaluation,
    dag=dag
)

# 6. Déploiement du modèle (conditionnel)
model_deployment = PythonOperator(
    task_id='model_deployment',
    python_callable=run_model_deployment,
    dag=dag
)

# 7. Tests de validation post-déploiement
validation_tests = PythonOperator(
    task_id='validation_tests',
    python_callable=run_model_validation_tests,
    dag=dag
)

# 8. Nettoyage des anciens modèles
cleanup_models = PythonOperator(
    task_id='cleanup_models',
    python_callable=cleanup_old_models,
    dag=dag,
    trigger_rule=TriggerRule.ALL_DONE  # Exécuter même si d'autres tâches échouent
)

# 9. Notification de succès
success_notification = PythonOperator(
    task_id='success_notification',
    python_callable=send_success_notification,
    dag=dag,
    trigger_rule=TriggerRule.ALL_SUCCESS
)

# 10. Notification d'échec
failure_notification = PythonOperator(
    task_id='failure_notification',
    python_callable=send_failure_notification,
    dag=dag,
    trigger_rule=TriggerRule.ONE_FAILED
)

# 11. Capteur pour vérifier la disponibilité de l'API
api_health_sensor = HttpSensor(
    task_id='api_health_check',
    http_conn_id='portfolio_api',  # À configurer dans Airflow
    endpoint='health',
    request_params={},
    poke_interval=30,
    timeout=300,
    dag=dag
)

# 12. Capteur pour vérifier la disponibilité de MLflow
mlflow_health_sensor = HttpSensor(
    task_id='mlflow_health_check',
    http_conn_id='mlflow_api',  # À configurer dans Airflow
    endpoint='',
    request_params={},
    poke_interval=30,
    timeout=300,
    dag=dag
)

# =============================================================================
# DÉFINITION DES DÉPENDANCES
# =============================================================================

# Flux principal du pipeline
data_quality_check >> data_ingestion >> data_preprocessing >> model_training >> model_evaluation

# Déploiement conditionnel (seulement si évaluation réussie)
model_evaluation >> model_deployment >> validation_tests

# Notifications parallèles
[validation_tests] >> success_notification
[data_quality_check, data_ingestion, data_preprocessing, model_training, model_evaluation, model_deployment, validation_tests] >> failure_notification

# Nettoyage en parallèle
validation_tests >> cleanup_models

# Capteurs de santé en parallèle (optionnels)
[api_health_sensor, mlflow_health_sensor] >> data_quality_check

# =============================================================================
# CONFIGURATION ADDITIONNELLE
# =============================================================================

# Variables de configuration pour le DAG
dag.doc_md = """
# Pipeline MLOps Portfolio RL

Ce pipeline orchestre l'ensemble du cycle de vie MLOps pour le système de trading par reinforcement learning.

## Étapes du pipeline:

1. **Vérification santé services**: Vérifie que l'API et MLflow sont disponibles
2. **Contrôle qualité données**: Valide la qualité des données avant traitement
3. **Ingestion données**: Télécharge les données financières récentes
4. **Preprocessing**: Prépare les données pour l'entraînement
5. **Entraînement**: Entraîne le modèle RL avec les nouvelles données
6. **Évaluation**: Évalue les performances du modèle entraîné
7. **Déploiement**: Déploie le modèle si les métriques sont satisfaisantes
8. **Tests validation**: Vérifie le bon fonctionnement du modèle déployé
9. **Nettoyage**: Supprime les anciens modèles pour économiser l'espace
10. **Notifications**: Informe de l'état du pipeline

## Métriques surveillées:

- **Reward moyen**: Performance de trading
- **Sharpe ratio**: Ratio rendement/risque
- **Drawdown maximum**: Perte maximale
- **Taux de gain**: Pourcentage de trades gagnants

## Configuration:

Le pipeline peut être configuré via les variables d'environnement Airflow:
- `PORTFOLIO_CONFIG_PATH`: Chemin vers le fichier de configuration
- `MLFLOW_TRACKING_URI`: URI du serveur MLflow
- `API_ENDPOINT`: Point d'entrée de l'API Portfolio

## Fréquence:

Par défaut quotidienne, mais configurable selon les besoins.
"""

# Configuration des connexions par défaut
dag.params = {
    "config_path": "/opt/airflow/config/default.json",
    "mlflow_uri": "http://mlflow:5000",
    "api_endpoint": "http://portfolio-api:8000",
    "notification_email": "admin@portfolio-rl.com",
    "max_training_time": 3600,  # 1 heure
    "quality_thresholds": {
        "min_reward": 0.1,
        "min_return": 5.0,
        "min_sharpe": 0.5,
        "max_drawdown": 0.2
    }
}