FROM python:3.9-slim

# Installer MLflow
RUN pip install --no-cache-dir mlflow>=2.0.0 psycopg2-binary

# Créer les répertoires pour MLflow
RUN mkdir -p /mlflow/mlruns /mlflow/artifacts

WORKDIR /mlflow

# Exposer le port du serveur MLflow
EXPOSE 5000

# Commande pour démarrer le serveur MLflow
CMD mlflow server \
    --host 0.0.0.0 \
    --port 5000 \
    --backend-store-uri ${BACKEND_STORE_URI:-sqlite:///mlflow.db} \
    --default-artifact-root ${DEFAULT_ARTIFACT_ROOT:-/mlflow/artifacts}