FROM apache/airflow:2.7.1

USER root

# Installer les dépendances système nécessaires
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    build-essential \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

USER airflow

# Installer les dépendances Python spécifiques au projet
COPY requirements-airflow.txt /opt/airflow/
RUN pip install --no-cache-dir -r /opt/airflow/requirements-airflow.txt

# Créer les répertoires nécessaires pour les données et modèles
RUN mkdir -p /opt/airflow/data/raw /opt/airflow/data/processed /opt/airflow/models