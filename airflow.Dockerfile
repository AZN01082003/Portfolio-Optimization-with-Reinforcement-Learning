FROM apache/airflow:2.5.0

USER root
RUN apt-get update && apt-get install -y \
    build-essential \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

USER airflow

# Copier et installer les requirements
COPY requirements.txt /tmp/
RUN pip install --no-cache-dir -r /tmp/requirements.txt

# Créer les répertoires nécessaires
RUN mkdir -p /opt/airflow/data/raw /opt/airflow/data/processed \
             /opt/airflow/models /opt/airflow/reports
