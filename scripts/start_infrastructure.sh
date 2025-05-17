#!/bin/bash

# Créer les répertoires nécessaires
mkdir -p data/raw data/processed logs models mlruns artifacts dags plugins

# Initialiser les variables d'environnement
export AIRFLOW_UID=$(id -u)
export AIRFLOW_GID=$(id -g)
export AIRFLOW_WWW_USER_USERNAME=airflow
export AIRFLOW_WWW_USER_PASSWORD=airflow

# Construire les images Docker
echo "Construction des images Docker..."
docker-compose build

# Démarrer les services
echo "Démarrage des services..."
docker-compose up -d

# Attendre que Airflow soit prêt
echo "Attente du démarrage d'Airflow..."
sleep 30

# Configurer les connexions Airflow
echo "Configuration des connexions Airflow..."
docker exec -it portfolio-airflow-webserver python /opt/airflow/scripts/setup_airflow_connections.py

echo "Infrastructure démarrée avec succès !"
echo "MLflow UI disponible à l'adresse: http://localhost:5000"
echo "Airflow UI disponible à l'adresse: http://localhost:8080"
echo "Kafka UI disponible à l'adresse: http://localhost:8081"
echo "API (lorsqu'elle sera mise en place) disponible à l'adresse: http://localhost:8000"