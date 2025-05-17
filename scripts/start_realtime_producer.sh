#!/bin/bash

# Ce script démarre le producteur Kafka en temps réel

# Répertoire du projet
PROJECT_DIR=$(dirname $(dirname $(realpath $0)))
cd $PROJECT_DIR

# Activer l'environnement virtuel si nécessaire
if [ -d "venv" ]; then
    source venv/bin/activate
fi

# Démarrer le producteur Kafka
echo "Démarrage du producteur Kafka en temps réel..."
python -m src.data.kafka_producer

echo "Producteur Kafka démarré !"