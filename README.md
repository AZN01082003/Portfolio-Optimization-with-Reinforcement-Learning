# Portfolio RL MLOps

## Structure des données

- `raw/`: Données brutes téléchargées
- `processed/`: Données préprocessées pour l'entraînement
- `models/`: Modèles entraînés sauvegardés
- `logs/`: Logs d'entraînement et d'évaluation

## Usage

1. Démarrer l'infrastructure: `start_development.bat` (Windows) ou `./start_development.sh` (Linux)
2. Accéder aux services:
   - API: http://localhost:8000
   - MLflow: http://localhost:5000
   - Grafana: http://localhost:3000 (admin/admin)
   - Prometheus: http://localhost:9090
