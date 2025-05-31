#!/bin/bash
echo "[DEPLOY] Démarrage en mode développement..."

python -c "import fastapi, mlflow, prometheus_client" 2>/dev/null || {
    echo "[ERROR] Dépendances manquantes. Installation..."
    pip install -r requirements.txt
}

echo "[DEPLOY] Démarrage des services..."
docker-compose -f docker-compose-production.yml up -d mlflow prometheus grafana

echo "[WAIT] Attente des services..."
sleep 10

echo "[SUCCESS] Services démarrés!"
echo "[INFO] API: http://localhost:8000"
echo "[INFO] MLflow: http://localhost:5000"
echo "[INFO] Prometheus: http://localhost:9090"
echo "[INFO] Grafana: http://localhost:3000 (admin/admin)"
echo ""
echo "[INFO] Pour arrêter: Ctrl+C puis docker-compose down"
