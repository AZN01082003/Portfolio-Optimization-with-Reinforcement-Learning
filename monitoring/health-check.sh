#!/bin/sh
# Script de verification de sante des services

check_service() {
    local service_name=$1
    local url=$2
    
    if curl -f -s "$url" > /dev/null 2>&1; then
        echo "[OK] $service_name: HEALTHY"
        return 0
    else
        echo "[FAIL] $service_name: FAILED"
        return 1
    fi
}

echo "[INFO] Verification de la sante des services..."

check_service "Prometheus" "http://prometheus:9090/-/healthy"
check_service "Grafana" "http://grafana:3000/api/health"
check_service "AlertManager" "http://alertmanager:9093/-/healthy"
check_service "Portfolio API" "http://portfolio-api-monitored:8000/health"
check_service "Metriques API" "http://portfolio-api-monitored:8001/metrics"

echo "Verification terminee a $(date)"
