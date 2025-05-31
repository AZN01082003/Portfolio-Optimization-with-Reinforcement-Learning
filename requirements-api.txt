FROM python:3.9-slim

WORKDIR /app

# Installer les dépendances système
RUN apt-get update && apt-get install -y \
    curl \
    git \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copier requirements.txt en premier pour le cache Docker
COPY requirements.txt .

# Installer pip avec retry et timeout plus longs
RUN pip install --no-cache-dir --upgrade pip

# Installer les packages de base d'abord (plus stables)
RUN pip install --no-cache-dir \
    fastapi>=0.100.0 \
    uvicorn[standard]>=0.20.0 \
    pydantic>=2.0.0 \
    numpy>=1.20.0 \
    pandas>=1.3.0 \
    requests>=2.25.1 \
    python-dotenv>=0.20.0 \
    pyyaml>=6.0

# Installer PyTorch CPU uniquement (plus léger et plus fiable)
RUN pip install --no-cache-dir \
    --timeout 1000 \
    --retries 3 \
    torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

# Installer les autres packages ML
RUN pip install --no-cache-dir \
    --timeout 1000 \
    --retries 3 \
    scikit-learn>=1.0.0 \
    mlflow>=2.0.0 \
    prometheus-client>=0.15.0 \
    psutil>=5.9.0

# Installer les packages moins critiques en dernier
RUN pip install --no-cache-dir \
    --timeout 1000 \
    --retries 3 \
    matplotlib>=3.5.0 \
    seaborn>=0.11.2 \
    yfinance>=0.1.70 || echo "Warning: Optional package failed"

# Installer gymnasium et stable-baselines3 (peuvent être problématiques)
RUN pip install --no-cache-dir \
    --timeout 1000 \
    --retries 3 \
    gymnasium>=0.28.1 \
    stable-baselines3 || echo "Warning: RL packages failed, will install at runtime"

# Copier le code source
COPY src/ ./src/
COPY config/ ./config/

# Créer les répertoires nécessaires
RUN mkdir -p logs/api data/processed models

# Créer un utilisateur non-root
RUN useradd -m -u 1000 apiuser && \
    chown -R apiuser:apiuser /app
USER apiuser

# Exposer les ports
EXPOSE 8000 8001

# Variables d'environnement
ENV PYTHONPATH=/app
ENV API_HOST=0.0.0.0
ENV API_PORT=8000

# Commande de démarrage
CMD ["python", "src/main_minimal.py"]