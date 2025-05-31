FROM python:3.9-slim

WORKDIR /app

# Installer les d�pendances syst�me
RUN apt-get update && apt-get install -y \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Copier les requirements
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copier le code source
COPY src/ ./src/
COPY config/ ./config/

# Exposer le port
EXPOSE 8000

# Variables d'environnement
ENV PYTHONPATH=/app
ENV API_HOST=0.0.0.0
ENV API_PORT=8000

# Commande de d�marrage
CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
