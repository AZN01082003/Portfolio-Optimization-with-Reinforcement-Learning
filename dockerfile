# Utilisation de l'image Python officielle
FROM python:3.9-slim

# Définir le répertoire de travail
WORKDIR /app

# Copier d'abord les fichiers de dépendances
COPY requirements.txt .
COPY setup.py .

# Installer les dépendances
RUN pip install --no-cache-dir -e ".[dev,mlops]"

# Copier le reste du code
COPY . .

# Créer les répertoires nécessaires
RUN mkdir -p data/raw data/processed models logs

# Exposer le port pour l'API FastAPI
EXPOSE 8000

# Exposer le port pour le serveur MLflow
EXPOSE 5000

# Commande par défaut
CMD ["python", "run_pipeline.py"]