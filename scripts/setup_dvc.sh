#!/bin/bash

# Ce script initialise DVC et configure le stockage distant

# Vérifier si DVC est installé
if ! command -v dvc &> /dev/null; then
    echo "DVC n'est pas installé. Installation en cours..."
    pip install dvc dvc-s3
fi

# Répertoire du projet
PROJECT_DIR=$(dirname $(dirname $(realpath $0)))
cd $PROJECT_DIR

# Initialiser DVC si ce n'est pas déjà fait
if [ ! -d ".dvc" ]; then
    echo "Initialisation de DVC..."
    dvc init
    git commit -m "Initialize DVC"
fi

# Configurer le stockage distant (S3, si configuré)
if grep -q "s3://" config/default.json; then
    # Extraire l'URL S3 du fichier de configuration
    S3_URL=$(grep -A 1 "remote:" config/default.json | grep "s3://" | tr -d ' ' | cut -d':' -f2-)
    
    if [ ! -z "$S3_URL" ]; then
        echo "Configuration du stockage distant S3: $S3_URL"
        dvc remote add -d s3remote $S3_URL
        
        # Demander les informations d'identification AWS si nécessaire
        if [ -z "$AWS_ACCESS_KEY_ID" ] || [ -z "$AWS_SECRET_ACCESS_KEY" ]; then
            echo "Configuration des informations d'identification AWS pour DVC..."
            
            read -p "AWS Access Key ID: " AWS_ACCESS_KEY_ID
            read -p "AWS Secret Access Key: " AWS_SECRET_ACCESS_KEY
            
            # Configurer les informations d'identification pour DVC
            dvc remote modify s3remote access_key_id $AWS_ACCESS_KEY_ID
            dvc remote modify s3remote secret_access_key $AWS_SECRET_ACCESS_KEY
        fi
        
        git add .dvc/config
        git commit -m "Configure DVC remote storage"
    else
        echo "URL S3 non trouvée dans la configuration"
    fi
else
    echo "Configuration d'un stockage local pour DVC..."
    mkdir -p .dvc/storage
    dvc remote add -d localremote .dvc/storage
    git add .dvc/config
    git commit -m "Configure DVC local storage"
fi

# Ajouter les répertoires de données
echo "Configuration des répertoires à versionner avec DVC..."
dvc add data/raw data/processed models
git add data/raw/.gitignore data/processed/.gitignore models/.gitignore
git add data/raw.dvc data/processed.dvc models.dvc
git commit -m "Add data and model directories to DVC"

echo "Configuration de DVC terminée avec succès!"