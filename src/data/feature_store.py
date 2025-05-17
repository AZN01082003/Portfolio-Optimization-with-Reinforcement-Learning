"""
Module pour la gestion du feature store.
Ce feature store permet de stocker, récupérer et réutiliser des features 
calculées à partir des données brutes.
"""
import os
import logging
import numpy as np
import pandas as pd
import pickle
import json
#import yaml
from datetime import datetime
import sqlite3
import hashlib
from typing import Dict, List, Union, Tuple, Any

# Configuration du logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def load_config(config_path="config/default.json"):
    """
    Charge la configuration depuis un fichier json.
    """
    with open(config_path, 'r') as file:
        config = json.load(file)
    return config

class FeatureStore:
    """
    Classe pour la gestion du feature store.
    """
    def __init__(self, base_dir="data/features", config_path="config/default.json"):
        """
        Initialise le feature store.
        
        Args:
            base_dir: Répertoire de base pour le stockage des features
            config_path: Chemin du fichier de configuration
        """
        self.base_dir = base_dir
        self.config_path = config_path
        self.config = load_config(config_path)
        
        # Créer le répertoire de base s'il n'existe pas
        os.makedirs(base_dir, exist_ok=True)
        
        # Initialiser la base de données de métadonnées
        self.db_path = os.path.join(base_dir, 'feature_store.db')
        self._init_db()
        
        logger.info(f"Feature store initialisé dans {base_dir}")
    
    def _init_db(self):
        """
        Initialise la base de données de métadonnées.
        """
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Créer la table des features
            cursor.execute('''
            CREATE TABLE IF NOT EXISTS features (
                id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                version TEXT NOT NULL,
                description TEXT,
                entity_type TEXT NOT NULL,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                parameters TEXT,
                stats TEXT,
                file_path TEXT NOT NULL,
                UNIQUE(name, version)
            )
            ''')
            
            # Créer la table des associations de features
            cursor.execute('''
            CREATE TABLE IF NOT EXISTS feature_sets (
                id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                version TEXT NOT NULL,
                description TEXT,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                features TEXT NOT NULL,
                file_path TEXT NOT NULL,
                UNIQUE(name, version)
            )
            ''')
            
            conn.commit()
            conn.close()
            
            logger.debug("Base de données de métadonnées initialisée")
        
        except Exception as e:
            logger.error(f"Erreur lors de l'initialisation de la base de données: {e}")
    
    def _generate_id(self, name, version, params=None):
        """
        Génère un ID unique pour une feature ou un feature set.
        
        Args:
            name: Nom de la feature ou du feature set
            version: Version
            params: Paramètres supplémentaires (optionnel)
            
        Returns:
            str: ID unique
        """
        # Créer une chaîne à hacher
        hash_string = f"{name}_{version}"
        if params:
            hash_string += f"_{json.dumps(params, sort_keys=True)}"
        
        # Générer un hash
        return hashlib.md5(hash_string.encode()).hexdigest()
    
    def save_feature(self, name: str, data: Union[np.ndarray, pd.DataFrame], 
                     entity_type: str, version: str="latest", 
                     description: str=None, parameters: Dict=None, stats: Dict=None):
        """
        Sauvegarde une feature dans le store.
        
        Args:
            name: Nom de la feature
            data: Données de la feature (numpy array ou DataFrame)
            entity_type: Type d'entité (ex: 'stock', 'portfolio')
            version: Version de la feature
            description: Description de la feature
            parameters: Paramètres utilisés pour générer la feature
            stats: Statistiques sur la feature
            
        Returns:
            str: ID de la feature
        """
        try:
            # Générer l'ID
            feature_id = self._generate_id(name, version, parameters)
            
            # Créer le répertoire pour cette feature
            feature_dir = os.path.join(self.base_dir, entity_type, name)
            os.makedirs(feature_dir, exist_ok=True)
            
            # Déterminer le chemin du fichier
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            if isinstance(data, pd.DataFrame):
                file_path = os.path.join(feature_dir, f"{version}_{timestamp}.parquet")
                data.to_parquet(file_path, index=False)
            else:
                file_path = os.path.join(feature_dir, f"{version}_{timestamp}.npy")
                np.save(file_path, data)
            
            # Calculer les statistiques si non fournies
            if stats is None and isinstance(data, pd.DataFrame):
                stats = {
                    'shape': data.shape,
                    'columns': list(data.columns),
                    'description': data.describe().to_dict()
                }
            elif stats is None and isinstance(data, np.ndarray):
                stats = {
                    'shape': data.shape,
                    'dtype': str(data.dtype),
                    'min': float(np.min(data)) if data.size > 0 else None,
                    'max': float(np.max(data)) if data.size > 0 else None,
                    'mean': float(np.mean(data)) if data.size > 0 else None,
                    'std': float(np.std(data)) if data.size > 0 else None
                }
            
            # Enregistrer les métadonnées
            now = datetime.now().isoformat()
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
            INSERT OR REPLACE INTO features (id, name, version, description, entity_type, 
                                           created_at, updated_at, parameters, stats, file_path)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                feature_id, name, version, description, entity_type, 
                now, now, 
                json.dumps(parameters) if parameters else None, 
                json.dumps(stats) if stats else None, 
                file_path
            ))
            
            conn.commit()
            conn.close()
            
            logger.info(f"Feature '{name}' (version: {version}) sauvegardée avec ID: {feature_id}")
            return feature_id
        
        except Exception as e:
            logger.error(f"Erreur lors de la sauvegarde de la feature '{name}': {e}")
            return None
    
    def load_feature(self, name: str=None, feature_id: str=None, version: str="latest", entity_type: str=None):
        """
        Charge une feature depuis le store.
        
        Args:
            name: Nom de la feature (requis si feature_id non fourni)
            feature_id: ID de la feature (requis si name non fourni)
            version: Version de la feature (ignoré si feature_id est fourni)
            entity_type: Type d'entité (requis si name est fourni et version est 'latest')
            
        Returns:
            tuple: (data, metadata) ou (None, None) en cas d'erreur
        """
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            if feature_id:
                # Récupérer la feature par ID
                cursor.execute('''
                SELECT id, name, version, description, entity_type, 
                       created_at, updated_at, parameters, stats, file_path
                FROM features
                WHERE id = ?
                ''', (feature_id,))
            elif name and version == "latest" and entity_type:
                # Récupérer la dernière version de la feature
                cursor.execute('''
                SELECT id, name, version, description, entity_type,
                       created_at, updated_at, parameters, stats, file_path
                FROM features
                WHERE name = ? AND entity_type = ?
                ORDER BY updated_at DESC
                LIMIT 1
                ''', (name, entity_type))
            elif name and version and entity_type:
                # Récupérer une version spécifique de la feature
                cursor.execute('''
                SELECT id, name, version, description, entity_type,
                       created_at, updated_at, parameters, stats, file_path
                FROM features
                WHERE name = ? AND version = ? AND entity_type = ?
                ''', (name, version, entity_type))
            else:
                raise ValueError("Paramètres insuffisants: fournir feature_id ou (name, version, entity_type)")
            
            row = cursor.fetchone()
            conn.close()
            
            if not row:
                logger.warning(f"Feature non trouvée avec les paramètres fournis")
                return None, None
            
            # Transformer la ligne en dictionnaire
            metadata = {
                'id': row[0],
                'name': row[1],
                'version': row[2],
                'description': row[3],
                'entity_type': row[4],
                'created_at': row[5],
                'updated_at': row[6],
                'parameters': json.loads(row[7]) if row[7] else None,
                'stats': json.loads(row[8]) if row[8] else None,
                'file_path': row[9]
            }
            
            # Charger les données
            file_path = metadata['file_path']
            if file_path.endswith('.parquet'):
                data = pd.read_parquet(file_path)
            elif file_path.endswith('.npy'):
                data = np.load(file_path)
            else:
                logger.error(f"Type de fichier non supporté: {file_path}")
                return None, None
            
            logger.info(f"Feature '{metadata['name']}' (version: {metadata['version']}) chargée")
            return data, metadata
        
        except Exception as e:
            logger.error(f"Erreur lors du chargement de la feature: {e}")
            return None, None
    
    def save_feature_set(self, name: str, features: List[Dict], 
                        version: str="latest", description: str=None):
        """
        Sauvegarde un ensemble de features.
        
        Args:
            name: Nom de l'ensemble de features
            features: Liste des métadonnées de features
            version: Version de l'ensemble
            description: Description de l'ensemble
            
        Returns:
            str: ID de l'ensemble de features
        """
        try:
            # Générer l'ID
            set_id = self._generate_id(name, version)
            
            # Créer le répertoire pour les ensembles de features
            sets_dir = os.path.join(self.base_dir, 'feature_sets')
            os.makedirs(sets_dir, exist_ok=True)
            
            # Déterminer le chemin du fichier
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            file_path = os.path.join(sets_dir, f"{name}_{version}_{timestamp}.json")
            
            # Sauvegarder la liste des features
            with open(file_path, 'w') as f:
                json.dump(features, f)
            
            # Enregistrer les métadonnées
            now = datetime.now().isoformat()
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
            INSERT OR REPLACE INTO feature_sets (id, name, version, description, 
                                               created_at, updated_at, features, file_path)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                set_id, name, version, description, 
                now, now, json.dumps(features), file_path
            ))
            
            conn.commit()
            conn.close()
            
            logger.info(f"Feature set '{name}' (version: {version}) sauvegardé avec ID: {set_id}")
            return set_id
        
        except Exception as e:
            logger.error(f"Erreur lors de la sauvegarde du feature set '{name}': {e}")
            return None
    
    def load_feature_set(self, name: str=None, set_id: str=None, version: str="latest", load_data: bool=True):
        """
        Charge un ensemble de features.
        
        Args:
            name: Nom de l'ensemble (requis si set_id non fourni)
            set_id: ID de l'ensemble (requis si name non fourni)
            version: Version de l'ensemble (ignoré si set_id est fourni)
            load_data: Si True, charge également les données des features
            
        Returns:
            dict: Métadonnées de l'ensemble et features chargées
        """
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            if set_id:
                # Récupérer l'ensemble par ID
                cursor.execute('''
                SELECT id, name, version, description, 
                       created_at, updated_at, features, file_path
                FROM feature_sets
                WHERE id = ?
                ''', (set_id,))
            elif name and version == "latest":
                # Récupérer la dernière version de l'ensemble
                cursor.execute('''
                SELECT id, name, version, description,
                       created_at, updated_at, features, file_path
                FROM feature_sets
                WHERE name = ?
                ORDER BY updated_at DESC
                LIMIT 1
                ''', (name,))
            elif name and version:
                # Récupérer une version spécifique de l'ensemble
                cursor.execute('''
                SELECT id, name, version, description,
                       created_at, updated_at, features, file_path
                FROM feature_sets
                WHERE name = ? AND version = ?
                ''', (name, version))
            else:
                raise ValueError("Paramètres insuffisants: fournir set_id ou (name, version)")
            
            row = cursor.fetchone()
            conn.close()
            
            if not row:
                logger.warning(f"Feature set non trouvé avec les paramètres fournis")
                return None
            
            # Transformer la ligne en dictionnaire
            metadata = {
                'id': row[0],
                'name': row[1],
                'version': row[2],
                'description': row[3],
                'created_at': row[4],
                'updated_at': row[5],
                'features': json.loads(row[6]),
                'file_path': row[7]
            }
            
            # Si demandé, charger les données des features
            if load_data:
                feature_data = {}
                for feature in metadata['features']:
                    feature_id = feature['id']
                    data, feature_metadata = self.load_feature(feature_id=feature_id)
                    if data is not None:
                        feature_data[feature['name']] = {
                            'data': data,
                            'metadata': feature_metadata
                        }
                
                metadata['feature_data'] = feature_data
            
            logger.info(f"Feature set '{metadata['name']}' (version: {metadata['version']}) chargé")
            return metadata
        
        except Exception as e:
            logger.error(f"Erreur lors du chargement du feature set: {e}")
            return None
    
    def list_features(self, entity_type: str=None, name: str=None):
        """
        Liste les features disponibles dans le store.
        
        Args:
            entity_type: Filtrer par type d'entité (optionnel)
            name: Filtrer par nom (optionnel)
            
        Returns:
            pd.DataFrame: DataFrame contenant les métadonnées des features
        """
        try:
            conn = sqlite3.connect(self.db_path)
            
            query = '''
            SELECT id, name, version, description, entity_type, 
                   created_at, updated_at, parameters, stats, file_path
            FROM features
            '''
            
            params = []
            where_clauses = []
            
            if entity_type:
                where_clauses.append("entity_type = ?")
                params.append(entity_type)
            
            if name:
                where_clauses.append("name = ?")
                params.append(name)
            
            if where_clauses:
                query += " WHERE " + " AND ".join(where_clauses)
            
            query += " ORDER BY updated_at DESC"
            
            # Exécuter la requête
            features_df = pd.read_sql_query(query, conn, params=params)
            conn.close()
            
            # Convertir les colonnes JSON
            if 'parameters' in features_df.columns:
                features_df['parameters'] = features_df['parameters'].apply(
                    lambda x: json.loads(x) if x else None
                )
            
            if 'stats' in features_df.columns:
                features_df['stats'] = features_df['stats'].apply(
                    lambda x: json.loads(x) if x else None
                )
            
            return features_df
        
        except Exception as e:
            logger.error(f"Erreur lors de la récupération de la liste des features: {e}")
            return pd.DataFrame()
    
    def list_feature_sets(self, name: str=None):
        """
        Liste les ensembles de features disponibles dans le store.
        
        Args:
            name: Filtrer par nom (optionnel)
            
        Returns:
            pd.DataFrame: DataFrame contenant les métadonnées des ensembles
        """
        try:
            conn = sqlite3.connect(self.db_path)
            
            query = '''
            SELECT id, name, version, description, 
                   created_at, updated_at, features, file_path
            FROM feature_sets
            '''
            
            params = []
            if name:
                query += " WHERE name = ?"
                params.append(name)
            
            query += " ORDER BY updated_at DESC"
            
            # Exécuter la requête
            sets_df = pd.read_sql_query(query, conn, params=params)
            conn.close()
            
            # Convertir les colonnes JSON
            if 'features' in sets_df.columns:
                sets_df['features'] = sets_df['features'].apply(
                    lambda x: json.loads(x) if x else None
                )
            
            return sets_df
        
        except Exception as e:
            logger.error(f"Erreur lors de la récupération de la liste des feature sets: {e}")
            return pd.DataFrame()