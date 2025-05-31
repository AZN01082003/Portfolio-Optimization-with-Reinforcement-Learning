#!/usr/bin/env python3
"""
Script pour finaliser le déploiement - construire l'image API manquante
"""
import subprocess
import sys
import time
from pathlib import Path

def check_existing_services():
    """Vérifie les services déjà en cours."""
    print("🔍 Vérification des services existants...")
    
    try:
        result = subprocess.run(["docker", "ps"], capture_output=True, text=True)
        print("📦 Services Docker actifs:")
        print(result.stdout)
        
        # Vérifier MLflow
        try:
            import requests
            response = requests.get("http://localhost:5000", timeout=5)
            print("✅ MLflow accessible sur http://localhost:5000")
        except:
            print("❌ MLflow non accessible")
        
        # Vérifier Prometheus
        try:
            response = requests.get("http://localhost:9090", timeout=5)
            print("✅ Prometheus accessible sur http://localhost:9090")
        except:
            print("❌ Prometheus non accessible")
            
    except Exception as e:
        print(f"❌ Erreur vérification: {e}")

def build_api_image_verbose():
    """Construit l'image API avec sortie verbose."""
    print("\n🔨 Construction de l'image API Portfolio...")
    print("📝 Cette étape peut prendre 5-15 minutes")
    
    try:
        # Vérifier que le Dockerfile existe
        dockerfile_path = Path("api.Dockerfile")
        if not dockerfile_path.exists():
            print("❌ api.Dockerfile introuvable!")
            return False
        
        print("📄 Dockerfile trouvé, démarrage de la construction...")
        
        # Construction avec sortie en temps réel
        process = subprocess.Popen([
            "docker", "build", 
            "-f", "api.Dockerfile",
            "-t", "portfolio-api",
            ".", 
            "--progress=plain"
        ], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
        
        print("🏗️ Construction en cours...")
        step_count = 0
        
        for line in iter(process.stdout.readline, ''):
            if not line:
                break
                
            line = line.strip()
            if line:
                # Afficher les étapes importantes
                if line.startswith('#'):
                    step_count += 1
                    print(f"[STEP {step_count}] {line}")
                elif 'Downloading' in line or 'Installing' in line:
                    print(f"[DOWNLOAD] {line[:80]}...")
                elif 'Successfully built' in line:
                    print(f"✅ {line}")
                elif 'ERROR' in line.upper() or 'FAILED' in line.upper():
                    print(f"❌ {line}")
                elif len(line) > 10:  # Éviter les lignes vides
                    print(f"[BUILD] {line[:100]}...")
        
        process.wait()
        
        if process.returncode == 0:
            print("✅ Image API construite avec succès!")
            return True
        else:
            print(f"❌ Construction échouée (code: {process.returncode})")
            return False
            
    except Exception as e:
        print(f"❌ Erreur construction: {e}")
        return False

def start_api_service():
    """Démarre le service API."""
    print("\n🚀 Démarrage du service API...")
    
    try:
        result = subprocess.run([
            "docker-compose", "-f", "docker-compose-production.yml",
            "up", "-d", "portfolio-api"
        ], capture_output=True, text=True)
        
        if result.returncode == 0:
            print("✅ Service API démarré!")
            print("🌐 API disponible sur: http://localhost:8000")
            return True
        else:
            print(f"❌ Erreur démarrage API: {result.stderr}")
            return False
            
    except Exception as e:
        print(f"❌ Erreur: {e}")
        return False

def verify_full_deployment():
    """Vérifie que tout fonctionne."""
    print("\n🧪 Vérification finale...")
    
    services = [
        ("MLflow", "http://localhost:5000"),
        ("Prometheus", "http://localhost:9090"), 
        ("API Portfolio", "http://localhost:8000")
    ]
    
    try:
        import requests
        
        for name, url in services:
            try:
                response = requests.get(url, timeout=10)
                if response.status_code in [200, 401]:
                    print(f"✅ {name}: {url}")
                else:
                    print(f"⚠️ {name}: {url} (code {response.status_code})")
            except:
                print(f"❌ {name}: {url} (non accessible)")
                
    except ImportError:
        print("⚠️ Module requests non disponible pour les tests")
    
    # Vérifier les conteneurs
    try:
        result = subprocess.run(["docker", "ps"], capture_output=True, text=True)
        running_containers = result.stdout.count("Up")
        print(f"📦 {running_containers} conteneurs actifs")
        
        if running_containers >= 3:
            print("🎉 DÉPLOIEMENT COMPLET RÉUSSI!")
            print("\n📋 Services disponibles:")
            print("   🌐 API: http://localhost:8000")
            print("   📊 MLflow: http://localhost:5000") 
            print("   📈 Prometheus: http://localhost:9090")
            print("   📉 Grafana: http://localhost:3000 (admin/admin)")
            return True
        else:
            print("⚠️ Déploiement partiel")
            return False
            
    except Exception as e:
        print(f"❌ Erreur vérification: {e}")
        return False

def main():
    """Point d'entrée principal."""
    print("🔧 FINALISATION DU DÉPLOIEMENT MLOps")
    print("=" * 50)
    
    # 1. Vérifier l'état actuel
    check_existing_services()
    
    # 2. Construire l'image API manquante
    if not build_api_image_verbose():
        print("\n💥 Échec construction image API")
        print("🔧 Essayez manuellement: docker build -f api.Dockerfile -t portfolio-api .")
        return False
    
    # 3. Démarrer le service API
    if not start_api_service():
        print("\n💥 Échec démarrage service API")
        return False
    
    # 4. Vérification finale
    time.sleep(10)  # Attendre que tout démarre
    if verify_full_deployment():
        print("\n🎉 SUCCÈS TOTAL!")
    else:
        print("\n⚠️ Déploiement partiel - vérifiez les logs")
    
    return True

if __name__ == "__main__":
    main()