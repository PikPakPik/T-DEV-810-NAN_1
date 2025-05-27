import os
import sys

# Vérifier le répertoire de travail actuel
print("Répertoire de travail actuel :", os.getcwd())

# Vérifier si le dossier common existe
if os.path.exists('common'):
    print("✅ Le dossier 'common' est trouvé")
    if os.path.exists('common/__init__.py'):
        print("✅ Le fichier '__init__.py' existe dans common/")
    else:
        print("❌ Le fichier '__init__.py' est manquant dans common/")
    
    if os.path.exists('common/fill_data.py'):
        print("✅ Le fichier 'fill_data.py' existe dans common/")
    else:
        print("❌ Le fichier 'fill_data.py' est manquant dans common/")
else:
    print("❌ Le dossier 'common' n'est pas trouvé dans le répertoire actuel")
    print("Vous devez naviguer vers le répertoire racine du projet")

# Ajouter le répertoire actuel au PYTHONPATH si nécessaire
if os.getcwd() not in sys.path:
    sys.path.insert(0, os.getcwd())
    print("✅ Répertoire ajouté au PYTHONPATH")

# Test d'importation
try:
    from common.fill_data import generate_augmented_images
    print("✅ Importation réussie !")
except ImportError as e:
    print(f"❌ Erreur d'importation : {e}") 