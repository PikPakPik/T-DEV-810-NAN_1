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
    
    if os.path.exists('common/data_augmentation.py'):
        print("✅ Le fichier 'data_augmentation.py' existe dans common/")
    else:
        print("❌ Le fichier 'data_augmentation.py' est manquant dans common/")
else:
    print("❌ Le dossier 'common' n'est pas trouvé dans le répertoire actuel")
    print("Vous devez naviguer vers le répertoire racine du projet")

# Ajouter le répertoire actuel au PYTHONPATH si nécessaire
if os.getcwd() not in sys.path:
    sys.path.insert(0, os.getcwd())
    print("✅ Répertoire ajouté au PYTHONPATH")

# Test d'importation
try:
    from common.data_augmentation import generate_augmented_images
    print("✅ Importation réussie !")
except ImportError as e:
    print(f"❌ Erreur d'importation : {e}") 