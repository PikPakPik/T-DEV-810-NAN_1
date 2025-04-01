import os
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import load_model
import json
from datetime import datetime
import numpy as np
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_curve, auc, classification_report
import matplotlib.pyplot as plt

# Définition du répertoire contenant les modèles sauvegardés
model_dir = os.path.join(os.getcwd(), 'models')
model_files = [f for f in os.listdir(model_dir) if f.endswith(".h5")]

# Vérification de la présence de modèles disponibles
if not model_files:
    print("Aucun modèle trouvé dans le dossier 'models'.")
    exit()

# Affichage de la liste des modèles disponibles
print("Modèles disponibles :")
for idx, m in enumerate(model_files):
    print(f"{idx} : {m}")

# Demande de sélection du modèle à charger
choice = input("Choisissez le numéro du modèle à utiliser : ")

# Validation de l'entrée utilisateur
try:
    choice_index = int(choice)
    chosen_model_file = model_files[choice_index]
except (ValueError, IndexError):
    print("Choix invalide.")
    exit()

# Chargement du modèle sélectionné
model_path = os.path.join(model_dir, chosen_model_file)
model = load_model(model_path)
print(f"Modèle chargé : {chosen_model_file}")

# Définition du chemin du dossier de test
test_dir = os.path.join(os.getcwd(), 'chest_xray', 'test')

# Paramètres d'entrée des images pour l'évaluation
IMG_SIZE = (224, 224)  # Taille des images d'entrée
BATCH_SIZE = 32  # Taille du batch pour le générateur d'images

# Préparation des images de test avec normalisation des pixels
test_datagen = ImageDataGenerator(rescale=1./255)

# Création du générateur d'images de test
test_generator = test_datagen.flow_from_directory(
    test_dir, # Chemin du dossier de test
    target_size=IMG_SIZE, # Redimensionnement des images
    batch_size=BATCH_SIZE, # Taille du batch
    class_mode='binary'  # Classification binaire (Pneumonie vs Normal)
)

# Évaluation du modèle sur l'ensemble de test
test_loss, test_acc, test_auc, test_categorical_acc, test_precision, test_recall = model.evaluate(test_generator)

# Récupération des classes réelles et des prédictions
y_true = test_generator.classes  # Classes réelles des images de test
y_pred = model.predict(test_generator)  # Prédictions du modèle
y_pred = np.round(y_pred).astype(int)  # Seuil de décision fixé à 0.5 pour classification binaire

# Génération de la matrice de confusion
cm = confusion_matrix(y_true, y_pred)

# Génération d'un identifiant de log basé sur la date et l'heure
date = datetime.now().strftime('%Y%m%d%H%M%S')
log_id = datetime.now().strftime("%Y-%m-%dT%H:%M:%S")

# Création d'une entrée de log avec les résultats du test
log_entry = {
    log_id: {
        "accuracy": test_acc,  # Précision finale sur l'ensemble de test
        "loss": test_loss,  # Perte finale sur l'ensemble de test
        "auc": test_auc,  # AUC sur l'ensemble de test
        "categorical_accuracy": test_categorical_acc,  # Précision catégorielle
        "precision": test_precision,  # Précision
        "recall": test_recall,  # Rappel
        "samples": test_generator.samples,  # Nombre d'échantillons testés
        "confusion_matrix": cm.tolist(),  # Matrice de confusion au format liste
        "graph_path": f"reports/graphs/confusion-matrix-{date}.png",  # Chemin du graphique sauvegardé
        "formatted_metrics": { # Métriques formatées pour affichage
            "formatted_accuracy": f"{test_acc:.2%}",  # Format en pourcentage
            "formatted_loss": f"{test_loss:.2%}", # Format en pourcentage
            "formatted_auc": f"{test_auc:.2%}", # Format en pourcentage
            "formatted_categorical_accuracy": f"{test_categorical_acc:.2%}", # Format en pourcentage
            "formatted_precision": f"{test_precision:.2%}", # Format en pourcentage
            "formatted_recall": f"{test_recall:.2%}", # Format en pourcentage
        }
    }
}

# Définition du chemin du fichier de log
log_file_path = "reports/test_log.json"

# Création du répertoire de logs s'il n'existe pas
os.makedirs("reports", exist_ok=True)

# Chargement des logs existants s'ils sont déjà présents
if os.path.exists(log_file_path):
    with open(log_file_path, "r") as f:
        existing_logs = json.load(f)
else:
    existing_logs = {}

# Mise à jour du fichier de log avec la nouvelle entrée
existing_logs.update(log_entry)

# Écriture des logs mis à jour dans le fichier JSON
with open(log_file_path, "w") as f:
    json.dump(existing_logs, f, indent=4)

# Création et affichage de la matrice de confusion avec seaborn
sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=["NORMAL", "PNEUMONIA"],
    yticklabels=["NORMAL", "PNEUMONIA"]
)
plt.xlabel("Prédictions")
plt.ylabel("Réelles")
plt.title("Matrice de Confusion")

# Création du dossier pour les graphiques s'il n'existe pas
os.makedirs("reports/graphs", exist_ok=True)

# Sauvegarde du graphique de la matrice de confusion
plt.savefig(f"reports/graphs/confusion-matrix-{date}.png")

# Fermeture de la figure pour éviter l'affichage multiple
plt.close()

report = classification_report(y_true, y_pred, target_names=['Class 0', 'Class 1'], labels=[0, 1], zero_division=0)

# Courbes ROC
fpr = dict()
tpr = dict()
roc_auc = dict()

# Affichage des courbes ROC
plt.figure(figsize=(8, 6))
for i in range(2):
    fpr[i], tpr[i], _ = roc_curve(y_true, [1 if x == i else 0 for x in y_pred])
    roc_auc[i] = auc(fpr[i], tpr[i])
    plt.plot(fpr[i], tpr[i], label=f'Classe {i} (AUC = {roc_auc[i]:0.2f})')

plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
plt.xlabel('Taux de Faux Positifs')
plt.ylabel('Taux de Vrais Positifs')
plt.title('Courbes ROC')
plt.legend(loc='lower right')

# Sauvegarde du graphique de la matrice de confusion
plt.savefig(f"reports/graphs/courbes-roc-{date}.png")

# Fermeture de la figure pour éviter l'affichage multiple
plt.close()