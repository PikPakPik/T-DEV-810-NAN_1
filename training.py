import os
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
from tensorflow.keras import layers, models
import json
from datetime import datetime
import time
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--epochs', dest='epochs', type=int, help='Set the number of epochs for training', default=30)
args = parser.parse_args()

cwd = os.getcwd()

# Data directories, follow this structure
train_dir = os.path.join(cwd, 'chest_xray', 'train')
validation_dir = os.path.join(cwd, 'chest_xray', 'val')

# Paramètres globaux de traitement des images
IMG_SIZE = (224, 224)  # Taille des images après redimensionnement
BATCH_SIZE = 32        # Taille du batch pour l'entraînement
EPOCHS = args.epochs            # Nombre epochs d'entraînement

# Générateur de données pour l'entraînement avec augmentation d'images
train_datagen = ImageDataGenerator(
    rescale=1./255,           # Normalisation des pixels entre 0 et 1
    rotation_range=20,        # Rotation aléatoire des images jusqu'à 20°
    width_shift_range=0.2,    # Décalage horizontal des images jusqu'à 20%
    height_shift_range=0.2,   # Décalage vertical des images jusqu'à 20%
    zoom_range=0.2,           # Zoom aléatoire jusqu'à 20%
    horizontal_flip=True      # Flip horizontal aléatoire
)

# Générateur de données pour la validation (sans augmentation)
val_datagen = ImageDataGenerator(rescale=1./255)

# Chargement des images d'entraînement à partir du répertoire
train_generator = train_datagen.flow_from_directory(
    train_dir, 
    target_size=IMG_SIZE,     # Redimensionnement des images
    batch_size=BATCH_SIZE,    # Taille du batch
    class_mode='binary'       # Classification binaire (pneumonie ou non)
)

# Chargement des images de validation à partir du répertoire
val_generator = val_datagen.flow_from_directory(
    validation_dir, 
    target_size=IMG_SIZE,     # Redimensionnement des images
    batch_size=BATCH_SIZE,    # Taille du batch
    class_mode='binary'       # Classification binaire (pneumonie ou non)
)

# Définition du modèle de réseau de neurones convolutionnel (CNN)
model = models.Sequential([
    # Première couche de convolution avec 32 filtres de taille 3x3 et activation ReLU
    layers.Conv2D(32, (3,3), activation='relu', input_shape=(224, 224, 3)),
    layers.MaxPooling2D((2,2)),  # Couche de pooling pour réduire la dimension spatiale

    # Deuxième couche de convolution avec 64 filtres
    layers.Conv2D(64, (3,3), activation='relu'),
    layers.MaxPooling2D((2,2)),

    # Troisième couche de convolution avec 128 filtres
    layers.Conv2D(128, (3,3), activation='relu'),
    layers.MaxPooling2D((2,2)),

    # Aplatissement des cartes de caractéristiques
    layers.Flatten(),
    
    # Couche dense entièrement connectée avec 128 neurones et activation ReLU
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.5),  # Dropout de 50% pour éviter le surapprentissage
    
    # Couche de sortie avec une activation sigmoïde pour une classification binaire
    layers.Dense(1, activation='sigmoid')  # Sortie binaire (0 = NORMAL, 1 = PNEUMONIA)
])

# Compilation du modèle avec l'optimiseur Adam et la fonction de perte adaptée à une classification binaire
model.compile(
    optimizer='adam',  # Optimiseur adaptatif qui ajuste le taux d'apprentissage automatiquement
    loss='binary_crossentropy',  # Fonction de perte pour la classification binaire
    metrics=[  # Liste des métriques utilisées pour évaluer la performance du modèle
        'accuracy',  # Précision globale des prédictions
        'categorical_accuracy',  # Exactitude pour chaque catégorie
        'precision',  # Précision des prédictions positives
        'recall',  # Rappel des prédictions positives
        'auc'  # Aire sous la courbe ROC (indicateur global de performance)
    ]
)

# Affichage du résumé du modèle pour visualiser les couches et le nombre de paramètres
model.summary()

# Enregistrement du temps de début de l'entraînement
start_time = time.time()

# Entraînement du modèle sur les données d'entraînement et validation sur les données de validation
history = model.fit(
    train_generator,  # Données d'entraînement générées
    validation_data=val_generator,  # Données de validation
    epochs=EPOCHS  # Nombre epochs définies précédemment
)

# Calcul de la durée totale de l'entraînement
end_time = time.time()
training_duration = end_time - start_time  # Durée en secondes

# Récupération des métriques d'entraînement et de validation au fil des epochs
acc = history.history['accuracy'] # Précision sur l'ensemble d'entraînement
auc = history.history['auc'] # Aire sous la courbe ROC
categorical_accuracy = history.history['categorical_accuracy'] # Exactitude pour chaque catégorie
loss = history.history['loss'] # Perte sur l'ensemble d'entraînement
precision = history.history['precision'] # Précision des prédictions positives
recall = history.history['recall'] # Rappel des prédictions positives
val_accuracy = history.history['val_accuracy'] # Précision sur l'ensemble de validation
val_auc = history.history['val_auc'] # Aire sous la courbe ROC sur l'ensemble de validation
val_categorical_accuracy = history.history['val_categorical_accuracy'] # Exactitude pour chaque catégorie sur l'ensemble de validation
val_loss = history.history['val_loss'] # Perte sur l'ensemble de validation
val_precision = history.history['val_precision'] # Précision des prédictions positives sur l'ensemble de validation
val_recall = history.history['val_recall'] # Rappel des prédictions positives sur l'ensemble de validation

# Définition de la plage des epochs utilisées pour l'entraînement
epochs_range = range(EPOCHS)

# Création du répertoire de sauvegarde du modèle s'il n'existe pas
os.makedirs("models", exist_ok=True)

# Génération d'un nom de fichier unique basé sur la date et l'heure
date = datetime.now().strftime('%Y%m%d%H%M%S')
model_save_path = f"models/pneumonia-{date}.h5"

# Sauvegarde du modèle entraîné au format HDF5
model.save(model_save_path)

# Enregistrement du temps d'entraînement dans un format lisible
log_id = datetime.now().strftime("%Y-%m-%dT%H:%M:%S")
hours, remainder = divmod(training_duration, 3600)
minutes, seconds = divmod(remainder, 60)
formatted_time = f"{int(hours)} hours, {int(minutes)} minutes, {seconds:.2f} seconds"

# Création d'une entrée de log contenant les informations clés de l'entraînement
log_entry = {
    log_id: {
        "model_path": model_save_path,  # Chemin du modèle sauvegardé
        "accuracy": acc[-1],  # Dernière valeur de la précision sur l'ensemble d'entraînement
        "loss": loss[-1],  # Dernière valeur de la perte sur l'ensemble d'entraînement
        "accuracy_validation": val_accuracy[-1],  # Dernière précision sur l'ensemble de validation
        "loss_validation": val_loss[-1],  # Dernière perte sur l'ensemble de validation
        "precision": precision[-1], # Précision des prédictions positives
        "recall": recall[-1], # Rappel des prédictions positives
        "auc": auc[-1], # Aire sous la courbe ROC
        "f1_score": 2 * (precision[-1] * recall[-1]) / (precision[-1] + recall[-1]), # Score F1
        "epochs": EPOCHS,  # Nombre total d'époques d'entraînement
        "batch_size": BATCH_SIZE,  # Taille du batch utilisé
        "img_size": IMG_SIZE,  # Taille des images utilisées
        "training_duration": formatted_time,  # Durée totale de l'entraînement
        "graph_path": f"reports/graphs/accuracy-loss-{date}.png",  # Chemin du graphique généré
        "dataset_info": { # Informations sur les données d'entraînement et de validation
            "train_samples": train_generator.samples, # Nombre d'échantillons dans l'ensemble d'entraînement
            "val_samples": val_generator.samples, # Nombre d'échantillons dans l'ensemble de validation
            "class_distribution": train_generator.class_indices # Distribution des classes
        },
        "metrics_per_epoch": { # Métriques par epoch pour l'entraînement et la validation
            "accuracy": acc, # Précision par epoch sur l'ensemble d'entraînement
            "val_accuracy": val_accuracy, # Précision par epoch sur l'ensemble de validation
            "loss": loss, # Perte par epoch sur l'ensemble d'entraînement
            "val_loss": val_loss, # Perte par epoch sur l'ensemble de validation
            "precision": precision, # Précision par epoch des prédictions positives
            "recall": recall, # Rappel par epoch des prédictions positives
            "auc": auc # Aire sous la courbe ROC par epoch
        },
        "formatted_metrics": { # Métriques formatées pour affichage
            "accuracy": [f"{a:.2%}" for a in acc], # Précision formatée par epoch
            "val_accuracy": [f"{a:.2%}" for a in val_accuracy], # Précision de validation formatée par epoch
            "loss": [f"{l:.4f}" for l in loss], # Perte formatée par epoch
            "val_loss": [f"{l:.4f}" for l in val_loss], # Perte de validation formatée par epoch
            "precision": [f"{p:.2%}" for p in precision], # Précision des prédictions positives formatée par epoch
            "recall": [f"{r:.2%}" for r in recall], # Rappel des prédictions positives formaté par epoch
            "auc": [f"{a:.4f}" for a in auc] # Aire sous la courbe ROC formatée par epoch
        }
    }
}

# Définition du chemin du fichier de log
log_file_path = "reports/training_log.json"

# Création du répertoire s'il n'existe pas
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
    json.dump(existing_logs, f, indent=4)  # Indentation pour une meilleure lisibilité

# Création d'une figure pour afficher les courbes d'apprentissage
plt.figure(figsize=(12, 5))  # Définition de la taille de la figure

# Première sous-figure : Précision du modèle
plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label='Training Accuracy', color='tab:blue')  # Précision sur l'entraînement
plt.plot(epochs_range, val_accuracy, label='Validation Accuracy', color='tab:red')  # Précision sur la validation
plt.xlabel('Epochs')  # Étiquette de l'axe des abscisses
plt.ylabel('Accuracy')  # Étiquette de l'axe des ordonnées
plt.title('Model Accuracy')  # Titre du graphique
plt.legend()  # Affichage de la légende

# Deuxième sous-figure : Perte du modèle
plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label='Training Loss', color='tab:blue')  # Perte sur l'entraînement
plt.plot(epochs_range, val_loss, label='Validation Loss', color='tab:red')  # Perte sur la validation
plt.xlabel('Epochs')  # Étiquette de l'axe des abscisses
plt.ylabel('Loss')  # Étiquette de l'axe des ordonnées
plt.title('Model Loss')  # Titre du graphique
plt.legend(loc=1)  # Affichage de la légende en haut à droite

# # Précision
# plt.subplot(1, 2, 1)
# plt.plot(epochs_range, acc, label='Training Accuracy')
# plt.plot(epochs_range, val_accuracy, label='Validation Accuracy')
# plt.legend(loc='lower right')
# plt.title('Accuracy')

# # Perte
# plt.subplot(1, 2, 2)
# plt.plot(epochs_range, loss, label='Training Loss')
# plt.plot(epochs_range, val_loss, label='Validation Loss')
# plt.legend(loc='upper right')
# plt.title('Loss')

# Création du répertoire s'il n'existe pas
os.makedirs("reports/graphs", exist_ok=True)

# Sauvegarde du graphique sous forme d'image PNG
plt.savefig(f"reports/graphs/accuracy-loss-{date}.png")

# Fermeture de la figure pour éviter l'affichage multiple
plt.close()