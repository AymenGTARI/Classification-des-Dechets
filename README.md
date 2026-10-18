# ♻️ Classification des déchets 

## 📌 Présentation du projet

Ce projet a pour objectif de développer un système intelligent capable de **classifier automatiquement les déchets à partir d'images** en utilisant le Deep Learning et la vision par ordinateur.

Le modèle repose sur l'architecture **ResNet50**, un réseau de neurones convolutif pré-entraîné sur ImageNet, associé à la technique du *Transfer Learning* (apprentissage par transfert).

Le développement et l'entraînement sont réalisés sur Google Colab avec TensorFlow et Keras. Le projet comprend la préparation des données, l'entraînement du modèle, l'évaluation de ses performances et la prédiction des catégories de déchets avec leurs scores de confiance.

## 🎯 Objectifs

- Automatiser la classification des déchets à partir d'images.
- Utiliser l'apprentissage par transfert avec ResNet50.
- Améliorer la généralisation du modèle grâce à l'augmentation des données.
- Gérer le déséquilibre entre les classes à l'aide de poids de classe.
- Évaluer les performances du modèle à l'aide de métriques adaptées.
- Afficher les probabilités de prédiction pour chaque catégorie de déchets.

## 🛠️ Technologies utilisées

- **Python** : langage de programmation.
- **Google Colab** : environnement de développement et d'entraînement.
- **TensorFlow et Keras** : création et entraînement du modèle de Deep Learning.
- **ResNet50** : architecture de réseau de neurones convolutif pré-entraînée.
- **NumPy** : calculs numériques.
- **Pandas** : manipulation des données.
- **Matplotlib et Seaborn** : visualisation des données et des résultats.
- **Pillow (PIL)** : traitement des images.
- **Scikit-learn** : calcul des poids de classe et évaluation du modèle.

## 🧠 Architecture du modèle

Le modèle utilise ResNet50 pré-entraîné sur ImageNet comme extracteur de caractéristiques. La base pré-entraînée est gelée pendant l'entraînement afin de conserver ses poids.

Les couches ajoutées sont les suivantes :

1. **ResNet50** : extraction des caractéristiques visuelles des images.
2. **Global Average Pooling** : réduction des cartes de caractéristiques.
3. **Couche Dense de 128 neurones** : apprentissage des caractéristiques spécifiques aux déchets.
4. **Batch Normalization** : stabilisation de l'entraînement.
5. **Activation ReLU** : introduction de la non-linéarité.
6. **Dropout de 50 %** : réduction du risque de surapprentissage.
7. **Couche de sortie Softmax** : calcul des probabilités associées aux différentes catégories.

Cette architecture permet de tirer parti des caractéristiques déjà apprises par ResNet50 tout en adaptant la classification au jeu de données utilisé.

## 📂 Jeu de données

Le projet utilise des images de déchets organisées dans des dossiers correspondant à leurs catégories.

Exemple de structure :

```text
TrashType_Image_Dataset/
├── cardboard/
├── glass/
├── metal/
├── paper/
├── plastic/
└── trash/
```

Un second répertoire, nommé `extras`, peut également être utilisé pour intégrer des images supplémentaires.

Le programme parcourt les dossiers, récupère les chemins des images et associe chaque image à son étiquette. Il vérifie ensuite la validité des fichiers et élimine les images corrompues ou qui ne sont pas au format RGB.

**Remarque :** les catégories réelles dépendent des dossiers présents dans les jeux de données. Les images ne sont pas automatiquement incluses dans ce dépôt GitHub.

## ⚙️ Étapes de réalisation

Le projet est organisé selon les étapes suivantes :

1. Montage de Google Drive et importation des bibliothèques nécessaires.
2. Définition des paramètres d'entraînement.
3. Chargement des images et création d'un DataFrame contenant les chemins et les catégories.
4. Nettoyage des données et vérification de la validité des images.
5. Visualisation de quelques exemples d'images.
6. Analyse de la distribution des classes.
7. Préparation des générateurs d'entraînement et de validation.
8. Application de techniques d'augmentation des données : rotation, translation, zoom, retournement horizontal et variation de luminosité.
9. Calcul des poids de classe pour limiter l'effet du déséquilibre des données.
10. Construction du modèle de classification basé sur ResNet50.
11. Entraînement du modèle avec la méthode Early Stopping.
12. Visualisation des courbes de précision et de perte.
13. Évaluation à l'aide d'un rapport de classification et d'une matrice de confusion.
14. Test du modèle sur des images individuelles avec affichage des scores de confiance.

## 🚀 Installation et utilisation

### 1. Ouvrir le notebook

Ouvrir le notebook `vpo_proj_v2.ipynb` dans [Google Colab](https://colab.research.google.com/).

### 2. Préparer le jeu de données

Importer les jeux de données dans Google Drive, puis adapter les chemins dans le code :

```python
dataset_paths = [
    "/content/drive/MyDrive/TrashType_Image_Dataset",
    "/content/drive/MyDrive/extras"
]
```

### 3. Configurer les paramètres

Les paramètres principaux sont définis comme suit :

```python
IMG_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 5
```

- `IMG_SIZE` : dimensions des images utilisées par le modèle.
- `BATCH_SIZE` : nombre d'images traitées à chaque lot.
- `EPOCHS` : nombre maximal de passages sur les données d'entraînement.

Ces paramètres peuvent être ajustés selon la taille du jeu de données et les ressources disponibles.

### 4. Exécuter le projet

Exécuter les cellules du notebook dans l'ordre afin de réaliser le prétraitement, l'entraînement, l'évaluation et les prédictions.

L'utilisation d'un environnement Google Colab avec accélération GPU est recommandée pour réduire le temps d'entraînement.

## 📊 Évaluation des performances

Les performances du modèle sont analysées à l'aide de plusieurs indicateurs :

- **Accuracy (précision globale)** : proportion d'images correctement classifiées.
- **Loss (fonction de perte)** : mesure l'erreur des prédictions du modèle.
- **Rapport de classification** : fournit la précision, le rappel et le score F1 pour chaque classe.
- **Matrice de confusion** : permet d'identifier les catégories fréquemment confondues.
- **Scores de confiance** : indiquent les probabilités prédites pour chaque catégorie lors de la classification d'une image.

Les résultats numériques dépendent de l'exécution du modèle et des données utilisées.

## 🔍 Prédiction sur une nouvelle image

Le modèle peut recevoir une image individuelle et calculer les probabilités associées à toutes les catégories apprises.

Exemple de format de sortie :

```text
Prédiction pour l'image : plastic_1049.jpg

cardboard : XX.XX %
glass     : XX.XX %
metal     : XX.XX %
paper     : XX.XX %
plastic   : XX.XX %
trash     : XX.XX %
```

Les valeurs `XX.XX %` sont des exemples de présentation et ne représentent pas des résultats réels.

## 📁 Structure du dépôt GitHub

```text
waste-classification/
├── vpo_proj_v2.ipynb
├── README.md
├── requirements.txt
└── models/
    └── .gitkeep
```

- `vpo_proj_v2.ipynb` : notebook contenant le code complet du projet.
- `README.md` : documentation du projet.
- `requirements.txt` : liste des bibliothèques Python nécessaires.
- `models/` : répertoire facultatif destiné au stockage du modèle entraîné.

## 📦 Bibliothèques nécessaires

Les principales dépendances peuvent être installées avec la commande suivante :

```bash
pip install tensorflow numpy pandas matplotlib seaborn pillow scikit-learn
```

Google Colab fournit déjà plusieurs de ces bibliothèques.

## 🔮 Perspectives d'amélioration

Plusieurs améliorations peuvent être envisagées :

- Affiner certaines couches de ResNet50 pour améliorer les performances.
- Augmenter la quantité et la diversité des images du jeu de données.
- Comparer ResNet50 avec d'autres architectures, telles que MobileNetV2 et EfficientNet.
- Exporter le modèle entraîné pour une utilisation indépendante de Google Colab.
- Développer une application web ou mobile permettant de classifier des déchets à partir de photographies.
- Intégrer le système dans une solution de gestion intelligente des déchets.

## 👨‍💻 Informations sur le projet

- **Domaine :** Intelligence artificielle et vision par ordinateur.
- **Approche :** Deep Learning et apprentissage par transfert.
- **Architecture :** ResNet50.
- **Environnement :** Google Colab.
- **Framework :** TensorFlow / Keras.
- **Langage :** Python.

---

*Ce projet illustre l'utilisation du Deep Learning pour automatiser la reconnaissance et la classification des déchets à partir d'images, dans le but de contribuer à des solutions intelligentes de gestion et de tri des déchets.*

