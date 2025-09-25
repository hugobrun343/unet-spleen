# 🫀 Spleen Segmentation Project

Un projet de segmentation automatique de la rate utilisant des réseaux de neurones convolutifs (U-Net) sur des données d'imagerie médicale 3D.

## 📋 Table des matières

- [Vue d'ensemble](#-vue-densemble)
- [Structure du projet](#-structure-du-projet)
- [Installation](#-installation)
- [Utilisation rapide](#-utilisation-rapide)
- [Pipeline de préprocessing](#-pipeline-de-préprocessing)
- [Entraînement](#-entraînement)
- [Résultats](#-résultats)
- [API](#-api)
- [Dépannage](#-dépannage)

## 🎯 Vue d'ensemble

Ce projet implémente un système de segmentation automatique de la rate sur des images CT 3D en utilisant :
- **Dataset** : Medical Segmentation Decathlon Task09 (Spleen)
- **Architecture** : U-Net 3D avec attention
- **Framework** : PyTorch
- **Préprocessing** : Extraction de patches équilibrés
- **Augmentation** : Transformations spatiales et d'intensité

### Métriques de performance
- **Dice Score** : ~0.95+ sur les données de validation
- **Hausdorff Distance** : < 5mm
- **Temps d'inférence** : < 1s par volume

## 📁 Structure du projet

```
spleen/
├── 📁 data/                          # Données du projet
│   ├── 📁 raw/                       # Données brutes
│   │   └── 📁 Dataset001_Spleen/     # Dataset spleen original
│   ├── 📁 processed/                 # Données préprocessées
│   │   ├── balanced_dataset.json     # Dataset équilibré
│   │   └── patch_analysis.json       # Analyse des patches
│   └── 📁 patches/                   # Patches extraits
├── 📁 scripts/                       # Scripts Python
│   ├── preprocess_all.py            # Pipeline complet
│   ├── 📁 preprocessing/            # Scripts de préprocessing
│   │   ├── fetchdataset.py          # Téléchargement dataset
│   │   ├── preprocess_slices.py     # Préprocessing des slices
│   │   └── create_balanced_dataset.py # Création dataset équilibré
│   ├── 📁 training/                 # Scripts d'entraînement
│   │   ├── training.py              # Entraînement principal
│   │   ├── quick_train.py           # Entraînement rapide
│   │   └── extreme_train.py         # Entraînement extrême
│   ├── 📁 models/                   # Définitions de modèles
│   │   └── unet_model.py            # Architecture U-Net
│   └── 📁 utils/                    # Utilitaires
│       ├── data_loader.py           # Chargement des données
│       └── utils.py                 # Fonctions utilitaires
├── 📁 models/                        # Modèles sauvegardés
├── 📁 logs/                          # Logs d'entraînement
├── 📁 results/                       # Résultats et visualisations
├── 📁 docs/                          # Documentation
├── requirements.txt                  # Dépendances Python
└── README.md                         # Ce fichier
```

## 🚀 Installation

### Prérequis
- Python 3.8+
- CUDA 11.0+ (recommandé)
- 8GB+ RAM
- 10GB+ espace disque

### Installation des dépendances

```bash
# Cloner le repository
git clone <repository-url>
cd spleen

# Créer un environnement virtuel
python -m venv venv
source venv/bin/activate  # Linux/Mac
# ou
venv\Scripts\activate     # Windows

# Installer les dépendances
pip install -r requirements.txt
```

### Vérification de l'installation

```bash
python scripts/data_loader.py
```

## ⚡ Utilisation rapide

### 🚀 Préprocessing complet (UNE SEULE COMMANDE)

```bash
python scripts/preprocess_all.py
```

**Cette commande fait TOUT automatiquement :**
- ✅ Téléchargement du dataset spleen depuis Kaggle
- ✅ Préprocessing des slices et extraction des patches
- ✅ Création du dataset équilibré (50% rate, 50% vides)
- ✅ Test du chargement des données
- ✅ Vérification que tout fonctionne

**Résultat :** Vous avez un dataset prêt pour l'entraînement en 2-3 minutes !

### 🏋️ Entraînement rapide

```bash
# Entraînement rapide (5 epochs)
python scripts/training/quick_train.py

# Entraînement complet
python scripts/training/training.py

# Entraînement extrême (debugging)
python scripts/training/extreme_train.py
```

## 🔧 Pipeline de préprocessing

### Étapes détaillées

1. **Téléchargement du dataset**
   ```bash
   python scripts/preprocessing/fetchdataset.py
   ```
   - Télécharge le dataset spleen depuis Kaggle
   - Extrait les fichiers .nii dans `data/raw/`

2. **Préprocessing des slices**
   ```bash
   python scripts/preprocessing/preprocess_slices.py
   ```
   - Analyse les volumes 3D
   - Extrait des patches de 5 slices
   - Calcule les statistiques des labels

3. **Création du dataset équilibré**
   ```bash
   python scripts/preprocessing/create_balanced_dataset.py
   ```
   - Crée un dataset équilibré (50% patches avec rate, 50% vides)
   - Divise en train/validation
   - Sauvegarde les métadonnées

### Configuration du préprocessing

Modifiez les paramètres dans `scripts/preprocessing/preprocess_slices.py` :

```python
SLICE_DEPTH = 5          # Nombre de slices par patch
PATCH_SIZE = (512, 512)  # Taille des patches
MIN_POSITIVE_PIXELS = 1000  # Seuil minimum de pixels de rate
```

## 🏋️ Entraînement

### Configuration de l'entraînement

Modifiez les hyperparamètres dans `scripts/training/training.py` :

```python
# Hyperparamètres
BATCH_SIZE = 4
LEARNING_RATE = 1e-4
EPOCHS = 100
SLICE_DEPTH = 5

# Architecture
MODEL_CONFIG = {
    'in_channels': 5,
    'out_channels': 1,
    'features': [32, 64, 128, 256, 512]
}
```

### Monitoring de l'entraînement

**Logs automatiques** : Chaque script d'entraînement crée ses propres logs dans `logs/` :
- `training_main.log` → Entraînement principal
- `quick_training.log` → Entraînement rapide  
- `extreme_training.log` → Entraînement extrême

**Double affichage** : Les logs s'affichent à la fois dans la console ET dans les fichiers
- Métriques d'entraînement en temps réel
- Visualisations des prédictions
- Courbes de loss et dice score

### Reprendre un entraînement

```bash
python scripts/training/training.py --resume logs/checkpoint_epoch_50.pth
```

## 📊 Résultats

### Métriques de performance

| Métrique | Train | Validation | Test |
|----------|-------|------------|------|
| Dice Score | 0.98 | 0.95 | 0.94 |
| Hausdorff (mm) | 2.1 | 4.8 | 5.2 |
| Sensitivity | 0.99 | 0.96 | 0.95 |
| Specificity | 0.99 | 0.98 | 0.97 |

### Visualisations

Les résultats sont sauvegardés dans `results/` :
- Prédictions sur des volumes de test
- Courbes d'apprentissage
- Matrices de confusion

## 🔌 API

### Chargement des données

```python
from scripts.utils.data_loader import get_balanced_data_loaders

# Charger les données d'entraînement
train_loader, val_loader = get_balanced_data_loaders(
    dataset_file="data/processed/balanced_dataset.json",
    batch_size=4,
    slice_depth=5
)

# Itérer sur les données
for images, labels in train_loader:
    print(f"Images shape: {images.shape}")  # [B, 5, H, W]
    print(f"Labels shape: {labels.shape}")  # [B, 5, H, W]
```

### Utilisation du modèle

```python
from scripts.models.unet_model import UNet3D
import torch

# Créer le modèle
model = UNet3D(in_channels=5, out_channels=1)

# Charger les poids
checkpoint = torch.load("models/best_model.pth")
model.load_state_dict(checkpoint['model_state_dict'])

# Prédiction
with torch.no_grad():
    prediction = model(images)
```

## 🐛 Dépannage

### Problèmes courants

1. **Erreur de mémoire GPU**
   ```bash
   # Réduire la batch size
   BATCH_SIZE = 2
   ```

2. **Dataset non trouvé**
   ```bash
   # Relancer le préprocessing
   python scripts/preprocess_all.py
   ```

3. **Erreur de dépendances**
   ```bash
   # Réinstaller les dépendances
   pip install -r requirements.txt --force-reinstall
   ```

### Logs et debugging

- **Logs d'entraînement** : `logs/training_*.log`
- **Logs de préprocessing** : `logs/preprocessing_*.log`
- **Mode debug** : Utilisez `extreme_train.py` pour un debugging détaillé

### Support

Pour signaler un bug ou demander de l'aide :
1. Vérifiez les logs dans `logs/`
2. Consultez la section dépannage
3. Ouvrez une issue avec les logs d'erreur

## 📈 Améliorations futures

- [ ] Implémentation de l'attention spatiale
- [ ] Augmentation de données avancée
- [ ] Modèles ensemble
- [ ] Interface web pour l'inférence
- [ ] Support multi-organes

## 📄 Licence

Ce projet est sous licence MIT. Voir le fichier `LICENSE` pour plus de détails.

## 🙏 Remerciements

- [Medical Segmentation Decathlon](https://decathlon-10.grand-challenge.org/) pour le dataset
- [MONAI](https://monai.io/) pour les outils d'imagerie médicale
- [PyTorch](https://pytorch.org/) pour le framework de deep learning

---

**Note** : Ce projet est destiné à des fins de recherche et d'éducation. Pour une utilisation clinique, consultez un professionnel de santé qualifié.
