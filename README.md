# 🩺 Segmentation de la Rate par 2D U-Net avec contexte 3D - Projet IA Médicale

## 📋 Vue d'ensemble du projet

Ce projet implémente un système de **segmentation de la rate** sur des images de tomodensitométrie (CT) médicales en utilisant un réseau de neurones U-Net 2D avec contexte 3D.

### 🎯 Objectif principal
Projet d'apprentissage en deep learning pour comprendre l'utilisation des CNN/U-Net et la segmentation d'images.

### 🔧 Approche technique
- **Architecture** : U-Net 2D avec 5 slices consécutives comme entrée
- **Datasets** : Deux stratégies (distribué vs adjacent) pour comparer les approches
- **Post-processing** : Reconstruction 3D avec connected components

## 🚀 Installation et utilisation

### 📦 Prérequis
```bash
# Installation des dépendances
pip install -r requirements.txt

# Vérification CUDA (optionnel mais recommandé)
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
```

### ⚙️ Utilisation

**1. Preprocessing des données :**
```bash
python scripts/preprocessing/preprocess_all.py
```

**2. Entraînement :**
```bash
# Dataset SPLIT (distribué)
python scripts/training/train_split.py
# --epochs 100          # Nombre d'epochs (défaut: 100)
# --batch_size 1        # Taille du batch (défaut: 1)
# --lr 0.001            # Learning rate (défaut: 1e-3)
# --num_workers 1       # Workers pour le data loader (défaut: 1)
# --train_patches N     # Nombre de patches train (défaut: tous)
# --val_patches N       # Nombre de patches val (défaut: tous)

# Dataset STACK (adjacent pour post-processing)
python scripts/training/train_stack.py
# --epochs 100          # Nombre d'epochs (défaut: 100)
# --batch_size 1        # Taille du batch (défaut: 1)
# --lr 0.001            # Learning rate (défaut: 1e-3)
# --num_workers 1       # Workers pour le data loader (défaut: 1)
# --train_volumes N     # Nombre de volumes train (défaut: tous)
# --val_volumes N       # Nombre de volumes val (défaut: tous)
```

**3. Évaluation :**
```bash
python scripts/postprocessing/evaluate_volume.py
# --checkpoint PATH     # Chemin vers le checkpoint (.pth) (requis)
# --num_volumes 5       # Nombre de volumes à évaluer (défaut: 5)
```

**4. Analyse des logs :**
```bash
python scripts/utils/analyze_logs.py logs/train_split.log
```

## 📊 Données utilisées

### 🏥 Source des données
- **Dataset** : Medical Segmentation (Spleen) - Kaggle
- **Lien** : https://www.kaggle.com/datasets/dhanvinsankaranand/spleen-segmentation-dataset
- **Type d'images** : Tomodensitométrie (CT) abdominale
- **Format** : Volumes 3D NIfTI (.nii)
- **Résolution** : 512×512 pixels par slice
- **Volumes d'entraînement** : 41 volumes avec annotations manuelles
- **Volumes de test** : 20 volumes pour évaluation finale

### 📈 Stratégies de dataset

**1. Dataset distribué (SPLIT)**
- **Objectif** : Entraînement généralisé sur toute la distribution des données
- **Contenu** : Tous les patches annotés + patches non-annotés répartis uniformément
- **Avantage** : Meilleure généralisation, évite le surapprentissage

**2. Dataset adjacent (STACK)**
- **Objectif** : Optimisation pour reconstruction 3D et post-processing
- **Contenu** : Patches annotés + patches adjacents dans un rayon de ±5 slices
- **Avantage** : Meilleure continuité spatiale pour reconstruction volumique

## 🧠 Architecture du modèle

### 🏗️ Structure du réseau
```
Input (5×512×512) → Encoder → Bottleneck → Decoder → Output (512×512)
     ↓                ↓           ↓          ↓
  5 channels      64→2048     2048→1024   1024→1
   (contexte)     (downsampling) (upsampling)
```

### 🧰 Spécifications techniques
- **Architecture** : U-Net 2D avec Batch Normalization
- **Entrée** : 5 slices consécutives (512×512×5)
- **Sortie** : Masque de segmentation binaire (512×512)
- **Framework** : PyTorch
- **Fonction de perte** : BCEWithLogitsLoss (Binary Cross Entropy)
- **Optimiseur** : Adam avec ReduceLROnPlateau
- **Paramètres** : ~31M paramètres entraînables

## 📁 Structure

```
spleen/
├── data/
│   ├── raw/                        # Dataset original
│   └── processed/
│       ├── patch_analysis.json     # Analyse des patches
│       ├── dataset_split.json      # Dataset distributed
│       └── dataset_stack.json      # Dataset adjacent
├── scripts/
│   ├── preprocessing/
│   │   ├── preprocess_all.py       # Pipeline complet
│   │   ├── fetchdataset.py
│   │   ├── preprocess_slices.py
│   │   ├── create_split_dataset.py
│   │   └── create_stack_dataset.py
│   ├── training/
│   │   ├── train_split.py          # Train sur split
│   │   └── train_stack.py          # Train sur stack
│   ├── postprocessing/
│   │   ├── evaluate_volume.py      # Évaluation 3D
│   │   └── utils.py                # Connected components 3D
│   ├── models/
│   │   └── unet_model.py           # U-Net 2D
│   └── utils/
│       ├── data_loader.py
│       ├── utils.py
│       └── analyze_logs.py         # Analyse des logs
├── logs/                           # Logs d'entraînement
│   ├── train_split.log
│   └── train_stack.log
├── checkpoints/                    # Modèles sauvegardés
│   ├── split/
│   │   ├── checkpoint_epoch_X.pth
│   │   └── best_model.pth
│   └── stack/
│       ├── checkpoint_epoch_X.pth
│       └── best_model.pth
└── results/                        # Résultats d'évaluation
```