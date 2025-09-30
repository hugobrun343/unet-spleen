# Spleen Segmentation - 2D U-Net avec contexte 3D

Segmentation automatique de la rate avec U-Net 2D (5 slices de contexte) sur des images CT médicales.

## 🚀 Quick Start

### 1. Préprocessing
```bash
# Preprocessing complet (télécharge, analyse, crée les 2 datasets)
python scripts/preprocessing/preprocess_all.py
```

### 2. Entraînement

**Dataset SPLIT (labeled répartis partout) :**
```bash
python scripts/training/train_split.py --epochs 100
```

**Dataset STACK (labeled + adjacents pour post-processing) :**
```bash
python scripts/training/train_stack.py --epochs 100
```

**Paramètres train_split.py (PATCHES) :**
```bash
--epochs 100          # Nombre d'epochs (défaut: 100)
--batch_size 1        # Taille du batch (défaut: 1)
--lr 0.001            # Learning rate (défaut: 1e-3)
--num_workers 1       # Workers pour le data loader (défaut: 1)
--train_patches 400   # Nombre de patches train (défaut: None = tous)
--val_patches 100     # Nombre de patches val (défaut: None = tous)
```

**Paramètres train_stack.py (VOLUMES) :**
```bash
--epochs 100          # Nombre d'epochs (défaut: 100)
--batch_size 1        # Taille du batch (défaut: 1)
--lr 0.001            # Learning rate (défaut: 1e-3)
--num_workers 1       # Workers pour le data loader (défaut: 1)
--train_volumes 20    # Nombre de volumes train (défaut: None = tous)
--val_volumes 5       # Nombre de volumes val (défaut: None = tous)
```

**Exemples d'utilisation :**
```bash
# SPLIT: Quick test avec 400 train patches + 100 val patches (50/50 labeled/unlabeled auto)
python scripts/training/train_split.py --epochs 50 --train_patches 400 --val_patches 100

# STACK: Train sur 20 volumes (tous leurs patches) + 5 volumes val
python scripts/training/train_stack.py --epochs 100 --train_volumes 20 --val_volumes 5

# Full training avec tous les volumes (pour post-processing optimal)
python scripts/training/train_stack.py --epochs 200

# Training avec batch size plus grand (si GPU le permet)
python scripts/training/train_split.py --epochs 100 --batch_size 4
```

### 3. Évaluation avec post-processing
```bash
# Évaluer avec connected components (vire les faux positifs)
python scripts/postprocessing/evaluate_volume.py \
    --checkpoint checkpoints/stack/best_model.pth \
    --num_volumes 5
```

**Paramètres évaluation :**
```bash
--checkpoint PATH     # Chemin vers le checkpoint (.pth) (requis)
--num_volumes 5       # Nombre de volumes à évaluer (défaut: 5)
```

### 4. Analyse des logs
```bash
# Générer résumé + graphiques depuis un log
python scripts/utils/analyze_logs.py logs/train_split.log
```

Génère automatiquement :
- `train_split_summary.txt` - Résumé texte détaillé
- `train_split_loss.png` - Graphique Loss
- `train_split_dice.png` - Graphique Dice Score
- `train_split_iou.png` - Graphique IoU
- `train_split_lr.png` - Graphique Learning Rate

## 📊 Datasets

**2 datasets disponibles :**

1. **`dataset_split.json`** (distributed)
   - TOUS les patches labeled (1207)
   - Même quantité d'unlabeled (933) répartis uniformément
   - Unlabeled évitent les bordures (10%)
   - ~28000 lignes

2. **`dataset_stack.json`** (adjacent for post-processing)
   - TOUS les patches labeled (1207)
   - Unlabeled AUTOUR des labeled (470) dans un rayon de ±5 slices
   - Optimisé pour reconstruction 3D et post-processing
   - ~21000 lignes

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

## 🧠 Architecture

- **Modèle** : U-Net 2D avec 5 slices de contexte (5 channels d'entrée)
- **Input** : 5 slices consécutives de 512x512 pixels
- **Output** : 1 slice de segmentation 512x512 pixels
- **Framework** : PyTorch
- **Loss** : BCEWithLogitsLoss
- **Optimizer** : Adam avec ReduceLROnPlateau

## 🧹 Post-processing

Le post-processing 3D avec **connected components** améliore les résultats :
- Reconstruction du volume 3D depuis les prédictions 2D
- Détection des composantes connexes
- Conservation uniquement du plus gros blob (la rate)
- Suppression des faux positifs

**Pourquoi dataset_stack ?** Les unlabeled adjacents aux labeled permettent une meilleure reconstruction du volume complet.

## 🔧 Configuration

**Dataset :**
- Source : Medical Segmentation Decathlon Task09 (Spleen)
- 41 volumes d'entraînement
- 20 volumes de test
- Résolution : 512x512 pixels
- Slice depth : 5

**Checkpoints :**
- Sauvegarde du dernier checkpoint à chaque epoch
- Sauvegarde du meilleur modèle (Val Dice) dans `best_model.pth`

**Logs :**
- Console + fichier log
- Métriques : Loss, Dice, IoU
- Learning rate à chaque epoch

**Analyse des logs :**
```bash
# Générer graphiques et résumé depuis un log
python scripts/utils/analyze_logs.py logs/train_split.log
```

## 💡 Problèmes courants

**CUDA out of memory :**
```bash
# Tuer les processus
pkill -f python

# Réduire le batch size
python scripts/training/train_split.py --batch_size 1
```

**Dataset manquant :**
```bash
python scripts/preprocess_all.py
```

## 📈 Résultats attendus

- **Dice Score** : ~0.60-0.70 (patch-based)
- **Amélioration post-processing** : +5-10% Dice
- **Temps d'entraînement** : ~2-5h pour 100 epochs (selon GPU)

## 🎯 Workflow recommandé

### Quick Start (test rapide)
```bash
# 1. Preprocessing complet
python scripts/preprocessing/preprocess_all.py

# 2. Quick training (500 patches, 50 epochs)
python scripts/training/train_stack.py --epochs 50 --max_patches 500

# 3. Analyser les résultats
python scripts/utils/analyze_logs.py logs/train_stack.log

# 4. Évaluer avec post-processing
python scripts/postprocessing/evaluate_volume.py \
    --checkpoint checkpoints/stack/best_model.pth \
    --num_volumes 3
```

### Full Training (production)
```bash
# 1. Train sur tous les patches
python scripts/training/train_stack.py --epochs 200

# 2. Évaluer sur plus de volumes
python scripts/postprocessing/evaluate_volume.py \
    --checkpoint checkpoints/stack/best_model.pth \
    --num_volumes 10

# 3. Analyser
python scripts/utils/analyze_logs.py logs/train_stack.log
```
