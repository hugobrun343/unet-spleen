# Spleen Segmentation - 3D U-Net

Segmentation automatique de la rate avec U-Net 3D sur des images CT médicales.

## Installation

```bash
pip install -r requirements.txt
```

## Utilisation

### Préprocessing complet
```bash
python scripts/preprocess_all.py
```

### Entraînement
```bash
# Rapide (100 epochs, 10 patches)
python scripts/training/quick_train.py

# Complet (dataset entier)
python scripts/training/training.py

# Debug (1 patch)
python scripts/training/extreme_train.py

# Personnalisé
python scripts/training/custom_train.py --epochs 200 --total_patches 50
```

## Structure

```
spleen/
├── data/                    # Données
│   ├── raw/                # Dataset original (ignoré par git)
│   └── processed/          # Dataset préprocessé
├── scripts/
│   ├── preprocess_all.py   # Pipeline complet
│   ├── preprocessing/      # Scripts de préprocessing
│   ├── training/           # Scripts d'entraînement
│   ├── models/             # Architecture U-Net
│   └── utils/              # Utilitaires
├── logs/                   # Logs d'entraînement
├── checkpoints/            # Modèles sauvegardés
└── results/                # Résultats
```

## Configuration

- **Dataset** : Medical Segmentation Decathlon Task09 (Spleen)
- **Architecture** : U-Net 3D
- **Framework** : PyTorch
- **Patches** : 5 slices, 512x512 pixels
- **Split** : 50% rate, 50% vides

## Problèmes courants

**CUDA out of memory** :
```bash
pkill -f python  # Tuer les processus en cours
# Puis réduire BATCH_SIZE = 1 dans les scripts
```

**Dataset manquant** :
```bash
python scripts/preprocess_all.py
```

## Résultats

- Dice Score : ~0.95
- Hausdorff Distance : < 5mm
- Temps d'inférence : < 1s par volume