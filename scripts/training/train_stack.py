#!/usr/bin/env python3
"""
Training on STACK dataset (all labeled + adjacent unlabeled)
- Uses dataset_stack.json
- All labeled patches + unlabeled around labeled regions
- Optimized for volume reconstruction and post-processing
"""

import sys
import time
from pathlib import Path
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau

sys.path.append(str(Path(__file__).parent.parent))

from models.unet_model import UNet
from utils.data_loader import get_balanced_data_loaders
from utils.utils import dice_coefficient, iou_score, save_checkpoint

LOG_FILE = None

def log_message(message):
    """Log to console and file"""
    print(message)
    if LOG_FILE:
        with open(LOG_FILE, 'a') as f:
            f.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')} - {message}\n")

def train_epoch(model, loader, criterion, optimizer, device, epoch, log_every=100):
    """Train one epoch"""
    model.train()
    total_loss, total_dice, total_iou, num_batches = 0.0, 0.0, 0.0, 0
    
    for batch_idx, (images, masks) in enumerate(loader, 1):
        images, masks = images.to(device), masks.to(device)
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, masks)
        loss.backward()
        optimizer.step()
        
        with torch.no_grad():
            batch_dice = dice_coefficient(outputs, masks)
            batch_iou = iou_score(outputs, masks)
            total_loss += loss.item()
            total_dice += batch_dice
            total_iou += batch_iou
            num_batches += 1
            
            # Log every N batches
            if batch_idx % log_every == 0:
                log_message(f"  Batch {batch_idx}/{len(loader)} - Loss: {loss.item():.4f}, Dice: {batch_dice:.4f}, IoU: {batch_iou:.4f}")
    
    return total_loss/num_batches, total_dice/num_batches, total_iou/num_batches

def validate_epoch(model, loader, criterion, device):
    """Validate one epoch"""
    model.eval()
    total_loss, total_dice, total_iou, num_batches = 0.0, 0.0, 0.0, 0
    
    with torch.no_grad():
        for images, masks in loader:
            images, masks = images.to(device), masks.to(device)
            outputs = model(images)
            
            total_loss += criterion(outputs, masks).item()
            total_dice += dice_coefficient(outputs, masks)
            total_iou += iou_score(outputs, masks)
            num_batches += 1
    
    return total_loss/num_batches, total_dice/num_batches, total_iou/num_batches

def main():
    parser = argparse.ArgumentParser(description="Train on stack dataset")
    parser.add_argument('--epochs', type=int, default=100, help="Number of epochs")
    parser.add_argument('--batch_size', type=int, default=1, help="Batch size")
    parser.add_argument('--lr', type=float, default=1e-3, help="Learning rate")
    parser.add_argument('--num_workers', type=int, default=1, help="Data loader workers")
    parser.add_argument('--train_volumes', type=int, default=None, help="Number of train volumes (None = all)")
    parser.add_argument('--val_volumes', type=int, default=None, help="Number of val volumes (None = all)")
    args = parser.parse_args()
    
    # Setup
    global LOG_FILE
    log_dir = Path('/teamspace/studios/this_studio/spleen/logs')
    log_dir.mkdir(parents=True, exist_ok=True)
    LOG_FILE = log_dir / 'train_stack.log'
    with open(LOG_FILE, 'w') as f:
        f.write(f"Training started at {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
    
    checkpoint_dir = Path('/teamspace/studios/this_studio/spleen/checkpoints/stack')
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    # Clean old checkpoints
    for ckpt in checkpoint_dir.glob("*.pth"):
        ckpt.unlink()
        log_message(f"🗑️  Removed: {ckpt.name}")
    
    log_message(f"{'='*70}")
    log_message(f"🚀 TRAINING ON STACK DATASET")
    log_message(f"{'='*70}")
    log_message(f"Epochs: {args.epochs}, Batch: {args.batch_size}, LR: {args.lr}")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    log_message(f"Device: {device}")
    
    # Load data
    dataset_file = "/teamspace/studios/this_studio/spleen/data/processed/dataset_stack.json"
    log_message(f"\n📂 Dataset: dataset_stack.json (STACK - full volumes with adjacent patches)")
    
    # Load and filter by volumes
    import json
    with open(dataset_file, 'r') as f:
        dataset = json.load(f)
    
    train_patches = dataset['train_patches']
    val_patches = dataset['val_patches']
    
    # Filter by number of volumes if specified
    if args.train_volumes:
        train_volume_indices = sorted(set(p['volume_idx'] for p in train_patches))[:args.train_volumes]
        train_patches = [p for p in train_patches if p['volume_idx'] in train_volume_indices]
        log_message(f"⚠️  Using {args.train_volumes} train volumes → {len(train_patches)} patches")
    
    if args.val_volumes:
        val_volume_indices = sorted(set(p['volume_idx'] for p in val_patches))[:args.val_volumes]
        val_patches = [p for p in val_patches if p['volume_idx'] in val_volume_indices]
        log_message(f"⚠️  Using {args.val_volumes} val volumes → {len(val_patches)} patches")
    
    # Update dataset
    dataset['train_patches'] = train_patches
    dataset['val_patches'] = val_patches
    
    # Save temp dataset
    import tempfile
    temp_dataset = tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False)
    json.dump(dataset, temp_dataset)
    temp_dataset.close()
    
    train_loader, val_loader = get_balanced_data_loaders(
        temp_dataset.name, batch_size=args.batch_size, 
        num_workers=args.num_workers, slice_depth=5
    )
    log_message(f"Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")
    
    # Model
    model = UNet(n_channels=1, n_classes=1, slice_depth=5).to(device)
    num_params = sum(p.numel() for p in model.parameters())
    log_message(f"Model parameters: {num_params:,}")
    
    # Training setup
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10)
    criterion = nn.BCEWithLogitsLoss()
    
    best_val_dice = 0.0
    
    # Training loop
    log_message(f"\n{'='*70}")
    log_message(f"🏋️  Training started")
    log_message(f"{'='*70}\n")
    
    for epoch in range(1, args.epochs + 1):
        log_message(f"\n{'='*70}")
        log_message(f"Epoch {epoch}/{args.epochs}")
        log_message(f"{'='*70}")
        
        train_loss, train_dice, train_iou = train_epoch(model, train_loader, criterion, optimizer, device, epoch)
        val_loss, val_dice, val_iou = validate_epoch(model, val_loader, criterion, device)
        
        scheduler.step(val_loss)
        lr = optimizer.param_groups[0]['lr']
        
        log_message(f"\n📊 Epoch {epoch} Summary:")
        log_message(f"Train - Loss: {train_loss:.4f}, Dice: {train_dice:.4f}, IoU: {train_iou:.4f}")
        log_message(f"Val   - Loss: {val_loss:.4f}, Dice: {val_dice:.4f}, IoU: {val_iou:.4f}")
        log_message(f"LR: {lr:.2e}")
        
        # Save latest
        for ckpt in checkpoint_dir.glob("checkpoint_*.pth"):
            ckpt.unlink()
        latest_path = checkpoint_dir / f"checkpoint_epoch_{epoch}.pth"
        save_checkpoint(model, optimizer, epoch, val_loss, latest_path)
        
        # Save best
        if val_dice > best_val_dice:
            best_val_dice = val_dice
            best_path = checkpoint_dir / "best_model.pth"
            save_checkpoint(model, optimizer, epoch, val_loss, best_path)
            log_message(f"\n🏆 New best Val Dice: {val_dice:.4f} (saved to best_model.pth)")
    
    log_message(f"\n{'='*70}")
    log_message(f"✅ Training complete! Best Val Dice: {best_val_dice:.4f}")
    log_message(f"📁 Checkpoints: {checkpoint_dir}")
    log_message(f"{'='*70}")

if __name__ == "__main__":
    main()

