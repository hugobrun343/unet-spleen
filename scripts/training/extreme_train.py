#!/usr/bin/env python3
"""
Extreme training script - 10000 epochs on a single patch
- Uses balanced_dataset.json
- Trains on only 1 labeled patch
- Extreme overfitting test
"""

import torch
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import os
import time
from pathlib import Path

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from models.unet_model import create_model
from utils.data_loader import get_single_patch_loader
from utils.utils import BCEDiceLoss, calculate_metrics, print_metrics, save_checkpoint, load_checkpoint, visualize_predictions

class ExtremeTrainer:
    def __init__(self, model, train_loader, val_loader, device, config):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.config = config
        
        # Loss and optimizer
        self.criterion = BCEDiceLoss()
        self.optimizer = optim.Adam(model.parameters(), lr=config['learning_rate'])
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.5, patience=100, min_lr=1e-8
        )
        
        # Checkpointing
        self.checkpoint_dir = Path('extreme_checkpoints')
        self.checkpoint_dir.mkdir(exist_ok=True)
        
        # Tensorboard
        self.writer = SummaryWriter('runs/extreme_training')
        
        # Log file
        self.log_file = '/teamspace/studios/this_studio/spleen/logs/extreme_training.log'
        # Clear previous log
        with open(self.log_file, 'w') as f:
            f.write(f"Extreme training started at {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        
        # Training state
        self.start_epoch = 0
        self.best_val_loss = float('inf')
    
    def log_message(self, message):
        """Log message to both console and file"""
        print(message)
        with open(self.log_file, 'a') as f:
            f.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')} - {message}\n")
        
    def train_epoch(self, epoch):
        """Train for one epoch"""
        self.model.train()
        total_loss = 0.0
        total_metrics = {'dice': 0.0, 'iou': 0.0, 'pixel_acc': 0.0}
        
        for batch_idx, (images, labels) in enumerate(self.train_loader):
            images = images.to(self.device)
            labels = labels.to(self.device)
            
            # Zero gradients
            self.optimizer.zero_grad()
            
            # Forward pass
            outputs = self.model(images)
            loss = self.criterion(outputs, labels)
            
            # Backward pass
            loss.backward()
            self.optimizer.step()
            
            # Calculate metrics
            metrics = calculate_metrics(outputs, labels)
            
            # Update totals
            total_loss += loss.item()
            for key in total_metrics:
                total_metrics[key] += metrics[key]
            
            # Print progress every 100 epochs
            if epoch % 100 == 0:
                print(f'Epoch {epoch}, Loss: {loss.item():.6f}, Dice: {metrics["dice"]:.6f}')
        
        # Calculate averages
        avg_loss = total_loss / len(self.train_loader)
        avg_metrics = {key: val / len(self.train_loader) for key, val in total_metrics.items()}
        
        return avg_loss, avg_metrics
    
    def validate_epoch(self, epoch):
        """Validate for one epoch"""
        self.model.eval()
        total_loss = 0.0
        total_metrics = {'dice': 0.0, 'iou': 0.0, 'pixel_acc': 0.0}
        
        with torch.no_grad():
            for batch_idx, (images, labels) in enumerate(self.val_loader):
                images = images.to(self.device)
                labels = labels.to(self.device)
                
                # Forward pass
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                
                # Calculate metrics
                metrics = calculate_metrics(outputs, labels)
                
                # Update totals
                total_loss += loss.item()
                for key in total_metrics:
                    total_metrics[key] += metrics[key]
        
        # Calculate averages
        avg_loss = total_loss / len(self.val_loader)
        avg_metrics = {key: val / len(self.val_loader) for key, val in total_metrics.items()}
        
        return avg_loss, avg_metrics
    
    def train(self, num_epochs):
        """Main training loop for extreme testing"""
        print(f"🔥 EXTREME TRAINING - Starting training for {num_epochs} epochs on 1 patch...")
        print(f"Device: {self.device}")
        print(f"Train samples: {len(self.train_loader.dataset)}")
        print(f"Val samples: {len(self.val_loader.dataset)}")
        print("⚠️  WARNING: This will overfit completely!")
        
        for epoch in range(self.start_epoch, num_epochs):
            start_time = time.time()
            
            # Train
            train_loss, train_metrics = self.train_epoch(epoch)
            
            # Validate
            val_loss, val_metrics = self.validate_epoch(epoch)
            
            # Learning rate scheduling
            self.scheduler.step(val_loss)
            
            # Log to tensorboard
            self.writer.add_scalar('Loss/Train', train_loss, epoch)
            self.writer.add_scalar('Loss/Val', val_loss, epoch)
            self.writer.add_scalar('Dice/Train', train_metrics['dice'], epoch)
            self.writer.add_scalar('Dice/Val', val_metrics['dice'], epoch)
            self.writer.add_scalar('IoU/Train', train_metrics['iou'], epoch)
            self.writer.add_scalar('IoU/Val', val_metrics['iou'], epoch)
            self.writer.add_scalar('PixelAcc/Train', train_metrics['pixel_acc'], epoch)
            self.writer.add_scalar('PixelAcc/Val', val_metrics['pixel_acc'], epoch)
            self.writer.add_scalar('Learning_Rate', self.optimizer.param_groups[0]['lr'], epoch)
            
            # Print epoch results every 100 epochs
            if epoch % 100 == 0:
                epoch_time = time.time() - start_time
                print(f'\nEpoch {epoch+1}/{num_epochs} ({epoch_time:.1f}s)')
                print(f'Train - Loss: {train_loss:.6f}, Dice: {train_metrics["dice"]:.6f}, IoU: {train_metrics["iou"]:.6f}')
                print(f'Val   - Loss: {val_loss:.6f}, Dice: {val_metrics["dice"]:.6f}, IoU: {val_metrics["iou"]:.6f}')
                print(f'LR: {self.optimizer.param_groups[0]["lr"]:.2e}')
            
            # Save checkpoint
            is_best = val_loss < self.best_val_loss
            if is_best:
                self.best_val_loss = val_loss
                if epoch % 100 == 0:
                    print(f'🎉 New best validation loss: {val_loss:.6f}')
            
            # Save checkpoint every 1000 epochs or if best
            if (epoch + 1) % 1000 == 0 or is_best:
                checkpoint_path = self.checkpoint_dir / f'extreme_checkpoint_epoch_{epoch+1}.pth'
                save_checkpoint(self.model, self.optimizer, epoch, val_loss, checkpoint_path)
                if epoch % 100 == 0:
                    print(f'💾 Checkpoint saved: {checkpoint_path}')
            
            # Cleanup old checkpoints
            self._cleanup_old_checkpoints()
            
            # Visualize predictions every 1000 epochs
            if (epoch + 1) % 1000 == 0:
                print("🎨 Visualizing predictions...")
                visualize_predictions(self.model, self.val_loader, self.device, num_samples=1)
        
        print(f"\n✅ Extreme training complete! Best validation loss: {self.best_val_loss:.6f}")
        print("🎯 Final overfitting achieved!")
        self.writer.close()
    
    def _cleanup_old_checkpoints(self):
        """Keep only the last 3 checkpoints for extreme training"""
        import glob
        checkpoint_files = glob.glob(str(self.checkpoint_dir / 'extreme_checkpoint_epoch_*.pth'))
        checkpoint_files.sort(key=lambda x: int(x.split('_')[-1].split('.')[0]))
        
        # Keep only last 3 checkpoints
        if len(checkpoint_files) > 3:
            for old_checkpoint in checkpoint_files[:-3]:
                os.remove(old_checkpoint)

def main():
    # Configuration for extreme testing
    config = {
        'batch_size': 1,
        'learning_rate': 1e-3,
        'num_epochs': 10000,
        'num_workers': 0,
        'slice_depth': 5
    }
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Dataset file
    dataset_file = "/teamspace/studios/this_studio/spleen/data/processed/balanced_dataset.json"
    
    # Create data loaders with 1 patch
    print("🔥 Loading 1 patch for extreme testing...")
    train_loader = get_single_patch_loader(
        dataset_file,
        batch_size=config['batch_size'],
        num_workers=config['num_workers'],
        slice_depth=config['slice_depth']
    )
    
    # Use same data for validation (single patch)
    val_loader = train_loader
    
    # Create model
    print("Creating model...")
    model = create_model(device, slice_depth=config['slice_depth'])
    
    # Create trainer
    trainer = ExtremeTrainer(model, train_loader, val_loader, device, config)
    
    # Visualize the single patch before training
    print("Visualizing the single patch before training...")
    visualize_predictions(model, val_loader, device, num_samples=1)
    
    # Start extreme training
    trainer.train(config['num_epochs'])

if __name__ == "__main__":
    main()
