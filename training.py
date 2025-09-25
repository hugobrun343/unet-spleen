import torch
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import os
import time
from pathlib import Path

from unet_model import create_model
from data_loader import get_data_loaders
from utils import BCEDiceLoss, calculate_metrics, print_metrics, save_checkpoint, load_checkpoint, visualize_predictions

class Trainer:
    def __init__(self, model, train_loader, val_loader, device, config):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.config = config
        
        # Loss and optimizer
        self.criterion = BCEDiceLoss(bce_weight=0.5, dice_weight=0.5)
        self.optimizer = optim.Adam(model.parameters(), lr=config['learning_rate'])
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.5, patience=10
        )
        
        # Create directories
        self.checkpoint_dir = Path(config['checkpoint_dir'])
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # Tensorboard
        self.writer = SummaryWriter(config['log_dir'])
        
        # Training state
        self.start_epoch = 0
        self.best_val_loss = float('inf')
        
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
            
            # Print progress
            if batch_idx % 10 == 0:
                print(f'Epoch {epoch}, Batch {batch_idx}/{len(self.train_loader)}, '
                      f'Loss: {loss.item():.4f}, Dice: {metrics["dice"]:.4f}')
        
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
        """Main training loop"""
        print(f"Starting training for {num_epochs} epochs...")
        print(f"Device: {self.device}")
        print(f"Train samples: {len(self.train_loader.dataset)}")
        print(f"Val samples: {len(self.val_loader.dataset)}")
        print("-" * 50)
        
        for epoch in range(self.start_epoch, num_epochs):
            start_time = time.time()
            
            # Train
            train_loss, train_metrics = self.train_epoch(epoch)
            
            # Validate
            val_loss, val_metrics = self.validate_epoch(epoch)
            
            # Update learning rate
            self.scheduler.step(val_loss)
            
            # Calculate epoch time
            epoch_time = time.time() - start_time
            
            # Print epoch results
            print(f"Epoch {epoch+1}/{num_epochs} ({epoch_time:.1f}s)")
            print(f"Train - Loss: {train_loss:.4f}, ", end="")
            print_metrics(train_metrics, "")
            print(f"Val   - Loss: {val_loss:.4f}, ", end="")
            print_metrics(val_metrics, "")
            print(f"LR: {self.optimizer.param_groups[0]['lr']:.6f}")
            print("-" * 50)
            
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
            
            # Save checkpoint
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                save_checkpoint(
                    self.model, self.optimizer, epoch, val_loss,
                    self.checkpoint_dir / 'best_model.pth'
                )
                print("New best model saved!")
            
            # Save regular checkpoint
            if (epoch + 1) % 10 == 0:
                save_checkpoint(
                    self.model, self.optimizer, epoch, val_loss,
                    self.checkpoint_dir / f'checkpoint_epoch_{epoch+1}.pth'
                )
        
        print("Training completed!")
        self.writer.close()
    
    def load_checkpoint(self, checkpoint_path):
        """Load checkpoint and resume training"""
        self.start_epoch, _ = load_checkpoint(checkpoint_path, self.model, self.optimizer)
        print(f"Resuming training from epoch {self.start_epoch + 1}")

def main():
    # Configuration
    config = {
        'batch_size': 1,
        'learning_rate': 1e-4,
        'num_epochs': 100,
        'train_split': 0.8,
        'num_workers': 4,
        'checkpoint_dir': 'checkpoints',
        'log_dir': 'runs/spleen_unet'
    }
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Dataset path
    dataset_path = "/teamspace/studios/this_studio/spleen/Dataset001_Spleen"
    
    # Create data loaders
    print("Loading data...")
    train_loader, val_loader = get_data_loaders(
        dataset_path,
        batch_size=config['batch_size'],
        train_split=config['train_split'],
        num_workers=config['num_workers'],
        slice_depth=5
    )
    
    # Create model
    print("Creating model...")
    model = create_model(device, slice_depth=5)  # 5 slices pour les blocs 3D
    
    # Create trainer
    trainer = Trainer(model, train_loader, val_loader, device, config)
    
    # Visualize some predictions before training
    print("Visualizing predictions before training...")
    visualize_predictions(model, val_loader, device, num_samples=2)
    
    # Start training
    trainer.train(config['num_epochs'])

if __name__ == "__main__":
    main()
