import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from sklearn.metrics import jaccard_score

class DiceLoss(nn.Module):
    """Dice Loss for segmentation"""
    def __init__(self, smooth=1e-5):
        super(DiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, inputs, targets):
        # Apply sigmoid to inputs
        inputs = torch.sigmoid(inputs)
        
        # Flatten tensors
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        
        # Calculate Dice coefficient
        intersection = (inputs * targets).sum()
        dice = (2. * intersection + self.smooth) / (inputs.sum() + targets.sum() + self.smooth)
        
        return 1 - dice

class BCEDiceLoss(nn.Module):
    """Combined BCE and Dice Loss"""
    def __init__(self, bce_weight=0.5, dice_weight=0.5):
        super(BCEDiceLoss, self).__init__()
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight
        self.bce = nn.BCEWithLogitsLoss()
        self.dice = DiceLoss()

    def forward(self, inputs, targets):
        bce_loss = self.bce(inputs, targets)
        dice_loss = self.dice(inputs, targets)
        return self.bce_weight * bce_loss + self.dice_weight * dice_loss

def dice_coefficient(pred, target, smooth=1e-5):
    """Calculate Dice coefficient"""
    pred = torch.sigmoid(pred)
    pred = (pred > 0.5).float()
    
    intersection = (pred * target).sum()
    dice = (2. * intersection + smooth) / (pred.sum() + target.sum() + smooth)
    return dice.item()

def iou_score(pred, target, smooth=1e-5):
    """Calculate IoU (Intersection over Union) score"""
    pred = torch.sigmoid(pred)
    pred = (pred > 0.5).float()
    
    intersection = (pred * target).sum()
    union = pred.sum() + target.sum() - intersection
    iou = (intersection + smooth) / (union + smooth)
    return iou.item()

def pixel_accuracy(pred, target):
    """Calculate pixel accuracy"""
    pred = torch.sigmoid(pred)
    pred = (pred > 0.5).float()
    correct = (pred == target).sum().item()
    total = target.numel()
    return correct / total

def save_checkpoint(model, optimizer, epoch, loss, filepath):
    """Save model checkpoint"""
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
    }
    torch.save(checkpoint, filepath)
    print(f"Checkpoint saved: {filepath}")

def load_checkpoint(filepath, model, optimizer=None):
    """Load model checkpoint"""
    checkpoint = torch.load(filepath)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    if optimizer:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    epoch = checkpoint['epoch']
    loss = checkpoint['loss']
    
    print(f"Checkpoint loaded: {filepath}")
    print(f"Epoch: {epoch}, Loss: {loss:.4f}")
    
    return epoch, loss

def calculate_metrics(pred, target):
    """Calculate all metrics for evaluation"""
    metrics = {}
    
    # Convert predictions to binary
    pred_binary = torch.sigmoid(pred)
    pred_binary = (pred_binary > 0.5).float()
    
    # Calculate metrics
    metrics['dice'] = dice_coefficient(pred, target)
    metrics['iou'] = iou_score(pred, target)
    metrics['pixel_acc'] = pixel_accuracy(pred, target)
    
    return metrics

def print_metrics(metrics, prefix=""):
    """Print metrics in a nice format"""
    print(f"{prefix}Dice: {metrics['dice']:.4f}, IoU: {metrics['iou']:.4f}, Pixel Acc: {metrics['pixel_acc']:.4f}")

def visualize_predictions(model, dataloader, device, num_samples=4):
    """Visualize model predictions on sample images"""
    model.eval()
    
    with torch.no_grad():
        for i, (images, labels) in enumerate(dataloader):
            if i >= num_samples:
                break
                
            images = images.to(device)
            labels = labels.to(device)
            
            # Get predictions
            outputs = model(images)
            preds = torch.sigmoid(outputs)
            preds = (preds > 0.5).float()
            
            # Print shapes and ranges
            print(f"Sample {i+1}:")
            print(f"  Image shape: {images.shape}")
            print(f"  Label shape: {labels.shape}")
            print(f"  Pred shape: {preds.shape}")
            print(f"  Image range: [{images.min():.3f}, {images.max():.3f}]")
            print(f"  Label range: [{labels.min():.3f}, {labels.max():.3f}]")
            print(f"  Pred range: [{preds.min():.3f}, {preds.max():.3f}]")
            print()

def test_utils():
    """Test utility functions"""
    print("Testing utility functions...")
    
    # Create dummy data
    pred = torch.randn(2, 1, 64, 64)
    target = torch.randint(0, 2, (2, 1, 64, 64)).float()
    
    # Test loss functions
    dice_loss = DiceLoss()
    bce_dice_loss = BCEDiceLoss()
    
    print(f"Dice Loss: {dice_loss(pred, target):.4f}")
    print(f"BCE+Dice Loss: {bce_dice_loss(pred, target):.4f}")
    
    # Test metrics
    metrics = calculate_metrics(pred, target)
    print_metrics(metrics, "Test ")
    
    print("Utils test completed!")

if __name__ == "__main__":
    test_utils()
