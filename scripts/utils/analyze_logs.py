#!/usr/bin/env python3
"""
Analyze training logs and generate summary with graphs
"""

import re
import matplotlib.pyplot as plt
import sys
from pathlib import Path

def parse_log_file(log_file):
    """Parse log file and extract epoch metrics"""
    epochs = []
    train_losses = []
    train_dices = []
    train_ious = []
    val_losses = []
    val_dices = []
    val_ious = []
    lrs = []
    
    with open(log_file, 'r') as f:
        lines = f.readlines()
    
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        
        # Look for epoch summary lines
        epoch_match = re.search(r'Epoch (\d+)/(\d+)', line)
        if epoch_match:
            epoch_num = int(epoch_match.group(1))
            
            # Get next lines for train and val metrics
            if i + 1 < len(lines):
                train_line = lines[i + 1].strip()
                train_match = re.search(r'Train - Loss: ([\d.]+), Dice: ([\d.]+), IoU: ([\d.]+)', train_line)
                if train_match:
                    epochs.append(epoch_num)
                    train_losses.append(float(train_match.group(1)))
                    train_dices.append(float(train_match.group(2)))
                    train_ious.append(float(train_match.group(3)))
            
            if i + 2 < len(lines):
                val_line = lines[i + 2].strip()
                val_match = re.search(r'Val   - Loss: ([\d.]+), Dice: ([\d.]+), IoU: ([\d.]+)', val_line)
                if val_match:
                    val_losses.append(float(val_match.group(1)))
                    val_dices.append(float(val_match.group(2)))
                    val_ious.append(float(val_match.group(3)))
            
            if i + 3 < len(lines):
                lr_line = lines[i + 3].strip()
                lr_match = re.search(r'LR: ([\d.e+-]+)', lr_line)
                if lr_match:
                    lrs.append(float(lr_match.group(1)))
        
        i += 1
    
    return {
        'epochs': epochs,
        'train_loss': train_losses,
        'train_dice': train_dices,
        'train_iou': train_ious,
        'val_loss': val_losses,
        'val_dice': val_dices,
        'val_iou': val_ious,
        'lr': lrs
    }

def generate_summary(metrics, output_file):
    """Generate text summary of metrics"""
    with open(output_file, 'w') as f:
        f.write("=" * 60 + "\n")
        f.write("TRAINING SUMMARY\n")
        f.write("=" * 60 + "\n\n")
        
        if not metrics['epochs']:
            f.write("No epoch data found in logs.\n")
            return
        
        f.write(f"Total epochs: {len(metrics['epochs'])}\n")
        f.write(f"Epochs range: {metrics['epochs'][0]} - {metrics['epochs'][-1]}\n\n")
        
        f.write("-" * 60 + "\n")
        f.write("TRAIN METRICS (Average)\n")
        f.write("-" * 60 + "\n")
        f.write(f"Loss: {sum(metrics['train_loss'])/len(metrics['train_loss']):.4f}\n")
        f.write(f"Dice: {sum(metrics['train_dice'])/len(metrics['train_dice']):.4f}\n")
        f.write(f"IoU:  {sum(metrics['train_iou'])/len(metrics['train_iou']):.4f}\n\n")
        
        f.write("-" * 60 + "\n")
        f.write("VALIDATION METRICS (Average)\n")
        f.write("-" * 60 + "\n")
        f.write(f"Loss: {sum(metrics['val_loss'])/len(metrics['val_loss']):.4f}\n")
        f.write(f"Dice: {sum(metrics['val_dice'])/len(metrics['val_dice']):.4f}\n")
        f.write(f"IoU:  {sum(metrics['val_iou'])/len(metrics['val_iou']):.4f}\n\n")
        
        f.write("-" * 60 + "\n")
        f.write("EPOCH BY EPOCH DETAILS\n")
        f.write("-" * 60 + "\n")
        f.write(f"{'Epoch':<8} {'Train Loss':<12} {'Train Dice':<12} {'Val Loss':<12} {'Val Dice':<12} {'LR':<10}\n")
        f.write("-" * 60 + "\n")
        
        for i, epoch in enumerate(metrics['epochs']):
            f.write(f"{epoch:<8} {metrics['train_loss'][i]:<12.4f} {metrics['train_dice'][i]:<12.4f} "
                   f"{metrics['val_loss'][i]:<12.4f} {metrics['val_dice'][i]:<12.4f} {metrics['lr'][i]:<10.2e}\n")
        
        f.write("\n" + "=" * 60 + "\n")
        f.write(f"Best Train Dice: {max(metrics['train_dice']):.4f} at epoch {metrics['epochs'][metrics['train_dice'].index(max(metrics['train_dice']))]}\n")
        f.write(f"Best Val Dice: {max(metrics['val_dice']):.4f} at epoch {metrics['epochs'][metrics['val_dice'].index(max(metrics['val_dice']))]}\n")
        f.write(f"Best Val Loss: {min(metrics['val_loss']):.4f} at epoch {metrics['epochs'][metrics['val_loss'].index(min(metrics['val_loss']))]}\n")

def plot_metrics(metrics, output_dir, base_name):
    """Generate separate plots for each metric"""
    
    # Plot 1: Loss
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(metrics['epochs'], metrics['train_loss'], label='Train Loss', marker='o', markersize=2, linewidth=2)
    ax.plot(metrics['epochs'], metrics['val_loss'], label='Val Loss', marker='s', markersize=2, linewidth=2)
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Loss', fontsize=12)
    ax.set_title('Loss over Epochs', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    loss_file = output_dir / f"{base_name}_loss.png"
    plt.savefig(loss_file, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"📊 Loss graph saved to: {loss_file}")
    
    # Plot 2: Dice
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(metrics['epochs'], metrics['train_dice'], label='Train Dice', marker='o', markersize=2, linewidth=2)
    ax.plot(metrics['epochs'], metrics['val_dice'], label='Val Dice', marker='s', markersize=2, linewidth=2)
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Dice Score', fontsize=12)
    ax.set_title('Dice Score over Epochs', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    dice_file = output_dir / f"{base_name}_dice.png"
    plt.savefig(dice_file, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"📊 Dice graph saved to: {dice_file}")
    
    # Plot 3: IoU
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(metrics['epochs'], metrics['train_iou'], label='Train IoU', marker='o', markersize=2, linewidth=2)
    ax.plot(metrics['epochs'], metrics['val_iou'], label='Val IoU', marker='s', markersize=2, linewidth=2)
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('IoU', fontsize=12)
    ax.set_title('IoU over Epochs', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    iou_file = output_dir / f"{base_name}_iou.png"
    plt.savefig(iou_file, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"📊 IoU graph saved to: {iou_file}")
    
    # Plot 4: Learning Rate
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(metrics['epochs'], metrics['lr'], label='Learning Rate', marker='o', markersize=2, linewidth=2, color='green')
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Learning Rate', fontsize=12)
    ax.set_title('Learning Rate over Epochs', fontsize=14, fontweight='bold')
    ax.set_yscale('log')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    lr_file = output_dir / f"{base_name}_lr.png"
    plt.savefig(lr_file, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"📊 Learning Rate graph saved to: {lr_file}")

def main():
    if len(sys.argv) < 2:
        print("Usage: python analyze_logs.py <log_file>")
        sys.exit(1)
    
    log_file = sys.argv[1]
    
    if not Path(log_file).exists():
        print(f"Error: Log file not found: {log_file}")
        sys.exit(1)
    
    print(f"📖 Analyzing log file: {log_file}")
    
    # Parse logs
    metrics = parse_log_file(log_file)
    
    if not metrics['epochs']:
        print("❌ No epoch data found in log file")
        sys.exit(1)
    
    print(f"✅ Found {len(metrics['epochs'])} epochs")
    
    # Generate summary in logs folder
    logs_dir = Path('/teamspace/studios/this_studio/spleen/logs')
    logs_dir.mkdir(parents=True, exist_ok=True)
    
    base_name = Path(log_file).stem
    summary_file = logs_dir / f"{base_name}_summary.txt"
    generate_summary(metrics, summary_file)
    print(f"📄 Summary saved to: {summary_file}")
    
    # Generate plots in logs folder
    plot_metrics(metrics, logs_dir, base_name)
    
    print("\n✅ Analysis complete!")

if __name__ == "__main__":
    main()

