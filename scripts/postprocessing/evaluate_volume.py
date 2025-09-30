#!/usr/bin/env python3
"""
Evaluation script with post-processing
- Load trained model
- Evaluate on validation volumes
- Compare metrics before/after connected component post-processing
"""

import sys
import os
import json
import argparse
from pathlib import Path
import torch
import nibabel as nib
import numpy as np

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from models.unet_model import UNet2D
from postprocessing.utils import (
    sliding_window_inference,
    postprocess_prediction,
    calculate_dice,
    calculate_iou
)

def load_model(checkpoint_path, device, in_channels=5):
    """Load trained model from checkpoint"""
    model = UNet2D(in_channels=in_channels, out_channels=1)
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    
    print(f"✅ Model loaded from: {checkpoint_path}")
    print(f"   Epoch: {checkpoint.get('epoch', 'N/A')}")
    print(f"   Loss: {checkpoint.get('loss', 'N/A'):.4f}")
    
    return model

def evaluate_volume(model, volume_path, label_path, device, slice_depth=5):
    """
    Evaluate model on a single volume with post-processing
    """
    print(f"\n📁 Evaluating: {Path(volume_path).name}")
    
    # Load volume and label
    try:
        volume_nii = nib.load(volume_path)
    except FileNotFoundError:
        volume_path_no_gz = str(volume_path).replace('.nii.gz', '.nii')
        volume_nii = nib.load(volume_path_no_gz)
    
    try:
        label_nii = nib.load(label_path)
    except FileNotFoundError:
        label_path_no_gz = str(label_path).replace('.nii.gz', '.nii')
        label_nii = nib.load(label_path_no_gz)
    
    volume_data = volume_nii.get_fdata()
    label_data = label_nii.get_fdata()
    
    print(f"   Volume shape: {volume_data.shape}")
    
    # Sliding window inference
    print(f"   🔄 Running sliding window inference...")
    pred_volume = sliding_window_inference(model, volume_data, device, slice_depth)
    
    # Binary predictions
    pred_binary = (pred_volume > 0.5).astype(np.float32)
    
    # Post-process
    print(f"   🧹 Applying post-processing (connected components)...")
    pred_postprocessed = postprocess_prediction(pred_volume)
    
    # Ground truth
    label_binary = (label_data > 0).astype(np.float32)
    
    # Calculate metrics
    dice_before = calculate_dice(pred_binary, label_binary)
    iou_before = calculate_iou(pred_binary, label_binary)
    
    dice_after = calculate_dice(pred_postprocessed, label_binary)
    iou_after = calculate_iou(pred_postprocessed, label_binary)
    
    improvement_dice = dice_after - dice_before
    improvement_iou = iou_after - iou_before
    
    print(f"   📊 Metrics:")
    print(f"      Before post-processing: Dice={dice_before:.4f}, IoU={iou_before:.4f}")
    print(f"      After post-processing:  Dice={dice_after:.4f}, IoU={iou_after:.4f}")
    print(f"      Improvement: Dice={improvement_dice:+.4f}, IoU={improvement_iou:+.4f}")
    
    return {
        'volume_name': Path(volume_path).name,
        'dice_before': float(dice_before),
        'dice_after': float(dice_after),
        'iou_before': float(iou_before),
        'iou_after': float(iou_after),
        'improvement_dice': float(improvement_dice),
        'improvement_iou': float(improvement_iou)
    }

def main():
    """Main evaluation function"""
    parser = argparse.ArgumentParser(description="Evaluate model with post-processing")
    parser.add_argument('--checkpoint', type=str, required=True, help="Path to checkpoint")
    parser.add_argument('--num_volumes', type=int, default=5, help="Number of volumes to evaluate")
    
    args = parser.parse_args()
    
    print("🔬 EVALUATION WITH POST-PROCESSING")
    print("=" * 70)
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load model
    print(f"\n📦 Loading model...")
    model = load_model(args.checkpoint, device, in_channels=5)
    
    # Load dataset to get validation volumes
    dataset_file = "/teamspace/studios/this_studio/spleen/data/processed/dataset_stack.json"
    print(f"\n📂 Loading dataset: {dataset_file}")
    
    with open(dataset_file, 'r') as f:
        dataset = json.load(f)
    
    # Get validation volume indices
    val_patches = dataset['val_patches']
    val_volume_indices = sorted(set(p['volume_idx'] for p in val_patches))[:args.num_volumes]
    
    print(f"   Evaluating on {len(val_volume_indices)} validation volumes")
    
    # Get volume names
    volume_names = {}
    for patch in val_patches:
        if patch['volume_idx'] in val_volume_indices:
            volume_names[patch['volume_idx']] = patch['volume_name']
    
    # Data root
    data_root = Path("/teamspace/studios/this_studio/spleen/data/raw")
    
    # Evaluate each volume
    results = []
    for vol_idx in val_volume_indices:
        volume_name = volume_names[vol_idx]
        volume_path = data_root / volume_name
        label_path = data_root / volume_name.replace('imagesTr', 'labelsTr')
        
        try:
            result = evaluate_volume(model, volume_path, label_path, device, slice_depth=5)
            results.append(result)
        except Exception as e:
            print(f"   ❌ Error: {e}")
    
    # Summary statistics
    print(f"\n{'='*70}")
    print("📊 SUMMARY STATISTICS")
    print(f"{'='*70}")
    
    if results:
        avg_dice_before = np.mean([r['dice_before'] for r in results])
        avg_dice_after = np.mean([r['dice_after'] for r in results])
        avg_iou_before = np.mean([r['iou_before'] for r in results])
        avg_iou_after = np.mean([r['iou_after'] for r in results])
        avg_improvement_dice = np.mean([r['improvement_dice'] for r in results])
        avg_improvement_iou = np.mean([r['improvement_iou'] for r in results])
        
        print(f"Average Dice (before):  {avg_dice_before:.4f}")
        print(f"Average Dice (after):   {avg_dice_after:.4f}")
        print(f"Average IoU (before):   {avg_iou_before:.4f}")
        print(f"Average IoU (after):    {avg_iou_after:.4f}")
        print(f"\n🎯 Average Improvement:")
        print(f"   Dice: {avg_improvement_dice:+.4f}")
        print(f"   IoU:  {avg_improvement_iou:+.4f}")
        
        # Count improvements
        num_improved_dice = sum(1 for r in results if r['improvement_dice'] > 0)
        num_improved_iou = sum(1 for r in results if r['improvement_iou'] > 0)
        
        print(f"\n✅ Volumes improved:")
        print(f"   Dice: {num_improved_dice}/{len(results)} ({num_improved_dice/len(results)*100:.1f}%)")
        print(f"   IoU:  {num_improved_iou}/{len(results)} ({num_improved_iou/len(results)*100:.1f}%)")
        
        # Save results
        results_dir = Path("/teamspace/studios/this_studio/spleen/results")
        results_dir.mkdir(parents=True, exist_ok=True)
        results_file = results_dir / "postprocessing_evaluation.json"
        
        with open(results_file, 'w') as f:
            json.dump({
                'checkpoint': str(args.checkpoint),
                'num_volumes': len(results),
                'summary': {
                    'avg_dice_before': float(avg_dice_before),
                    'avg_dice_after': float(avg_dice_after),
                    'avg_iou_before': float(avg_iou_before),
                    'avg_iou_after': float(avg_iou_after),
                    'avg_improvement_dice': float(avg_improvement_dice),
                    'avg_improvement_iou': float(avg_improvement_iou)
                },
                'results': results
            }, f, indent=2)
        
        print(f"\n💾 Results saved to: {results_file}")
    
    print(f"\n{'='*70}")
    print("✅ Evaluation complete!")

if __name__ == "__main__":
    main()

