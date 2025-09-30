#!/usr/bin/env python3
"""
Create SPLIT dataset
- Keep ALL labeled patches from each volume
- Add same number of unlabeled patches DISTRIBUTED across the volume
- Unlabeled patches avoid borders and are spread uniformly
"""

import json
import numpy as np
from pathlib import Path
from datetime import datetime
import random

def load_patch_analysis(analysis_file):
    """Load patch analysis results"""
    with open(analysis_file, 'r') as f:
        analysis = json.load(f)
    return analysis

def filter_border_patches(patches, volume_shape, border_ratio=0.1):
    """
    Filter out patches that are at the volume borders
    Args:
        patches: List of patches
        volume_shape: Shape of the volume (H, W, D)
        border_ratio: Ratio of slices to exclude from top/bottom (0.1 = 10%)
    """
    total_slices = volume_shape[2]
    border_size = int(total_slices * border_ratio)
    
    # Keep only patches that are not in the border zones
    filtered_patches = [
        p for p in patches 
        if p['start_slice'] >= border_size and p['end_slice'] < (total_slices - border_size)
    ]
    
    return filtered_patches

def select_patches_from_volume(volume, border_ratio=0.1):
    """
    Select patches from a single volume:
    - Take ALL labeled patches
    - Take approximately same number of unlabeled patches (avoiding borders)
    """
    labeled_patches = volume['labeled_patches'].copy()
    empty_patches = volume['empty_patches'].copy()
    volume_shape = volume['volume_shape']
    
    # Add volume info to labeled patches
    for patch in labeled_patches:
        patch['volume_idx'] = volume['volume_idx']
        patch['volume_name'] = volume['volume_name']
        patch['volume_shape'] = volume_shape
        patch['is_labeled'] = True
    
    labeled_count = len(labeled_patches)
    
    # Filter out border patches from empty patches
    filtered_empty = filter_border_patches(empty_patches, volume_shape, border_ratio)
    
    # Calculate how many empty patches to take (approximately same as labeled)
    empty_needed = min(labeled_count, len(filtered_empty))
    
    # Randomly select empty patches
    if empty_needed > 0 and len(filtered_empty) > 0:
        selected_empty = random.sample(filtered_empty, empty_needed)
    else:
        selected_empty = []
    
    # Add volume info to empty patches
    for patch in selected_empty:
        patch['volume_idx'] = volume['volume_idx']
        patch['volume_name'] = volume['volume_name']
        patch['volume_shape'] = volume_shape
        patch['is_labeled'] = False
    
    all_patches = labeled_patches + selected_empty
    
    return all_patches, labeled_count, len(selected_empty)

def create_full_labeled_dataset(analysis_file, output_file, train_ratio=0.8, border_ratio=0.1):
    """
    Create dataset with ALL labeled patches + balanced unlabeled
    Args:
        analysis_file: Path to patch_analysis.json
        output_file: Path to save dataset
        train_ratio: Ratio for train/val split (0.8 = 80% train)
        border_ratio: Ratio of border slices to exclude from unlabeled (0.1 = 10%)
    """
    print("🔍 Loading patch analysis...")
    analysis = load_patch_analysis(analysis_file)
    
    volumes = analysis['volumes']
    total_volumes = len(volumes)
    
    print(f"📊 Found {total_volumes} volumes")
    print(f"🎯 Strategy: Keep ALL labeled patches + ~same number of unlabeled (excluding {border_ratio*100:.0f}% border)")
    
    # Process each volume
    all_patches = []
    volume_stats = []
    
    print(f"\n🔍 Processing volumes...")
    
    for i, volume in enumerate(volumes):
        print(f"\n📁 Volume {i+1}/{total_volumes}: {volume['volume_name']}")
        print(f"   Shape: {volume['volume_shape']}")
        print(f"   Available: {len(volume['labeled_patches'])} labeled, {len(volume['empty_patches'])} empty")
        
        # Select patches from this volume
        volume_patches, labeled_count, empty_count = select_patches_from_volume(volume, border_ratio)
        
        all_patches.extend(volume_patches)
        
        volume_stats.append({
            'volume_idx': volume['volume_idx'],
            'volume_name': volume['volume_name'],
            'volume_shape': volume['volume_shape'],
            'labeled_patches': labeled_count,
            'empty_patches': empty_count,
            'total_patches': len(volume_patches)
        })
        
        print(f"   ✅ Selected: {labeled_count} labeled, {empty_count} empty = {len(volume_patches)} total")
    
    # Calculate global statistics
    total_labeled = sum(vs['labeled_patches'] for vs in volume_stats)
    total_empty = sum(vs['empty_patches'] for vs in volume_stats)
    total_patches = len(all_patches)
    
    print(f"\n📊 Global patch counts:")
    print(f"   Labeled: {total_labeled}")
    print(f"   Empty: {total_empty}")
    print(f"   Total: {total_patches}")
    print(f"   Balance: {total_labeled/total_patches*100:.1f}% labeled, {total_empty/total_patches*100:.1f}% empty")
    
    # Split into train/val by VOLUME (not by patch)
    # This ensures patches from same volume stay together
    volume_indices = list(range(total_volumes))
    random.shuffle(volume_indices)
    
    train_volume_count = int(total_volumes * train_ratio)
    train_volume_indices = set(volume_indices[:train_volume_count])
    val_volume_indices = set(volume_indices[train_volume_count:])
    
    train_patches = [p for p in all_patches if p['volume_idx'] in train_volume_indices]
    val_patches = [p for p in all_patches if p['volume_idx'] in val_volume_indices]
    
    # Calculate train/val statistics
    train_labeled = len([p for p in train_patches if p['is_labeled']])
    train_empty = len([p for p in train_patches if not p['is_labeled']])
    val_labeled = len([p for p in val_patches if p['is_labeled']])
    val_empty = len([p for p in val_patches if not p['is_labeled']])
    
    # Create dataset
    dataset = {
        'timestamp': datetime.now().isoformat(),
        'source_analysis': str(analysis_file),
        'train_ratio': train_ratio,
        'border_ratio': border_ratio,
        'total_volumes': total_volumes,
        'train_volumes': len(train_volume_indices),
        'val_volumes': len(val_volume_indices),
        'train_patches': train_patches,
        'val_patches': val_patches,
        'volume_stats': volume_stats,
        'train_stats': {
            'total_patches': len(train_patches),
            'labeled_patches': train_labeled,
            'empty_patches': train_empty
        },
        'val_stats': {
            'total_patches': len(val_patches),
            'labeled_patches': val_labeled,
            'empty_patches': val_empty
        }
    }
    
    # Save dataset
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        json.dump(dataset, f, indent=2)
    
    print(f"\n✅ Full labeled dataset created!")
    print(f"💾 Saved to: {output_path}")
    
    # Print comprehensive statistics
    print(f"\n📊 Dataset Statistics:")
    print(f"🚂 Train: {len(train_patches)} patches from {len(train_volume_indices)} volumes")
    print(f"   - Labeled: {train_labeled} ({train_labeled/len(train_patches)*100:.1f}%)")
    print(f"   - Empty: {train_empty} ({train_empty/len(train_patches)*100:.1f}%)")
    print(f"✅ Val: {len(val_patches)} patches from {len(val_volume_indices)} volumes")
    print(f"   - Labeled: {val_labeled} ({val_labeled/len(val_patches)*100:.1f}%)")
    print(f"   - Empty: {val_empty} ({val_empty/len(val_patches)*100:.1f}%)")
    
    # Print per-volume statistics
    print(f"\n📁 Per-Volume Statistics:")
    for stats in volume_stats:
        split = "TRAIN" if stats['volume_idx'] in train_volume_indices else "VAL"
        print(f"   [{split}] Volume {stats['volume_idx']+1}: {stats['total_patches']} patches "
              f"({stats['labeled_patches']} labeled, {stats['empty_patches']} empty)")
    
    return dataset

def main():
    """Main function"""
    analysis_file = "/teamspace/studios/this_studio/spleen/data/processed/patch_analysis.json"
    output_file = "/teamspace/studios/this_studio/spleen/data/processed/dataset_split.json"
    
    # Configuration
    config = {
        'train_ratio': 0.8,  # 80% volumes for train, 20% for val
        'border_ratio': 0.1  # Exclude 10% top/bottom slices from unlabeled selection
    }
    
    print("🎯 Create Split Dataset")
    print("=" * 70)
    print(f"Analysis file: {analysis_file}")
    print(f"Output file: {output_file}")
    print(f"Config: {config}")
    print("=" * 70)
    
    # Set random seed for reproducibility
    random.seed(42)
    np.random.seed(42)
    
    # Create dataset
    dataset = create_full_labeled_dataset(analysis_file, output_file, **config)
    
    print(f"\n🎉 Split dataset creation complete!")
    print(f"📁 Dataset file: {output_file}")
    print(f"\n💡 Unlabeled patches are distributed across the volume (borders excluded)")

if __name__ == "__main__":
    main()

