#!/usr/bin/env python3
"""
Create STACK dataset for post-processing
- Keep ALL labeled patches from each volume
- Add unlabeled patches AROUND/ADJACENT to labeled regions
- This creates a "stack" of patches that can be reconstructed into volumes
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

def get_adjacent_unlabeled_patches(labeled_patches, empty_patches, vicinity_range=5):
    """
    Get unlabeled patches that are adjacent/near to labeled patches
    Args:
        labeled_patches: List of labeled patches
        empty_patches: List of empty patches
        vicinity_range: Number of slices around labeled patches to consider
    """
    # Get slice ranges of labeled patches
    labeled_slices = set()
    for patch in labeled_patches:
        for s in range(patch['start_slice'], patch['end_slice'] + 1):
            labeled_slices.add(s)
    
    # Expand to include vicinity
    expanded_slices = set()
    for s in labeled_slices:
        for offset in range(-vicinity_range, vicinity_range + 1):
            expanded_slices.add(s + offset)
    
    # Select empty patches in this vicinity
    adjacent_empty = []
    for patch in empty_patches:
        patch_slice = patch['start_slice']
        if patch_slice in expanded_slices:
            adjacent_empty.append(patch)
    
    return adjacent_empty

def select_patches_from_volume(volume, vicinity_range=5):
    """
    Select patches from a single volume:
    - Take ALL labeled patches
    - Take unlabeled patches AROUND labeled regions
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
    
    # Get adjacent unlabeled patches
    adjacent_empty = get_adjacent_unlabeled_patches(
        labeled_patches, empty_patches, vicinity_range
    )
    
    # Add volume info to empty patches
    for patch in adjacent_empty:
        patch['volume_idx'] = volume['volume_idx']
        patch['volume_name'] = volume['volume_name']
        patch['volume_shape'] = volume_shape
        patch['is_labeled'] = False
    
    all_patches = labeled_patches + adjacent_empty
    
    return all_patches, labeled_count, len(adjacent_empty)

def create_stack_dataset(analysis_file, output_file, train_ratio=0.8, vicinity_range=5):
    """
    Create stack dataset with ALL labeled + adjacent unlabeled
    Args:
        analysis_file: Path to patch_analysis.json
        output_file: Path to save dataset
        train_ratio: Ratio for train/val split (0.8 = 80% train)
        vicinity_range: Number of slices around labeled to include unlabeled
    """
    print("🔍 Loading patch analysis...")
    analysis = load_patch_analysis(analysis_file)
    
    volumes = analysis['volumes']
    total_volumes = len(volumes)
    
    print(f"📊 Found {total_volumes} volumes")
    print(f"🎯 Strategy: Keep ALL labeled patches + unlabeled within {vicinity_range} slices")
    
    # Process each volume
    all_patches = []
    volume_stats = []
    
    print(f"\n🔍 Processing volumes...")
    
    for i, volume in enumerate(volumes):
        print(f"\n📁 Volume {i+1}/{total_volumes}: {volume['volume_name']}")
        print(f"   Shape: {volume['volume_shape']}")
        print(f"   Available: {len(volume['labeled_patches'])} labeled, {len(volume['empty_patches'])} empty")
        
        # Select patches from this volume
        volume_patches, labeled_count, empty_count = select_patches_from_volume(
            volume, vicinity_range
        )
        
        all_patches.extend(volume_patches)
        
        volume_stats.append({
            'volume_idx': volume['volume_idx'],
            'volume_name': volume['volume_name'],
            'volume_shape': volume['volume_shape'],
            'labeled_patches': labeled_count,
            'empty_patches': empty_count,
            'total_patches': len(volume_patches)
        })
        
        print(f"   ✅ Selected: {labeled_count} labeled, {empty_count} empty (adjacent) = {len(volume_patches)} total")
    
    # Calculate global statistics
    total_labeled = sum(vs['labeled_patches'] for vs in volume_stats)
    total_empty = sum(vs['empty_patches'] for vs in volume_stats)
    total_patches = len(all_patches)
    
    print(f"\n📊 Global patch counts:")
    print(f"   Labeled: {total_labeled}")
    print(f"   Empty (adjacent): {total_empty}")
    print(f"   Total: {total_patches}")
    print(f"   Ratio: {total_labeled/total_patches*100:.1f}% labeled, {total_empty/total_patches*100:.1f}% empty")
    
    # Split into train/val by VOLUME
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
        'vicinity_range': vicinity_range,
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
    
    print(f"\n✅ Stack dataset created!")
    print(f"💾 Saved to: {output_path}")
    
    # Print comprehensive statistics
    print(f"\n📊 Dataset Statistics:")
    print(f"🚂 Train: {len(train_patches)} patches from {len(train_volume_indices)} volumes")
    print(f"   - Labeled: {train_labeled} ({train_labeled/len(train_patches)*100:.1f}%)")
    print(f"   - Empty (adjacent): {train_empty} ({train_empty/len(train_patches)*100:.1f}%)")
    print(f"✅ Val: {len(val_patches)} patches from {len(val_volume_indices)} volumes")
    print(f"   - Labeled: {val_labeled} ({val_labeled/len(val_patches)*100:.1f}%)")
    print(f"   - Empty (adjacent): {val_empty} ({val_empty/len(val_patches)*100:.1f}%)")
    
    # Print per-volume statistics
    print(f"\n📁 Per-Volume Statistics:")
    for stats in volume_stats:
        split = "TRAIN" if stats['volume_idx'] in train_volume_indices else "VAL"
        print(f"   [{split}] Volume {stats['volume_idx']+1}: {stats['total_patches']} patches "
              f"({stats['labeled_patches']} labeled, {stats['empty_patches']} adjacent empty)")
    
    return dataset

def main():
    """Main function"""
    analysis_file = "/teamspace/studios/this_studio/spleen/data/processed/patch_analysis.json"
    output_file = "/teamspace/studios/this_studio/spleen/data/processed/dataset_stack.json"
    
    # Configuration
    config = {
        'train_ratio': 0.8,
        'vicinity_range': 5  # Include unlabeled patches within 5 slices of labeled
    }
    
    print("🎯 Create Stack Dataset (for post-processing)")
    print("=" * 70)
    print(f"Analysis file: {analysis_file}")
    print(f"Output file: {output_file}")
    print(f"Config: {config}")
    print("=" * 70)
    
    # Set random seed for reproducibility
    random.seed(42)
    np.random.seed(42)
    
    # Create dataset
    dataset = create_stack_dataset(analysis_file, output_file, **config)
    
    print(f"\n🎉 Stack dataset creation complete!")
    print(f"📁 Dataset file: {output_file}")
    print(f"\n💡 This dataset is optimized for volume reconstruction and post-processing")

if __name__ == "__main__":
    main()
