#!/usr/bin/env python3
"""
Create balanced dataset from patch analysis
- Balances labeled vs empty patches (50/50)
- Ensures spatial uniformity across z-axis (height)
- Creates reproducible balanced dataset
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

def divide_volume_into_zones(volume_shape, num_zones=3):
    """
    Divide volume into spatial zones (beginning, middle, end)
    Returns zone boundaries for slice indices
    """
    total_slices = volume_shape[2]
    zone_size = total_slices // num_zones
    
    zones = []
    for i in range(num_zones):
        start_slice = i * zone_size
        if i == num_zones - 1:  # Last zone gets remaining slices
            end_slice = total_slices - 1
        else:
            end_slice = (i + 1) * zone_size - 1
        
        zones.append({
            'zone_id': i,
            'start_slice': start_slice,
            'end_slice': end_slice,
            'name': ['beginning', 'middle', 'end'][i]
        })
    
    return zones

def assign_patches_to_zones(patches, zones):
    """
    Assign patches to spatial zones based on their slice positions
    """
    zone_patches = {zone['zone_id']: [] for zone in zones}
    
    for patch in patches:
        patch_slice = patch['start_slice']
        
        # Find which zone this patch belongs to
        for zone in zones:
            if zone['start_slice'] <= patch_slice <= zone['end_slice']:
                zone_patches[zone['zone_id']].append(patch)
                break
    
    return zone_patches

def select_balanced_patches_from_volume(volume, zones, target_patches_per_volume):
    """
    Select balanced patches from a single volume
    - Take ALL labeled patches available
    - Select empty patches to balance (50/50 globally)
    - Distribute empty patches spatially across zones
    """
    labeled_patches = volume['labeled_patches']
    empty_patches = volume['empty_patches']
    
    # Take ALL labeled patches
    selected_labeled = labeled_patches.copy()
    labeled_count = len(selected_labeled)
    
    # Calculate how many empty patches we need for 50/50 balance
    total_needed = target_patches_per_volume
    empty_needed = min(len(empty_patches), total_needed - labeled_count)
    
    # If we don't have enough labeled patches, take what we can
    if labeled_count > total_needed:
        selected_labeled = random.sample(labeled_patches, total_needed)
        labeled_count = len(selected_labeled)
        empty_needed = 0
    
    # Distribute empty patches across zones
    selected_empty = []
    if empty_needed > 0:
        # Calculate empty patches per zone
        empty_per_zone = empty_needed // len(zones)
        remaining_empty = empty_needed % len(zones)
        
        for i, zone in enumerate(zones):
            # Get empty patches for this zone
            zone_empty = [p for p in empty_patches 
                         if zone['start_slice'] <= p['start_slice'] <= zone['end_slice']]
            
            # Calculate how many to take from this zone
            zone_empty_count = empty_per_zone
            if i < remaining_empty:  # Distribute remaining
                zone_empty_count += 1
            
            # Take what we can from this zone
            zone_empty_count = min(zone_empty_count, len(zone_empty))
            
            if zone_empty_count > 0:
                selected_zone_empty = random.sample(zone_empty, zone_empty_count)
                selected_empty.extend(selected_zone_empty)
    
    # Add zone information to labeled patches
    for patch in selected_labeled:
        patch['zone_id'] = None  # Will be assigned based on slice position
        patch['zone_name'] = None
        patch['is_labeled'] = True
        
        # Find zone for this patch
        for zone in zones:
            if zone['start_slice'] <= patch['start_slice'] <= zone['end_slice']:
                patch['zone_id'] = zone['zone_id']
                patch['zone_name'] = zone['name']
                break
    
    # Add zone information to empty patches
    for patch in selected_empty:
        patch['zone_id'] = None  # Will be assigned based on slice position
        patch['zone_name'] = None
        patch['is_labeled'] = False
        
        # Find zone for this patch
        for zone in zones:
            if zone['start_slice'] <= patch['start_slice'] <= zone['end_slice']:
                patch['zone_id'] = zone['zone_id']
                patch['zone_name'] = zone['name']
                break
    
    selected_patches = selected_labeled + selected_empty
    
    # Print zone distribution
    zone_stats = {zone['zone_id']: {'labeled': 0, 'empty': 0} for zone in zones}
    for patch in selected_patches:
        if patch['zone_id'] is not None:
            if patch['is_labeled']:
                zone_stats[patch['zone_id']]['labeled'] += 1
            else:
                zone_stats[patch['zone_id']]['empty'] += 1
    
    print(f"    Total: {len(selected_patches)} patches ({labeled_count} labeled, {len(selected_empty)} empty)")
    for zone in zones:
        stats = zone_stats[zone['zone_id']]
        print(f"    Zone {zone['name']}: {stats['labeled']} labeled, {stats['empty']} empty")
    
    return selected_patches

def create_balanced_dataset(analysis_file, output_file, train_ratio=0.8, patches_per_volume=30):
    """
    Create balanced dataset from patch analysis
    Args:
        analysis_file: Path to patch_analysis.json
        output_file: Path to save balanced dataset
        train_ratio: Ratio for train/val split (0.8 = 80% train)
        patches_per_volume: Target number of patches per volume
    """
    print("🔍 Loading patch analysis...")
    analysis = load_patch_analysis(analysis_file)
    
    volumes = analysis['volumes']
    total_volumes = len(volumes)
    
    print(f"📊 Found {total_volumes} volumes")
    print(f"🎯 Target: {patches_per_volume} patches per volume")
    
    # Create zones (using first volume as reference)
    reference_volume = volumes[0]
    zones = divide_volume_into_zones(reference_volume['volume_shape'])
    
    print(f"🗺️  Spatial zones: {[z['name'] for z in zones]}")
    
    # First pass: collect all labeled patches and count empty patches
    all_labeled_patches = []
    all_empty_patches = []
    volume_stats = []
    
    print(f"\n🔍 First pass: Collecting all patches...")
    
    for i, volume in enumerate(volumes):
        print(f"📁 Volume {i+1}/{total_volumes}: {volume['volume_name']}")
        
        # Collect all labeled patches
        volume_labeled = volume['labeled_patches'].copy()
        volume_empty = volume['empty_patches'].copy()
        
        # Add volume info
        for patch in volume_labeled + volume_empty:
            patch['volume_idx'] = volume['volume_idx']
            patch['volume_name'] = volume['volume_name']
            patch['volume_shape'] = volume['volume_shape']
        
        all_labeled_patches.extend(volume_labeled)
        all_empty_patches.extend(volume_empty)
        
        volume_stats.append({
            'volume_idx': volume['volume_idx'],
            'volume_name': volume['volume_name'],
            'total_patches': len(volume_labeled) + len(volume_empty),
            'labeled_patches': len(volume_labeled),
            'empty_patches': len(volume_empty)
        })
        
        print(f"  ✅ Volume {i+1}: {len(volume_labeled)} labeled, {len(volume_empty)} empty")
    
    # Calculate global balance
    total_labeled = len(all_labeled_patches)
    total_empty = len(all_empty_patches)
    
    print(f"\n📊 Global patch counts:")
    print(f"  Labeled: {total_labeled}")
    print(f"  Empty: {total_empty}")
    
    # Calculate how many patches we can use for 50/50 balance
    max_usable = min(total_labeled, total_empty) * 2  # 50/50 split
    target_labeled = max_usable // 2
    target_empty = max_usable // 2
    
    print(f"🎯 Target: {target_labeled} labeled, {target_empty} empty = {max_usable} total")
    
    # Select labeled patches (take all if we don't have enough)
    selected_labeled = all_labeled_patches.copy()
    if len(selected_labeled) > target_labeled:
        selected_labeled = random.sample(all_labeled_patches, target_labeled)
    
    # Select empty patches to balance
    selected_empty = random.sample(all_empty_patches, target_empty)
    
    # Combine and assign zones
    all_selected_patches = selected_labeled + selected_empty
    
    # Assign zones to all patches
    for patch in all_selected_patches:
        patch['is_labeled'] = patch in selected_labeled
        patch['zone_id'] = None
        patch['zone_name'] = None
        
        # Find zone for this patch
        for zone in zones:
            if zone['start_slice'] <= patch['start_slice'] <= zone['end_slice']:
                patch['zone_id'] = zone['zone_id']
                patch['zone_name'] = zone['name']
                break
    
    # Split into train/val
    total_patches = len(all_selected_patches)
    train_count = int(total_patches * train_ratio)
    val_count = total_patches - train_count
    
    # Shuffle and split
    random.shuffle(all_selected_patches)
    train_patches = all_selected_patches[:train_count]
    val_patches = all_selected_patches[train_count:]
    
    # Create balanced dataset
    balanced_dataset = {
        'timestamp': datetime.now().isoformat(),
        'source_analysis': str(analysis_file),
        'train_ratio': train_ratio,
        'patches_per_volume': patches_per_volume,
        'total_volumes': total_volumes,
        'zones': zones,
        'train_patches': train_patches,
        'val_patches': val_patches,
        'volume_stats': volume_stats,
        'train_stats': {
            'total_patches': len(train_patches),
            'labeled_patches': len([p for p in train_patches if p['is_labeled']]),
            'empty_patches': len([p for p in train_patches if not p['is_labeled']])
        },
        'val_stats': {
            'total_patches': len(val_patches),
            'labeled_patches': len([p for p in val_patches if p['is_labeled']]),
            'empty_patches': len([p for p in val_patches if not p['is_labeled']])
        }
    }
    
    # Save balanced dataset
    output_path = Path(output_file)
    output_path.parent.mkdir(exist_ok=True)
    
    with open(output_path, 'w') as f:
        json.dump(balanced_dataset, f, indent=2)
    
    print(f"\n✅ Balanced dataset created!")
    print(f"💾 Saved to: {output_path}")
    
    # Print comprehensive statistics
    print(f"\n📊 Dataset Statistics:")
    print(f"🚂 Train: {balanced_dataset['train_stats']['total_patches']} patches")
    print(f"  - Labeled: {balanced_dataset['train_stats']['labeled_patches']}")
    print(f"  - Empty: {balanced_dataset['train_stats']['empty_patches']}")
    print(f"✅ Val: {balanced_dataset['val_stats']['total_patches']} patches")
    print(f"  - Labeled: {balanced_dataset['val_stats']['labeled_patches']}")
    print(f"  - Empty: {balanced_dataset['val_stats']['empty_patches']}")
    
    # Print per-zone statistics
    print(f"\n🗺️  Spatial Distribution (Train):")
    for zone in zones:
        train_zone = [p for p in train_patches if p['zone_id'] == zone['zone_id']]
        labeled_zone = [p for p in train_zone if p['is_labeled']]
        empty_zone = [p for p in train_zone if not p['is_labeled']]
        print(f"  {zone['name']}: {len(train_zone)} total ({len(labeled_zone)} labeled, {len(empty_zone)} empty)")
    
    print(f"\n🗺️  Spatial Distribution (Val):")
    for zone in zones:
        val_zone = [p for p in val_patches if p['zone_id'] == zone['zone_id']]
        labeled_zone = [p for p in val_zone if p['is_labeled']]
        empty_zone = [p for p in val_zone if not p['is_labeled']]
        print(f"  {zone['name']}: {len(val_zone)} total ({len(labeled_zone)} labeled, {len(empty_zone)} empty)")
    
    # Print per-volume statistics
    print(f"\n📁 Per-Volume Statistics:")
    for stats in volume_stats:
        print(f"  Volume {stats['volume_idx']+1}: {stats['total_patches']} patches ({stats['labeled_patches']} labeled)")
    
    return balanced_dataset

def main():
    """Main function"""
    analysis_file = "/teamspace/studios/this_studio/spleen/data/processed/patch_analysis.json"
    output_file = "/teamspace/studios/this_studio/spleen/data/processed/balanced_dataset.json"
    
    # Configuration
    config = {
        'train_ratio': 0.8,
        'patches_per_volume': 30  # Adjust based on your needs
    }
    
    print("🎯 Create Balanced Dataset")
    print("=" * 50)
    print(f"Analysis file: {analysis_file}")
    print(f"Output file: {output_file}")
    print(f"Config: {config}")
    print("=" * 50)
    
    # Set random seed for reproducibility
    random.seed(42)
    np.random.seed(42)
    
    # Create balanced dataset
    balanced_dataset = create_balanced_dataset(analysis_file, output_file, **config)
    
    print(f"\n🎉 Balanced dataset creation complete!")
    print(f"📁 Balanced dataset: {output_file}")

if __name__ == "__main__":
    main()
