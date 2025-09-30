#!/usr/bin/env python3
"""
Preprocessing script - just compute patch coordinates and stats
- Loads volumes one by one
- Computes which patches have positive labels
- Saves only coordinates and stats (no actual data)
"""

import json
import numpy as np
import nibabel as nib
from pathlib import Path
from datetime import datetime

def count_positive_label(label_block):
    """Check if block contains any positive labels"""
    positive_pixels = np.sum(label_block > 0)
    return positive_pixels

def analyze_volume(img_path, label_path, slice_depth=5):
    """
    Analyze a single volume and return patch coordinates
    """
    # Load image and label
    try:
        img_nii = nib.load(img_path)
    except FileNotFoundError:
        img_path_no_gz = str(img_path).replace('.nii.gz', '.nii')
        img_nii = nib.load(img_path_no_gz)
    
    try:
        label_nii = nib.load(label_path)
    except FileNotFoundError:
        label_path_no_gz = str(label_path).replace('.nii.gz', '.nii')
        label_nii = nib.load(label_path_no_gz)
    
    img_data = img_nii.get_fdata()
    label_data = label_nii.get_fdata()
    
    # Ensure same shape
    if img_data.shape != label_data.shape:
        print(f"Warning: Shape mismatch {img_data.shape} vs {label_data.shape}")
        return None
    
    volume_shape = img_data.shape
    max_start_slice = volume_shape[2] - slice_depth
    
    print(f"📊 Volume shape: {volume_shape}, analyzing {max_start_slice + 1} possible patches")
    
    # Analyze all possible patches
    labeled_patches = []
    empty_patches = []
    
    for start_slice in range(max_start_slice + 1):
        # Extract block: [H, W, slice_depth]
        label_block = label_data[:, :, start_slice:start_slice + slice_depth]
        
        # Check if patch contains any positive labels
        positive_pixels = count_positive_label(label_block)
        
        patch_info = {
            'start_slice': int(start_slice),
            'end_slice': int(start_slice + slice_depth - 1),
            'positive_pixels': int(positive_pixels)
        }
        
        if positive_pixels > 0:
            labeled_patches.append(patch_info)
        else:
            empty_patches.append(patch_info)
    
    print(f"✅ Found {len(labeled_patches)} labeled patches, {len(empty_patches)} empty patches")
    
    # Clear data from memory
    del img_data, label_data
    
    return {
        'volume_shape': volume_shape,
        'labeled_patches': labeled_patches,
        'empty_patches': empty_patches,
        'total_patches': len(labeled_patches) + len(empty_patches)
    }

def preprocess_dataset(dataset_path, output_path, slice_depth=5):
    """
    Simple preprocessing - just analyze volumes and save patch coordinates
    """
    dataset_path = Path(dataset_path)
    output_path = Path(output_path)
    output_path.mkdir(exist_ok=True)
    
    # Load dataset.json
    with open(dataset_path / 'dataset.json', 'r') as f:
        dataset_info = json.load(f)
    
    total_volumes = len(dataset_info['training'])
    volume_analyses = []
    
    print(f"🚀 Analyzing {total_volumes} volumes...")
    print(f"Slice depth: {slice_depth}")
    
    for i, file_pair in enumerate(dataset_info['training']):
        img_path = dataset_path / file_pair['image']
        label_path = dataset_path / file_pair['label']
        
        print(f"\n📁 Volume {i+1}/{total_volumes}: {file_pair['image']}")
        
        # Analyze this volume
        volume_analysis = analyze_volume(img_path, label_path, slice_depth)
        
        if volume_analysis:
            volume_analysis['volume_idx'] = i
            volume_analysis['volume_name'] = file_pair['image']
            volume_analyses.append(volume_analysis)
            
            print(f"  ✅ Volume {i+1} complete: {volume_analysis['total_patches']} patches ({len(volume_analysis['labeled_patches'])} labeled)")
        
        # Force garbage collection
        import gc
        gc.collect()
    
    # Save analysis results
    analysis_file = output_path / 'patch_analysis.json'
    print(f"\n💾 Saving analysis to {analysis_file}...")
    
    with open(analysis_file, 'w') as f:
        json.dump({
            'timestamp': datetime.now().isoformat(),
            'slice_depth': slice_depth,
            'total_volumes': len(volume_analyses),
            'volumes': volume_analyses
        }, f, indent=2)
    
    print(f"\n✅ Analysis complete!")
    print(f"📊 Analyzed {len(volume_analyses)} volumes")
    print(f"💾 Saved to: {analysis_file}")
    
    # Print summary statistics
    total_labeled = sum(len(v['labeled_patches']) for v in volume_analyses)
    total_empty = sum(len(v['empty_patches']) for v in volume_analyses)
    total_patches = total_labeled + total_empty
    
    print(f"🎯 Total labeled patches: {total_labeled}")
    print(f"⚪ Total empty patches: {total_empty}")
    print(f"📈 Labeled ratio: {total_labeled/total_patches:.2%}")
    
    return analysis_file

def main():
    """Main preprocessing function"""
    dataset_path = "/teamspace/studios/this_studio/spleen/data/raw"
    output_path = "/teamspace/studios/this_studio/spleen/data/processed"
    
    # Configuration
    config = {
        'slice_depth': 5
    }
    
    print("🔧 Spleen Dataset Analysis")
    print("=" * 50)
    print(f"Dataset: {dataset_path}")
    print(f"Output: {output_path}")
    print(f"Config: {config}")
    print("=" * 50)
    
    # Run analysis
    analysis_file = preprocess_dataset(dataset_path, output_path, **config)
    
    print(f"\n🎉 Analysis complete! Use the patch coordinates for training.")
    print(f"📁 Analysis file: {analysis_file}")

if __name__ == "__main__":
    main()