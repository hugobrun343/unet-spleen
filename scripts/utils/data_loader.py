#!/usr/bin/env python3
"""
Balanced data loader for spleen segmentation
- Loads patches from balanced_dataset.json
- Supports train/val splits
- Supports test mode with limited patches
"""

import torch
from torch.utils.data import Dataset, DataLoader
import nibabel as nib
import numpy as np
import json
from pathlib import Path
import torchvision.transforms as transforms
from datetime import datetime

def load_balanced_dataset(dataset_file):
    """Load balanced dataset from JSON file"""
    with open(dataset_file, 'r') as f:
        dataset = json.load(f)
    return dataset

class BalancedSpleenDataset(Dataset):
    def __init__(self, dataset_file, split='train', transform=None, slice_depth=5, test_patches=None):
        """
        Dataset for balanced spleen segmentation
        Args:
            dataset_file: Path to balanced_dataset.json
            split: 'train', 'val', or 'test'
            transform: Optional transform to be applied on images
            slice_depth: Number of slices to stack (should match dataset)
            test_patches: Number of patches for test mode (if None, use all)
        """
        self.dataset_file = Path(dataset_file)
        self.transform = transform
        self.slice_depth = slice_depth
        self.split = split
        self.test_patches = test_patches
        
        # Load balanced dataset
        self.dataset = load_balanced_dataset(self.dataset_file)
        
        # Get patches based on split
        if split == 'train':
            self.patches = self.dataset['train_patches']
        elif split == 'val':
            self.patches = self.dataset['val_patches']
        elif split == 'test':
            # For test, use labeled patches only
            all_patches = self.dataset['train_patches'] + self.dataset['val_patches']
            labeled_patches = [p for p in all_patches if p['is_labeled']]
            
            if test_patches is not None:
                self.patches = labeled_patches[:test_patches]
            else:
                self.patches = labeled_patches
        else:
            raise ValueError("split must be 'train', 'val', or 'test'")
        
        print(f"📊 Loaded {len(self.patches)} {split} patches")
        
        # Print statistics
        if split in ['train', 'val']:
            labeled_count = len([p for p in self.patches if p['is_labeled']])
            empty_count = len([p for p in self.patches if not p['is_labeled']])
            print(f"🎯 Labeled: {labeled_count}, Empty: {empty_count}")
            print(f"📈 Labeled ratio: {labeled_count/len(self.patches):.2%}")
        else:
            print(f"🧪 Test mode: {len(self.patches)} labeled patches")
    
    def __len__(self):
        return len(self.patches)
    
    def __getitem__(self, idx):
        patch_info = self.patches[idx]
        
        # Load image and label from original volume
        volume_path = Path("/teamspace/studios/this_studio/spleen/data/raw/Dataset001_Spleen")
        img_path = volume_path / patch_info['volume_name']
        
        # Load image
        try:
            img_nii = nib.load(img_path)
        except FileNotFoundError:
            img_path_no_gz = str(img_path).replace('.nii.gz', '.nii')
            img_nii = nib.load(img_path_no_gz)
        
        img_data = img_nii.get_fdata()
        
        # Load label
        label_name = patch_info['volume_name'].lstrip('./').lstrip('._').replace('imagesTr', 'labelsTr')
        label_path = volume_path / label_name
        
        try:
            label_nii = nib.load(label_path)
        except FileNotFoundError:
            label_path_no_gz = str(label_path).replace('.nii.gz', '.nii')
            label_nii = nib.load(label_path_no_gz)
        
        label_data = label_nii.get_fdata()
        
        # Extract patch from volume
        start_slice = patch_info['start_slice']
        end_slice = patch_info['end_slice']
        
        # Extract 3D block: [H, W, slice_depth]
        img_block = img_data[:, :, start_slice:end_slice + 1]
        label_block = label_data[:, :, start_slice:end_slice + 1]
        
        # Reshape to [slice_depth, H, W] for channel dimension
        img_block = np.transpose(img_block, (2, 0, 1))  # [slice_depth, H, W]
        label_block = np.transpose(label_block, (2, 0, 1))  # [slice_depth, H, W]
        
        # Convert to torch tensors
        img_tensor = torch.FloatTensor(img_block)  # [slice_depth, H, W]
        label_tensor = torch.FloatTensor(label_block)  # [slice_depth, H, W]
        
        # Normalize image to [0, 1]
        img_tensor = (img_tensor - img_tensor.min()) / (img_tensor.max() - img_tensor.min())
        
        # Apply transforms if any
        if self.transform:
            img_tensor = self.transform(img_tensor)
            label_tensor = self.transform(label_tensor)
        
        return img_tensor, label_tensor

def get_balanced_data_loaders(dataset_file, batch_size=4, num_workers=4, slice_depth=5, max_patches=None):
    """
    Create train and validation data loaders from balanced dataset
    """
    # Create datasets
    train_dataset = BalancedSpleenDataset(dataset_file, split='train', slice_depth=slice_depth)
    val_dataset = BalancedSpleenDataset(dataset_file, split='val', slice_depth=slice_depth)
    
    # Limit number of patches if specified
    if max_patches is not None:
        # Calculate train/val split
        total_patches = min(len(train_dataset) + len(val_dataset), max_patches)
        train_size = int(total_patches * 0.8)
        val_size = total_patches - train_size
        
        # Limit datasets
        train_dataset.patches = train_dataset.patches[:train_size]
        val_dataset.patches = val_dataset.patches[:val_size]
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return train_loader, val_loader

def get_mixed_data_loaders(dataset_file, batch_size=4, num_workers=4, slice_depth=5, labeled_patches=3, unlabeled_patches=2):
    """
    Create train and validation data loaders with mixed labeled/unlabeled patches
    """
    # Load dataset info
    with open(dataset_file, 'r') as f:
        data = json.load(f)
    
    # Get all patches from train and val
    all_train_patches = data['train_patches']
    all_val_patches = data['val_patches']
    all_patches = all_train_patches + all_val_patches
    
    # Get labeled patches (from labeled volumes)
    labeled_patches_list = []
    for patch in all_patches:
        if patch['is_labeled']:
            labeled_patches_list.append(patch)
    
    # Get unlabeled patches (from unlabeled volumes) 
    unlabeled_patches_list = []
    for patch in all_patches:
        if not patch['is_labeled']:
            unlabeled_patches_list.append(patch)
    
    # Select the requested number of patches
    selected_labeled = labeled_patches_list[:labeled_patches]
    selected_unlabeled = unlabeled_patches_list[:unlabeled_patches]
    
    # Combine all patches
    all_patches = selected_labeled + selected_unlabeled
    
    # Split 80/20
    train_size = int(len(all_patches) * 0.8)
    train_patches = all_patches[:train_size]
    val_patches = all_patches[train_size:]
    
    # Create datasets
    train_dataset = BalancedSpleenDataset(dataset_file, split='train', slice_depth=slice_depth)
    val_dataset = BalancedSpleenDataset(dataset_file, split='val', slice_depth=slice_depth)
    
    # Override patches
    train_dataset.patches = train_patches
    val_dataset.patches = val_patches
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return train_loader, val_loader

def get_test_loader(dataset_file, test_patches=20, batch_size=4, num_workers=4, slice_depth=5):
    """
    Create test data loader with limited patches
    """
    test_dataset = BalancedSpleenDataset(
        dataset_file, 
        split='test', 
        slice_depth=slice_depth, 
        test_patches=test_patches
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return test_loader

def get_single_patch_loader(dataset_file, batch_size=1, num_workers=0, slice_depth=5):
    """
    Create loader with only 1 patch for extreme testing
    """
    single_dataset = BalancedSpleenDataset(
        dataset_file, 
        split='test', 
        slice_depth=slice_depth, 
        test_patches=1
    )
    
    single_loader = DataLoader(
        single_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return single_loader

def test_balanced_loader():
    """Test function to verify balanced data loading works"""
    dataset_file = "/teamspace/studios/this_studio/spleen/data/processed/balanced_dataset.json"
    
    try:
        print("=" * 60)
        print("🔍 TESTING BALANCED DATA LOADER")
        print("=" * 60)
        
        # Test train/val loaders
        train_loader, val_loader = get_balanced_data_loaders(dataset_file, batch_size=2)
        
        print(f"\n📊 DATASET STATISTICS:")
        print(f"Train samples: {len(train_loader.dataset)}")
        print(f"Val samples: {len(val_loader.dataset)}")
        
        # Test one batch
        print(f"\n🔬 SAMPLE ANALYSIS:")
        for images, labels in train_loader:
            print(f"Image shape: {images.shape}")
            print(f"Label shape: {labels.shape}")
            print(f"Image range: [{images.min():.3f}, {images.max():.3f}]")
            print(f"Label range: [{labels.min():.3f}, {labels.max():.3f}]")
            print(f"Number of channels (slices): {images.shape[1]}")
            
            # Count non-zero labels
            non_zero_pixels = (labels > 0).sum().item()
            total_pixels = labels.numel()
            print(f"Non-zero label pixels: {non_zero_pixels}/{total_pixels} ({100*non_zero_pixels/total_pixels:.2f}%)")
            break
        
        # Test test loader
        test_loader = get_test_loader(dataset_file, test_patches=5)
        print(f"\n🧪 Test samples: {len(test_loader.dataset)}")
        
        # Test single patch loader
        single_loader = get_single_patch_loader(dataset_file)
        print(f"🎯 Single patch samples: {len(single_loader.dataset)}")
        
        print("\n✅ Balanced data loading test successful!")
        print("=" * 60)
        
    except Exception as e:
        print(f"❌ Error in balanced data loading: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_balanced_loader()
