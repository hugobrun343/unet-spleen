import torch
from torch.utils.data import Dataset, DataLoader
import nibabel as nib
import numpy as np
import os
import json
from pathlib import Path
import torchvision.transforms as transforms
from datetime import datetime

class SpleenDataset(Dataset):
    def __init__(self, dataset_path, split='training', transform=None, slice_depth=5):
        """
        Dataset for spleen segmentation with 3D blocks using dataset.json
        Args:
            dataset_path: Path to dataset directory containing dataset.json
            split: 'training' or 'test'
            transform: Optional transform to be applied on images
            slice_depth: Number of slices to stack (5 or 6)
        """
        self.dataset_path = Path(dataset_path)
        self.transform = transform
        self.slice_depth = slice_depth
        self.split = split
        
        # Load dataset.json
        with open(self.dataset_path / 'dataset.json', 'r') as f:
            self.dataset_info = json.load(f)
        
        # Get file list based on split
        if split == 'training':
            self.file_pairs = self.dataset_info['training']
        elif split == 'test':
            # For test, we only have images, no labels
            self.file_pairs = [{'image': f} for f in self.dataset_info['test']]
        else:
            raise ValueError("split must be 'training' or 'test'")
        
        print(f"Loaded {len(self.file_pairs)} {split} samples")
        
    def __len__(self):
        return len(self.file_pairs)
    
    def __getitem__(self, idx):
        # Get file paths from dataset.json
        file_pair = self.file_pairs[idx]
        img_path = self.dataset_path / file_pair['image']
        
        # Load image (try .nii first, then .nii.gz)
        try:
            img_nii = nib.load(img_path)
        except FileNotFoundError:
            # Try without .gz extension
            img_path_no_gz = str(img_path).replace('.nii.gz', '.nii')
            img_nii = nib.load(img_path_no_gz)
        
        img_data = img_nii.get_fdata()
        
        # Load label if available (training only)
        if self.split == 'training' and 'label' in file_pair:
            label_path = self.dataset_path / file_pair['label']
            try:
                label_nii = nib.load(label_path)
            except FileNotFoundError:
                # Try without .gz extension
                label_path_no_gz = str(label_path).replace('.nii.gz', '.nii')
                label_nii = nib.load(label_path_no_gz)
            label_data = label_nii.get_fdata()
        else:
            # For test set, create dummy labels
            label_data = np.zeros_like(img_data)
        
        # Get random slice position for 3D block
        original_shape = img_data.shape
        max_slice = img_data.shape[2] - self.slice_depth
        if max_slice <= 0:
            # If volume is too small, pad it
            pad_size = self.slice_depth - img_data.shape[2]
            img_data = np.pad(img_data, ((0, 0), (0, 0), (0, pad_size)), mode='edge')
            label_data = np.pad(label_data, ((0, 0), (0, 0), (0, pad_size)), mode='edge')
            start_slice = 0
            print(f"Volume {idx}: {original_shape} -> padded to {img_data.shape}, using slice 0-{self.slice_depth-1}")
        else:
            start_slice = np.random.randint(0, max_slice + 1)
            print(f"Volume {idx}: {original_shape}, using slices {start_slice}-{start_slice + self.slice_depth - 1}")
        
        # Extract 3D block: [H, W, slice_depth]
        img_block = img_data[:, :, start_slice:start_slice + self.slice_depth]
        label_block = label_data[:, :, start_slice:start_slice + self.slice_depth]
        
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

def save_data_splits(full_dataset, train_subset, val_subset, dataset_path):
    """
    Save train/val splits to file for reproducibility
    """
    # Get indices for train and val
    train_indices = train_subset.indices
    val_indices = val_subset.indices
    
    # Get file names for train and val
    train_files = [full_dataset.file_pairs[i] for i in train_indices]
    val_files = [full_dataset.file_pairs[i] for i in val_indices]
    
    # Create splits info
    splits_info = {
        'timestamp': datetime.now().isoformat(),
        'total_samples': len(full_dataset),
        'train_samples': len(train_subset),
        'val_samples': len(val_subset),
        'train_split_ratio': len(train_subset) / len(full_dataset),
        'train_files': train_files,
        'val_files': val_files,
        'train_indices': train_indices.tolist() if hasattr(train_indices, 'tolist') else list(train_indices),
        'val_indices': val_indices.tolist() if hasattr(val_indices, 'tolist') else list(val_indices)
    }
    
    # Save to file
    splits_file = Path(dataset_path) / 'data_splits.json'
    with open(splits_file, 'w') as f:
        json.dump(splits_info, f, indent=2)
    
    print(f"Data splits saved to: {splits_file}")
    print(f"Train: {len(train_subset)} samples, Val: {len(val_subset)} samples")

def get_data_loaders(dataset_path, batch_size=4, train_split=0.8, num_workers=4, slice_depth=5, save_splits=True, max_samples=None):
    """
    Create train and validation data loaders using dataset.json
    Args:
        max_samples: If provided, limit the dataset to this many samples (for testing)
    """
    # Create training dataset
    train_dataset = SpleenDataset(dataset_path, split='training', slice_depth=slice_depth)
    
    # Limit dataset size if max_samples is provided
    if max_samples is not None and max_samples < len(train_dataset):
        print(f"🔬 TESTING MODE: Limiting dataset to {max_samples} samples")
        train_dataset.file_pairs = train_dataset.file_pairs[:max_samples]
        print(f"Limited to {len(train_dataset.file_pairs)} samples")
    
    # Split training data into train/val
    train_size = int(train_split * len(train_dataset))
    val_size = len(train_dataset) - train_size
    
    train_subset, val_subset = torch.utils.data.random_split(
        train_dataset, [train_size, val_size]
    )
    
    # Save train/val splits for reproducibility
    if save_splits:
        save_data_splits(train_dataset, train_subset, val_subset, dataset_path)
    
    # Create data loaders
    train_loader = DataLoader(
        train_subset, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_subset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=num_workers,
        pin_memory=True
    )
    
    return train_loader, val_loader

def get_test_loader(dataset_path, batch_size=4, num_workers=4, slice_depth=5):
    """
    Create test data loader using dataset.json
    """
    test_dataset = SpleenDataset(dataset_path, split='test', slice_depth=slice_depth)
    
    test_loader = DataLoader(
        test_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=num_workers,
        pin_memory=True
    )
    
    return test_loader

def test_data_loader():
    """Test function to verify data loading works"""
    dataset_path = "/teamspace/studios/this_studio/spleen/Dataset001_Spleen"
    
    try:
        print("=" * 60)
        print("🔍 ANALYZING DATASET STRUCTURE")
        print("=" * 60)
        
        # Test training data
        train_loader, val_loader = get_data_loaders(dataset_path, batch_size=2, slice_depth=5)
        
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
        
        # Test test data
        test_loader = get_test_loader(dataset_path, batch_size=2, slice_depth=5)
        print(f"\nTest samples: {len(test_loader.dataset)}")
        
        for images, labels in test_loader:
            print(f"Test Image shape: {images.shape}")
            print(f"Test Label shape: {labels.shape}")
            break
            
        print("\n✅ Data loading test successful!")
        print("=" * 60)
        
    except Exception as e:
        print(f"❌ Error in data loading: {e}")

if __name__ == "__main__":
    test_data_loader()
