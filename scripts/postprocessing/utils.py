#!/usr/bin/env python3
"""
Post-processing utilities for 3D medical image segmentation
- Connected component analysis
- Keep largest component
- Volume reconstruction
"""

import numpy as np
import torch
from scipy import ndimage
from skimage import measure

def keep_largest_component(mask_3d):
    """
    Keep only the largest connected component in 3D mask
    Args:
        mask_3d: 3D binary mask (numpy array)
    Returns:
        mask_3d: 3D binary mask with only largest component
    """
    # Label connected components
    labeled_mask, num_components = ndimage.label(mask_3d)
    
    if num_components == 0:
        return mask_3d
    
    # Find the largest component
    component_sizes = np.bincount(labeled_mask.ravel())
    
    # Skip background (label 0)
    if len(component_sizes) > 1:
        component_sizes[0] = 0
        largest_component = component_sizes.argmax()
        
        # Keep only the largest component
        mask_3d = (labeled_mask == largest_component).astype(np.float32)
    
    return mask_3d

def postprocess_prediction(pred_volume):
    """
    Apply post-processing to prediction volume
    Args:
        pred_volume: 3D prediction volume (numpy array or torch tensor)
    Returns:
        processed_volume: Post-processed 3D volume
    """
    # Convert to numpy if needed
    if torch.is_tensor(pred_volume):
        pred_volume = pred_volume.cpu().numpy()
    
    # Ensure binary mask
    binary_mask = (pred_volume > 0.5).astype(np.float32)
    
    # Apply connected component filtering
    processed_mask = keep_largest_component(binary_mask)
    
    return processed_mask

def reconstruct_volume_from_patches(patches_dict, volume_shape, slice_depth=5, overlap_mode='max'):
    """
    Reconstruct full 3D volume from overlapping patches using sliding window
    Args:
        patches_dict: Dictionary {start_slice: prediction_patch}
        volume_shape: Shape of the full volume (H, W, D)
        slice_depth: Depth of each patch
        overlap_mode: How to handle overlaps ('max', 'mean', 'vote')
    Returns:
        volume: Reconstructed 3D volume
    """
    H, W, D = volume_shape
    
    # Initialize volume and count array for averaging
    volume = np.zeros((H, W, D), dtype=np.float32)
    count = np.zeros((H, W, D), dtype=np.float32)
    
    # Fill in patches
    for start_slice, patch in patches_dict.items():
        end_slice = min(start_slice + slice_depth, D)
        patch_depth = end_slice - start_slice
        
        # Handle patch that might be tensor
        if torch.is_tensor(patch):
            patch = patch.cpu().numpy()
        
        # Remove batch and channel dimensions if present
        if patch.ndim == 4:  # [B, C, H, W]
            patch = patch[0, 0]  # Take first batch, first channel
        elif patch.ndim == 3:  # [C, H, W]
            patch = patch[0]  # Take first channel
        
        # For sliding window, we only use the middle slice(s)
        # To avoid artifacts at patch boundaries
        middle_idx = slice_depth // 2
        if patch_depth == slice_depth:
            # Use only middle slice
            volume[:, :, start_slice + middle_idx] += patch
            count[:, :, start_slice + middle_idx] += 1
    
    # Average overlapping regions
    count = np.maximum(count, 1)  # Avoid division by zero
    volume = volume / count
    
    return volume

def sliding_window_inference(model, volume_data, device, slice_depth=5, batch_size=1):
    """
    Perform sliding window inference on full 3D volume
    Args:
        model: Trained model
        volume_data: Full 3D volume (H, W, D)
        device: torch device
        slice_depth: Depth of sliding window
        batch_size: Batch size for inference
    Returns:
        prediction_volume: Full 3D prediction volume
    """
    H, W, D = volume_data.shape
    predictions = {}
    
    model.eval()
    with torch.no_grad():
        # Slide through the volume
        for start_slice in range(D - slice_depth + 1):
            # Extract patch
            patch = volume_data[:, :, start_slice:start_slice + slice_depth]
            
            # Transpose to [C, H, W] where C = slice_depth
            patch = np.transpose(patch, (2, 0, 1))  # [D, H, W] -> [D, H, W]
            
            # Normalize
            patch = (patch - patch.mean()) / (patch.std() + 1e-8)
            
            # Convert to tensor and add batch dimension
            patch_tensor = torch.from_numpy(patch).float().unsqueeze(0).to(device)
            
            # Predict
            output = model(patch_tensor)
            pred = torch.sigmoid(output)
            
            # Store prediction for middle slice
            middle_idx = slice_depth // 2
            predictions[start_slice + middle_idx] = pred[0, 0].cpu().numpy()
    
    # Reconstruct full volume
    pred_volume = np.zeros((H, W, D), dtype=np.float32)
    for slice_idx, pred_slice in predictions.items():
        pred_volume[:, :, slice_idx] = pred_slice
    
    return pred_volume

def evaluate_with_postprocessing(model, volume_path, label_path, device, slice_depth=5):
    """
    Evaluate model on full volume with and without post-processing
    Args:
        model: Trained model
        volume_path: Path to volume file
        label_path: Path to label file
        device: torch device
        slice_depth: Depth of sliding window
    Returns:
        metrics: Dictionary with metrics before/after post-processing
    """
    import nibabel as nib
    
    # Load volume and label
    volume_nii = nib.load(volume_path)
    label_nii = nib.load(label_path)
    
    volume_data = volume_nii.get_fdata()
    label_data = label_nii.get_fdata()
    
    # Get prediction
    pred_volume = sliding_window_inference(model, volume_data, device, slice_depth)
    
    # Binary predictions
    pred_binary = (pred_volume > 0.5).astype(np.float32)
    
    # Post-process
    pred_postprocessed = postprocess_prediction(pred_volume)
    
    # Calculate metrics
    label_binary = (label_data > 0).astype(np.float32)
    
    # Dice before post-processing
    dice_before = calculate_dice(pred_binary, label_binary)
    
    # Dice after post-processing
    dice_after = calculate_dice(pred_postprocessed, label_binary)
    
    return {
        'dice_before': dice_before,
        'dice_after': dice_after,
        'improvement': dice_after - dice_before,
        'pred_before': pred_binary,
        'pred_after': pred_postprocessed
    }

def calculate_dice(pred, target, smooth=1e-5):
    """Calculate Dice coefficient"""
    intersection = (pred * target).sum()
    union = pred.sum() + target.sum()
    dice = (2.0 * intersection + smooth) / (union + smooth)
    return dice

def calculate_iou(pred, target, smooth=1e-5):
    """Calculate IoU (Intersection over Union)"""
    intersection = (pred * target).sum()
    union = pred.sum() + target.sum() - intersection
    iou = (intersection + smooth) / (union + smooth)
    return iou

