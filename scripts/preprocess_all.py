#!/usr/bin/env python3
"""
Complete preprocessing pipeline for spleen segmentation
- Downloads dataset if not present
- Preprocesses slices and creates patches
- Creates balanced dataset
- Tests data loading
"""

import os
import sys
import subprocess
from pathlib import Path
import json
from datetime import datetime

# Add current directory to path for imports
sys.path.append(str(Path(__file__).parent))

def run_script(script_name, description, subfolder=""):
    """Run a Python script and handle errors"""
    print(f"\n{'='*60}")
    print(f"🔄 {description}")
    print(f"{'='*60}")
    
    if subfolder:
        script_path = Path(__file__).parent / subfolder / script_name
    else:
        script_path = Path(__file__).parent / script_name
    
    if not script_path.exists():
        print(f"❌ Script not found: {script_path}")
        return False
    
    try:
        result = subprocess.run([
            sys.executable, str(script_path)
        ], capture_output=True, text=True, check=True)
        
        print("✅ Success!")
        if result.stdout:
            print(result.stdout)
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"❌ Error running {script_name}:")
        print(f"Return code: {e.returncode}")
        if e.stdout:
            print(f"STDOUT: {e.stdout}")
        if e.stderr:
            print(f"STDERR: {e.stderr}")
        return False

def check_dataset_exists():
    """Check if dataset is already downloaded"""
    dataset_path = Path("/teamspace/studios/this_studio/spleen/data/raw/Dataset001_Spleen")
    images_path = dataset_path / "imagesTr"
    
    if images_path.exists() and len(list(images_path.glob("*.nii*"))) > 0:
        print(f"✅ Dataset already exists at {dataset_path}")
        return True
    return False

def main():
    """Main preprocessing pipeline"""
    print("🚀 SPLEEN SEGMENTATION PREPROCESSING PIPELINE")
    print("=" * 60)
    print(f"📅 Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Create necessary directories
    os.makedirs("/teamspace/studios/this_studio/spleen/data/raw", exist_ok=True)
    os.makedirs("/teamspace/studios/this_studio/spleen/data/processed", exist_ok=True)
    os.makedirs("/teamspace/studios/this_studio/spleen/data/patches", exist_ok=True)
    os.makedirs("/teamspace/studios/this_studio/spleen/logs", exist_ok=True)
    os.makedirs("/teamspace/studios/this_studio/spleen/models", exist_ok=True)
    
    success_count = 0
    total_steps = 4
    
    # Step 1: Download dataset if needed
    if not check_dataset_exists():
        print("\n📥 Step 1/4: Downloading dataset...")
        if run_script("fetchdataset.py", "Downloading spleen dataset from Kaggle", "preprocessing"):
            success_count += 1
        else:
            print("❌ Failed to download dataset. Exiting.")
            return
    else:
        print("\n✅ Step 1/4: Dataset already exists, skipping download")
        success_count += 1
    
    # Step 2: Preprocess slices
    print("\n🔧 Step 2/4: Preprocessing slices...")
    if run_script("preprocess_slices.py", "Preprocessing slices and creating patches", "preprocessing"):
        success_count += 1
    else:
        print("❌ Failed to preprocess slices. Exiting.")
        return
    
    # Step 3: Create balanced dataset
    print("\n⚖️ Step 3/4: Creating balanced dataset...")
    if run_script("create_balanced_dataset.py", "Creating balanced dataset for training", "preprocessing"):
        success_count += 1
    else:
        print("❌ Failed to create balanced dataset. Exiting.")
        return
    
    # Step 4: Test data loading
    print("\n🧪 Step 4/4: Testing data loading...")
    if run_script("data_loader.py", "Testing balanced data loader", "utils"):
        success_count += 1
    else:
        print("❌ Data loading test failed.")
        return
    
    # Final summary
    print(f"\n{'='*60}")
    print("🎉 PREPROCESSING PIPELINE COMPLETED!")
    print(f"{'='*60}")
    print(f"✅ Successfully completed: {success_count}/{total_steps} steps")
    print(f"📅 Finished at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    if success_count == total_steps:
        print("\n🚀 Ready for training! You can now run:")
        print("   python scripts/training/training.py")
        print("   python scripts/training/quick_train.py")
        print("   python scripts/training/extreme_train.py")
    else:
        print(f"\n⚠️ Some steps failed. Please check the logs above.")

if __name__ == "__main__":
    main()
