import os, shutil
import sys
from pathlib import Path
import random

# Source directory with your material classes
src_dir = Path('data_simplified')
# Target directory for the organized dataset
trg_dir = Path('data_simplified2')

# Function to create train/validation/test split from a source directory
def organize_material_data(material_name, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15):
    # Source directory for this material
    material_src = src_dir.joinpath(material_name)
    
    # Create target directories
    train_dir = trg_dir.joinpath('train', material_name)
    val_dir = trg_dir.joinpath('validation', material_name)
    test_dir = trg_dir.joinpath('test', material_name)
    
    # Make sure all directories exist
    Path.mkdir(train_dir, parents=True, exist_ok=True)
    Path.mkdir(val_dir, parents=True, exist_ok=True)
    Path.mkdir(test_dir, parents=True, exist_ok=True)
    
    # Get all image files from the source directory
    image_files = [f for f in os.listdir(material_src) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    
    # Shuffle the files to ensure random distribution
    random.shuffle(image_files)
    
    # Calculate split indices
    total_files = len(image_files)
    train_end = int(total_files * train_ratio)
    val_end = train_end + int(total_files * val_ratio)
    
    # Split the files
    train_files = image_files[:train_end]
    val_files = image_files[train_end:val_end]
    test_files = image_files[val_end:]
    
    # Copy files to their respective directories
    for fname in train_files:
        shutil.copyfile(str(material_src.joinpath(fname)), str(train_dir.joinpath(fname)))
    
    for fname in val_files:
        shutil.copyfile(str(material_src.joinpath(fname)), str(val_dir.joinpath(fname)))
    
    for fname in test_files:
        shutil.copyfile(str(material_src.joinpath(fname)), str(test_dir.joinpath(fname)))
    
    print(f"Organized {material_name}: {len(train_files)} training, {len(val_files)} validation, {len(test_files)} test images")

# Process each material class
for material in ['Brass', 'nylon', 'Steel']:
    organize_material_data(material)

print("Data organization complete!") 