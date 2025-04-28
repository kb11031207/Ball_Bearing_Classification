import os
import shutil
from PIL import Image

def reorganize_dataset(source_path="data2", target_path="data_simplified"):
    """
    Reorganize the dataset to flatten the hierarchy and focus only on material.
    This will copy all images from subdirectories into their parent material directory.
    """
    # Create target directory if it doesn't exist
    if not os.path.exists(target_path):
        os.makedirs(target_path)

    # Check if source path exists
    if not os.path.exists(source_path):
        print(f"Source path {source_path} does not exist.")
        return
    
    # Get all material directories
    material_dirs = [d for d in os.listdir(source_path) 
                    if os.path.isdir(os.path.join(source_path, d))]
    
    for material in material_dirs:
        # Create material directory in target
        material_target = os.path.join(target_path, material)
        if not os.path.exists(material_target):
            os.makedirs(material_target)
        
        # Walk through all subdirectories
        material_source = os.path.join(source_path, material)
        for root, _, files in os.walk(material_source):
            for file in files:
                if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                    # Copy image to target directory
                    source_file = os.path.join(root, file)
                    # Create a unique filename to avoid overwriting
                    subdir = os.path.relpath(root, material_source).replace(os.path.sep, '_')
                    if subdir == '.':
                        target_file = os.path.join(material_target, file)
                    else:
                        target_file = os.path.join(material_target, f"{subdir}_{file}")
                    
                    try:
                        # Verify image is valid before copying
                        with Image.open(source_file) as img:
                            img.verify()
                        shutil.copy2(source_file, target_file)
                        print(f"Copied: {source_file} -> {target_file}")
                    except Exception as e:
                        print(f"Skipped corrupted image {source_file}: {e}")
    
    print(f"\nDataset reorganized at: {target_path}")

if __name__ == "__main__":
    reorganize_dataset() 