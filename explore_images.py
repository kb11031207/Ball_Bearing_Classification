import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import random

def explore_dataset(dataset_path="data2", samples_per_class=5):
    """Visualize sample images from each class to understand the dataset."""
    
    # Get all class directories
    class_dirs = [d for d in os.listdir(dataset_path) 
                 if os.path.isdir(os.path.join(dataset_path, d))]
    
    # Set up the plot
    fig, axes = plt.subplots(len(class_dirs), samples_per_class, 
                            figsize=(samples_per_class*3, len(class_dirs)*3))
    
    # For each class
    for i, class_name in enumerate(class_dirs):
        class_path = os.path.join(dataset_path, class_name)
        
        # Get all image files from this class (including subdirectories)
        image_files = []
        for root, _, files in os.walk(class_path):
            for file in files:
                if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                    image_files.append(os.path.join(root, file))
        
        # Sample random images
        if len(image_files) >= samples_per_class:
            sampled_files = random.sample(image_files, samples_per_class)
        else:
            sampled_files = image_files
            print(f"Warning: Only {len(image_files)} images found for class {class_name}")
        
        # Display each sampled image
        for j, img_path in enumerate(sampled_files):
            img = Image.open(img_path)
            axes[i, j].imshow(img)
            axes[i, j].set_title(f"{class_name}\n{os.path.basename(img_path)}")
            axes[i, j].axis('off')
    
    plt.tight_layout()
    plt.savefig('sample_images.png')
    plt.show()

if __name__ == "__main__":
    explore_dataset() 