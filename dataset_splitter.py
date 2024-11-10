import os
import shutil
import random
from pathlib import Path

# Set paths and split ratios
original_dataset_dir = 'Dataset'  # Original dataset path
output_dir = './split'  # Output path for train/val/test split
train_ratio = 0.8
val_ratio = 0.1
test_ratio = 0.1

# Create directories for train, val, and test sets
for split in ['train', 'val', 'test']:
    for class_dir in os.listdir(original_dataset_dir):
        Path(f"{output_dir}/{split}/{class_dir}").mkdir(parents=True, exist_ok=True)

# Split images into train, val, and test
for class_dir in os.listdir(original_dataset_dir):
    class_path = os.path.join(original_dataset_dir, class_dir)
    if os.path.isdir(class_path):
        images = os.listdir(class_path)
        random.shuffle(images)
        
        # Calculate split sizes
        train_split = int(len(images) * train_ratio)
        val_split = int(len(images) * val_ratio)
        
        # Get subsets of images
        train_images = images[:train_split]
        val_images = images[train_split:train_split + val_split]
        test_images = images[train_split + val_split:]
        
        # Move images to corresponding folders
        for img in train_images:
            shutil.copy(os.path.join(class_path, img), os.path.join(output_dir, 'train', class_dir, img))
        
        for img in val_images:
            shutil.copy(os.path.join(class_path, img), os.path.join(output_dir, 'val', class_dir, img))
        
        for img in test_images:
            shutil.copy(os.path.join(class_path, img), os.path.join(output_dir, 'test', class_dir, img))

print("Dataset split completed successfully!")
