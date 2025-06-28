"""
Dataset Balancing Script
Limits each class to maximum 500 images and removes excess files
"""

import os
import random
from collections import Counter
import shutil

# Food class names matching the model
CLASSES = [
    'Apple', 'Asparagus', 'Avocado', 'Banana', 'Beans', 'Beetroot', 'Bell_pepper', 
    'Blueberry', 'Broccoli', 'Brussel_sprouts', 'Cabbage', 'Carrot', 'Cauliflower', 
    'Celery', 'Cucumber', 'Eggplant', 'Galia', 'Garlic', 'Ginger', 'Grapefruit', 
    'Grapes', 'Kaki', 'Kiwi', 'Lemon', 'Lettuce', 'Mango', 'Melon', 'Mushroom', 
    'Onion', 'Orange', 'Passion_fruit', 'Peach', 'Pear', 'Peas', 'Pineapple', 
    'Plum', 'Pomegranate', 'Potato', 'Pumpkin', 'Radish', 'Raspberry', 'Strawberry', 
    'Tomato', 'Watermelon', 'Zucchini'
]

def get_most_common_class(file_path):
    """Get the most common class in a label file"""
    try:
        with open(file_path, 'r') as file:
            class_ids = [int(line.split()[0]) for line in file if line.strip()]
        if not class_ids:
            return None
        return max(set(class_ids), key=class_ids.count)
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
        return None

def balance_dataset(labels_folder, images_folder, max_images_per_class=500, backup=True):
    """
    Balance dataset by limiting each class to max_images_per_class
    
    Args:
        labels_folder: Path to labels directory
        images_folder: Path to images directory  
        max_images_per_class: Maximum images to keep per class
        backup: Whether to create backup before deletion
    """
    
    print(f"Balancing dataset in {labels_folder}")
    print(f"Maximum images per class: {max_images_per_class}")
    
    # Create backup if requested
    if backup:
        backup_dir = f"{labels_folder}_backup"
        if not os.path.exists(backup_dir):
            print(f"Creating backup at {backup_dir}")
            shutil.copytree(labels_folder, backup_dir)
    
    # Dictionary to store files for each class
    class_files = {class_name: [] for class_name in CLASSES}
    
    # Scan all label files and categorize by primary class
    print("Scanning label files...")
    total_files = 0
    for filename in os.listdir(labels_folder):
        if filename.endswith('.txt'):
            file_path = os.path.join(labels_folder, filename)
            most_common_class = get_most_common_class(file_path)
            
            if most_common_class is not None and 0 <= most_common_class < len(CLASSES):
                class_files[CLASSES[most_common_class]].append(filename)
                total_files += 1
    
    print(f"Total files found: {total_files}")
    
    # Print initial distribution
    print("\nInitial class distribution:")
    for class_name, files in class_files.items():
        if len(files) > 0:
            print(f"  {class_name}: {len(files)} files")
    
    # Randomly select files to keep for each class
    files_to_keep = set()
    files_to_remove = set()
    
    for class_name, files in class_files.items():
        if len(files) > max_images_per_class:
            # Randomly sample files to keep
            selected_files = random.sample(files, max_images_per_class)
            files_to_keep.update(selected_files)
            
            # Mark remaining files for removal
            files_to_remove.update(set(files) - set(selected_files))
            
            print(f"  {class_name}: keeping {len(selected_files)}, removing {len(files) - len(selected_files)}")
        else:
            # Keep all files if under the limit
            files_to_keep.update(files)
            print(f"  {class_name}: keeping all {len(files)} files")
    
    # Remove excess files
    print(f"\nRemoving {len(files_to_remove)} excess files...")
    removed_count = 0
    
    for filename in files_to_remove:
        # Remove label file
        label_path = os.path.join(labels_folder, filename)
        if os.path.exists(label_path):
            os.remove(label_path)
            removed_count += 1
        
        # Remove corresponding image file
        base_name = os.path.splitext(filename)[0]
        for ext in ['.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG']:
            image_path = os.path.join(images_folder, base_name + ext)
            if os.path.exists(image_path):
                os.remove(image_path)
                break
    
    print(f"Removed {removed_count} label files and corresponding images")
    
    # Count final distribution
    final_class_counts = Counter()
    for filename in os.listdir(labels_folder):
        if filename.endswith('.txt'):
            file_path = os.path.join(labels_folder, filename)
            most_common_class = get_most_common_class(file_path)
            if most_common_class is not None and 0 <= most_common_class < len(CLASSES):
                final_class_counts[CLASSES[most_common_class]] += 1
    
    # Print final results
    print("\nFinal class distribution:")
    total_kept = 0
    for class_name in CLASSES:
        count = final_class_counts[class_name]
        if count > 0:
            print(f"  {class_name}: {count} files")
            total_kept += count
    
    print(f"\nTotal files kept: {total_kept}")
    print("Dataset balancing complete!")
    
    return final_class_counts

def main():
    """Main execution function"""
    # Configuration
    dataset_dir = 'path/to/seg/dataset'
    labels_folder = os.path.join(dataset_dir, 'labels')
    images_folder = os.path.join(dataset_dir, 'images')
    
    # Validate directories exist
    if not os.path.exists(labels_folder):
        print(f"Error: Labels folder not found at {labels_folder}")
        return
    
    if not os.path.exists(images_folder):
        print(f"Error: Images folder not found at {images_folder}")
        return
    
    # Set random seed for reproducibility
    random.seed(42)
    
    # Run balancing
    final_counts = balance_dataset(
        labels_folder=labels_folder,
        images_folder=images_folder,
        max_images_per_class=500,
        backup=True
    )
    
    # Generate statistics
    print("\n" + "="*50)
    print("DATASET BALANCING SUMMARY")
    print("="*50)
    
    non_empty_classes = sum(1 for count in final_counts.values() if count > 0)
    total_images = sum(final_counts.values())
    avg_images_per_class = total_images / non_empty_classes if non_empty_classes > 0 else 0
    
    print(f"Classes with data: {non_empty_classes}/45")
    print(f"Total images: {total_images}")
    print(f"Average images per class: {avg_images_per_class:.1f}")
    print(f"Target images per class: 500")
    
    # Classes needing more data
    classes_needing_data = [cls for cls, count in final_counts.items() if count < 400]
    if classes_needing_data:
        print(f"\nClasses needing more data (< 400 images):")
        for cls in classes_needing_data:
            print(f"  {cls}: {final_counts[cls]} images")

if __name__ == "__main__":
    main()
