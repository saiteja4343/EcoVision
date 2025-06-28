"""
Train/Test/Validation Split Script
Splits balanced dataset into train (70%), test (20%), and validation (10%) sets
"""

import os
import shutil
import random
from collections import defaultdict
from pathlib import Path

def get_primary_class(label_file_path, class_names):
    """Get the primary (most common) class from a label file"""
    try:
        with open(label_file_path, 'r') as f:
            class_ids = [int(line.split()[0]) for line in f if line.strip()]
        
        if not class_ids:
            return None
            
        # Get most common class
        most_common_id = max(set(class_ids), key=class_ids.count)
        return class_names[most_common_id] if 0 <= most_common_id < len(class_names) else None
    except Exception as e:
        print(f"Error reading {label_file_path}: {e}")
        return None

def create_train_test_split(source_dir, output_dir, train_ratio=0.7, test_ratio=0.2, val_ratio=0.1):
    """
    Create train/test/validation split maintaining class distribution
    
    Args:
        source_dir: Directory containing images/ and labels/ folders
        output_dir: Output directory for split dataset
        train_ratio: Fraction for training set (default: 0.7)
        test_ratio: Fraction for test set (default: 0.2)  
        val_ratio: Fraction for validation set (default: 0.1)
    """
    
    # Validate ratios
    if abs(train_ratio + test_ratio + val_ratio - 1.0) > 0.001:
        raise ValueError("Ratios must sum to 1.0")
    
    # Class names
    class_names = [
        'Apple', 'Asparagus', 'Avocado', 'Banana', 'Beans', 'Beetroot', 'Bell_pepper', 
        'Blueberry', 'Broccoli', 'Brussel_sprouts', 'Cabbage', 'Carrot', 'Cauliflower', 
        'Celery', 'Cucumber', 'Eggplant', 'Galia', 'Garlic', 'Ginger', 'Grapefruit', 
        'Grapes', 'Kaki', 'Kiwi', 'Lemon', 'Lettuce', 'Mango', 'Melon', 'Mushroom', 
        'Onion', 'Orange', 'Passion_fruit', 'Peach', 'Pear', 'Peas', 'Pineapple', 
        'Plum', 'Pomegranate', 'Potato', 'Pumpkin', 'Radish', 'Raspberry', 'Strawberry', 
        'Tomato', 'Watermelon', 'Zucchini'
    ]
    
    # Setup paths
    source_images = os.path.join(source_dir, 'images')
    source_labels = os.path.join(source_dir, 'labels')
    
    # Validate source directories
    if not os.path.exists(source_images) or not os.path.exists(source_labels):
        raise ValueError(f"Source directories not found: {source_images}, {source_labels}")
    
    # Create output directory structure
    splits = ['train', 'test', 'valid']
    for split in splits:
        os.makedirs(os.path.join(output_dir, split, 'images'), exist_ok=True)
        os.makedirs(os.path.join(output_dir, split, 'labels'), exist_ok=True)
    
    print(f"Creating train/test/validation split")
    print(f"Ratios - Train: {train_ratio}, Test: {test_ratio}, Val: {val_ratio}")
    print(f"Source: {source_dir}")
    print(f"Output: {output_dir}")
    
    # Group files by class
    class_files = defaultdict(list)
    
    # Scan all label files
    for label_file in os.listdir(source_labels):
        if not label_file.endswith('.txt'):
            continue
            
        label_path = os.path.join(source_labels, label_file)
        primary_class = get_primary_class(label_path, class_names)
        
        if primary_class:
            # Check if corresponding image exists
            base_name = os.path.splitext(label_file)[0]
            image_found = False
            
            for ext in ['.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG']:
                image_path = os.path.join(source_images, base_name + ext)
                if os.path.exists(image_path):
                    class_files[primary_class].append((label_file, base_name + ext))
                    image_found = True
                    break
            
            if not image_found:
                print(f"Warning: No image found for {label_file}")
    
    # Print class distribution before splitting
    print(f"\nClass distribution before splitting:")
    total_files = 0
    for class_name, files in class_files.items():
        print(f"  {class_name}: {len(files)} files")
        total_files += len(files)
    print(f"Total files: {total_files}")
    
    # Perform stratified split for each class
    split_stats = {split: defaultdict(int) for split in splits}
    
    for class_name, files in class_files.items():
        if not files:
            continue
            
        # Shuffle files for this class
        random.shuffle(files)
        
        # Calculate split indices
        n_files = len(files)
        n_train = int(n_files * train_ratio)
        n_test = int(n_files * test_ratio)
        n_val = n_files - n_train - n_test  # Remainder goes to validation
        
        # Split files
        train_files = files[:n_train]
        test_files = files[n_train:n_train + n_test]
        val_files = files[n_train + n_test:]
        
        # Copy files to respective directories
        split_data = [
            ('train', train_files),
            ('test', test_files),
            ('valid', val_files)
        ]
        
        for split_name, file_list in split_data:
            for label_file, image_file in file_list:
                # Copy label file
                src_label = os.path.join(source_labels, label_file)
                dst_label = os.path.join(output_dir, split_name, 'labels', label_file)
                shutil.copy2(src_label, dst_label)
                
                # Copy image file
                src_image = os.path.join(source_images, image_file)
                dst_image = os.path.join(output_dir, split_name, 'images', image_file)
                shutil.copy2(src_image, dst_image)
                
                split_stats[split_name][class_name] += 1
    
    # Print split statistics
    print(f"\n" + "="*60)
    print("TRAIN/TEST/VALIDATION SPLIT RESULTS")
    print("="*60)
    
    for split_name in splits:
        total_split = sum(split_stats[split_name].values())
        percentage = (total_split / total_files * 100) if total_files > 0 else 0
        print(f"\n{split_name.upper()} SET: {total_split} files ({percentage:.1f}%)")
        
        # Print class distribution for this split
        for class_name in class_names:
            count = split_stats[split_name][class_name]
            if count > 0:
                print(f"  {class_name}: {count}")
    
    # Verify file counts
    print(f"\nVerification:")
    for split in splits:
        images_count = len(os.listdir(os.path.join(output_dir, split, 'images')))
        labels_count = len(os.listdir(os.path.join(output_dir, split, 'labels')))
        print(f"  {split}: {images_count} images, {labels_count} labels")
        
        if images_count != labels_count:
            print(f"  WARNING: Mismatch in {split} set!")
    
    print("\nDataset split completed successfully!")

def main():
    """Main execution function"""
    # Configuration
    source_directory = 'path/to/seg/dataset'  # After balancing
    output_directory = 'path/to/output/split/dataset'  # Where to save the split dataset
    
    # Set random seed for reproducibility
    random.seed(42)
    
    # Create the split
    create_train_test_split(
        source_dir=source_directory,
        output_dir=output_directory,
        train_ratio=0.7,
        test_ratio=0.2,
        val_ratio=0.1
    )

if __name__ == "__main__":
    main()
