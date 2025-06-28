"""
Dataset Statistics Script
Analyzes and reports comprehensive statistics about the processed dataset
"""

import os
import json
from collections import Counter, defaultdict
from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd

def count_files_in_directory(directory):
    """Count files by extension in a directory"""
    if not os.path.exists(directory):
        return {}
    
    file_counts = defaultdict(int)
    for file in os.listdir(directory):
        ext = os.path.splitext(file)[1].lower()
        file_counts[ext] += 1
    
    return dict(file_counts)

def analyze_class_distribution(labels_dir, class_names):
    """Analyze class distribution in a labels directory"""
    class_counts = Counter()
    
    if not os.path.exists(labels_dir):
        return class_counts
    
    for label_file in os.listdir(labels_dir):
        if not label_file.endswith('.txt'):
            continue
            
        file_path = os.path.join(labels_dir, label_file)
        try:
            with open(file_path, 'r') as f:
                class_ids = [int(line.split()[0]) for line in f if line.strip()]
            
            # Count primary class (most common in file)
            if class_ids:
                primary_class_id = max(set(class_ids), key=class_ids.count)
                if 0 <= primary_class_id < len(class_names):
                    class_counts[class_names[primary_class_id]] += 1
        except Exception as e:
            print(f"Error reading {label_file}: {e}")
    
    return class_counts

def generate_dataset_statistics(dataset_root):
    """Generate comprehensive dataset statistics"""
    
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
    
    # Dataset splits to analyze
    splits = ['train', 'test', 'valid']
    
    # Statistics container
    stats = {
        'dataset_root': dataset_root,
        'total_classes': len(class_names),
        'splits': {}
    }
    
    print("="*60)
    print("DATASET STATISTICS ANALYSIS")
    print("="*60)
    print(f"Dataset root: {dataset_root}")
    print(f"Total classes: {len(class_names)}")
    
    # Analyze each split
    for split in splits:
        split_dir = os.path.join(dataset_root, split)
        images_dir = os.path.join(split_dir, 'images')
        labels_dir = os.path.join(split_dir, 'labels')
        
        print(f"\n{split.upper()} SET:")
        print("-" * 20)
        
        if not os.path.exists(split_dir):
            print(f"  Directory not found: {split_dir}")
            continue
        
        # Count files
        image_counts = count_files_in_directory(images_dir)
        label_counts = count_files_in_directory(labels_dir)
        
        total_images = sum(image_counts.values())
        total_labels = label_counts.get('.txt', 0)
        
        print(f"  Images: {total_images}")
        print(f"  Labels: {total_labels}")
        
        # Check for mismatches
        if total_images != total_labels:
            print(f"  ⚠️  WARNING: Image/label count mismatch!")
        
        # Analyze class distribution
        class_distribution = analyze_class_distribution(labels_dir, class_names)
        
        # Calculate statistics
        non_empty_classes = len([c for c in class_distribution.values() if c > 0])
        min_images = min(class_distribution.values()) if class_distribution else 0
        max_images = max(class_distribution.values()) if class_distribution else 0
        avg_images = total_labels / non_empty_classes if non_empty_classes > 0 else 0
        
        print(f"  Classes with data: {non_empty_classes}/{len(class_names)}")
        print(f"  Min images per class: {min_images}")
        print(f"  Max images per class: {max_images}")
        print(f"  Avg images per class: {avg_images:.1f}")
        
        # Store statistics
        stats['splits'][split] = {
            'total_images': total_images,
            'total_labels': total_labels,
            'image_extensions': image_counts,
            'classes_with_data': non_empty_classes,
            'min_images_per_class': min_images,
            'max_images_per_class': max_images,
            'avg_images_per_class': avg_images,
            'class_distribution': dict(class_distribution)
        }
        
        # Print top and bottom classes
        if class_distribution:
            sorted_classes = sorted(class_distribution.items(), key=lambda x: x[1], reverse=True)
            
            print(f"  Top 5 classes:")
            for class_name, count in sorted_classes[:5]:
                print(f"    {class_name}: {count}")
            
            print(f"  Bottom 5 classes:")
            for class_name, count in sorted_classes[-5:]:
                if count > 0:
                    print(f"    {class_name}: {count}")
    
    # Overall statistics
    print(f"\n" + "="*60)
    print("OVERALL DATASET SUMMARY")
    print("="*60)
    
    total_images_all = sum(stats['splits'][split]['total_images'] for split in splits if split in stats['splits'])
    total_labels_all = sum(stats['splits'][split]['total_labels'] for split in splits if split in stats['splits'])
    
    print(f"Total images across all splits: {total_images_all}")
    print(f"Total labels across all splits: {total_labels_all}")
    
    # Calculate split ratios
    if total_images_all > 0:
        print(f"\nSplit ratios:")
        for split in splits:
            if split in stats['splits']:
                ratio = stats['splits'][split]['total_images'] / total_images_all
                print(f"  {split}: {ratio:.1%}")
    
    # Combined class distribution
    combined_distribution = Counter()
    for split in splits:
        if split in stats['splits']:
            for class_name, count in stats['splits'][split]['class_distribution'].items():
                combined_distribution[class_name] += count
    
    print(f"\nCombined class distribution:")
    sorted_combined = sorted(combined_distribution.items(), key=lambda x: x[1], reverse=True)
    
    for class_name, count in sorted_combined:
        if count > 0:
            print(f"  {class_name}: {count}")
    
    # Classes needing attention
    low_data_classes = [cls for cls, count in combined_distribution.items() if count < 100]
    if low_data_classes:
        print(f"\n⚠️  Classes with < 100 images:")
        for cls in low_data_classes:
            print(f"  {cls}: {combined_distribution[cls]}")
    
    # Store overall statistics
    stats['total_images'] = total_images_all
    stats['total_labels'] = total_labels_all
    stats['combined_class_distribution'] = dict(combined_distribution)
    stats['classes_needing_attention'] = low_data_classes
    
    return stats

def save_statistics_report(stats, output_file):
    """Save statistics to JSON file"""
    with open(output_file, 'w') as f:
        json.dump(stats, f, indent=2)
    print(f"\nStatistics saved to: {output_file}")

def create_visualization(stats, output_dir):
    """Create visualization plots for the dataset"""
    import matplotlib.pyplot as plt
    
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Class distribution plot
    combined_dist = stats['combined_class_distribution']
    classes = list(combined_dist.keys())
    counts = list(combined_dist.values())
    
    plt.figure(figsize=(20, 8))
    plt.bar(range(len(classes)), counts)
    plt.xlabel('Food Classes')
    plt.ylabel('Number of Images')
    plt.title('Combined Class Distribution Across All Splits')
    plt.xticks(range(len(classes)), classes, rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'class_distribution.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. Split distribution plot
    splits = ['train', 'test', 'valid']
    split_counts = [stats['splits'][split]['total_images'] for split in splits if split in stats['splits']]
    split_labels = [split.title() for split in splits if split in stats['splits']]
    
    plt.figure(figsize=(8, 6))
    plt.pie(split_counts, labels=split_labels, autopct='%1.1f%%', startangle=90)
    plt.title('Dataset Split Distribution')
    plt.axis('equal')
    plt.savefig(os.path.join(output_dir, 'split_distribution.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Visualization plots saved to: {output_dir}")

def main():
    """Main execution function"""
    # Configuration
    dataset_root = '/path/to/dataset'  # Update with your dataset root path
    output_dir = '/path/to/dataset_analysis'
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate statistics
    stats = generate_dataset_statistics(dataset_root)
    
    # Save report
    report_file = os.path.join(output_dir, 'dataset_statistics.json')
    save_statistics_report(stats, report_file)
    
    # Create visualizations
    viz_dir = os.path.join(output_dir, 'visualizations')
    create_visualization(stats, viz_dir)
    
    print(f"\n✅ Dataset analysis complete!")
    print(f"📊 Reports saved to: {output_dir}")

if __name__ == "__main__":
    main()
