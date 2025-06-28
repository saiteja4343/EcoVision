"""
Lang-SAM Segmentation for Synthetic Images
Generates segmentation masks for synthetic images using language-guided SAM
"""

import os
import shutil
from PIL import Image
import numpy as np
import cv2
from lang_sam import LangSAM
import torch

def segmentation_pipeline(input_main_folder, output_main_folder, class_list, 
                         area_ratio_threshold=0.05, overlap_threshold=0.8, 
                         dedup_iou_threshold=0.7, child_area_threshold=0.02, 
                         target_size=(640, 640)):
    """
    Main segmentation pipeline using Lang-SAM for synthetic images
    
    Args:
        input_main_folder: Directory containing class subfolders with synthetic images
        output_main_folder: Output directory for processed images and labels
        class_list: List of food class names
        area_ratio_threshold: Minimum area ratio for child objects
        overlap_threshold: Threshold for detecting overlapping masks
        dedup_iou_threshold: IoU threshold for deduplication
        child_area_threshold: Minimum area threshold for child objects
        target_size: Target image size (width, height)
    """
    
    # Initialize the LangSAM model
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    model = LangSAM(sam_type="sam2.1_hiera_large", device=device)
    
    # Prepare output directories
    images_output_dir = os.path.join(output_main_folder, 'images')
    labels_output_dir = os.path.join(output_main_folder, 'labels')
    os.makedirs(images_output_dir, exist_ok=True)
    os.makedirs(labels_output_dir, exist_ok=True)
    
    # Create a mapping from class name to class index
    class_to_index = {cls_name: idx for idx, cls_name in enumerate(class_list)}
    
    def mask_iou(mask1, mask2):
        """Calculate Intersection over Union for two binary masks"""
        intersection = np.logical_and(mask1, mask2).sum()
        union = np.logical_or(mask1, mask2).sum()
        if union == 0:
            return 0
        return intersection / union
    
    # Process each class folder
    total_processed = 0
    for class_name in os.listdir(input_main_folder):
        class_folder_path = os.path.join(input_main_folder, class_name)
        if not os.path.isdir(class_folder_path):
            continue
        if class_name not in class_list:
            print(f"Skipping folder {class_name} as it is not in the class list.")
            continue
        
        # Use the folder name as the text prompt
        text_prompt = class_name
        class_processed = 0
        
        print(f"\nProcessing class: {class_name}")
        print(f"Text prompt: {text_prompt}")
        
        # Process each image in the subfolder
        for image_filename in os.listdir(class_folder_path):
            if not image_filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                continue
            
            image_path = os.path.join(class_folder_path, image_filename)
            
            try:
                image_pil = Image.open(image_path).convert('RGB')
            except Exception as e:
                print(f"Error loading image {image_filename}: {e}")
                continue
            
            # Resize image to target size (640x640) before processing
            original_width, original_height = image_pil.size
            resized_image = image_pil.resize(target_size, Image.Resampling.LANCZOS)
            
            # Run segmentation with LangSAM on the resized image
            try:
                results = model.predict([resized_image], [text_prompt])
            except Exception as e:
                print(f'Error during segmentation for {image_filename}: {e}')
                continue
            
            if len(results) == 0:
                print(f'No results for image {image_filename} in class {class_name}')
                continue
            
            result = results[0]  # Assuming single image result
            masks = result.get('masks', [])
            
            if len(masks) == 0:
                print(f'No masks detected for image {image_filename} in class {class_name}')
                continue
            
            # Convert masks to binary numpy arrays (already in 640x640 resolution)
            binary_masks = [(np.array(mask) > 0.5).astype(np.uint8) for mask in masks]
            
            # Step 1: Deduplicate masks for the same object
            keep_mask_flags = [True] * len(binary_masks)
            areas = [np.sum(mask) for mask in binary_masks]
            sorted_indices = np.argsort(areas)[::-1]  # Largest first
            
            for i in range(len(binary_masks)):
                if not keep_mask_flags[sorted_indices[i]]:
                    continue
                for j in range(i + 1, len(binary_masks)):
                    if not keep_mask_flags[sorted_indices[j]]:
                        continue
                    iou = mask_iou(binary_masks[sorted_indices[i]], binary_masks[sorted_indices[j]])
                    if iou > dedup_iou_threshold:  # High overlap indicates same object
                        keep_mask_flags[sorted_indices[j]] = False  # Discard smaller mask
            
            # Step 2: Filter out combined masks (e.g., two objects in contact)
            temp_masks = [mask for mask, keep in zip(binary_masks, keep_mask_flags) if keep]
            temp_flags = [True] * len(temp_masks)
            temp_areas = [np.sum(mask) for mask in temp_masks]
            temp_sorted_indices = np.argsort(temp_areas)[::-1]  # Largest first
            
            for i in range(len(temp_masks)):
                if not temp_flags[temp_sorted_indices[i]]:
                    continue
                overlap_count = 0
                for j in range(len(temp_masks)):
                    if i == j or not temp_flags[temp_sorted_indices[j]]:
                        continue
                    iou = mask_iou(temp_masks[temp_sorted_indices[i]], temp_masks[temp_sorted_indices[j]])
                    if iou > overlap_threshold / 2:  # Partial overlap with individual mask
                        overlap_count += 1
                if overlap_count >= 2:  # Overlaps with 2 or more masks, likely a combined mask
                    temp_flags[temp_sorted_indices[i]] = False
            
            # Update keep_mask_flags with combined mask filtering results
            final_flags = [False] * len(binary_masks)
            temp_idx = 0
            for orig_idx, keep in enumerate(keep_mask_flags):
                if keep:
                    if temp_idx < len(temp_flags) and temp_flags[temp_idx]:
                        final_flags[orig_idx] = True
                    temp_idx += 1
            final_masks = [mask for mask, keep in zip(binary_masks, final_flags) if keep]
            
            # Step 3: Enhanced Hierarchy and Area Filtering
            image_width, image_height = target_size  # Use resized dimensions (640x640)
            total_image_area = image_width * image_height
            
            # Process contours for hierarchy filtering
            all_contours = []
            all_hierarchies = []
            
            for mask_bin in final_masks:
                contours, hierarchy = cv2.findContours(mask_bin, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
                all_contours.append(contours)
                all_hierarchies.append(hierarchy[0] if hierarchy is not None and len(hierarchy) > 0 else [])
            
            # Flatten contours and hierarchy for all masks
            flat_contours = []
            flat_hierarchy = []
            contour_mask_idx = []
            
            for mask_idx, (contours, hierarchy) in enumerate(zip(all_contours, all_hierarchies)):
                for i, cnt in enumerate(contours):
                    flat_contours.append(cnt)
                    contour_mask_idx.append(mask_idx)
                    if len(hierarchy) > i:
                        flat_hierarchy.append(hierarchy[i])
                    else:
                        flat_hierarchy.append([-1, -1, -1, -1])
            
            # Filter contours: prioritize top-level contours and exclude tiny child contours
            filtered_contours = []
            processed_mask_indices = set()
            
            for i, (cnt, hier, mask_idx) in enumerate(zip(flat_contours, flat_hierarchy, contour_mask_idx)):
                if mask_idx in processed_mask_indices:
                    continue
                    
                parent_idx = hier[3]  # Parent index in hierarchy
                area = cv2.contourArea(cnt)
                
                if parent_idx == -1:  # Top-level contour
                    if area / total_image_area > 0.01:  # At least 1% of image area
                        filtered_contours.append(cnt)
                        processed_mask_indices.add(mask_idx)
                elif parent_idx >= 0 and parent_idx < len(flat_contours):
                    parent_area = cv2.contourArea(flat_contours[parent_idx])
                    if parent_area > 0:
                        area_ratio = area / parent_area
                        area_to_image = area / total_image_area
                        # Only keep child if it's significant relative to parent and image
                        if area_ratio > area_ratio_threshold and area_to_image > child_area_threshold:
                            filtered_contours.append(cnt)
                            processed_mask_indices.add(mask_idx)
            
            # Prepare annotation lines
            annotation_lines = []
            class_index = class_to_index[class_name]
            
            # For each filtered contour, convert to normalized polygon points
            for contour in filtered_contours:
                if len(contour) < 6:  # Need at least 3 points (x,y) pairs for a valid polygon
                    continue
                    
                segmentation = []
                for point in contour.squeeze():
                    x_norm = point[0] / image_width  # Already in 640x640 space
                    y_norm = point[1] / image_height  # Already in 640x640 space
                    segmentation.append(x_norm)
                    segmentation.append(y_norm)
                
                # Format line: class_index followed by segmentation coordinates
                line = f'{class_index} ' + ' '.join(f'{coord:.6f}' for coord in segmentation)
                annotation_lines.append(line)
            
            # Only save if we have valid annotations
            if annotation_lines:
                # Save annotation to .txt file in labels folder
                base_filename = os.path.splitext(image_filename)[0]
                annotation_path = os.path.join(labels_output_dir, f'{base_filename}.txt')
                with open(annotation_path, 'w') as f:
                    f.write('\n'.join(annotation_lines))
                
                # Save resized image to output images folder
                resized_image.save(os.path.join(images_output_dir, image_filename))
                
                class_processed += 1
                total_processed += 1
                
                if class_processed % 50 == 0:
                    print(f'  Processed {class_processed} images for {class_name}')
            else:
                print(f'  No valid annotations for {image_filename}, skipping')
        
        print(f'Completed {class_name}: {class_processed} images processed')
    
    print(f'\nPipeline processing complete. Total images processed: {total_processed}')

def main():
    """Main execution function"""
    # Define your folder paths and class list
    input_folder = "path/to/gen/images"
    output_folder = "path/to/output/seg/images"
    
    # Class list matching your food categories
    classes = [
        'Apple', 'Asparagus', 'Avocado', 'Banana', 'Beans', 'Beetroot', 'Bell peppers', 
        'Blueberries', 'Broccoli', 'Brussel sprouts', 'Cabbage', 'Carrot', 'Cauliflower', 
        'Celeries', 'Cucumber', 'Eggplant', 'Galia melons', 'Garlic', 'Ginger', 
        'Grapefruits', 'Grapes', 'Kaki', 'Kiwi', 'Lemon', 'Lettuce', 'Mango', 
        'Cantaloupes', 'Mushrooms', 'Onion', 'Orange', 'Passion fruits', 'Peach', 
        'Pear', 'Peas', 'Pineapple', 'Plums', 'Pomegranate', 'Potato', 'Pumpkin', 
        'Radish', 'Raspberries', 'Strawberry', 'Tomato', 'Watermelon', 'Zucchini'
    ]
    
    # Create output directory
    os.makedirs(output_folder, exist_ok=True)
    
    # Run the pipeline
    segmentation_pipeline(input_folder, output_folder, classes)

if __name__ == "__main__":
    main()
