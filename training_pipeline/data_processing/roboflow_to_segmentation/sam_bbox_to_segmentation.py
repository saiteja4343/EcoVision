"""
SAM 2.1 Bounding Box to Segmentation Conversion
Converts YOLO bounding box annotations to segmentation masks using SAM 2.1
"""

import os
import cv2
import numpy as np
from ultralytics import SAM
from pathlib import Path

def main():
    # Define paths
    main_dir = Path("path/to/bb/dataset")
    output_dir = Path("path/to/output/seg/dataset")

    input_images_dir = main_dir / "images"
    input_labels_dir = main_dir / "labels"
    output_images_dir = output_dir / "images"
    output_labels_dir = output_dir / "labels"

    # Create output directories
    output_images_dir.mkdir(parents=True, exist_ok=True)
    output_labels_dir.mkdir(parents=True, exist_ok=True)

    # Initialize SAM 2.1 Large model
    print("Loading SAM 2.1 model...")
    model = SAM("sam2.1_l.pt")

    # Process each image
    for label_file in input_labels_dir.glob("*.txt"):
        process_image(label_file, input_images_dir, output_images_dir, output_labels_dir, model)

    print("SAM conversion complete!")

def yolo_to_absolute_bbox(bbox, img_width, img_height):
    """Convert YOLO bbox to SAM prompt format (absolute pixel coordinates)"""
    center_x, center_y, width, height = bbox
    center_x_abs = center_x * img_width
    center_y_abs = center_y * img_height
    width_abs = width * img_width
    height_abs = height * img_height
    x1 = center_x_abs - (width_abs / 2)
    y1 = center_y_abs - (height_abs / 2)
    x2 = center_x_abs + (width_abs / 2)
    y2 = center_y_abs + (height_abs / 2)
    return [x1, y1, x2, y2]

def simplify_contour(contour, epsilon_factor=0.001):
    """Simplify contour while preserving details"""
    perimeter = cv2.arcLength(contour, True)
    epsilon = epsilon_factor * perimeter
    approx = cv2.approxPolyDP(contour, epsilon, True)
    return approx

def process_image(label_file, input_images_dir, output_images_dir, output_labels_dir, model):
    """Process a single image through SAM conversion"""
    # Get corresponding image file
    image_name = label_file.stem + ".jpg"
    image_path = input_images_dir / image_name
    
    if not image_path.exists():
        print(f"Image {image_path} not found, skipping...")
        return
    
    # Read image to get dimensions
    img = cv2.imread(str(image_path))
    img_height, img_width = img.shape[:2]
    
    # Read bounding boxes from label file
    bboxes = []
    class_ids = []
    with open(label_file, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 5:
                continue
            class_id = int(parts[0])
            bbox = list(map(float, parts[1:5]))
            class_ids.append(class_id)
            bboxes.append(bbox)
    
    if not bboxes:
        print(f"No bounding boxes found in {label_file}, skipping...")
        return
    
    # Convert YOLO bboxes to absolute coordinates for SAM
    abs_bboxes = [yolo_to_absolute_bbox(bbox, img_width, img_height) for bbox in bboxes]
    
    # Run SAM model with bounding box prompts
    try:
        results = model.predict(image_path, bboxes=abs_bboxes, conf=0.25, iou=0.7)
    except Exception as e:
        print(f"Error processing {image_name} with SAM: {e}")
        return
    
    # Extract segmentation masks and convert to YOLO segmentation format
    segmentation_lines = []
    for i, result in enumerate(results):
        if hasattr(result, 'masks') and result.masks is not None and i < len(class_ids):
            masks = result.masks.data
            if len(masks) == 0:
                # Fallback: Convert bbox to rectangular polygon
                segmentation_lines.append(bbox_to_polygon(abs_bboxes[i], class_ids[i], img_width, img_height))
                continue
            
            # Process each mask
            for mask_idx, mask in enumerate(masks):
                if mask_idx >= len(class_ids):
                    break
                mask_np = mask.cpu().numpy().astype(np.uint8) * 255
                
                # Clean up mask with morphological operations
                kernel = np.ones((5, 5), np.uint8)
                mask_np = cv2.morphologyEx(mask_np, cv2.MORPH_CLOSE, kernel)
                contours, _ = cv2.findContours(mask_np, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
                
                if contours:
                    # Get largest contour
                    contour = max(contours, key=cv2.contourArea)
                    if len(contour) < 3:
                        segmentation_lines.append(bbox_to_polygon(abs_bboxes[i], class_ids[i], img_width, img_height))
                        continue
                    
                    # Simplify and normalize contour
                    contour = simplify_contour(contour, epsilon_factor=0.001)
                    contour_normalized = contour.reshape(-1, 2) / np.array([img_width, img_height])
                    contour_str = " ".join([f"{x:.6f}" for x in contour_normalized.flatten()])
                    segmentation_lines.append(f"{class_ids[i]} {contour_str}")
                else:
                    segmentation_lines.append(bbox_to_polygon(abs_bboxes[i], class_ids[i], img_width, img_height))
        else:
            segmentation_lines.append(bbox_to_polygon(abs_bboxes[i], class_ids[i], img_width, img_height))
    
    # Save segmentation labels
    output_label_path = output_labels_dir / label_file.name
    with open(output_label_path, 'w') as f:
        f.write("\n".join(segmentation_lines))
    
    # Copy image to output directory
    output_image_path = output_images_dir / image_name
    cv2.imwrite(str(output_image_path), img)
    print(f"Processed {image_name}")

def bbox_to_polygon(abs_bbox, class_id, img_width, img_height):
    """Convert bbox to polygon as fallback"""
    x1, y1, x2, y2 = abs_bbox
    rect_points = np.array([[x1, y1], [x2, y1], [x2, y2], [x1, y2]])
    rect_normalized = rect_points / np.array([img_width, img_height])
    contour_str = " ".join([f"{x:.6f}" for x in rect_normalized.flatten()])
    return f"{class_id} {contour_str}"

if __name__ == "__main__":
    main()
