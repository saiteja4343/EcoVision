# src/logic/weight_estimation.py
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw
from transformers import pipeline
import torch
import cv2
import streamlit as st

# Class mapping exactly as you provided
CLASS_NAMES = [
    'Apple', 'Asparagus', 'Avocado', 'Banana', 'Beans', 'Beetroot', 'Bell_pepper', 'Blueberry', 'Broccoli',
    'Brussel_sprouts', 'Cabbage', 'Carrot', 'Cauliflower', 'Celery', 'Cucumber', 'Eggplant', 'Galia', 'Garlic',
    'Ginger', 'Grapefruit', 'Grapes', 'Kaki', 'Kiwi', 'Lemon', 'Lettuce', 'Mango', 'Melon', 'Mushroom', 'Onion',
    'Orange', 'Passion_fruit', 'Peach', 'Pear', 'Peas', 'Pineapple', 'Plums', 'Pomegranate', 'Potato', 'Pumpkin',
    'Radish', 'Raspberry', 'Strawberry', 'Tomato', 'Watermelon', 'Zucchini'
]

class_to_id = {name: idx for idx, name in enumerate(CLASS_NAMES)}
id_to_class = {v: k for k, v in class_to_id.items()}

@st.cache_resource
def load_depth_pipeline():
    """Load depth estimation pipeline (cached for performance)"""
    try:
        return pipeline(
            task="depth-estimation", 
            model="depth-anything/Depth-Anything-V2-Small-hf"
        )
    except Exception as e:
        st.error(f"Failed to load depth estimation model: {str(e)}")
        return None

def parse_yolo_mask_from_results(yolo_result, img_width, img_height):
    """Parse YOLO segmentation results into objects list - your exact approach"""
    objects = []
    
    if not hasattr(yolo_result, 'masks') or yolo_result.masks is None:
        return objects
    
    for i, (box, mask) in enumerate(zip(yolo_result.boxes, yolo_result.masks.data)):
        class_id = int(box.cls)
        
        # Convert mask to polygon coordinates
        mask_np = mask.cpu().numpy().astype(np.uint8)
        
        # Resize mask to match image dimensions if needed
        if mask_np.shape != (img_height, img_width):
            mask_np = cv2.resize(mask_np, (img_width, img_height), interpolation=cv2.INTER_NEAREST)
        
        # Find contours
        contours, _ = cv2.findContours(mask_np, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if contours:
            # Get largest contour (main object)
            largest_contour = max(contours, key=cv2.contourArea)
            
            # Convert contour to polygon coordinates - exact format you used
            polygon = [(int(point[0][0]), int(point[0][1])) for point in largest_contour]
            
            if len(polygon) > 2:  # Valid polygon
                objects.append({
                    'class_id': class_id,
                    'polygon': polygon
                })
    
    return objects

def polygon_area(polygon):
    """Compute polygon area using shoelace formula - your exact implementation"""
    if len(polygon) < 3:
        return 0
    
    x, y = zip(*polygon)
    return 0.5 * np.abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))

def estimate_scale(depth_map, mask, polygon):
    """Estimate scale using depth map - your exact implementation"""
    # Mask out the object in the depth map
    mask_img = Image.new('L', (depth_map.shape[1], depth_map.shape[0]), 0)
    ImageDraw.Draw(mask_img).polygon(polygon, outline=1, fill=1)
    mask_arr = np.array(mask_img)
    masked_depth = depth_map * mask_arr
    # Use median depth of the object
    valid_depths = masked_depth[mask_arr > 0]
    if len(valid_depths) > 0:
        object_depth = np.median(valid_depths)
        return object_depth  # in meters (if depth map is metric)
    else:
        return 1.0  # Default depth

def estimate_real_area(polygon, depth, img_width, img_height, fov=60):
    """Estimate real-world area - your exact implementation"""
    # FOV is camera horizontal field of view in degrees (adjust as needed)
    # Calculate pixel size at given depth
    fov_rad = np.deg2rad(fov)
    sensor_width = 2 * depth * np.tan(fov_rad / 2)
    pixel_size = sensor_width / img_width  # meters per pixel
    area_pixels = polygon_area(polygon)
    area_m2 = area_pixels * (pixel_size ** 2)
    return area_m2

def estimate_volume(area_m2, depth, thickness_ratio=0.8):
    """Estimate volume - your exact implementation"""
    # Assume roughly spheroid/ellipsoid: volume = area * thickness
    # thickness_ratio is a heuristic (typically 0.6-1.0 for roundish fruits)
    thickness = np.sqrt(area_m2) * thickness_ratio
    volume = area_m2 * thickness  # m^3
    return volume

def estimate_weights_from_yolo_results(image, yolo_result, co2_data):
    """
    Main function using your exact weight estimation logic
    
    Args:
        image: PIL Image or numpy array
        yolo_result: YOLO segmentation result object
        co2_data: DataFrame with food data including density
    
    Returns:
        dict: Results with individual weights for each mask
    """
    try:
        # Convert image to PIL if needed
        if isinstance(image, np.ndarray):
            if image.shape[2] == 3:  # BGR to RGB
                image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            else:
                image_rgb = image
            pil_image = Image.fromarray(image_rgb)
        else:
            pil_image = image
        
        img_width, img_height = pil_image.size
        
        # Load depth estimation pipeline
        depth_pipe = load_depth_pipeline()
        if depth_pipe is None:
            st.error("Depth estimation model not available")
            return {}
        
        # Get depth map
        with st.spinner("Estimating depth for weight calculation..."):
            depth_out = depth_pipe(pil_image)
            depth_map = np.array(depth_out['depth'])
        
        # Normalize depth to [0, 1] and scale to plausible depth range - your exact approach
        depth_map = depth_map.astype(np.float32)
        depth_map = (depth_map - depth_map.min()) / (depth_map.max() - depth_map.min())
        min_depth, max_depth = 0.3, 2.0  # meters, adjust as needed - your exact values
        depth_map = min_depth + depth_map * (max_depth - min_depth)
        
        # Create density lookup from co2_data
        density_lookup = dict(zip(co2_data['Foodstuff'], co2_data['Density']))
        
        # Parse YOLO results into objects
        objects = parse_yolo_mask_from_results(yolo_result, img_width, img_height)
        
        if not objects:
            return {}
        
        # Process each detected object - your exact logic
        results = {}
        individual_weights = {}  # Store individual mask weights
        
        for i, obj in enumerate(objects):
            class_id = obj['class_id']
            
            # Get class name from YOLO model names
            if hasattr(yolo_result, 'names') and class_id in yolo_result.names:
                class_name = yolo_result.names[class_id]
            else:
                # Fallback to predefined class names
                class_name = id_to_class.get(class_id, f"Class_{class_id}")
            
            polygon = obj['polygon']
            
            # Skip if polygon is too small
            if len(polygon) < 3:
                continue
            
            # Area in pixels - your exact calculation
            area_px = polygon_area(polygon)
            if area_px < 100:  # Skip very small objects
                continue
            
            # Estimate scale using depth - your exact function
            median_depth = estimate_scale(depth_map, None, polygon)
            
            # Real-world area (m^2) - your exact function
            area_m2 = estimate_real_area(polygon, median_depth, img_width, img_height)
            
            # Estimate volume (m^3) - your exact function
            volume_m3 = estimate_volume(area_m2, median_depth)
            
            # Estimate weight (kg) - your exact approach
            if class_name in density_lookup:
                density = density_lookup[class_name]
            else:
                # Try to match with co2_data
                matched = co2_data[co2_data['Foodstuff'].str.lower() == class_name.lower()]
                if not matched.empty:
                    density = matched['Density'].values[0]
                else:
                    density = 0.8  # Default density
            
            weight_kg = volume_m3 * density
            weight_g = weight_kg * 1000
            
            # Store individual weight for this mask
            mask_id = f"{class_name}_{i}"
            individual_weights[mask_id] = {
                'class_name': class_name,
                'weight_kg': weight_kg,
                'weight_g': weight_g,
                'area_px': area_px,
                'area_m2': area_m2,
                'volume_m3': volume_m3,
                'depth': median_depth
            }
            
            # Aggregate by class - your exact approach
            if class_name not in results:
                results[class_name] = {'count': 0, 'weight_g': 0.0, 'weight_kg': 0.0}
            results[class_name]['count'] += 1
            results[class_name]['weight_g'] += weight_g
            results[class_name]['weight_kg'] += weight_kg
        
        # Store individual weights for detailed analysis
        results['_individual_weights'] = individual_weights
        
        return results
        
    except Exception as e:
        st.error(f"Error in weight estimation: {str(e)}")
        print(f"Detailed error in weight estimation: {e}")
        return {}

def estimate_weights_legacy(image_path, mask_txt_path, excel_path):
    """
    Your original function for file-based processing (exact copy)
    """
    # Load image
    image = Image.open(image_path)
    img_width, img_height = image.size

    # Load depth map
    depth_pipe = pipeline(task="depth-estimation", model="depth-anything/Depth-Anything-V2-Small-hf")
    depth_out = depth_pipe(image)
    depth_map = np.array(depth_out['depth'])
    # Normalize depth to [0, 1] and scale to a plausible depth range (e.g., 0.3-2.0m)
    depth_map = depth_map.astype(np.float32)
    depth_map = (depth_map - depth_map.min()) / (depth_map.max() - depth_map.min())
    min_depth, max_depth = 0.3, 2.0  # meters, adjust as needed
    depth_map = min_depth + depth_map * (max_depth - min_depth)

    # Load class data
    class_data = pd.read_excel(excel_path)
    density_lookup = dict(zip(class_data['Foodstuff'], class_data['Density']))

    # Parse mask file
    def parse_yolo_mask(mask_txt_path, img_width, img_height):
        objects = []
        with open(mask_txt_path, 'r') as f:
            lines = f.readlines()
        for line in lines:
            parts = line.strip().split()
            class_id = int(parts[0])
            coords = list(map(float, parts[1:]))
            xy = [(int(x * img_width), int(y * img_height)) for x, y in zip(coords[::2], coords[1::2])]
            objects.append({'class_id': class_id, 'polygon': xy})
        return objects

    objects = parse_yolo_mask(mask_txt_path, img_width, img_height)

    # Aggregate results
    results = {}
    for obj in objects:
        class_id = obj['class_id']
        class_name = id_to_class[class_id]
        polygon = obj['polygon']
        # Area in pixels
        area_px = polygon_area(polygon)
        # Estimate scale using depth
        median_depth = estimate_scale(depth_map, None, polygon)
        # Real-world area (m^2)
        area_m2 = estimate_real_area(polygon, median_depth, img_width, img_height)
        # Estimate volume (m^3)
        volume_m3 = estimate_volume(area_m2, median_depth)
        # Estimate weight (kg)
        density = density_lookup[class_name]
        weight_kg = volume_m3 * density
        weight_g = weight_kg * 1000
        # Aggregate by class
        if class_name not in results:
            results[class_name] = {'count': 0, 'weight_g': 0.0}
        results[class_name]['count'] += 1
        results[class_name]['weight_g'] += weight_g

    return results
