# src/logic/weight_estimation.py
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw
from transformers import pipeline
import torch
import cv2
from scipy.spatial import ConvexHull
from sklearn.cluster import DBSCAN
import open3d as o3d
import streamlit as st

# Class mapping for YOLO model - your exact list
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

def get_camera_intrinsics(img_width, img_height, fov_horizontal=60, fov_vertical=None):
    """
    Calculate camera intrinsic parameters from FOV - your exact function
    """
    if fov_vertical is None:
        # Assume 4:3 aspect ratio for vertical FOV calculation
        fov_vertical = fov_horizontal * (img_height / img_width) * (3/4)
    
    # Convert FOV to focal length
    fx = img_width / (2 * np.tan(np.deg2rad(fov_horizontal) / 2))
    fy = img_height / (2 * np.tan(np.deg2rad(fov_vertical) / 2))
    cx = img_width / 2
    cy = img_height / 2
    
    return np.array([[fx, 0, cx],
                     [0, fy, cy],
                     [0, 0, 1]])

def depth_to_pointcloud(depth_map, camera_intrinsics, mask=None):
    """
    Convert depth map to 3D point cloud using camera intrinsics - your exact function
    """
    h, w = depth_map.shape
    
    # Create pixel coordinates
    u, v = np.meshgrid(np.arange(w), np.arange(h))
    
    # Apply mask if provided
    if mask is not None:
        u = u[mask > 0]
        v = v[mask > 0]
        depth = depth_map[mask > 0]
    else:
        u = u.flatten()
        v = v.flatten()
        depth = depth_map.flatten()
    
    # Convert to homogeneous coordinates
    pixel_coords = np.vstack([u, v, np.ones_like(u)])
    
    # Convert to camera coordinates
    cam_coords = np.linalg.inv(camera_intrinsics) @ pixel_coords
    
    # Scale by depth
    points_3d = cam_coords * depth.reshape(1, -1)
    
    return points_3d.T  # Return as Nx3 array

def estimate_volume_divespot(points_3d, method='convex_hull'):
    """
    DIVESPOT-inspired volume estimation using different methods - your exact function
    """
    if len(points_3d) < 4:
        return 0.0
    
    try:
        if method == 'convex_hull':
            # Method 1: Convex Hull Volume (most robust)
            hull = ConvexHull(points_3d)
            return hull.volume
            
        elif method == 'alpha_shape':
            # Method 2: Alpha Shape Volume (more accurate for complex shapes)
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(points_3d)
            
            # Compute alpha shape
            alpha = 0.1  # Adjust based on point density
            mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_alpha_shape(pcd, alpha)
            
            if mesh.is_watertight():
                return mesh.get_volume()
            else:
                # Fallback to convex hull
                return estimate_volume_divespot(points_3d, 'convex_hull')
                
        elif method == 'voxel_grid':
            # Method 3: Voxel-based Volume Estimation
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(points_3d)
            
            # Create voxel grid
            voxel_size = 0.01  # 1cm voxels
            voxel_grid = o3d.geometry.VoxelGrid.create_from_point_cloud(pcd, voxel_size)
            
            num_voxels = len(voxel_grid.get_voxels())
            return num_voxels * (voxel_size ** 3)
            
    except Exception as e:
        print(f"Volume estimation failed with method {method}: {e}")
        # Fallback to simple bounding box volume
        bbox_min = np.min(points_3d, axis=0)
        bbox_max = np.max(points_3d, axis=0)
        return np.prod(bbox_max - bbox_min) * 0.5  # Assume 50% fill ratio

def process_depth_map(depth_map, polygon, img_width, img_height, scale_factor=1.0):
    """
    Process depth map with better scaling and filtering - your exact function
    """
    # Create mask from polygon
    mask_img = Image.new('L', (img_width, img_height), 0)
    ImageDraw.Draw(mask_img).polygon(polygon, outline=1, fill=1)
    mask_arr = np.array(mask_img)
    
    # Apply morphological operations to clean mask
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask_arr = cv2.morphologyEx(mask_arr.astype(np.uint8), cv2.MORPH_CLOSE, kernel)
    
    # Resize depth map to match image dimensions if needed
    if depth_map.shape[:2] != (img_height, img_width):
        depth_map = cv2.resize(depth_map, (img_width, img_height))
    
    # Normalize and scale depth map
    depth_map = depth_map.astype(np.float32)
    depth_map = (depth_map - depth_map.min()) / (depth_map.max() - depth_map.min())
    
    # Apply realistic depth scaling based on camera setup
    min_depth, max_depth = 0.2, 3.0  # Adjust based on your camera setup
    depth_map = min_depth + depth_map * (max_depth - min_depth)
    depth_map *= scale_factor
    
    # Apply Gaussian smoothing to reduce noise
    depth_map = cv2.GaussianBlur(depth_map, (3, 3), 0)
    
    return depth_map, mask_arr

def get_density_for_class(class_name, co2_data):
    """Get density value for a class from the CO2 data"""
    try:
        # Match class name with foodstuff in data
        matched = co2_data[co2_data['Foodstuff'].str.lower() == class_name.lower()]
        if not matched.empty:
            return matched['Density'].values[0]  # Density in g/cm³
        else:
            return 0.8  # Default density
    except Exception:
        return 0.8  # Default density on error

def estimate_weights_from_yolo_results(image, yolo_result, co2_data, camera_fov=60, scale_factor=1.0):
    """
    Enhanced weight estimation using DIVESPOT-inspired volume calculation - your exact approach
    
    Args:
        image: PIL Image or numpy array
        yolo_result: YOLO segmentation result object
        co2_data: DataFrame with food data including density
        camera_fov: Horizontal field of view in degrees
        scale_factor: Depth scale correction factor
    
    Returns:
        dict: Results with class names as keys and weight info as values
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
        
        # Get depth map using Depth Anything V2
        with st.spinner("Estimating depth for enhanced volume calculation..."):
            depth_out = depth_pipe(pil_image)
            depth_map = np.array(depth_out['depth'])
        
        # Get camera intrinsics
        camera_intrinsics = get_camera_intrinsics(img_width, img_height, camera_fov)
        
        # Parse YOLO results into objects
        objects = parse_yolo_mask_from_results(yolo_result, img_width, img_height)
        
        if not objects:
            return {}
        
        # Process each detected object
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
            
            # Process depth map for this object - your exact function
            processed_depth, mask = process_depth_map(depth_map, polygon, img_width, img_height, scale_factor)
            
            # Generate 3D point cloud for the object - your exact function
            points_3d = depth_to_pointcloud(processed_depth, camera_intrinsics, mask)
            
            if len(points_3d) < 10:  # Skip if too few points
                continue
            
            # DIVESPOT-inspired volume estimation - your exact approach
            # Try multiple methods and use the most reliable result
            volumes = []
            for method in ['convex_hull', 'voxel_grid']:
                try:
                    vol = estimate_volume_divespot(points_3d, method)
                    if vol > 0:
                        volumes.append(vol)
                except:
                    continue
            
            if not volumes:
                continue
                
            # Use median volume for robustness - your exact approach
            volume_m3 = np.median(volumes)
            
            # Get density and calculate weight
            density = get_density_for_class(class_name, co2_data)  # kg/m³
            weight_kg = volume_m3 * density
            weight_g = weight_kg * 1000
            
            # Store individual weight for this mask
            mask_id = f"{class_name}_{i}"
            individual_weights[mask_id] = {
                'class_name': class_name,
                'weight_kg': weight_kg,
                'weight_g': weight_g,
                'volume_m3': volume_m3,
                'volume_cm3': volume_m3 * 1e6,
                'points_3d_count': len(points_3d),
                'methods_used': len(volumes)
            }
            
            # Aggregate results by class - your exact approach
            if class_name not in results:
                results[class_name] = {
                    'count': 0,
                    'total_weight_g': 0.0,
                    'weight_kg': 0.0,
                    'avg_volume_cm3': 0.0,
                    'volumes': []
                }
            
            results[class_name]['count'] += 1
            results[class_name]['total_weight_g'] += weight_g
            results[class_name]['weight_kg'] += weight_kg
            results[class_name]['volumes'].append(volume_m3 * 1e6)  # Convert to cm³
            results[class_name]['avg_volume_cm3'] = np.mean(results[class_name]['volumes'])
        
        # Store individual weights for detailed analysis
        results['_individual_weights'] = individual_weights
        
        return results
        
    except Exception as e:
        st.error(f"Enhanced weight estimation error: {str(e)}")
        print(f"Detailed error in enhanced weight estimation: {e}")
        return {}

def estimate_weights_legacy(image_path, mask_txt_path, excel_path, camera_fov=60, scale_factor=1.0):
    """
    Your original enhanced function for file-based processing (exact copy)
    """
    # Load image
    image = Image.open(image_path)
    img_width, img_height = image.size
    
    # Load depth map using Depth Anything V2
    depth_pipe = pipeline(task="depth-estimation", model="depth-anything/Depth-Anything-V2-Small-hf")
    depth_out = depth_pipe(image)
    depth_map = np.array(depth_out['depth'])
    
    # Get camera intrinsics
    camera_intrinsics = get_camera_intrinsics(img_width, img_height, camera_fov)
    
    # Load density data
    class_data = pd.read_excel(excel_path)
    density_lookup = dict(zip(class_data['Foodstuff'], class_data['Density']))
    
    # Parse segmentation masks (your original function)
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
    
    # Process each object
    results = {}
    for obj in objects:
        class_id = obj['class_id']
        class_name = id_to_class[class_id]
        polygon = obj['polygon']
        
        # Process depth map for this object
        processed_depth, mask = process_depth_map(depth_map, polygon, img_width, img_height, scale_factor)
        
        # Generate 3D point cloud for the object
        points_3d = depth_to_pointcloud(processed_depth, camera_intrinsics, mask)
        
        if len(points_3d) < 10:  # Skip if too few points
            continue
        
        # DIVESPOT-inspired volume estimation
        # Try multiple methods and use the most reliable result
        volumes = []
        for method in ['convex_hull', 'voxel_grid']:
            try:
                vol = estimate_volume_divespot(points_3d, method)
                if vol > 0:
                    volumes.append(vol)
            except:
                continue
        
        if not volumes:
            continue
            
        # Use median volume for robustness
        volume_m3 = np.median(volumes)
        
        # Get density and calculate weight
        if class_name in density_lookup:
            density = density_lookup[class_name]  # kg/m³
            weight_kg = volume_m3 * density
            weight_g = weight_kg * 1000
            
            # Aggregate results
            if class_name not in results:
                results[class_name] = {
                    'count': 0, 
                    'total_weight_g': 0.0, 
                    'avg_volume_cm3': 0.0,
                    'volumes': []
                }
            
            results[class_name]['count'] += 1
            results[class_name]['total_weight_g'] += weight_g
            results[class_name]['volumes'].append(volume_m3 * 1e6)  # Convert to cm³
            results[class_name]['avg_volume_cm3'] = np.mean(results[class_name]['volumes'])
    
    return results
