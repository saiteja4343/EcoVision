# src/logic/image_processing.py
import streamlit as st
import numpy as np
import cv2
from PIL import Image

# Import the new weight estimation module
from src.logic.weight_estimation import estimate_weights_from_yolo_results

def process_image(img, model, confidence, class_filter=None, co2_data=None, detection_type="Bounding boxes"):
    """Process image with YOLO model - handles both detection and segmentation"""
    try:
        # Determine if this is a segmentation model
        is_segmentation = "-seg" in str(model.ckpt_path).lower() or detection_type == "Segmentation"
        
        if is_segmentation:
            return process_segmentation_image(img, model, confidence, class_filter, co2_data)
        else:
            return process_detection_image(img, model, confidence, class_filter, co2_data)
            
    except Exception as e:
        st.error(f"Processing error: {str(e)}")
        return None, None

def process_detection_image(img, model, confidence, class_filter=None, co2_data=None):
    """Process image for bounding box detection"""
    try:
        # Handle class filtering for detection
        class_indices = None
        if class_filter and len(class_filter) > 0:
            class_indices = []
            for food_name in class_filter:
                for idx, class_name in model.names.items():
                    if class_name.lower() == food_name.lower():
                        class_indices.append(idx)
        
        # Run YOLO inference
        results = model.predict(
            source=img,
            conf=confidence,
            classes=class_indices,
            imgsz=640,
            device=model.device.type,
            verbose=False,
            stream=False
        )
        
        # Get the first result
        result = results[0]
        
        # Plot results on image
        plotted = result.plot()
        
        # Convert BGR to RGB for proper display
        if len(plotted.shape) == 3:
            plotted = plotted[:, :, ::-1]  # BGR to RGB
            
        return result, plotted
        
    except Exception as e:
        st.error(f"Detection processing error: {str(e)}")
        return None, None

def process_segmentation_image(img, model, confidence, class_filter=None, co2_data=None):
    """Process image for segmentation with advanced weight estimation"""
    try:
        # Handle class filtering for segmentation
        class_indices = None
        if class_filter and len(class_filter) > 0:
            class_indices = []
            for food_name in class_filter:
                for idx, class_name in model.names.items():
                    if class_name.lower() == food_name.lower():
                        class_indices.append(idx)
        
        # Run YOLO segmentation inference
        results = model.predict(
            source=img,
            conf=confidence,
            classes=class_indices,
            imgsz=640,
            device=model.device.type,
            verbose=False,
            stream=False
        )
        
        # Get the first result
        result = results[0]
        
        # Check if we have masks (segmentation results)
        if hasattr(result, 'masks') and result.masks is not None:
            # Process with advanced weight estimation
            result_with_weights = process_segmentation_with_advanced_weights(img, result, co2_data)
            
            # Plot results on image
            plotted = result.plot()
            
            # Convert BGR to RGB for proper display
            if len(plotted.shape) == 3:
                plotted = plotted[:, :, ::-1]  # BGR to RGB
                
            return result_with_weights, plotted
        else:
            # No masks found, treat as regular detection
            plotted = result.plot()
            if len(plotted.shape) == 3:
                plotted = plotted[:, :, ::-1]
            return result, plotted
        
    except Exception as e:
        st.error(f"Segmentation processing error: {str(e)}")
        return None, None

def process_segmentation_with_advanced_weights(img, result, co2_data):
    """Process segmentation results with advanced depth-based weight estimation"""
    try:
        # Use the new weight estimation approach
        with st.spinner("Calculating weights using depth estimation..."):
            estimated_weights = estimate_weights_from_yolo_results(img, result, co2_data)
        
        # Add weight information to result object
        result.estimated_weights = estimated_weights
        
        return result
        
    except Exception as e:
        st.error(f"Advanced weight processing error: {str(e)}")
        return result  # Return original result on error
