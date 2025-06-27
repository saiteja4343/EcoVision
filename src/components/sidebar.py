# src/components/sidebar.py
import streamlit as st

def create_sidebar(co2_data, models):
    with st.sidebar:
        st.header("🔧 Settings")
        
        # Detection type selection (new section at the beginning)
        st.subheader("🎯 Detection Type")
        detection_type = st.selectbox(
            "Select Detection Type",
            ["Bounding boxes", "Segmentation"],
            help="Choose between bounding box detection and segmentation"
        )
        
        # Filter models based on detection type
        if detection_type == "Segmentation":
            # Show only models with "-seg" in the name
            filtered_models = [model for model in models if "-seg" in model.lower()]
        else:
            # Show only models without "-seg" in the name (Bounding boxes)
            filtered_models = [model for model in models if "-seg" not in model.lower()]
        
        # Handle case where no models are available for selected type
        if not filtered_models:
            st.error(f"No {detection_type.lower()} models available")
            # Fall back to all models to prevent crash
            filtered_models = models
        
        # Model selection (filtered based on detection type)
        st.subheader("🤖 Model Selection")
        selected_model = st.selectbox(
            "Select YOLO Model", 
            filtered_models,
            help=f"Available {detection_type.lower()} models"
        )
        
        # Detection parameters
        st.subheader("⚙️ Detection Parameters")
        confidence = st.slider(
            "Confidence Threshold", 
            0.0, 1.0, 0.5,
            help="Minimum confidence for detections"
        )
        
        # Class filtering
        available_classes = co2_data['Foodstuff'].unique()
        class_filter = st.multiselect(
            "Filter Food Classes", 
            available_classes,
            help="Select specific foods to detect (leave empty for all)"
        )
        
        # Removed: Current Selection info box
        
        return selected_model, confidence, class_filter, detection_type
