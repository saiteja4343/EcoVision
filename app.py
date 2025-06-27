# app.py
import os
import sys
import streamlit as st
from src.config import THEME
from src.components.sidebar import create_sidebar
from src.logic.model_loader import load_model
from src.logic.data_loader import load_co2_data
import pandas as pd
from huggingface_hub import hf_hub_download, list_repo_files

def main():
    # Initialize app configuration
    st.set_page_config(
        page_title="EcoVision AI",
        page_icon="🌍",
        layout="wide"
    )
    
    # Configuration
    REPO_ID = "nagasaiteja999/EcoVision"
    CACHE_DIR = "model_cache"
    
    # Apply custom theme
    apply_custom_theme()
    
    # Initialize session state FIRST
    init_session_state()
    
    # Load required data
    co2_data = load_co2_data()
    models = get_available_models(repo_id=REPO_ID)
    
    # Create sidebar components (now returns detection_type)
    selected_model, confidence, class_filter, detection_type = create_sidebar(co2_data, models)
    
    # Load model from Hugging Face Hub
    model_path = hf_hub_download(
        repo_id=REPO_ID,
        filename=selected_model,
        cache_dir=CACHE_DIR,
    )
    
    # Load selected model
    model = load_model(model_path)
    
    # Main app interface
    st.title("🌱 EcoVision AI")
    st.markdown("### AI-Powered Food Recognition & Carbon Footprint Calculator")
    
    # Removed: Current mode display
    
    # Page routing
    page = st.sidebar.selectbox("Choose Mode", ["Image Analysis", "Live Detection"])
    
    if page == "Image Analysis":
        from pages.image_analysis import image_analysis_page
        image_analysis_page(model, confidence, class_filter, co2_data, detection_type)
    else:
        from pages.live_detection import live_detection_page
        live_detection_page(model, confidence, class_filter, co2_data, detection_type)
    
    st.markdown("---")
    st.caption("Copyright © 2025 Kuenneth Research Group, University of Bayreuth. All rights reserved. Created by Naga Sai Teja Kolakaleti.")

def apply_custom_theme():
    """Applies custom CSS theme from config"""
    st.markdown(f"""
    <style>
        .stApp {{
            background-color: {THEME['background_color']};
        }}
        .stSelectbox > div > div {{
            background-color: {THEME['secondary_background_color']};
        }}
        .stButton > button {{
            background-color: {THEME['primary_color']};
            color: {THEME['button_text_color']};
            border-radius: {THEME['border_radius']};
        }}
        .stMetric {{
            background-color: {THEME['secondary_background_color']};
            padding: 10px;
            border-radius: {THEME['border_radius']};
        }}
    </style>
    """, unsafe_allow_html=True)

def init_session_state():
    """Initialize ALL session state variables"""
    defaults = {
        'camera_active': False,
        'captured_frame': None,
        'emissions_df': pd.DataFrame(),
        'total_co2': 0,
        'last_frame': None,
        # WebRTC specific state
        'webrtc_emissions': {
            'emissions_df': None,
            'total_co2': 0,
            'captured_frame': None,
            'processing_active': False
        }
    }
    
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

def get_available_models(repo_id="nagasaiteja999/EcoVision"):
    """Retrieves available YOLO models from Hugging Face repository"""
    models = [f for f in list_repo_files(repo_id) if f.endswith('.pt')]
    if not models:
        st.error("No models found in repository")
        st.stop()
    return models

if __name__ == "__main__":
    # Add project root to Python path
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    main()
