# app.py
import os
import sys
import streamlit as st
from src.config import PATHS, THEME
from src.components.sidebar import create_sidebar
from src.logic.model_loader import load_model
from src.logic.data_loader import load_co2_data
import pandas as pd
from src.components.camera_controls import get_camera_settings, get_twilio_client

import os

# Add Twilio credentials to environment
os.environ["TWILIO_ACCOUNT_SID"] = "your-twilio-key-here"
os.environ["TWILIO_AUTH_TOKEN"] = "your-twilio-key-here"


def main():
    # Initialize app configuration
    st.set_page_config(
        page_title="EcoVision AI",
        page_icon="🌍",
        layout="wide"
    )

    try:
        twilio_client = get_twilio_client()
        twilio_client.messages.list(limit=1)  # Simple API call to validate credentials
    except Exception as e:
        st.error(f"Twilio connection failed: {str(e)}")
        st.stop()

    # Apply custom theme
    apply_custom_theme()

    # Initialize session state
    init_session_state()

    # Load required data
    co2_data = load_co2_data()
    models = get_available_models()

    # # Add camera settings to session state
    # st.session_state.camera_config = get_camera_settings()



    # Create sidebar components
    #selected_model, confidence, class_filter, cam_choice = create_sidebar(co2_data, models)
    selected_model, confidence, class_filter = create_sidebar(co2_data, models)

    # Load selected model
    model = load_model(os.path.join(PATHS['models'], selected_model))

    # Main app interface
    st.title("🌱 EcoVision AI")

    # Page routing
    page = st.sidebar.selectbox("Choose Mode", ["Image Analysis", "Live Detection"])

    if page == "Image Analysis":
        from pages.image_analysis import image_analysis_page
        image_analysis_page(model, confidence, class_filter, co2_data)
    else:
        from pages.live_detection import live_detection_page
        #live_detection_page(model, confidence, class_filter, co2_data, cam_choice)
        live_detection_page(model, confidence, class_filter, co2_data)


def apply_custom_theme():
    """Applies custom CSS theme from config"""
    st.markdown(f"""
    <style>
        .stApp {{
            background-color: {THEME['background_color']};
            color: {THEME['text_color']};
        }}
        .stButton>button {{
            background-color: {THEME['primary_color']} !important;
            color: {THEME['text_color']} !important;
            border-radius: 8px;
            transition: transform 0.2s;
        }}
        .stButton>button:hover {{
            transform: scale(1.05);
            opacity: 0.9;
        }}
        .stSelectbox label, .stSlider label {{
            font-weight: 600;
            color: {THEME['text_color']};
        }}
        .dataframe {{
            background-color: {THEME['secondary_background_color']} !important;
        }}
    </style>
    """, unsafe_allow_html=True)


def init_session_state():
    """Initializes all required session state variables"""
    defaults = {
        'camera_active': False,
        'captured_frame': None,
        'emissions_df': pd.DataFrame(),
        'total_co2': 0,
        'last_frame': None
    }

    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def get_available_models():
    """Retrieves available YOLO models from models directory"""
    models = [f for f in os.listdir(PATHS['models']) if f.endswith('.pt')]
    if not models:
        st.error("No models found in models directory")
        st.stop()
    return models


if __name__ == "__main__":
    # Add project root to Python path
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    main()
