import streamlit as st
from src.components.camera_controls import list_available_cameras


def create_sidebar(co2_data, models):
    with st.sidebar:
        st.header("Settings")

        # Model selection
        selected_model = st.selectbox("Select Model", models)

        # Camera selection
        available_cams = list_available_cameras()
        cam_choice = st.selectbox("Select Camera", available_cams,
                                  format_func=lambda x: f"Camera {x}")

        # Detection parameters
        confidence = st.slider("Confidence Threshold", 0.0, 1.0, 0.5)
        class_filter = st.multiselect("Filter Classes", co2_data['Foodstuff'].unique())

    return selected_model, confidence, class_filter, cam_choice
