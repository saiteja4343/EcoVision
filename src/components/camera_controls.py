# src/components/camera_controls.py
import streamlit as st

def get_webrtc_config():
    """Get WebRTC configuration for cloud deployment"""
    return {
        "iceServers": [
            {"urls": ["stun:stun.l.google.com:19302"]},
            # Add more STUN servers for better connectivity
            {"urls": ["stun:stun1.l.google.com:19302"]},
        ]
    }

def display_camera_info():
    """Display information about WebRTC camera access"""
    st.info("""
    🎥 **Camera Access**: This app uses WebRTC to access your camera directly in the browser.
    
    **First time setup**:
    1. Click 'START' on the video component
    2. Allow camera permissions when prompted
    3. Your video will appear and processing will begin
    
    **Cloud deployment ready** - works on any hosting platform!
    """)
