# import cv2
#
# def list_available_cameras():
#     cameras = []
#     for i in range(4):
#         cap = cv2.VideoCapture(i)
#         if cap.isOpened():
#             cameras.append(i)
#             cap.release()
#     return cameras

# src/components/camera_controls.py
import os
import cv2
from streamlit_webrtc import webrtc_streamer, WebRtcMode
from twilio.rest import Client


def get_twilio_client():
    """Initialize Twilio client for STUN/TURN servers"""
    return Client(
        os.getenv("TWILIO_ACCOUNT_SID"),
        os.getenv("TWILIO_AUTH_TOKEN")
    )

def create_webrtc_component(processor_factory):
    """Main camera connection component"""
    return webrtc_streamer(
        key="eco_vision_cam",
        mode=WebRtcMode.SENDRECV,
        rtc_configuration={
            "iceServers": [
                {
                    "urls": ["stun:stun.l.google.com:19302"],
                    "username": os.getenv("TWILIO_ACCOUNT_SID"),
                    "credential": os.getenv("TWILIO_AUTH_TOKEN")
                }
            ]
        },
        media_stream_constraints={
            "video": {
                "width": {"ideal": 1920},
                "height": {"ideal": 1080},
                "frameRate": {"ideal": 60},
                "aspectRatio": 16/9
            },
            "audio": False
        },
        video_processor_factory=processor_factory,
        async_processing=True
    )

def get_camera_settings():
    """Centralized camera configuration"""
    return {
        "resolution": (1280, 720),
        "fps": 30,
        "codec": "h264",
        "color_format": "bgr24"
    }

