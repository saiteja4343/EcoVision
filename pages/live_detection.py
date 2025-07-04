# pages/live_detection.py
import streamlit as st
import cv2
import numpy as np
from streamlit_webrtc import webrtc_streamer, RTCConfiguration, VideoProcessorBase
import av
import threading
from PIL import Image
import io
import time

from src.logic.image_processing import process_image
from src.logic.emissions_calculator import calculate_emissions, export_data

RTC_CONFIGURATION = RTCConfiguration({"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]})

def live_detection_page(model, confidence, class_filter, co2_data, detection_type):
    # Display detection type info with enhanced weight estimation method
    st.header("Capture and Process")
    
    # Show detection type specific info with enhanced processing details
    if detection_type == "Segmentation":
        st.info("🎯 **Enhanced Segmentation Mode**: Using DIVESPOT-inspired 3D point cloud analysis for precise weight estimation")
        st.markdown("""
        **Enhanced Weight Estimation Process:**
        - 🔍 **Segmentation**: Pixel-level food boundary detection
        - 📏 **Depth Analysis**: Depth-Anything-V2-Small-hf model (0.2-3.0m range)
        - 🎥 **Camera Calibration**: Dynamic intrinsic parameter calculation
        - ☁️ **Point Cloud Generation**: 3D point cloud from depth + camera intrinsics
        - 📐 **Volume Estimation**: Multiple methods (Convex Hull, Voxel Grid, Alpha Shape)
        - ⚖️ **Weight Calculation**: Volume × Food Density with robust median filtering
        - 🌱 **CO₂ Calculation**: Enhanced Weight × CO₂ Factor
        """)
    else:
        st.info("📦 **Standard Detection Mode**: Using average weights for quick estimation")
    
    handle_capture_webrtc_process(model, confidence, class_filter, co2_data, detection_type)

def handle_capture_webrtc_process(model, confidence, class_filter, co2_data, detection_type):
    st.subheader("📹 Camera Capture")

    class CaptureVideoProcessor(VideoProcessorBase):
        def __init__(self):
            self.lock = threading.Lock()
            self.captured_frame = None
            self.capture_requested = False

        def recv(self, frame):
            img = frame.to_ndarray(format="bgr24")

            with self.lock:
                if self.capture_requested:
                    self.captured_frame = img.copy()
                    st.session_state.webrtc_emissions['captured_frame'] = img.copy()
                    self.capture_requested = False

            return av.VideoFrame.from_ndarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), format="rgb24")

        def request_capture(self):
            with self.lock:
                self.capture_requested = True

        def get_captured_frame(self):
            with self.lock:
                return self.captured_frame.copy() if self.captured_frame is not None else None

    webrtc_ctx = webrtc_streamer(
        key="simple_capture",
        video_processor_factory=CaptureVideoProcessor,
        rtc_configuration=RTC_CONFIGURATION,
        media_stream_constraints={"video": True, "audio": False},
    )

    if webrtc_ctx.state.playing:
        st.success("✅ Camera is active - ready to capture!")

        if st.button("📸 Capture Frame", key="simple_capture_btn", type="primary"):
            if webrtc_ctx.video_processor:
                webrtc_ctx.video_processor.request_capture()
                st.success("📸 Frame captured! Processing...")
                time.sleep(0.5)
                captured_frame = webrtc_ctx.video_processor.get_captured_frame()
                if captured_frame is not None:
                    st.session_state.webrtc_emissions['captured_frame'] = captured_frame
                st.rerun()
            else:
                st.error("❌ Video processor not available")

    if st.session_state.webrtc_emissions.get('captured_frame') is not None:
        st.markdown("---")
        st.subheader("🔬 Processing Captured Frame")

        frame = st.session_state.webrtc_emissions['captured_frame']

        # Show different processing messages based on detection type
        if detection_type == "Segmentation":
            processing_message = "🧠 Processing with DIVESPOT-inspired 3D volume estimation..."
        else:
            processing_message = "⚡ Processing with standard detection and average weights..."
            
        with st.spinner(processing_message):
            results, output = process_image(frame, model, confidence, class_filter, co2_data, detection_type)

        if results is not None and output is not None:
            col1, col2 = st.columns(2)

            with col1:
                st.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), caption="📸 Captured Frame", use_column_width=True)

            with col2:
                st.image(output, caption="🔍 Processed Result", use_column_width=True)

            emissions_df, total_co2 = calculate_emissions(results, model, co2_data)

            if not emissions_df.empty:
                st.subheader("🌱 Carbon Emissions Report")
                
                # Display method-specific information
                if detection_type == "Segmentation":
                    st.success("✨ **Enhanced Analysis**: Weight calculated using DIVESPOT-inspired 3D point cloud volume estimation")
                    
                    # Show enhanced weight estimation details if available
                    if hasattr(results, 'estimated_weights') and results.estimated_weights:
                        individual_weights = results.estimated_weights.get('_individual_weights', {})
                        if individual_weights:
                            st.markdown("**Enhanced Weight Estimation Details:**")
                            for mask_id, info in individual_weights.items():
                                st.markdown(f"• **{mask_id}**: {info['weight_g']:.1f}g (vol: {info['volume_cm3']:.1f}cm³, 3D points: {info['points_3d_count']}, methods: {info['methods_used']})")
                        
                        st.markdown("**Total Weight per Class:**")
                        for food, info in results.estimated_weights.items():
                            if food != '_individual_weights':
                                st.markdown(f"• **{food}**: {info['total_weight_g']:.1f}g total from {info['count']} mask(s) (avg vol: {info['avg_volume_cm3']:.1f}cm³)")
                else:
                    st.info("📊 **Standard Analysis**: Weight estimated using database average weights")
                
                # Format dataframe differently based on detection type
                if detection_type == "Segmentation":
                    # For segmentation: no Unit Weight column
                    formatted_columns = {
                        'Total Weight (kg)': '{:.3f}',
                        'CO2 emissions (kg)- class': '{:.3f}',
                        'CO₂ Emissions (kg)': '{:.3f}'
                    }
                else:
                    # For detection: include Unit Weight column
                    formatted_columns = {
                        'Unit Weight (kg)': '{:.3f}',
                        'Total Weight (kg)': '{:.3f}',
                        'CO2 emissions (kg)- class': '{:.3f}',
                        'CO₂ Emissions (kg)': '{:.3f}'
                    }
                
                st.dataframe(
                    emissions_df.style.format(formatted_columns),
                    use_container_width=True
                )

                st.metric("Total CO₂ Impact", f"{total_co2:.2f} kg CO₂ eq")

                st.markdown("### 📥 Export Results")
                export_data(emissions_df, detection_type)

                buf = io.BytesIO()
                Image.fromarray(output).save(buf, format="PNG")
                st.download_button("📥 Download Processed Image", buf.getvalue(), "processed_image.png", "image/png")
            else:
                st.info("ℹ️ No food items detected in the captured frame")
        else:
            st.error("❌ Failed to process the captured frame. Please try again.")

        if st.button("🔄 Capture New Frame", key="reset_simple_capture"):
            st.session_state.webrtc_emissions['captured_frame'] = None
            st.rerun()
