import av
import numpy as np
from streamlit_webrtc import VideoProcessorBase, webrtc_streamer
from src.logic.model_loader import load_model
from src.logic.emissions_calculator import calculate_emissions, export_data
import torch
import streamlit as st
import os
import cv2
from src.components.camera_controls import get_camera_settings, create_webrtc_component
from src.logic.image_processing import process_image


class YOLOVideoProcessor(VideoProcessorBase):
    def __init__(self, model, confidence, class_filter, co2_data):
        self.model = model
        self.confidence = confidence
        self.class_filter = class_filter
        self.co2_data = co2_data
        self.camera_config = get_camera_settings()
        self.last_results = None

    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        try:
            # Convert frame to RGB first
            img = frame.to_ndarray(format=self.camera_config["color_format"])
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert to RGB

            # # Process with YOLO
            # results = self.model.predict(
            #     source=img_rgb,
            #     conf=self.confidence,
            #     classes=self.class_filter,
            #     imgsz=self.camera_config["resolution"],
            #     device='cuda' if torch.cuda.is_available() else 'cpu'
            # )
            #
            # # Store results for emissions calculation
            # self.last_results = results[0]
            #
            # # Convert back to BGR for video output
            # plotted = cv2.cvtColor(results[0].plot(), cv2.COLOR_RGB2BGR)
            _, plotted = process_image(img_rgb, self.model, self.confidence, self.class_filter, self.co2_data)

            return av.VideoFrame.from_ndarray(plotted, format=self.camera_config["color_format"])

        except Exception as e:
            st.error(f"Frame processing error: {str(e)}")
            return frame


def live_detection_page(model, confidence, class_filter, co2_data):
    st.header("Live Sustainability Analysis")

    # # Initialize WebRTC component
    # ctx = webrtc_streamer(
    #     key="livestream",
    #     video_processor_factory=lambda: YOLOVideoProcessor(model, confidence, class_filter, co2_data),
    #     rtc_configuration={
    #         "iceServers": [
    #             {
    #                 "urls": ["stun:stun.l.google.com:19302"],
    #                 "username": os.environ["TWILIO_ACCOUNT_SID"],
    #                 "credential": os.environ["TWILIO_AUTH_TOKEN"]
    #             }
    #         ]
    #     },
    #     media_stream_constraints={"video": True, "audio": False},
    #     async_processing=True
    # )
    #
    # # Emissions calculation and display
    # if ctx.video_processor and ctx.video_processor.last_results:
    #     results = ctx.video_processor.last_results
    #     emissions_df, total_co2 = calculate_emissions(results, co2_data)
    #
    #     st.subheader("Real-time Carbon Impact")
    #     st.dataframe(
    #         emissions_df.style.format({
    #             'Unit Weight (kg)': '{:.3f}',
    #             'Total Weight (kg)': '{:.3f}',
    #             'CO₂ Emissions (kg)': '{:.3f}'
    #         }),
    #         use_container_width=True
    #     )
    #     st.metric("Total CO₂ Impact", f"{total_co2:.2f} kg CO₂ eq")

    # Create WebRTC component with video processor
    ctx = create_webrtc_component(lambda: YOLOVideoProcessor(model, confidence, class_filter, co2_data))

    # Display emissions results when available
    if ctx.video_processor and ctx.video_processor.last_results:
        results = ctx.video_processor.last_results
        emissions_df, total_co2 = calculate_emissions(results, ctx.video_processor.model, ctx.video_processor.co2_data)

        if not emissions_df.empty:
            st.subheader("Real-time Carbon Impact Report")

            # Display dataframe with formatting
            st.dataframe(
                emissions_df.style.format({
                    'Unit Weight (kg)': '{:.3f}',
                    'Total Weight (kg)': '{:.3f}',
                    'CO₂ Emissions (kg)': '{:.3f}'
                }),
                use_container_width=True
            )

            # Metrics and export section
            col1, col2 = st.columns([1, 2])
            with col1:
                st.metric("Total CO₂ Impact", f"{total_co2:.2f} kg CO₂ eq")
            with col2:
                st.markdown("### Export Options")
                export_data(emissions_df)

# # pages/live_detection.py
# import cv2
# import time
# import streamlit as st
# from datetime import datetime
# from PIL import Image
# import io
# from src.logic.image_processing import process_image
# from src.logic.emissions_calculator import calculate_emissions, export_data
#
# def live_detection_page(model, confidence, class_filter, co2_data, cam_choice):
#     st.header("Live Camera Analysis")
#     # processing_mode = st.radio("Mode:", ["Live Detection", "Capture & Process"], horizontal=True)
#     #
#     # # Create fixed layout containers
#     # frame_placeholder = st.empty()
#     # status_placeholder = st.empty()
#     #
#     # if processing_mode == "Live Detection":
#     #     handle_live_detection(frame_placeholder, status_placeholder, model,
#     #                         confidence, class_filter, co2_data, cam_choice)
#     # else:
#     #     handle_capture_process(frame_placeholder, status_placeholder, model,
#     #                          confidence, class_filter, co2_data, cam_choice)
#
#     # Radio button container at the top
#     with st.container():
#         processing_mode = st.radio("Mode:", ["Live Detection", "Capture & Process"],
#                                    horizontal=True, key="proc_mode")
#
#     # Fixed controls container below radio buttons
#     controls_container = st.container()
#
#     # Create fixed frame placeholder below controls
#     frame_placeholder = st.empty()
#     status_placeholder = st.empty()
#
#     if processing_mode == "Live Detection":
#         with controls_container:
#             col1, col2 = st.columns(2)
#             with col1:
#                 if st.button("Start Camera Feed", key="start_cam"):
#                     st.session_state.camera_active = True
#             with col2:
#                 if st.button("Stop Camera", key="stop_cam"):
#                     st.session_state.camera_active = False
#
#         handle_live_detection(frame_placeholder, status_placeholder, model,
#                               confidence, class_filter, co2_data, cam_choice)
#     else:
#         with controls_container:
#             col1, col2 = st.columns(2)
#             with col1:
#                 if st.button("Start Camera Feed", key="start_cam"):
#                     st.session_state.camera_active = True
#             with col2:
#                 if st.button("Stop Camera", key="stop_cam"):
#                     st.session_state.camera_active = False
#
#         handle_capture_process(frame_placeholder, status_placeholder, model,
#                                confidence, class_filter, co2_data, cam_choice)
#
# def handle_live_detection(frame_placeholder, status_placeholder, model, confidence,
#                         class_filter, co2_data, cam_choice):
#     emissions_table = st.empty()
#     emissions_metric = st.empty()
#     export_placeholder = st.empty()
#
#     # # Create fixed container for camera controls at the top
#     # button_container = st.container()
#     #
#     # with button_container:
#     #
#     #     # Camera control buttons
#     #     col1, col2, _ = st.columns([1,1,6])
#     #     with col1:
#     #         if st.button("Start Camera Feed", key="start_cam"):
#     #             st.session_state.camera_active = True
#     #     with col2:
#     #         if st.button("Stop Camera", key="stop_cam"):
#     #             st.session_state.camera_active = False
#
#     # # Fixed column definition outside conditional blocks
#     # control_col1, control_col2 = st.columns([1, 1])
#     #
#     # # Persistent button container
#     # with control_col1.container():
#     #     if not st.session_state.get('camera_active', False):
#     #         if st.button("Start Camera Feed", key="start_cam"):
#     #             st.session_state.camera_active = True
#     #             st.rerun()
#     #
#     # with control_col2.container():
#     #     if st.session_state.get('camera_active', False):
#     #         if st.button("Stop Camera", key="stop_cam"):
#     #             st.session_state.camera_active = False
#     #             st.rerun()
#
#     if st.session_state.get('camera_active', False):
#         status_placeholder.info("Live detection active - processing frames...")
#         cap = None
#         try:
#             cap = cv2.VideoCapture(cam_choice)
#             if not cap.isOpened():
#                 status_placeholder.error(f"Failed to open camera {cam_choice}")
#                 st.session_state.camera_active = False
#                 return
#
#             cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
#             cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
#
#             frame_skip = 2
#             frame_count = 0
#             last_update_time = time.time()
#
#             while cap.isOpened() and st.session_state.camera_active:
#                 ret, frame = cap.read()
#                 if not ret:
#                     status_placeholder.error("Failed to capture frame")
#                     time.sleep(0.5)
#                     continue
#
#                 frame_count += 1
#                 st.session_state.last_frame = frame.copy()
#
#                 if frame_count % frame_skip == 0:
#                     results, output = process_image(frame, model, confidence, class_filter, co2_data)
#
#                     if results is not None:
#                         frame_placeholder.image(output, channels="RGB", use_column_width=True)#, use_container_width=True)
#
#                         current_time = time.time()
#                         if current_time - last_update_time > 1:
#                             emissions_df, total_co2 = calculate_emissions(results, model, co2_data)
#                             st.session_state.emissions_df = emissions_df
#                             st.session_state.total_co2 = total_co2
#                             last_update_time = current_time
#
#                             if not emissions_df.empty:
#                                 emissions_table.dataframe(
#                                     emissions_df.style.format({
#                                         'Unit Weight (kg)': '{:.3f}',
#                                         'Total Weight (kg)': '{:.3f}',
#                                         'CO₂ Emissions (kg)': '{:.3f}'
#                                     }),
#                                     use_container_width=True
#                                 )
#                                 emissions_metric.metric("Current CO₂ Impact",
#                                                         f"{total_co2:.2f} kg CO₂ eq")
#                     else:
#                         display_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#                         frame_placeholder.image(display_frame, channels="RGB", use_column_width=True)#, use_container_width=True)
#
#                 time.sleep(0.03)
#
#         except Exception as e:
#             status_placeholder.error(f"Camera error: {str(e)}")
#             st.session_state.camera_active = False
#
#         finally:
#             if cap and cap.isOpened():
#                 cap.release()
#
#     if not st.session_state.get('camera_active', False) and st.session_state.get('emissions_df') is not None:
#         if not st.session_state.emissions_df.empty:
#             status_placeholder.success("Camera stopped - results frozen")
#
#             if st.session_state.last_frame is not None:
#                 results, output = process_image(st.session_state.last_frame, model, confidence, class_filter, co2_data)
#                 if output is not None:
#                     frame_placeholder.image(output, channels="RGB", use_column_width=True)#, use_container_width=True)
#
#             emissions_table.dataframe(
#                 st.session_state.emissions_df.style.format({
#                     'Unit Weight (kg)': '{:.3f}',
#                     'Total Weight (kg)': '{:.3f}',
#                     'CO₂ Emissions (kg)': '{:.3f}'
#                 }),
#                 use_container_width=True
#             )
#             emissions_metric.metric("Final CO₂ Impact",
#                                     f"{st.session_state.total_co2:.2f} kg CO₂ eq")
#             export_placeholder.markdown("### Export Results")
#             export_data(st.session_state.emissions_df)
#
# def handle_capture_process(frame_placeholder, status_placeholder, model, confidence,
#                          class_filter, co2_data, cam_choice):
#
#
#
#     # # Fixed control container
#     # control_container = st.container()
#     #
#     # with control_container:
#     #
#     #     # Camera control buttons
#     #     col1, col2, _ = st.columns([1,1,6])
#     #     with col1:
#     #         if st.button("Start Camera Feed", key="start_cam"):
#     #             st.session_state.camera_active = True
#     #     with col2:
#     #         if st.button("Stop Camera", key="stop_cam"):
#     #             st.session_state.camera_active = False
#
#     """Handles frame capture and processing workflow"""
#     capture_btn = st.empty()
#
#     if st.session_state.get('camera_active', False):
#         status_placeholder.info("Camera active - ready to capture")
#         cap = None
#         try:
#             # Initialize camera with error handling
#             cap = cv2.VideoCapture(cam_choice)
#             if not cap.isOpened():
#                 status_placeholder.error(f"Failed to open camera {cam_choice}")
#                 st.session_state.camera_active = False
#                 return
#
#             # Configure camera resolution
#             cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
#             cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
#
#             # Capture button with visual feedback
#             if capture_btn.button("📸 Capture Frame", key="capture_btn_cp"):
#                 ret, frame = cap.read()
#                 if ret:
#                     st.session_state.captured_frame = frame.copy()
#                     st.session_state.camera_active = False
#                 else:
#                     status_placeholder.error("Failed to capture frame")
#                     time.sleep(0.5)
#
#             # Camera feed display loop
#             while cap.isOpened() and st.session_state.camera_active:
#                 ret, frame = cap.read()
#                 if not ret:
#                     status_placeholder.error("Failed to read from camera")
#                     time.sleep(0.5)
#                     continue
#
#                 display_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#                 frame_placeholder.image(display_frame,
#                                         channels="RGB",
#                                         use_column_width=True)#, use_container_width=True)
#                 time.sleep(0.03)  # Prevent UI freeze
#
#         except Exception as e:
#             status_placeholder.error(f"Camera error: {str(e)}")
#             st.session_state.camera_active = False
#
#         finally:
#             # Ensure camera resource cleanup
#             if cap and cap.isOpened():
#                 cap.release()
#
#     # Process captured frame after camera stops
#     if not st.session_state.get('camera_active', False) and \
#             st.session_state.get('captured_frame') is not None:
#
#         status_placeholder.info("Processing captured frame...")
#         frame = st.session_state.captured_frame
#
#         # Process image through model
#         results, output = process_image(frame, model, confidence, class_filter, co2_data)
#
#         if results is not None and output is not None:
#             col1, col2 = st.columns(2)
#             with col1:
#                 st.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB),
#                          caption="Captured Frame",
#                          use_column_width=True)#, use_container_width=True)
#             with col2:
#                 st.image(output,
#                          caption="Processed Result",
#                          use_column_width=True)#, use_container_width=True)
#
#             # Generate emissions report
#             emissions_df, total_co2 = calculate_emissions(results, model, co2_data)
#             if not emissions_df.empty:
#                 st.subheader("Carbon Emissions Report")
#                 st.dataframe(
#                     emissions_df.style.format({
#                         'Unit Weight (kg)': '{:.3f}',
#                         'Total Weight (kg)': '{:.3f}',
#                         'CO₂ Emissions (kg)': '{:.3f}'
#                     }),
#                     use_container_width=True
#                 )
#
#                 # Export functionality
#                 st.markdown("### Export Results")
#                 export_data(emissions_df)
#
#                 st.metric("Total CO₂ Impact",
#                           f"{total_co2:.2f} kg CO₂ eq")
#
#             # Image download
#             buf = io.BytesIO()
#             Image.fromarray(output).save(buf, format="PNG")
#             st.download_button("Download Result", buf.getvalue(),
#                                "processed_image.png", "image/png")
#
#             # Reset mechanism
#             if st.button("Clear and return to camera", key="clear_capture"):
#                 st.session_state.captured_frame = None
#                 st.rerun()
