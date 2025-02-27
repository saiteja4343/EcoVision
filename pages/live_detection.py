# pages/live_detection.py
import cv2
import time
import streamlit as st
from datetime import datetime
from PIL import Image
import io
from src.logic.image_processing import process_image
from src.logic.emissions_calculator import calculate_emissions, export_data

def live_detection_page(model, confidence, class_filter, co2_data, cam_choice):
    st.header("Live Camera Analysis")
    processing_mode = st.radio("Mode:", ["Live Detection", "Capture & Process"], horizontal=True)

    # Create fixed layout containers
    frame_placeholder = st.empty()
    status_placeholder = st.empty()

    if processing_mode == "Live Detection":
        handle_live_detection(frame_placeholder, status_placeholder, model, 
                            confidence, class_filter, co2_data, cam_choice)
    else:
        handle_capture_process(frame_placeholder, status_placeholder, model,
                             confidence, class_filter, co2_data, cam_choice)

def handle_live_detection(frame_placeholder, status_placeholder, model, confidence,
                        class_filter, co2_data, cam_choice):
    emissions_table = st.empty()
    emissions_metric = st.empty()
    export_placeholder = st.empty()

    # Camera control buttons
    col1, col2 = st.columns(2)
    with col1:
        if st.button("Start Camera Feed", key="start_cam"):
            st.session_state.camera_active = True
    with col2:
        if st.button("Stop Camera", key="stop_cam"):
            st.session_state.camera_active = False

    if st.session_state.get('camera_active', False):
        status_placeholder.info("Live detection active - processing frames...")
        cap = None
        try:
            cap = cv2.VideoCapture(cam_choice)
            if not cap.isOpened():
                status_placeholder.error(f"Failed to open camera {cam_choice}")
                st.session_state.camera_active = False
                return

            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

            frame_skip = 2
            frame_count = 0
            last_update_time = time.time()

            while cap.isOpened() and st.session_state.camera_active:
                ret, frame = cap.read()
                if not ret:
                    status_placeholder.error("Failed to capture frame")
                    time.sleep(0.5)
                    continue

                frame_count += 1
                st.session_state.last_frame = frame.copy()

                if frame_count % frame_skip == 0:
                    results, output = process_image(frame, model, confidence, class_filter, co2_data)

                    if results is not None:
                        frame_placeholder.image(output, channels="RGB", use_container_width=True)

                        current_time = time.time()
                        if current_time - last_update_time > 1:
                            emissions_df, total_co2 = calculate_emissions(results, model, co2_data)
                            st.session_state.emissions_df = emissions_df
                            st.session_state.total_co2 = total_co2
                            last_update_time = current_time

                            if not emissions_df.empty:
                                emissions_table.dataframe(
                                    emissions_df.style.format({
                                        'Unit Weight (kg)': '{:.3f}',
                                        'Total Weight (kg)': '{:.3f}',
                                        'CO₂ Emissions (kg)': '{:.3f}'
                                    }),
                                    use_container_width=True
                                )
                                emissions_metric.metric("Current CO₂ Impact",
                                                        f"{total_co2:.2f} kg CO₂ eq")
                    else:
                        display_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        frame_placeholder.image(display_frame, channels="RGB", use_container_width=True)

                time.sleep(0.03)

        except Exception as e:
            status_placeholder.error(f"Camera error: {str(e)}")
            st.session_state.camera_active = False

        finally:
            if cap and cap.isOpened():
                cap.release()

    if not st.session_state.get('camera_active', False) and st.session_state.get('emissions_df') is not None:
        if not st.session_state.emissions_df.empty:
            status_placeholder.success("Camera stopped - results frozen")

            if st.session_state.last_frame is not None:
                results, output = process_image(st.session_state.last_frame, model, confidence, class_filter, co2_data)
                if output is not None:
                    frame_placeholder.image(output, channels="RGB", use_container_width=True)

            emissions_table.dataframe(
                st.session_state.emissions_df.style.format({
                    'Unit Weight (kg)': '{:.3f}',
                    'Total Weight (kg)': '{:.3f}',
                    'CO₂ Emissions (kg)': '{:.3f}'
                }),
                use_container_width=True
            )
            emissions_metric.metric("Final CO₂ Impact",
                                    f"{st.session_state.total_co2:.2f} kg CO₂ eq")
            export_placeholder.markdown("### Export Results")
            export_data(st.session_state.emissions_df)

def handle_capture_process(frame_placeholder, status_placeholder, model, confidence,
                         class_filter, co2_data, cam_choice):

    """Handles frame capture and processing workflow"""
    capture_btn = st.empty()


    # Camera control buttons
    col1, col2 = st.columns(2)
    with col1:
        if st.button("Start Camera Feed", key="start_cam"):
            st.session_state.camera_active = True
    with col2:
        if st.button("Stop Camera", key="stop_cam"):
            st.session_state.camera_active = False

    if st.session_state.get('camera_active', False):
        status_placeholder.info("Camera active - ready to capture")
        cap = None
        try:
            # Initialize camera with error handling
            cap = cv2.VideoCapture(cam_choice)
            if not cap.isOpened():
                status_placeholder.error(f"Failed to open camera {cam_choice}")
                st.session_state.camera_active = False
                return

            # Configure camera resolution
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

            # Capture button with visual feedback
            if capture_btn.button("📸 Capture Frame", key="capture_btn_cp"):
                ret, frame = cap.read()
                if ret:
                    st.session_state.captured_frame = frame.copy()
                    st.session_state.camera_active = False
                else:
                    status_placeholder.error("Failed to capture frame")
                    time.sleep(0.5)

            # Camera feed display loop
            while cap.isOpened() and st.session_state.camera_active:
                ret, frame = cap.read()
                if not ret:
                    status_placeholder.error("Failed to read from camera")
                    time.sleep(0.5)
                    continue

                display_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame_placeholder.image(display_frame,
                                        channels="RGB",
                                        use_container_width=True)
                time.sleep(0.03)  # Prevent UI freeze

        except Exception as e:
            status_placeholder.error(f"Camera error: {str(e)}")
            st.session_state.camera_active = False

        finally:
            # Ensure camera resource cleanup
            if cap and cap.isOpened():
                cap.release()

    # Process captured frame after camera stops
    if not st.session_state.get('camera_active', False) and \
            st.session_state.get('captured_frame') is not None:

        status_placeholder.info("Processing captured frame...")
        frame = st.session_state.captured_frame

        # Process image through model
        results, output = process_image(frame, model, confidence, class_filter, co2_data)

        if results is not None and output is not None:
            col1, col2 = st.columns(2)
            with col1:
                st.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB),
                         caption="Captured Frame",
                         use_container_width=True)
            with col2:
                st.image(output,
                         caption="Processed Result",
                         use_container_width=True)

            # Generate emissions report
            emissions_df, total_co2 = calculate_emissions(results, model, co2_data)
            if not emissions_df.empty:
                st.subheader("Carbon Emissions Report")
                st.dataframe(
                    emissions_df.style.format({
                        'Unit Weight (kg)': '{:.3f}',
                        'Total Weight (kg)': '{:.3f}',
                        'CO₂ Emissions (kg)': '{:.3f}'
                    }),
                    use_container_width=True
                )

                # Export functionality
                st.markdown("### Export Results")
                export_data(emissions_df)

                st.metric("Total CO₂ Impact",
                          f"{total_co2:.2f} kg CO₂ eq")

            # Image download
            buf = io.BytesIO()
            Image.fromarray(output).save(buf, format="PNG")
            st.download_button("Download Result", buf.getvalue(),
                               "processed_image.png", "image/png")

            # Reset mechanism
            if st.button("Clear and return to camera", key="clear_capture"):
                st.session_state.captured_frame = None
                st.rerun()
