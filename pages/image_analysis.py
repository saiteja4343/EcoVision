# pages/image_analysis.py
import streamlit as st
import cv2
import numpy as np
from PIL import Image
import io

from src.logic.image_processing import process_image
from src.logic.emissions_calculator import calculate_emissions, export_data

def image_analysis_page(model, confidence, class_filter, co2_data, detection_type):
    st.header("Image Analysis")
    
    # Removed: Mode info boxes and advanced features description

    uploaded_file = st.file_uploader("Upload Food Image", type=["jpg", "png", "jpeg"])

    if uploaded_file:
        img = cv2.imdecode(np.frombuffer(uploaded_file.read(), np.uint8), 1)
        
        # Simple processing message
        with st.spinner("Processing image..."):
            results, output = process_image(img, model, confidence, class_filter, co2_data, detection_type)

        if results and output is not None:
            col1, col2 = st.columns(2)

            with col1:
                st.image(cv2.cvtColor(img, cv2.COLOR_BGR2RGB),
                        caption="Original Image", use_column_width=True)

            with col2:
                st.image(output, caption="Processed Result", use_column_width=True)

            # Emissions report
            emissions_df, total_co2 = calculate_emissions(results, model, co2_data)

            if not emissions_df.empty:
                st.subheader("🌱 Carbon Emissions Report")
                
                # Removed: Detection method info
                # Removed: Method-specific information
                
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

                export_data(emissions_df, detection_type)
                st.metric("Total CO₂ Impact", f"{total_co2:.2f} kg CO₂ eq")

                # Download processed image
                buf = io.BytesIO()
                Image.fromarray(output).save(buf, format="PNG")
                st.download_button("📥 Download Result", buf.getvalue(),
                                 "processed_image.png", "image/png")
