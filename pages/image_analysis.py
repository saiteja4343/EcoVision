import streamlit as st
import cv2
import numpy as np
from PIL import Image
import io
from src.logic.image_processing import process_image
from src.logic.emissions_calculator import calculate_emissions, export_data


def image_analysis_page(model, confidence, class_filter, co2_data):
    st.header("Image Analysis")
    uploaded_file = st.file_uploader("Upload Food Image", type=["jpg", "png", "jpeg"])

    if uploaded_file:
        img = cv2.imdecode(np.frombuffer(uploaded_file.read(), np.uint8), 1)
        results, output = process_image(img, model, confidence, class_filter, co2_data)

        if results and output is not None:
            col1, col2 = st.columns(2)
            with col1:
                st.image(cv2.cvtColor(img, cv2.COLOR_BGR2RGB),
                         caption="Original Image", use_container_width=True)
            with col2:
                st.image(output, caption="Processed Result", use_container_width=True)

            # Emissions report
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
                export_data(emissions_df)
                st.metric("Total CO₂ Impact", f"{total_co2:.2f} kg CO₂ eq")

            # Download processed image
            buf = io.BytesIO()
            Image.fromarray(output).save(buf, format="PNG")
            st.download_button("Download Result", buf.getvalue(),
                               "processed_image.png", "image/png")
