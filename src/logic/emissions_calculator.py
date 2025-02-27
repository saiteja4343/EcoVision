import pandas as pd
import streamlit as st
import io
from datetime import datetime

def calculate_emissions(detections, model, co2_data):
    try:
        detection_counts = {}
        for box in detections.boxes:
            cls_name = model.names[int(box.cls)]
            matched = co2_data[co2_data['Foodstuff'].str.lower() == cls_name.lower()]
            if not matched.empty:
                detection_counts[cls_name] = detection_counts.get(cls_name, 0) + 1

        emissions = []
        total_co2 = 0
        for food, count in detection_counts.items():
            data = co2_data[co2_data['Foodstuff'].str.lower() == food.lower()]
            if not data.empty:
                avg_weight = data['avg_weight'].values[0]
                co2_per_kg = data['CO2 Footprint (kg CO2 eq/kg food)'].values[0]
                total_weight = avg_weight * count
                food_co2 = total_weight * co2_per_kg
                total_co2 += food_co2
                emissions.append({
                    "Food Item": food,
                    "Quantity": count,
                    "Unit Weight (kg)": avg_weight,
                    "Total Weight (kg)": total_weight,
                    "CO₂ Emissions (kg)": food_co2
                })
        return pd.DataFrame(emissions), total_co2
    except Exception as e:
        st.error(f"Emissions calculation error: {str(e)}")
        return pd.DataFrame(), 0

def export_data(df):
    """Export functionality with dropdown format selection"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Create columns layout
    col1, col2 = st.columns([1, 3])

    with col1:
        # Format selection dropdown
        export_format = st.selectbox(
            "Select export format:",
            ["CSV", "Excel"],
            index=0,
            help="Choose file format for download"
        )

    with col2:
        # Add a spacer to align with the selectbox label
        st.write('<div style="height: 30px;"></div>', unsafe_allow_html=True)

        # Generate data based on selected format
        if export_format == "CSV":
            data = df.to_csv(index=False).encode('utf-8')
            file_name = f"emissions_{timestamp}.csv"
            mime_type = "text/csv"
        else:
            output = io.BytesIO()
            with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
                df.to_excel(writer, index=False, sheet_name='Emissions')
                writer.close()
            data = output.getvalue()
            file_name = f"emissions_{timestamp}.xlsx"
            mime_type = "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"

        # Single download button that adapts to selection
        st.download_button(
            label=f"Download {export_format}",
            data=data,
            file_name=file_name,
            mime=mime_type,
            key=f"download_{timestamp}"
        )
