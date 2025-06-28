# src/logic/emissions_calculator.py
import pandas as pd
import streamlit as st
import io
from datetime import datetime

def calculate_emissions(detections, model, co2_data):
    """Calculate emissions for both detection and segmentation results"""
    try:
        # Check if this is segmentation with estimated weights
        if hasattr(detections, 'estimated_weights') and detections.estimated_weights:
            return calculate_advanced_segmentation_emissions(detections, co2_data)
        else:
            return calculate_detection_emissions(detections, model, co2_data)
            
    except Exception as e:
        st.error(f"Emissions calculation error: {str(e)}")
        return pd.DataFrame(), 0

def calculate_detection_emissions(detections, model, co2_data):
    """Calculate emissions for bounding box detection using avg_weight"""
    try:
        detection_counts = {}
        
        # Count detections by class
        for box in detections.boxes:
            cls_name = model.names[int(box.cls)]
            matched = co2_data[co2_data['Foodstuff'].str.lower() == cls_name.lower()]
            if not matched.empty:
                detection_counts[cls_name] = detection_counts.get(cls_name, 0) + 1
        
        # Calculate emissions using avg_weight
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
                    "CO2 emissions (kg)- class": co2_per_kg,
                    "CO₂ Emissions (kg)": food_co2
                    # Removed: Method column
                })
        
        return pd.DataFrame(emissions), total_co2
        
    except Exception as e:
        st.error(f"Detection emissions calculation error: {str(e)}")
        return pd.DataFrame(), 0

def calculate_advanced_segmentation_emissions(detections, co2_data):
    """Calculate emissions for segmentation using total weight of all masks per class"""
    try:
        emissions = []
        total_co2 = 0
        
        # Use estimated weights from advanced segmentation
        for food_name, weight_info in detections.estimated_weights.items():
            if food_name == '_individual_weights':
                continue  # Skip the individual weights metadata
                
            data = co2_data[co2_data['Foodstuff'].str.lower() == food_name.lower()]
            if not data.empty:
                co2_per_kg = data['CO2 Footprint (kg CO2 eq/kg food)'].values[0]
                
                # Get total weight in kg (sum of all individual masks for this class)
                total_weight_kg = weight_info['weight_kg']  # This is already the sum of all masks
                count = weight_info['count']  # Number of masks for this class
                
                food_co2 = total_weight_kg * co2_per_kg
                total_co2 += food_co2
                
                # For segmentation: no unit weight column, just total weight
                emissions.append({
                    "Food Item": food_name,
                    "Quantity": count,
                    "Total Weight (kg)": total_weight_kg,  # Total weight of all masks
                    "CO2 emissions (kg)- class": co2_per_kg,
                    "CO₂ Emissions (kg)": food_co2
                    # Removed: Method column
                })
        
        return pd.DataFrame(emissions), total_co2
        
    except Exception as e:
        st.error(f"Advanced segmentation emissions calculation error: {str(e)}")
        return pd.DataFrame(), 0

def export_data(df, detection_type="Bounding boxes"):
    """Export functionality with mode suffix in filename"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Determine mode suffix
    mode_suffix = "_seg" if detection_type == "Segmentation" else "_bb"
    
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
        st.write('', unsafe_allow_html=True)
    
    # Generate data based on selected format
    if export_format == "CSV":
        data = df.to_csv(index=False).encode('utf-8')
        file_name = f"emissions_{timestamp}{mode_suffix}.csv"
        mime_type = "text/csv"
    else:
        output = io.BytesIO()
        with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
            df.to_excel(writer, index=False, sheet_name='Emissions')
            writer.close()
        data = output.getvalue()
        file_name = f"emissions_{timestamp}{mode_suffix}.xlsx"
        mime_type = "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    
    # Single download button that adapts to selection
    st.download_button(
        label=f"Download {export_format}",
        data=data,
        file_name=file_name,
        mime=mime_type,
        key=f"download_{timestamp}"
    )
