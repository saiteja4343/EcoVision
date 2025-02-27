import pandas as pd
import streamlit as st
from src.config import PATHS


@st.cache_data
def load_co2_data():
    """Load and validate CO2 emissions data"""
    try:
        co2_df = pd.read_excel(PATHS['data'])

        # Validate required columns
        required_columns = ['Foodstuff', 'avg_weight', 'CO2 Footprint (kg CO2 eq/kg food)']
        if not all(col in co2_df.columns for col in required_columns):
            st.error("Missing required columns in Excel file")
            st.stop()

        return co2_df

    except FileNotFoundError:
        st.error(f"Data file not found at {PATHS['data']}")
        st.stop()

    except Exception as e:
        st.error(f"Error loading Excel file: {str(e)}")
        st.stop()
