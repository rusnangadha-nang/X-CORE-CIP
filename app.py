"""
X-CORE's Corrosion Prediction Web App
A Streamlit application for predicting general and pitting corrosion rates
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import warnings
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="X-CORE's Corrosion Rate Predictor",
    page_icon="⚗️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .prediction-box {
        background-color: #f0f2f6;
        padding: 20px;
        border-radius: 10px;
        border-left: 5px solid #1f77b4;
        margin: 10px 0;
    }
    .input-section {
        background-color: #ffffff;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        margin: 10px 0;
    }
    .result-general {
        color: #2e7d32;
        font-size: 1.2rem;
        font-weight: bold;
    }
    .result-pitting {
        color: #d32f2f;
        font-size: 1.2rem;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

# Load models and scaler
@st.cache_resource
def load_models():
    """Load the trained models and scaler"""
    try:
        model_general = joblib.load('model_general_corrosion.pkl')
        model_pitting = joblib.load('model_pitting_corrosion.pkl')
        scaler = joblib.load('scaler.pkl')
        return model_general, model_pitting, scaler
    except FileNotFoundError:
        st.error("❌ Model files not found. Please run Model.py first to train the models.")
        st.stop()

# Load models
model_general, model_pitting, scaler = load_models()

# Feature columns (must match training data)
FEATURE_COLUMNS = ['Temperature (F)', 'Pressure (psig)', 'Velocity (ft/s)',
                   'WaterCut', 'CO2 (%mol)', 'Chloride (ppm)', 'Sand (ppm)', '%wt-Gas']

def predict_corrosion(temperature_f, pressure_psig, velocity, water_cut, co2, chloride, sand, gas_fraction):
    """
    Predict general and pitting corrosion rates

    Parameters:
    -----------
    temperature_f : float - Temperature in °F
    pressure_psig : float - Pressure in psig
    velocity : float - Fluid velocity in ft/s
    water_cut : float - Water cut percentage
    co2 : float - CO2 concentration in %mol
    chloride : float - Chloride concentration in ppm
    sand : float - Sand content in ppm
    gas_fraction : float - Gas fraction (%wt-Gas)

    Returns:
    --------
    tuple : (general_corrosion_rate, pitting_corrosion_rate) in mpy
    """
    # Create DataFrame with proper column names
    input_data = pd.DataFrame([[temperature_f, pressure_psig, velocity, water_cut,
                               co2, chloride, sand, gas_fraction]], columns=FEATURE_COLUMNS)

    # Scale the input
    input_scaled = scaler.transform(input_data)

    # Make predictions
    general_rate = model_general.predict(input_scaled)[0]
    pitting_rate = model_pitting.predict(input_scaled)[0]

    return general_rate, pitting_rate

def main():
    # Header
    st.markdown('<h1 class="main-header">⚗️ X-CORE\'s Corrosion Rate Predictor</h1>', unsafe_allow_html=True)
    st.markdown("---")

    # Description
    st.markdown("""
    **Predict General and Pitting Corrosion Rates** using machine learning models trained on industrial corrosion data.

    This tool uses Gradient Boosting Regression to predict corrosion rates based on:
    - Temperature (°F)
    - Pressure (psig)
    - Fluid Velocity (ft/s)
    - Water Cut (%)
    - CO2 Concentration (%mol)
    - Chloride Concentration (ppm)
    - Sand Content (ppm)
    - Gas Fraction (%wt-Gas)
    """)

    # Create two columns for layout
    col1, col2 = st.columns([1, 1])

    with col1:
        st.markdown('<div class="input-section">', unsafe_allow_html=True)
        st.subheader("🔧 Input Parameters")

        # Input fields with validation
        temperature = st.number_input(
            "Temperature (°F)",
            min_value=32.0,
            max_value=500.0,
            value=150.0,
            step=1.0,
            help="Operating temperature in Fahrenheit"
        )

        pressure = st.number_input(
            "Pressure (psig)",
            min_value=0.0,
            max_value=10000.0,
            value=500.0,
            step=10.0,
            help="Operating pressure in psig"
        )

        velocity = st.number_input(
            "Fluid Velocity (ft/s)",
            min_value=0.1,
            max_value=50.0,
            value=5.0,
            step=0.1,
            help="Fluid velocity in feet per second"
        )

        water_cut = st.slider(
            "Water Cut (%)",
            min_value=0.0,
            max_value=100.0,
            value=90.0,
            step=0.1,
            help="Percentage of water in the fluid"
        )

        st.markdown('</div>', unsafe_allow_html=True)

    with col2:
        st.markdown('<div class="input-section">', unsafe_allow_html=True)
        st.subheader("🧪 Chemical Parameters")

        co2 = st.number_input(
            "CO2 Concentration (%mol)",
            min_value=0.0,
            max_value=50.0,
            value=2.0,
            step=0.1,
            help="CO2 concentration in mole percent"
        )

        chloride = st.number_input(
            "Chloride Concentration (ppm)",
            min_value=0,
            max_value=50000,
            value=1000,
            step=100,
            help="Chloride concentration in parts per million"
        )

        sand = st.number_input(
            "Sand Content (ppm)",
            min_value=0,
            max_value=10000,
            value=100,
            step=10,
            help="Sand content in parts per million"
        )

        gas_fraction = st.slider(
            "Gas Fraction (%wt-Gas)",
            min_value=0.0,
            max_value=100.0,
            value=15.0,
            step=0.1,
            help="Weight percentage of gas in the fluid"
        )

        st.markdown('</div>', unsafe_allow_html=True)

    # Prediction button
    st.markdown("---")
    col_center = st.columns([1, 2, 1])[1]
    with col_center:
        predict_button = st.button("🔮 Predict Corrosion Rates", type="primary", use_container_width=True)

    # Make prediction when button is clicked
    if predict_button:
        with st.spinner("Analyzing corrosion conditions..."):
            try:
                # Get predictions
                general_rate, pitting_rate = predict_corrosion(
                    temperature_f=temperature,
                    pressure_psig=pressure,
                    velocity=velocity,
                    water_cut=water_cut,
                    co2=co2,
                    chloride=chloride,
                    sand=sand,
                    gas_fraction=gas_fraction
                )

                # Display results
                st.markdown("---")
                st.subheader("📊 Prediction Results")

                # General corrosion result
                st.markdown('<div class="prediction-box">', unsafe_allow_html=True)
                st.markdown(f'<p class="result-general">🟢 General Corrosion Rate: {general_rate:.3f} mpy</p>', unsafe_allow_html=True)

                # Risk assessment for general corrosion
                if general_rate < 1.0:
                    st.success("✅ Low corrosion risk - Good conditions")
                elif general_rate < 5.0:
                    st.warning("⚠️ Moderate corrosion risk - Monitor closely")
                else:
                    st.error("🚨 High corrosion risk - Take immediate action")
                st.markdown('</div>', unsafe_allow_html=True)

                # Pitting corrosion result
                st.markdown('<div class="prediction-box">', unsafe_allow_html=True)
                st.markdown(f'<p class="result-pitting">🔴 Pitting Corrosion Rate: {pitting_rate:.3f} mpy</p>', unsafe_allow_html=True)

                # Risk assessment for pitting corrosion
                if pitting_rate < 2.0:
                    st.success("✅ Low pitting risk - Good conditions")
                elif pitting_rate < 10.0:
                    st.warning("⚠️ Moderate pitting risk - Monitor closely")
                else:
                    st.error("🚨 High pitting risk - Take immediate action")
                st.markdown('</div>', unsafe_allow_html=True)

                # Summary table
                st.subheader("📋 Summary")
                summary_data = {
                    "Parameter": ["Temperature", "Pressure", "Velocity", "Water Cut", "CO2", "Chloride", "Sand", "Gas Fraction"],
                    "Value": [f"{temperature} °F", f"{pressure} psig", f"{velocity} ft/s", f"{water_cut}%",
                             f"{co2} %mol", f"{chloride} ppm", f"{sand} ppm", f"{gas_fraction}%"],
                    "General CR": [f"{general_rate:.3f} mpy", "", "", "", "", "", "", ""],
                    "Pitting CR": [f"{pitting_rate:.3f} mpy", "", "", "", "", "", "", ""]
                }
                summary_df = pd.DataFrame(summary_data)
                st.table(summary_df)

            except Exception as e:
                st.error(f"❌ Prediction failed: {str(e)}")
                st.info("Please check your input values and try again.")

    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #666;'>
        <p><strong>X-CORE's Corrosion Rate Predictor v1.0</strong></p>
        <p>Built with Streamlit and Gradient Boosting Regression</p>
        <p>⚠️ This tool provides predictions based on trained models. Always consult corrosion experts for critical applications.</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()