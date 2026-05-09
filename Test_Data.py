"""
Testing and Prediction Module for Corrosion Rate Models
"""

import pandas as pd
import numpy as np
import joblib

# Load saved models and scaler
try:
    model_general = joblib.load('model_general_corrosion.pkl')
    model_pitting = joblib.load('model_pitting_corrosion.pkl')
    scaler = joblib.load('scaler.pkl')
    print("✓ Models loaded successfully!")
except FileNotFoundError:
    print("ERROR: Models not found. Please run Model.py first to train the models.")
    exit()

# Load original data for reference
df = pd.read_excel("Corrosion_AI_Industrial.xlsx", sheet_name="AI_Model")
print(f"\nDataset loaded: {df.shape[0]} records")

FEATURE_COLUMNS = ['Temperature (F)', 'Pressure (psig)', 'Velocity (ft/s)', 'WaterCut', 'CO2 (%mol)', 'Chloride (ppm)', 'Sand (ppm)', '%wt-Gas']

def predict_corrosion(temperature_f, pressure_psig, velocity, water_cut, co2, chloride, sand, gas_fraction):
    """
    Predict general and pitting corrosion rates for given conditions.
    
    Parameters:
    -----------
    temperature_f : float
        Temperature in Fahrenheit
    pressure_psig : float
        Pressure in psig
    velocity : float
        Fluid velocity in ft/s
    water_cut : float
        Water cut percentage
    co2 : float
        CO2 concentration (%mol)
    chloride : float
        Chloride concentration in ppm
    sand : float
        Sand content in ppm
    gas_fraction : float
        Gas fraction (%wt-Gas)
    
    Returns:
    --------
    tuple : (general_corrosion_rate, pitting_corrosion_rate) in mpy
    """
    # Create DataFrame with proper column names to avoid sklearn warnings
    input_data = pd.DataFrame([[temperature_f, pressure_psig, velocity, water_cut, co2, chloride, sand, gas_fraction]], 
                             columns=FEATURE_COLUMNS)
    input_scaled = scaler.transform(input_data)
    
    general_rate = model_general.predict(input_scaled)[0]
    pitting_rate = model_pitting.predict(input_scaled)[0]
    
    return general_rate, pitting_rate
    """
    Predict general and pitting corrosion rates for given conditions.
    
    Parameters:
    -----------
    temperature : float
        Temperature in °C
    pressure : float
        Pressure in bar
    fluid_velocity : float
        Fluid velocity in m/s
    gas_fraction : float
        Gas fraction (0-1)
    chloride : float
        Chloride concentration in mg/L
    sand : float
        Sand content
    gas_fraction : float
        Gas fraction (%wt-Gas)
    
    Returns:
    --------
    tuple : (general_corrosion_rate, pitting_corrosion_rate) in mm/year
    """
    # Create DataFrame with proper column names to avoid sklearn warnings
    input_data = pd.DataFrame([[velocity, water_cut, co2, chloride, sand, gas_fraction]], 
                             columns=FEATURE_COLUMNS)
    input_scaled = scaler.transform(input_data)
    
    general_rate = model_general.predict(input_scaled)[0]
    pitting_rate = model_pitting.predict(input_scaled)[0]
    
    return general_rate, pitting_rate

def batch_predict(test_df):
    """
    Predict corrosion rates for multiple test cases.
    
    Parameters:
    -----------
    test_df : pd.DataFrame
        DataFrame with columns: Temperature (F), Pressure (psig), Velocity (ft/s), WaterCut, CO2 (%mol), Chloride (ppm), Sand (ppm), %wt-Gas
    
    Returns:
    --------
    pd.DataFrame : Original data with added predictions in mpy
    """
    predictions = []
    
    for idx, row in test_df.iterrows():
        general, pitting = predict_corrosion(
            temperature_f=row['Temperature (F)'],
            pressure_psig=row['Pressure (psig)'],
            velocity=row['Velocity (ft/s)'],
            water_cut=row['WaterCut'],
            co2=row['CO2 (%mol)'],
            chloride=row['Chloride (ppm)'],
            sand=row['Sand (ppm)'],
            gas_fraction=row['%wt-Gas']
        )
        predictions.append({
            'General_Corrosion_Rate': general,
            'Pitting_Corrosion_Rate': pitting
        })
    
    results = pd.concat([test_df.reset_index(drop=True), 
                        pd.DataFrame(predictions)], axis=1)
    return results

# ============ TEST CASES ============
print("\n" + "="*70)
print("TESTING CORROSION PREDICTION MODEL")
print("="*70)

# Test Case 1: Low corrosion conditions
print("\n[Test Case 1] Low Corrosion Conditions:")
print("-" * 50)
gen_rate1, pit_rate1 = predict_corrosion(
    temperature_f=77.0,
    pressure_psig=50.0,
    velocity=2.0, 
    water_cut=95.0, 
    co2=1.0,
    chloride=100, 
    sand=10,
    gas_fraction=5.0
)
print(f"Temperature: 77°F | Pressure: 50 psig | Velocity: 2.0 ft/s | Water Cut: 95% | CO2: 1.0")
print(f"Chloride: 100 mg/L | Sand: 10 | Gas: 5.0%")
print(f"→ General Corrosion Rate: {gen_rate1:.4f} mpy")
print(f"→ Pitting Corrosion Rate: {pit_rate1:.4f} mpy")

# Test Case 2: Medium corrosion conditions
print("\n[Test Case 2] Medium Corrosion Conditions:")
print("-" * 50)
gen_rate2, pit_rate2 = predict_corrosion(
    temperature_f=122.0,
    pressure_psig=100.0,
    velocity=3.0, 
    water_cut=98.0, 
    co2=2.5,
    chloride=500, 
    sand=50,
    gas_fraction=15.0
)
print(f"Temperature: 122°F | Pressure: 100 psig | Velocity: 3.0 ft/s | Water Cut: 98% | CO2: 2.5")
print(f"Chloride: 500 mg/L | Sand: 50 | Gas: 15.0%")
print(f"→ General Corrosion Rate: {gen_rate2:.4f} mpy")
print(f"→ Pitting Corrosion Rate: {pit_rate2:.4f} mpy")

# Test Case 3: High corrosion conditions (harsh environment)
print("\n[Test Case 3] High Corrosion Conditions (Harsh Environment):")
print("-" * 50)
gen_rate3, pit_rate3 = predict_corrosion(
    temperature_f=176.0,
    pressure_psig=200.0,
    velocity=4.0, 
    water_cut=99.5, 
    co2=5.0,
    chloride=1000, 
    sand=150,
    gas_fraction=30.0
)
print(f"Temperature: 176°F | Pressure: 200 psig | Velocity: 4.0 ft/s | Water Cut: 99.5% | CO2: 5.0")
print(f"Chloride: 1000 mg/L | Sand: 150 | Gas: 30.0%")
print(f"→ General Corrosion Rate: {gen_rate3:.4f} mpy")
print(f"→ Pitting Corrosion Rate: {pit_rate3:.4f} mpy")

# Test Case 4: Extreme conditions
print("\n[Test Case 4] Extreme Conditions:")
print("-" * 50)
gen_rate4, pit_rate4 = predict_corrosion(
    temperature_f=212.0,
    pressure_psig=500.0,
    velocity=6.0, 
    water_cut=99.8, 
    co2=8.0,
    chloride=2000, 
    sand=300,
    gas_fraction=50.0
)
print(f"Temperature: 212°F | Pressure: 500 psig | Velocity: 6.0 ft/s | Water Cut: 99.8% | CO2: 8.0")
print(f"Chloride: 2000 mg/L | Sand: 300 | Gas: 50.0%")
print(f"→ General Corrosion Rate: {gen_rate4:.4f} mpy")
print(f"→ Pitting Corrosion Rate: {pit_rate4:.4f} mpy")

# Summary comparison
print("\n" + "="*70)
print("SUMMARY COMPARISON")
print("="*70)
summary_df = pd.DataFrame({
    'Scenario': ['Low', 'Medium', 'High', 'Extreme'],
    'Temp (°F)': [77.0, 122.0, 176.0, 212.0],
    'Pressure (psig)': [50.0, 100.0, 200.0, 500.0],
    'Velocity (ft/s)': [2.0, 3.0, 4.0, 6.0],
    'General Corrosion (mpy)': [gen_rate1, gen_rate2, gen_rate3, gen_rate4],
    'Pitting Corrosion (mpy)': [pit_rate1, pit_rate2, pit_rate3, pit_rate4]
})
print(summary_df.to_string(index=False))

print("\n✓ Testing completed successfully!")
print("\nTo use the prediction function for your own data:")
print("  general_rate, pitting_rate = predict_corrosion(temp_f, pressure_psig, velocity, water_cut, co2, chloride, sand, gas_fraction)")
