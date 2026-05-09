"""
Predictive Model for General and Pitting Corrosion Rates
Features: Temperature, Pressure, Fluid Velocity, Gas Fraction, Chloride, CO2 Fraction, Sand
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import joblib
import warnings
warnings.filterwarnings('ignore')

# Load data
print("Loading corrosion data...")
df = pd.read_excel("Corrosion_AI_Industrial.xlsx", sheet_name="AI_Model")
print(f"Dataset shape: {df.shape}")
print(f"\nFirst few rows:")
print(df.head())
print(f"\nColumn names: {df.columns.tolist()}")
print(f"\nData info:")
print(df.info())

# Define feature columns and target columns (updated to match your Excel file)
FEATURE_COLUMNS = ['Temperature (F)', 'Pressure (psig)', 'Velocity (ft/s)', 'WaterCut', 'CO2 (%mol)', 'Chloride (ppm)', 'Sand (ppm)', '%wt-Gas']
TARGET_GENERAL = 'Measured_General CR (mpy)'  # General corrosion rate column
TARGET_PITTING = 'Measured_Pitting CR (mpy)'  # Pitting corrosion rate column

# Check if columns exist and adjust if needed
available_columns = df.columns.tolist()
print(f"\nAvailable columns in dataset: {available_columns}")

# Prepare data - remove rows with missing values in key columns
df_clean = df.dropna(subset=FEATURE_COLUMNS + [TARGET_GENERAL, TARGET_PITTING])
print(f"Clean dataset shape after removing NaN: {df_clean.shape}")

if len(df_clean) == 0:
    print("ERROR: No complete data available. Please check your Excel file structure.")
else:
    X = df_clean[FEATURE_COLUMNS]
    y_general = df_clean[TARGET_GENERAL]
    y_pitting = df_clean[TARGET_PITTING]
    
    # Split data into training and testing sets
    X_train, X_test, y_gen_train, y_gen_test, y_pit_train, y_pit_test = train_test_split(
        X, y_general, y_pitting, test_size=0.2, random_state=42
    )
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    print(f"\n--- Training General Corrosion Rate Model ---")
    # Model 1: General Corrosion Rate
    model_general = GradientBoostingRegressor(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=5,
        random_state=42,
        verbose=1
    )
    model_general.fit(X_train_scaled, y_gen_train)
    
    # Predictions
    y_gen_pred_train = model_general.predict(X_train_scaled)
    y_gen_pred_test = model_general.predict(X_test_scaled)
    
    # Evaluation
    gen_r2_train = r2_score(y_gen_train, y_gen_pred_train)
    gen_r2_test = r2_score(y_gen_test, y_gen_pred_test)
    gen_rmse_test = np.sqrt(mean_squared_error(y_gen_test, y_gen_pred_test))
    gen_mae_test = mean_absolute_error(y_gen_test, y_gen_pred_test)
    
    print(f"General Corrosion - Train R²: {gen_r2_train:.4f}")
    print(f"General Corrosion - Test R²: {gen_r2_test:.4f}")
    print(f"General Corrosion - Test RMSE: {gen_rmse_test:.4f}")
    print(f"General Corrosion - Test MAE: {gen_mae_test:.4f}")
    
    print(f"\n--- Training Pitting Corrosion Rate Model ---")
    # Model 2: Pitting Corrosion Rate
    model_pitting = GradientBoostingRegressor(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=5,
        random_state=42,
        verbose=1
    )
    model_pitting.fit(X_train_scaled, y_pit_train)
    
    # Predictions
    y_pit_pred_train = model_pitting.predict(X_train_scaled)
    y_pit_pred_test = model_pitting.predict(X_test_scaled)
    
    # Evaluation
    pit_r2_train = r2_score(y_pit_train, y_pit_pred_train)
    pit_r2_test = r2_score(y_pit_test, y_pit_pred_test)
    pit_rmse_test = np.sqrt(mean_squared_error(y_pit_test, y_pit_pred_test))
    pit_mae_test = mean_absolute_error(y_pit_test, y_pit_pred_test)
    
    print(f"Pitting Corrosion - Train R²: {pit_r2_train:.4f}")
    print(f"Pitting Corrosion - Test R²: {pit_r2_test:.4f}")
    print(f"Pitting Corrosion - Test RMSE: {pit_rmse_test:.4f}")
    print(f"Pitting Corrosion - Test MAE: {pit_mae_test:.4f}")
    
    # Feature importance
    print(f"\n--- Feature Importance (General Corrosion) ---")
    gen_importance = pd.DataFrame({
        'Feature': FEATURE_COLUMNS,
        'Importance': model_general.feature_importances_
    }).sort_values('Importance', ascending=False)
    print(gen_importance)
    
    print(f"\n--- Feature Importance (Pitting Corrosion) ---")
    pit_importance = pd.DataFrame({
        'Feature': FEATURE_COLUMNS,
        'Importance': model_pitting.feature_importances_
    }).sort_values('Importance', ascending=False)
    print(pit_importance)
    
    # Save models
    joblib.dump(model_general, 'model_general_corrosion.pkl')
    joblib.dump(model_pitting, 'model_pitting_corrosion.pkl')
    joblib.dump(scaler, 'scaler.pkl')
    print("\n✓ Models saved successfully!")
    
    def predict_corrosion(temperature_f, pressure_psig, velocity, water_cut, co2, chloride, sand, gas_fraction):
        """Predict general and pitting corrosion rates"""
        # Create DataFrame with proper column names to avoid sklearn warnings
        input_data = pd.DataFrame([[temperature_f, pressure_psig, velocity, water_cut, co2, chloride, sand, gas_fraction]], 
                                 columns=FEATURE_COLUMNS)
        input_scaled = scaler.transform(input_data)
        
        general_rate = model_general.predict(input_scaled)[0]
        pitting_rate = model_pitting.predict(input_scaled)[0]
        
        return general_rate, pitting_rate
    
    # Example prediction
    print("\n--- Example Prediction ---")
    example_general, example_pitting = predict_corrosion(
        temperature_f=122.0,
        pressure_psig=100.0,
        velocity=3.0, 
        water_cut=99.0, 
        co2=2.5,
        chloride=500, 
        sand=50,
        gas_fraction=15.0
    )
    print(f"Predicted General Corrosion Rate: {example_general:.4f} mpy")
    print(f"Predicted Pitting Corrosion Rate: {example_pitting:.4f} mpy")
    
    print("\n✓ Corrosion AI model is ready!")