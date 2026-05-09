# X-CORE's Corrosion Rate Predictor Web App

A modern web application for predicting general and pitting corrosion rates using machine learning.

## Features

- 🎯 **Dual Predictions**: Predict both general and pitting corrosion rates
- 📊 **Interactive Interface**: User-friendly web interface with input validation
- ⚡ **Real-time Results**: Instant predictions with risk assessment
- 📈 **Risk Assessment**: Color-coded risk levels (Low/Moderate/High)
- 📋 **Detailed Summary**: Comprehensive results table

## Prerequisites

Make sure you have trained the models first:

```bash
python Model.py
```

This will create the required model files:
- `model_general_corrosion.pkl`
- `model_pitting_corrosion.pkl`
- `scaler.pkl`

## Installation

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Run the web app:**
   ```bash
   streamlit run app.py
   ```

3. **Open your browser** to `http://localhost:8501`

## Input Parameters

The model requires 8 input parameters:

| Parameter | Unit | Range | Description |
|-----------|------|-------|-------------|
| Temperature | °F | 32-500 | Operating temperature |
| Pressure | psig | 0-10000 | Operating pressure |
| Velocity | ft/s | 0.1-50 | Fluid velocity |
| Water Cut | % | 0-100 | Water percentage |
| CO2 | %mol | 0-50 | CO2 concentration |
| Chloride | ppm | 0-50000 | Chloride concentration |
| Sand | ppm | 0-10000 | Sand content |
| Gas Fraction | %wt-Gas | 0-100 | Gas weight percentage |

## Output

The app provides:
- **General Corrosion Rate** (mpy)
- **Pitting Corrosion Rate** (mpy)
- **Risk Assessment** with color coding
- **Detailed Summary Table**

## Risk Levels

### General Corrosion:
- 🟢 **Low Risk**: < 1.0 mpy
- 🟡 **Moderate Risk**: 1.0 - 5.0 mpy
- 🔴 **High Risk**: > 5.0 mpy

### Pitting Corrosion:
- 🟢 **Low Risk**: < 2.0 mpy
- 🟡 **Moderate Risk**: 2.0 - 10.0 mpy
- 🔴 **High Risk**: > 10.0 mpy

## Model Details

- **Algorithm**: Gradient Boosting Regression
- **Training Data**: Industrial corrosion measurements
- **Features**: 8 process parameters
- **Targets**: General and pitting corrosion rates

## Usage Example

```python
# Example prediction
general_rate, pitting_rate = predict_corrosion(
    temperature_f=150,
    pressure_psig=500,
    velocity=5.0,
    water_cut=90.0,
    co2=2.0,
    chloride=1000,
    sand=100,
    gas_fraction=15.0
)
```

## Disclaimer

⚠️ **This tool provides predictions based on trained machine learning models. Always consult qualified corrosion engineers and follow industry standards for critical applications.**

## License

This project is for educational and research purposes.