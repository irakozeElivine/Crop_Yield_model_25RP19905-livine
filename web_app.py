import streamlit as st
import numpy as np
import joblib
import os

# -------------------------
# Load Model
# -------------------------
model_path = os.path.join(os.path.dirname(__file__), "crop_yield_model_25RP19905.pkl")
model = joblib.load(model_path)

# -------------------------
# Page Configuration
# -------------------------
st.set_page_config(
    page_title="🌾 Crop Yield Prediction",
    layout="wide"
)

# -------------------------
# Sidebar Input
# -------------------------
st.sidebar.header("Input Parameters")
temperature = st.sidebar.slider(
    "Temperature (°C)", 0.0, 50.0, 25.0, 0.1
)

# -------------------------
# Main Section
# -------------------------
st.title("🌾 Crop Yield Prediction System")

if st.button("Predict Crop Yield"):
    prediction = model.predict([[temperature]])
    st.success(f"Predicted Crop Yield: {prediction[0]:.2f}")
