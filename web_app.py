import streamlit as st
import numpy as np
import joblib
import os

# -------------------------
# Load Model
# -------------------------
model_path = os.path.join(os.getcwd(), "crop_yield_model_25RP19905.pkl")
model = joblib.load(model_path)

# -------------------------
# Page Configuration
# -------------------------
st.set_page_config(
    page_title="🌾 Crop Yield Prediction",
    layout="wide",
    initial_sidebar_state="expanded",
)

# -------------------------
# Sidebar for Input
# -------------------------
st.sidebar.header("Input Parameters")
temperature = st.sidebar.slider(
    "Temperature (°C)",
    min_value=0.0,
    max_value=50.0,
    value=25.0,
    step=0.1
)

# -------------------------
# Main Title & Description
# -------------------------
st.title("🌾 Crop Yield Prediction System")
st.markdown(
    """
This intelligent web app predicts the **crop yield** based on **temperature** using a trained **Random Forest Regressor**.  
It is designed to support farmers and agricultural planners with **data-driven insights** for optimal decision-making.

- Enter the temperature in the sidebar to get a prediction.  
- The system uses historical data to provide accurate crop yield estimates.  
"""
)

# -------------------------
# Prediction
# -------------------------
if st.button("Predict Crop Yield"):
    prediction = model.predict(np.array([[temperature]]))
    st.success(f"✅ Predicted Crop Yield: **{prediction[0]:.2f} units**")

# -------------------------
# Additional Features (Optional)
# -------------------------
st.markdown("---")
st.subheader("How It Works")
st.markdown(
    """
1. **Data Collection:** Historical temperature and crop yield data are collected.  
2. **Model Training:** A Random Forest Regressor is trained to learn the relationship.  
3. **Prediction:** Input temperature is used to forecast crop yield.  
"""
)

# -------------------------
# Footer
# -------------------------
st.markdown(
    """
---
Developed by **Mugabe Edson | 25RP19905**  
Powered by **Python, Streamlit & Scikit-learn**
"""
)
