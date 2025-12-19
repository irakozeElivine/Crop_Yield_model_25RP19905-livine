import streamlit as st
import numpy as np
import joblib
import os

# -------------------------
# Load Model (DEPLOYMENT SAFE)
# -------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(BASE_DIR, "crop_yield_model_25RP19905.pkl")
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
# Sidebar Input
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
# Main Content
# -------------------------
st.title("🌾 Crop Yield Prediction System")

st.markdown(
    """
This intelligent web application predicts **crop yield** based on **temperature**
using a trained **Random Forest Regressor**.

### 🔍 How to Use
- Adjust the temperature using the slider on the left
- Click **Predict Crop Yield**
- Get an instant prediction based on historical data
"""
)

# -------------------------
# Prediction
# -------------------------
if st.button("Predict Crop Yield"):
    prediction = model.predict(np.array([[temperature]]))
    st.success(f"✅ Predicted Crop Yield: **{prediction[0]:.2f} units**")

# -------------------------
# Explanation Section
# -------------------------
st.markdown("---")
st.subheader("Model Overview")

st.markdown(
    """
- **Algorithm:** Random Forest Regressor  
- **Input Feature:** Temperature  
- **Output:** Estimated Crop Yield  
- **Use Case:** Smart agriculture decision support
"""
)

# -------------------------
# Footer
# -------------------------
st.markdown(
    """
---
👨‍💻 Developed by **Mugabe Edson**  
🆔 Registration Number: **25RP19905**  
⚙️ Powered by **Python • Streamlit • Scikit-learn**
"""
)
