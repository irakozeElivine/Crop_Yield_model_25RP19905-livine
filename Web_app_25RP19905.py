import streamlit as st
import joblib
import numpy as np

# Load the pre-trained model
model = joblib.load("25RP19906_model.joblib")

# Page config
st.set_page_config(
    page_title="🌱 Crop Yield Predictor",
    page_icon="🌾",
    layout="centered",
    initial_sidebar_state="expanded"
)

# Main title with style
st.markdown("""
    <h1 style='text-align: center; color: green; font-size: 48px;'>
        🌾 Crop Yield Prediction Based on Temperature 🌡️
    </h1>
""", unsafe_allow_html=True)

st.markdown("---")

# Sidebar input
st.sidebar.header("Input Temperature")
temp = st.sidebar.number_input(
    "Temperature (°C)", min_value=-50.0, max_value=60.0, value=20.0, step=0.1
)

# Optional: Add some context info in sidebar
st.sidebar.markdown("""
---
Developed with Machine Learning  
Using Linear Regression Model  
""")

# Predict button with custom style using markdown
button = st.button("Predict Crop Yield 🌱")

if button:
    input_array = np.array([[temp]])
    prediction = model.predict(input_array)
    
    # Display the prediction in a big friendly box
    st.markdown(f"""
        <div style="
            background-color: #DFF2BF; 
            border-radius: 10px; 
            padding: 20px; 
            margin-top: 30px;
            font-size: 28px;
            text-align: center;
            color: #4F8A10;
            font-weight: bold;
        ">
            Predicted Crop Yield: {prediction[0]:.2f} 🌾
        </div>
    """, unsafe_allow_html=True)

# Optional: Add a relevant image below (replace URL with your own if you want)
st.markdown("""
<div style="text-align:center; margin-top: 50px;">
    <img src="https://images.unsplash.com/photo-1506744038136-46273834b3fb?auto=format&fit=crop&w=800&q=80" width="600" alt="Farming">
</div>
""", unsafe_allow_html=True)
