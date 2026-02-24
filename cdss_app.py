import streamlit as st
import pandas as pd
import joblib

# Load trained model (replace with your actual model path if different)
@st.cache_resource
def load_model():
    model = joblib.load("logistic_model.pkl")
    return model

model = load_model()

st.title("🦷 Dental Implant Outcome Predictor")

st.markdown("Enter the implant features below to predict success or failure:")

# User Inputs
bbox_count = st.number_input("Bounding Box Count", min_value=0, step=1)
avg_width = st.number_input("Average Width", format="%.5f")
avg_height = st.number_input("Average Height", format="%.5f")
mean_y_center = st.number_input("Mean Y Center", format="%.5f")
failure_ratio = st.number_input("Failure Ratio", format="%.5f")

if st.button("Predict"):
    input_data = pd.DataFrame({
        "bbox_count": [bbox_count],
        "avg_width": [avg_width],
        "avg_height": [avg_height],
        "mean_y_center": [mean_y_center],
        "failure_ratio": [failure_ratio]
    })

    prediction = model.predict(input_data)[0]
    prob = model.predict_proba(input_data)[0][1]  # Probability of failure

    if prediction == 1:
        st.error(f"⚠️ Predicted Outcome: **Failure** with probability {prob:.2f}")
    else:
        st.success(f"✅ Predicted Outcome: **Success** with probability {1 - prob:.2f}")

