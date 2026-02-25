import os
import streamlit as st
import pandas as pd
import joblib

# ---------------------------
# PAGE CONFIG (MUST BE FIRST)
# ---------------------------
st.set_page_config(
    page_title="Dental Implant CDSS",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# ---------------------------
# FORCE DARK BLACK + BOLD CSS
# ---------------------------
st.markdown("""
<style>

/* Import professional font */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;700;800;900&display=swap');

/* Apply globally */
html, body, [class*="css"], .stMarkdown, .stText, p, span, div {
    font-family: 'Inter', sans-serif !important;
    color: #000000 !important;
    font-weight: 800 !important;
}

/* Header */
.header-block {
    background: #ffffff;
    border: 2px solid #000000;
    border-radius: 12px;
    padding: 20px;
    margin-bottom: 15px;
}

.title {
    font-size: 32px;
    font-weight: 900 !important;
    color: #000000 !important;
}

/* Labels */
label {
    color: #000000 !important;
    font-weight: 900 !important;
    font-size: 17px !important;
}

/* Expander */
.stExpander summary {
    font-weight: 900 !important;
    color: #000000 !important;
    font-size: 18px !important;
}

.stExpander div {
    color: #000000 !important;
    font-weight: 800 !important;
}

/* Input values */
input {
    color: #000000 !important;
    font-weight: 900 !important;
}

/* Button */
button {
    color: #000000 !important;
    font-weight: 900 !important;
}

/* Card */
.card {
    border: 2px solid #000000;
    border-radius: 10px;
    padding: 15px;
    background: #ffffff;
}

/* Success */
.success-box {
    background-color: #dcfce7;
    border: 2px solid #000000;
    padding: 15px;
    border-radius: 10px;
    font-weight: 900;
    color: #000000;
}

/* Failure */
.failure-box {
    background-color: #fee2e2;
    border: 2px solid #000000;
    padding: 15px;
    border-radius: 10px;
    font-weight: 900;
    color: #000000;
}

/* Footer */
.footer {
    margin-top: 25px;
    font-size: 15px;
    font-weight: 900;
    color: #000000;
    border-top: 2px solid #000000;
    padding-top: 10px;
}

/* Dataframe text */
[data-testid="stDataFrame"] {
    color: #000000 !important;
    font-weight: 900 !important;
}

</style>
""", unsafe_allow_html=True)

# ---------------------------
# CHECK MODEL FILE EXISTS
# ---------------------------
MODEL_FILE = "logistic_model.pkl"

if not os.path.exists(MODEL_FILE):
    st.error("ERROR: logistic_model.pkl not found. Put it in the same folder as cdss_app.py")
    st.stop()

# ---------------------------
# LOAD MODEL
# ---------------------------
@st.cache_resource
def load_model():
    return joblib.load(MODEL_FILE)

model = load_model()

# ---------------------------
# HEADER
# ---------------------------
st.markdown("""
<div class="header-block">
<div class="title">
Dental Implant Clinical Decision Support System (CDSS)
</div>
</div>
""", unsafe_allow_html=True)

st.write("Enter radiographic-derived features to estimate implant outcome and probability.")

# ---------------------------
# FEATURE DEFINITIONS
# ---------------------------
with st.expander("Feature Definitions (for manuscript clarity)"):
    st.write("""
Bounding Box Count: Number of detected implant-related regions.

Average Width: Mean horizontal size of detected regions (normalized).

Average Height: Mean vertical size of detected regions (normalized).

Mean Y Center: Average vertical location (0–1 scale).

Failure Ratio: Failure-related proportional feature used in dataset.
""")

# ---------------------------
# INPUTS
# ---------------------------
bbox_count = st.number_input(
    "Bounding Box Count (count)",
    min_value=0,
    step=1
)

avg_width = st.number_input(
    "Average Width (normalized units)",
    format="%.5f"
)

avg_height = st.number_input(
    "Average Height (normalized units)",
    format="%.5f"
)

mean_y_center = st.number_input(
    "Mean Y Center (0–1 scale)",
    format="%.5f"
)

failure_ratio = st.number_input(
    "Failure Ratio (0–1 scale)",
    format="%.5f"
)

# ---------------------------
# PREDICTION
# ---------------------------
if st.button("Run Prediction"):

    input_data = pd.DataFrame({
        "bbox_count": [bbox_count],
        "avg_width": [avg_width],
        "avg_height": [avg_height],
        "mean_y_center": [mean_y_center],
        "failure_ratio": [failure_ratio]
    })

    prediction = int(model.predict(input_data)[0])

    prob_failure = float(model.predict_proba(input_data)[0][1])
    prob_success = 1 - prob_failure

    st.markdown('<div class="card">', unsafe_allow_html=True)

    if prediction == 0:

        st.markdown(
            f'<div class="success-box">Predicted Outcome: SUCCESS<br>Probability of Success: {prob_success:.2f}</div>',
            unsafe_allow_html=True
        )

        st.progress(prob_success)

    else:

        st.markdown(
            f'<div class="failure-box">Predicted Outcome: FAILURE<br>Probability of Failure: {prob_failure:.2f}</div>',
            unsafe_allow_html=True
        )

        st.progress(prob_failure)

    st.markdown('</div>', unsafe_allow_html=True)

# ---------------------------
# FOOTER
# ---------------------------
st.markdown("""
<div class="footer">
Clinical Decision Support System Prototype for Dental Implant Outcome Prediction.
</div>
""", unsafe_allow_html=True)
