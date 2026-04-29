import streamlit as st
import numpy as np
import pandas as pd
import joblib

st.set_page_config(page_title="Cyber Threat Prediction", layout="wide")

st.title("🔐 Cyber Threat Prediction Dashboard")
st.write("Predict next-day Local Infection using 14-day historical threat indicators.")

# -----------------------------
# Load Model
# -----------------------------
@st.cache_resource
def load_model():
    return joblib.load("artifacts/best_regression_model.joblib")

model = load_model()

# -----------------------------
# Feature Setup
# -----------------------------
WINDOW = 14

features = [
    'Spam',
    'Ransomware',
    'Exploit',
    'Malicious Mail',
    'Network Attack',
    'On Demand Scan',
    'Web Threat',
    'Local Infection'
]

st.markdown("## 📥 Input 14-Day Historical Data")

st.write("Enter 14 comma-separated values for each feature (oldest → newest).")

input_data = {}

for feature in features:
    values = st.text_area(
        f"{feature} (14 values)",
        placeholder="0.01, 0.02, 0.015, ...",
        key=feature
    )
    input_data[feature] = values

# -----------------------------
# Prediction
# -----------------------------
if st.button("Predict Next-Day Local Infection"):

    try:
        # Build feature vector matching training format
        feature_vector = []

        for feature in features:
            values = [float(v.strip()) for v in input_data[feature].split(",")]

            if len(values) != WINDOW:
                st.error(f"{feature} must contain exactly 14 values.")
                st.stop()

            # Create lag features (lag_1 is most recent)
            for lag in range(WINDOW, 0, -1):
                feature_vector.append(values[-lag])

        feature_vector = np.array(feature_vector).reshape(1, -1)

        prediction = model.predict(feature_vector)

        st.success(f"📈 Predicted Local Infection (Next Day): {prediction[0]:.6f}")

    except Exception as e:
        st.error("Invalid input format. Please ensure 14 numeric values per feature.")