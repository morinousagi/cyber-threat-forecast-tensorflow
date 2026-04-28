import streamlit as st
import numpy as np
import joblib
import tensorflow as tf
from config import WINDOW_SIZE

st.title("Cyber Threat Forecasting")

model = tf.keras.models.load_model("artifacts/best_model.h5")
scaler = joblib.load("artifacts/scaler.joblib")

user_input = st.text_area(f"Enter {WINDOW_SIZE} values separated by commas")

if st.button("Predict"):
    values = np.array([float(v) for v in user_input.split(",")])
    values = values.reshape(1, WINDOW_SIZE, 1)

    pred = model.predict(values)
    pred_inv = scaler.inverse_transform(pred)

    st.success(f"Predicted Local Infection: {pred_inv[0][0]:.6f}")