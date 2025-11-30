import streamlit as st
import pandas as pd
import numpy as np
import joblib

# -------------------------------
# Load trained model + scaler + features
# -------------------------------
model = joblib.load("churn_model.pkl")
scaler = joblib.load("scaler.pkl")
features = joblib.load("features.pkl")

st.set_page_config(page_title="Customer Churn Prediction", layout="centered")
st.title("📊 Customer Churn Prediction App")
st.write("Enter the customer details below to predict churn.")

# -------------------------------
# Create input form
# -------------------------------
st.header("Customer Input Details")

input_data = {}

for feature in features:
    input_data[feature] = st.number_input(
        f"Enter {feature}",
        min_value=0.0,
        value=0.0,
        step=1.0
    )

# -------------------------------
# Prediction
# -------------------------------
if st.button("Predict Churn"):

    # Convert to DataFrame
    input_df = pd.DataFrame([input_data])

    # Scale numeric data
    input_scaled = scaler.transform(input_df)

    # Predict
    pred = model.predict(input_scaled)[0]
    prob = model.predict_proba(input_scaled)[0][1]

    st.write("---")
    st.subheader("📌 Prediction Result")

    if pred == 1:
        st.error(f"🔴 Customer is likely to CHURN (Probability: {prob:.2f})")
    else:
        st.success(f"🟢 Customer will NOT churn (Probability: {prob:.2f})")
