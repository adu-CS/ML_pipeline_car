# app.py
import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Load artifacts
model = joblib.load("artifacts/model.pkl")
encoder = joblib.load("artifacts/encoder.pkl")

st.title("🚗 Used Car Price Prediction")
st.write("Predict your car's resale value instantly!")

# Input fields
car_name = st.text_input("Car Name")
yr_mfr = st.number_input("Year of Manufacture", min_value=1990, max_value=2025, value=2018)
kms_run = st.number_input("Kilometers Driven", min_value=0)
fuel_type = st.selectbox("Fuel Type", ["Petrol", "Diesel", "CNG", "Electric", "missing"])
city = st.text_input("City")
total_owners = st.selectbox("Number of Owners", ["First", "Second", "Third", "Fourth & Above"])
body_type = st.text_input("Body Type (e.g., SUV, Sedan)")
transmission = st.selectbox("Transmission", ["Manual", "Automatic", "missing"])
make = st.text_input("Make (e.g., Maruti, Hyundai, Toyota)")
model_name = st.text_input("Model (e.g., Swift, i20, Fortuner)")
original_price = st.number_input("Original Price (₹)", min_value=0.0)

if st.button("Predict Price"):
    df = pd.DataFrame({
        "car_name": [car_name],
        "fuel_type": [fuel_type],
        "city": [city],
        "total_owners": [total_owners],
        "body_type": [body_type],
        "transmission": [transmission],
        "make": [make],
        "model": [model_name],
        "kms_run": [kms_run],
        "original_price": [original_price],
        "car_age": [2025 - yr_mfr]
    })

    # Encode categorical features
    cat_cols = ["car_name", "fuel_type", "city", "total_owners",
                "body_type", "transmission", "make", "model"]
    df[cat_cols] = encoder.transform(df[cat_cols])

    # Predict
    pred_log = model.predict(df)
    predicted_price = np.expm1(pred_log)[0]

    st.success(f"💰 Estimated Resale Price: ₹{predicted_price:,.0f}")
