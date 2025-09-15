import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
import joblib
import os

from models import load_models, recommend_for_farmer

# Load models safely
disease_model, DISEASE_CLASSES, price_model, model_columns = load_models()

st.title("🌾 Smart Farming Assistant")

# Farmer profile section
st.header("👨‍🌾 Farmer Profile")
farmer_id = st.text_input("Enter Farmer ID")
crop = st.selectbox("Crop you sowed", ["wheat", "rice", "maize", "cotton"])
sowing_date = st.date_input("Sowing Date")
harvest_date = st.date_input("Expected Harvest Date")
quantity = st.number_input("Quantity (kg)", min_value=0.0)

if st.button("Save Profile"):
    st.success(f"Profile saved for Farmer {farmer_id} ✅")

# Disease detection section
st.header("🦠 Crop Disease Detection")
uploaded_img = st.file_uploader("Upload crop image", type=["jpg", "png", "jpeg"])

if uploaded_img:
    from tensorflow.keras.preprocessing import image
    img = image.load_img(uploaded_img, target_size=(224, 224))
    img_array = np.expand_dims(image.img_to_array(img) / 255.0, axis=0)
    preds = disease_model.predict(img_array)
    pred_class = DISEASE_CLASSES[np.argmax(preds)]
    st.write(f"Prediction: *{pred_class}*")

# Price prediction section
st.header("💰 Crop Price Prediction")
location = st.text_input("Enter Location")
month = st.selectbox("Month", list(range(1, 13)))
demand = st.slider("Market Demand (1-10)", 1, 10, 5)
temp_max = st.number_input("Max Temperature", value=30)
temp_min = st.number_input("Min Temperature", value=20)
rainfall = st.number_input("Rainfall (mm)", value=100)

if st.button("Predict Price"):
    features = pd.DataFrame([[crop, temp_max, temp_min, rainfall, month, demand]],
                            columns=model_columns)
    pred_price = price_model.predict(features)[0]
    st.success(f"Predicted Price: ₹{pred_price:.2f} per kg")

# Recommendations
st.header("📊 Smart Recommendations")
if st.button("Get Recommendations"):
    rec = recommend_for_farmer(crop)
    st.write(rec)