import streamlit as st
import pandas as pd
import numpy as np
import uuid
import os
import datetime
import requests
import joblib
import tensorflow as tf
from PIL import Image

# =====================
# CONFIG & FILES
# =====================
FARMER_FILE = "farmers.csv"
DISEASE_MODEL_PATH = "plant_disease_mobilenetv2.h5"
CLASS_FILE = "class_names.txt"
PRICE_MODEL_PATH = "crop_price_model.pkl"
MODEL_COLS_PATH = "model_columns.pkl"
IMG_SIZE = (224, 224)

# =====================
# DATA STORAGE
# =====================
if not os.path.exists(FARMER_FILE):
    df = pd.DataFrame(columns=["farmer_id", "name", "location", "crop", "acres",
                               "sowing_date", "harvest_date", "harvest_amount",
                               "fertilizer", "yield_rate"])
    df.to_csv(FARMER_FILE, index=False)

# =====================
# LOAD MODELS
# =====================
@st.cache_resource
def load_models():
    disease_model = tf.keras.models.load_model(DISEASE_MODEL_PATH)
    with open(CLASS_FILE, "r") as f:
        disease_classes = [line.strip() for line in f]
    price_model = joblib.load(PRICE_MODEL_PATH)
    model_columns = joblib.load(MODEL_COLS_PATH)
    return disease_model, disease_classes, price_model, model_columns

disease_model, DISEASE_CLASSES, price_model, model_columns = load_models()

# =====================
# HELPER FUNCTIONS
# =====================
def register_farmer(name, location):
    farmer_id = str(uuid.uuid4())[:8]
    df = pd.read_csv(FARMER_FILE)
    new_farmer = pd.DataFrame([[farmer_id, name, location, "", 0, "", "", 0, "", 0]], columns=df.columns)
    df = pd.concat([df, new_farmer], ignore_index=True)
    df.to_csv(FARMER_FILE, index=False)
    return farmer_id

def add_crop(farmer_id, crop, acres, sowing_date, harvest_date, harvest_amount, fertilizer):
    df = pd.read_csv(FARMER_FILE)
    if farmer_id not in df['farmer_id'].values:
        return "❌ Farmer ID not found!"
    try:
        acres = float(acres)
        harvest_amount = float(harvest_amount)  # in kg
    except:
        return "❌ Please enter valid numbers."
    yield_rate = round(harvest_amount / acres, 2) if acres > 0 else 0
    df.loc[df['farmer_id'] == farmer_id,
           ["crop", "acres", "sowing_date", "harvest_date", "harvest_amount", "fertilizer", "yield_rate"]] = \
          [crop, acres, sowing_date, harvest_date, harvest_amount, fertilizer, yield_rate]
    df.to_csv(FARMER_FILE, index=False)

    # Recommendations
    if yield_rate < 1000:
        rec = "⚠ Low yield. Try nitrogen-rich fertilizers & crop rotation."
    else:
        rec = "✅ Good yield! Keep same practice."
    return f"✅ Data saved.\n📊 Yield Rate: {yield_rate} kg/acre\nRecommendation: {rec}"

def preprocess_pil(img_pil):
    img = img_pil.resize(IMG_SIZE)
    arr = np.array(img).astype("float32")
    if arr.shape[-1] == 4:
        arr = arr[..., :3]
    arr = tf.keras.applications.mobilenet_v2.preprocess_input(arr)
    arr = np.expand_dims(arr, 0)
    return arr

def analyze_disease(img):
    if img is None:
        return "❌ Please upload an image.", None
    arr = preprocess_pil(img)
    preds = disease_model.predict(arr)[0]
    top_idx = preds.argsort()[-3:][::-1]
    result = {DISEASE_CLASSES[i]: float(preds[i]) for i in top_idx}
    top_class = DISEASE_CLASSES[top_idx[0]]
    return f"Detected: {top_class} ({preds[top_idx[0]]*100:.1f}%)", result

def get_weather(lat, lon, date):
    url = f"https://archive-api.open-meteo.com/v1/archive?latitude={lat}&longitude={lon}&start_date={date}&end_date={date}&daily=temperature_2m_max,temperature_2m_min,precipitation_sum&timezone=auto"
    r = requests.get(url).json()
    if "daily" not in r:
        return None
    return {
        "temp_max": r["daily"]["temperature_2m_max"][0],
        "temp_min": r["daily"]["temperature_2m_min"][0],
        "rainfall": r["daily"]["precipitation_sum"][0]
    }

def predict_price(lat, lon, date, crop):
    weather = get_weather(lat, lon, date)
    if not weather:
        return "❌ Weather unavailable"
    features = pd.DataFrame([{
        "temp_max": weather["temp_max"],
        "temp_min": weather["temp_min"],
        "rainfall": weather["rainfall"],
        "month": datetime.datetime.strptime(date, "%Y-%m-%d").month,
        "crop": crop,
        "demand": 120  # dummy demand index
    }])
    features = pd.get_dummies(features)
    features = features.reindex(columns=model_columns, fill_value=0)
    price = price_model.predict(features)[0]
    return f"📊 Predicted {crop} price on {date}: ₹{round(price,2)} / quintal"

def view_history(farmer_id):
    df = pd.read_csv(FARMER_FILE)
    if farmer_id not in df['farmer_id'].values:
        return None
    return df[df['farmer_id'] == farmer_id]

# =====================
# STREAMLIT UI
# =====================
st.set_page_config(page_title="🌱 Smart Farming Assistant", layout="wide")
st.title("🌱 Smart Farming Assistant")

menu = st.sidebar.radio("Navigation", ["👤 Register Farmer", "🌾 Crop Data", "🩺 Disease Detection", "💰 Price Prediction", "📊 Farmer History"])

if menu == "👤 Register Farmer":
    st.header("Register Farmer")
    name = st.text_input("Farmer Name")
    location = st.text_input("Location")
    if st.button("Register"):
        if name and location:
            fid = register_farmer(name, location)
            st.success(f"✅ Farmer registered! ID: {fid}")
        else:
            st.error("Please enter all details.")

elif menu == "🌾 Crop Data":
    st.header("Add Crop Data")
    fid = st.text_input("Farmer ID")
    crop = st.text_input("Crop Name")
    acres = st.number_input("Acres", min_value=0.1, step=0.1)
    sowing = st.date_input("Sowing Date")
    harvest = st.date_input("Harvest Date")
    amount = st.number_input("Harvest Amount (kg)", min_value=0.0, step=10.0)
    fert = st.text_input("Fertilizer Used")
    if st.button("Save Crop Data"):
        result = add_crop(fid, crop, acres, sowing, harvest, amount, fert)
        st.info(result)

elif menu == "🩺 Disease Detection":
    st.header("Upload Crop Image")
    img = st.file_uploader("Upload Leaf Image", type=["jpg", "jpeg", "png"])
    if img:
        image = Image.open(img)
        st.image(image, caption="Uploaded Image", use_column_width=True)
        if st.button("Analyze"):
            res, top3 = analyze_disease(image)
            st.success(res)
            st.write("Top Predictions:", top3)

elif menu == "💰 Price Prediction":
    st.header("Predict Crop Price")
    lat = st.number_input("Latitude", value=28.6)
    lon = st.number_input("Longitude", value=77.2)
    date = st.date_input("Date")
    crop2 = st.text_input("Crop Name")
    if st.button("Predict Price"):
        result = predict_price(lat, lon, str(date), crop2)
        st.info(result)

elif menu == "📊 Farmer History":
    st.header("Farmer History")
    fid = st.text_input("Farmer ID")
    if st.button("View History"):
        hist = view_history(fid)
        if hist is None or hist.empty:
            st.error("❌ No data found for this farmer.")
        else:
            st.dataframe(hist)