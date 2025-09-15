# agri_ews_streamlit_app.py
# Agriculture Early Warning System (EWS) demo app
# Streamlit single-file implementation with dummy ML models

import streamlit as st
import pandas as pd
import numpy as np
import os
import joblib
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, accuracy_score
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

# --------------------- Dummy Dataset Generators ---------------------
def generate_price_dataset(n=1000):
    rng = np.random.default_rng(42)
    crops = ["Wheat", "Rice", "Maize", "Tomato", "Potato", "Cotton"]
    rows = []
    start_date = datetime(2022, 1, 1)
    for i in range(n):
        crop = rng.choice(crops)
        rainfall = rng.normal(80, 20)
        temp = rng.normal(26, 4)
        demand = rng.integers(50, 150)
        price = 1500 + 10*demand - 2*rainfall + rng.normal(0, 50)
        date = start_date + timedelta(days=i % 365)
        rows.append([date, crop, rainfall, temp, demand, price])
    df = pd.DataFrame(rows, columns=['date','crop','rainfall_mm','temperature_c','demand_index','price_pkr'])
    return df

def generate_disease_dataset(n=800):
    rng = np.random.default_rng(0)
    crops = ["Tomato","Potato","Wheat"]
    diseases = {
        "Tomato": ["Leaf Blight","Yellow Curl","Healthy"],
        "Potato": ["Late Blight","Early Blight","Healthy"],
        "Wheat": ["Rust","Smut","Healthy"]
    }
    rows=[]
    for i in range(n):
        crop = rng.choice(crops)
        possibilities = diseases[crop]
        if len(possibilities) == 3:
            p = [0.25,0.25,0.5]
        else:
            p = [1.0/len(possibilities)] * len(possibilities)
        label = rng.choice(possibilities, p=p)
        spots = int(rng.poisson(2 if label!='Healthy' else 0))
        if label=='Healthy':
            yellowing = int(rng.choice([0,1], p=[0.8,0.2]))
            wilting = int(rng.choice([0,1], p=[0.85,0.15]))
            lesion_size = float(abs(rng.normal(0.05,0.05)))
        else:
            yellowing = int(rng.choice([0,1], p=[0.3,0.7]))
            wilting = int(rng.choice([0,1], p=[0.3,0.7]))
            lesion_size = float(abs(rng.normal(0.5,1.0)))
        humidity = float(rng.normal(70,10))
        temperature = float(rng.normal(24,5))
        rows.append([crop, spots, yellowing, wilting, lesion_size, humidity, temperature, label])
    df = pd.DataFrame(rows, columns=['crop','spots_count','yellowing','wilting','lesion_size_cm','humidity_pct','temperature_c','disease'])
    return df

# --------------------- Model Paths ---------------------
MODEL_DIR = "models"
os.makedirs(MODEL_DIR, exist_ok=True)
PRICE_MODEL_PATH = os.path.join(MODEL_DIR,'price_model.joblib')
DISEASE_MODEL_PATH = os.path.join(MODEL_DIR,'disease_model.joblib')
SCALER_PATH = os.path.join(MODEL_DIR,'scaler.joblib')  # price scaler
DISEASE_SCALER_PATH = os.path.join(MODEL_DIR,'disease_scaler.joblib')
DISEASE_FEATURES_PATH = os.path.join(MODEL_DIR,'disease_feature_cols.joblib')

# --------------------- Model training / load ---------------------
@st.cache_resource
def train_price_model(df):
    df2 = pd.get_dummies(df.copy(), columns=['crop'], drop_first=True)
    X = df2[['rainfall_mm','temperature_c','demand_index'] + [c for c in df2.columns if c.startswith('crop_')]]
    y = df2['price_pkr']
    X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2, random_state=1)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    model = RandomForestRegressor(n_estimators=120, random_state=1)
    model.fit(X_train_scaled, y_train)
    preds = model.predict(X_test_scaled)
    rmse = mean_squared_error(y_test, preds, squared=False)
    joblib.dump(model, PRICE_MODEL_PATH)
    joblib.dump(scaler, SCALER_PATH)
    return model, scaler, rmse

@st.cache_resource
def train_disease_model(df):
    df2 = pd.get_dummies(df.copy(), columns=['crop'], drop_first=True)
    feature_cols = ['spots_count','yellowing','wilting','lesion_size_cm','humidity_pct','temperature_c'] + [c for c in df2.columns if c.startswith('crop_')]
    X = df2[feature_cols]
    y = df2['disease']
    X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2, random_state=2)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    clf = RandomForestClassifier(n_estimators=100, random_state=2)
    clf.fit(X_train_scaled, y_train)
    preds = clf.predict(X_test_scaled)
    acc = accuracy_score(y_test, preds)
    joblib.dump(clf, DISEASE_MODEL_PATH)
    joblib.dump(scaler, DISEASE_SCALER_PATH)
    joblib.dump(feature_cols, DISEASE_FEATURES_PATH)
    return clf, scaler, acc

def load_price_model():
    if os.path.exists(PRICE_MODEL_PATH) and os.path.exists(SCALER_PATH):
        model = joblib.load(PRICE_MODEL_PATH)
        scaler = joblib.load(SCALER_PATH)
        return model, scaler
    return None, None

def load_disease_model():
    if os.path.exists(DISEASE_MODEL_PATH) and os.path.exists(DISEASE_SCALER_PATH) and os.path.exists(DISEASE_FEATURES_PATH):
        clf = joblib.load(DISEASE_MODEL_PATH)
        scaler = joblib.load(DISEASE_SCALER_PATH)
        feature_cols = joblib.load(DISEASE_FEATURES_PATH)
        return clf, scaler, feature_cols
    return None, None, None

# --------------------- Simple Chatbot ---------------------
class SimpleChatbot:
    def _init_(self, diseases_df=None, price_df=None):
        self.diseases_df = diseases_df
        self.price_df = price_df

    def reply(self, text):
        text_l = text.lower()

        # Price queries
        if self.price_df is not None and ('price' in text_l or 'market' in text_l):
            for c in self.price_df['crop'].unique():
                if c.lower() in text_l:
                    recent = self.price_df[self.price_df['crop']==c].sort_values('date',ascending=False).iloc[:7]
                    mean_price = int(recent['price_pkr'].mean())
                    return f"📊 Recent average {c} price ≈ ₹{mean_price} per quintal."
            return "Which crop price do you want? (e.g., Wheat, Rice, Tomato)."

        # Disease queries
        if self.diseases_df is not None:
            for _, row in self.diseases_df.iterrows():
                if row['disease'].lower() in text_l:
                    return f"🦠 Disease: {row['disease']} (symptoms: {row['spots_count']} spots, yellowing={row['yellowing']}, wilting={row['wilting']})"

        if 'help' in text_l:
            return "You can ask about prices, diseases, or weather."

        return "❌ Sorry, I didn’t understand. Try asking about crop prices or diseases."

# --------------------- Farmer Recommendation System ---------------------
def recommend_for_farmer(crop):
    recs = {
        "Wheat": {"fertilizer": "Use nitrogen-rich fertilizer (urea).","rotation": "Rotate with rice or pulses.","extra": "Irrigate at crown root initiation stage."},
        "Rice": {"fertilizer": "Apply phosphate fertilizers.","rotation": "Rotate with maize.","extra": "Maintain 2–3 cm water during tillering."},
        "Maize": {"fertilizer": "Balanced NPK recommended.","rotation": "Rotate with cotton or pulses.","extra": "Avoid waterlogging; requires good drainage."},
        "Tomato": {"fertilizer": "Use potassium-rich fertilizer.","rotation": "Rotate with leafy vegetables.","extra": "Stake plants to prevent fungal infection."},
        "Potato": {"fertilizer": "Nitrogen and potash recommended.","rotation": "Rotate with maize or wheat.","extra": "Monitor soil moisture regularly."},
        "Cotton": {"fertilizer": "Potassium & micronutrient sprays help.","rotation": "Rotate with wheat.","extra": "Avoid continuous cotton cropping."}
    }
    return recs.get(crop, {"fertilizer": "Get soil tested for best fertilizer advice.","rotation": "Rotate with legumes to restore soil nitrogen.","extra": "Follow local agri advisories."})

# --------------------- Streamlit Layout ---------------------
st.set_page_config(page_title='Agri - Streamlit', layout='wide')

price_df = generate_price_dataset(600)
disease_df = generate_disease_dataset(400)
chatbot = SimpleChatbot(diseases_df=disease_df, price_df=price_df)

st.sidebar.title('Navigation')
page = st.sidebar.radio('Go to', [
    'Home','Farm Profile','Price Predictor','Disease Detector','Weather','Chatbot','Marketplace','Research','Admin'
])

# --- HOME ---
if page=='Home':
    st.title('🌾 Agriculture Recommendations System ')
    st.write('Demo Streamlit app with dummy models for crop price prediction, disease detection, and more.')

# --- FARM PROFILE ---
elif page=='Farm Profile':
    st.header("👨‍🌾 Farmer Profile & Recommendations")
    location = st.text_input("Location (e.g., Punjab)")
    soil = st.selectbox("Soil type", ["Loamy","Sandy","Clay","Black Soil"])
    season = st.selectbox("Season", ["Kharif","Rabi","Zaid"])
    crop = st.selectbox("Current Crop", ["Wheat","Rice","Maize","Tomato","Potato","Cotton"])
    if st.button("Get Recommendations"):
        rec = recommend_for_farmer(crop)
        st.success(f"👉 Recommendations for {crop}:")
        st.write(f"🌱 Fertilizer Advice: {rec['fertilizer']}")
        st.write(f"🔄 Crop Rotation: {rec['rotation']}")
        st.write(f"💡 Extra Tip: {rec['extra']}")

# --- PRICE PREDICTOR ---
elif page=='Price Predictor':
    st.header("📈 Crop Price Prediction")
    crop = st.selectbox("Select crop", price_df['crop'].unique())
    rainfall = st.number_input("Rainfall (mm)", value=80.0)
    temp = st.number_input("Temperature (°C)", value=25.0)
    demand = st.slider("Demand Index", 50, 150, 100)
    if st.button("Predict Price"):
        model, scaler = load_price_model()
        df_cols = pd.get_dummies(price_df[['crop']], columns=['crop'], drop_first=True)
        crop_cols = [c for c in df_cols.columns if c.startswith('crop_')]
        inp = {"rainfall_mm":rainfall,"temperature_c":temp,"demand_index":demand}
        for col in crop_cols:
            inp[col] = 1 if col==f'crop_{crop}' else 0
        X = pd.DataFrame([inp])
        ordered_cols = ['rainfall_mm','temperature_c','demand_index'] + crop_cols
        X = X[ordered_cols]
        if model is None or scaler is None:
            st.warning('Model not trained yet. Go to Admin -> Retrain models.')
        else:
            X_scaled = scaler.transform(X)
            pred = model.predict(X_scaled)
            st.success(f'Predicted price for {crop} ≈ ₹{int(pred[0])} per quintal.')

# --- DISEASE DETECTOR ---
elif page=='Disease Detector':
    st.header("🦠 Crop Disease Detector")
    crop = st.selectbox("Crop", ["Tomato","Potato","Wheat"])
    spots = st.number_input("Spots count", value=0)
    yellowing = st.selectbox("Yellowing", [0,1])
    wilting = st.selectbox("Wilting", [0,1])
    lesion = st.number_input("Lesion size (cm)", value=0.0)
    humidity = st.number_input("Humidity (%)", value=70.0)
    temp = st.number_input("Temperature (°C)", value=24.0)
    if st.button('Predict Disease'):
        clf, scaler_d, feat_cols = load_disease_model()
        if clf is None:
            st.warning('Model not trained yet. Go to Admin -> Retrain models.')
        else:
            inp = {'spots_count':spots,'yellowing':yellowing,'wilting':wilting,
                   'lesion_size_cm':lesion,'humidity_pct':humidity,'temperature_c':temp}
            crop_dummy_cols = [c for c in feat_cols if c.startswith('crop_')]
            for col in crop_dummy_cols:
                inp[col] = 1 if col==f'crop_{crop}' else 0
            X = pd.DataFrame([inp])[feat_cols]
            X_scaled = scaler_d.transform(X)
            pred = clf.predict(X_scaled)
            proba = clf.predict_proba(X_scaled)
            st.success(f'Predicted: {pred[0]}')
            st.write('Probabilities:', dict(zip(clf.classes_, proba[0])))

# --- WEATHER (Mock) ---
elif page=='Weather':
    st.header("⛅ Weather Forecast (Mock Data)")
    days = [datetime.today()+timedelta(days=i) for i in range(5)]
    temps = np.random.normal(28,3,len(days))
    rains = np.random.normal(70,15,len(days))
    df = pd.DataFrame({"Date":days,"Temp (°C)":temps,"Rainfall (mm)":rains})
    st.table(df)
    fig, ax = plt.subplots()
    ax.plot(df['Date'], df['Temp (°C)'], label='Temp')
    ax.set_ylabel('Temp (°C)')
    ax2 = ax.twinx()
    ax2.bar(df['Date'], df['Rainfall (mm)'], alpha=0.3, label='Rain')
    st.pyplot(fig)

# --- CHATBOT ---
elif page=='Chatbot':
    st.header("🤖 Agri Chatbot")
    user_input = st.text_input("Ask the bot:")
    if st.button("Send"):
        reply = chatbot.reply(user_input)
        st.markdown(f"You: {user_input}")
        st.markdown(f"Bot: {reply}")

# --- MARKETPLACE ---
elif page=='Marketplace':
    st.header("🛒 Marketplace (Mock)")
    st.info("Feature coming soon: Farmers can list and buy agri-products here.")

# --- RESEARCH ---
elif page=='Research':
    st.header("📚 Research & Knowledge")
    st.write("Example dataset summaries and links.")
    st.write("- Paper on crop yield improvement.")
    st.write("- Disease resistance research.")

# --- ADMIN ---
elif page=='Admin':
    st.header("⚙ Admin Panel")
    if st.button("Retrain Price Model"):
        model, scaler, rmse = train_price_model(price_df)
        st.success(f'Price model retrained. RMSE ≈ {rmse:.2f}')
    if st.button("Retrain Disease Model"):
        clf, sc_d, acc = train_disease_model(disease_df)
        st.success(f'Disease model retrained. Accuracy ≈ {acc*100:.1f}%')
    st.subheader("Model Status")
    pm, sc = load_price_model()
    dm, sc_d, feats = load_disease_model()
    st.write('Price model present:', bool(pm))
    st.write('Disease model present:', bool(dm))
    if feats is not None:
        st.write('Disease feature count:', len(feats))

# Footer
st.sidebar.markdown('---')
st.sidebar.caption('Agri demo — single-file Streamlit app.')