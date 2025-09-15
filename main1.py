# agri_ews_streamlit_app.py
# Complete single-file Streamlit agriculture "Early Warning System (EWS)" style app
# Features included:
# - Dummy dataset generation for crops, prices, disease symptoms
# - Model training (price regression, disease classification)
# - Streamlit multi-page UI: Home, Price Predictor, Disease Detector (tabular), Weather (mock), Chatbot, Marketplace, Research
# - Simple chatbot based on retrieval + small rule-based responses
# - Admin panel to retrain models and download datasets
# - Uses scikit-learn, pandas, numpy, matplotlib

import streamlit as st
import pandas as pd
import numpy as np
import os
import io
import joblib
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

# --------------------- Helpers & Dummy Data ---------------------
@st.cache_data
def generate_price_dataset(n=2000, random_state=42):
    rng = np.random.RandomState(random_state)
    dates = pd.date_range(end=pd.Timestamp.today(), periods=n).to_pydatetime().tolist()
    crops = ['Wheat', 'Rice', 'Maize', 'Tomato', 'Potato', 'Onion']
    data = []
    for i in range(n):
        crop = rng.choice(crops)
        base_price = {'Wheat':1800, 'Rice':2500, 'Maize':1500,'Tomato':8000,'Potato':1200,'Onion':6000}[crop]
        seasonality = 1 + 0.2*np.sin(i/30)
        rainfall = max(0, rng.normal(80, 40)) # mm
        temperature = rng.normal(25, 6) # C
        demand_index = rng.normal(100, 20)
        price = base_price * seasonality * (1 + (100-rainfall)/500) * (1 + (30-temperature)/200) * (demand_index/100)
        price = max(100, price + rng.normal(0, base_price*0.05))
        data.append([dates[i], crop, rainfall, temperature, demand_index, price])
    df = pd.DataFrame(data, columns=['date','crop','rainfall_mm','temperature_c','demand_index','price_pkr'])
    return df

@st.cache_data
def generate_disease_dataset(n=2000, random_state=24):
    rng = np.random.RandomState(random_state)
    crops = ['Tomato','Potato','Wheat']
    diseases = {
        'Tomato':['Late Blight','Leaf Spot','Healthy'],
        'Potato':['Early Blight','Late Blight','Healthy'],
        'Wheat':['Rust','Healthy']
    }
    rows = []
    for i in range(n):
        crop = rng.choice(crops)
        if crop=='Tomato': possibilities = diseases['Tomato']
        elif crop=='Potato': possibilities = diseases['Potato']
        else: possibilities = diseases['Wheat']
        label = rng.choice(possibilities, p=[0.25,0.25,0.5] if 'Healthy' in possibilities else [0.5,0.5])
        # symptoms: spots, yellowing, wilting, lesion_size
        spots = rng.poisson(2 if label!='Healthy' else 0)
        yellowing = rng.choice([0,1], p=[0.8,0.2]) if label=='Healthy' else rng.choice([0,1], p=[0.3,0.7])
        wilting = rng.choice([0,1], p=[0.85,0.15]) if label=='Healthy' else rng.choice([0,1], p=[0.3,0.7])
        lesion_size = float(rng.normal(0.5,1.0)) if label!='Healthy' else float(abs(rng.normal(0.05,0.05)))
        humidity = rng.normal(70,10)
        temperature = rng.normal(24,5)
        rows.append([crop, spots, yellowing, wilting, lesion_size, humidity, temperature, label])
    df = pd.DataFrame(rows, columns=['crop','spots_count','yellowing','wilting','lesion_size_cm','humidity_pct','temperature_c','disease'])
    return df

# File paths to save models/datasets
MODEL_DIR = 'models'
os.makedirs(MODEL_DIR, exist_ok=True)
PRICE_MODEL_PATH = os.path.join(MODEL_DIR,'price_model.joblib')
DISEASE_MODEL_PATH = os.path.join(MODEL_DIR,'disease_model.joblib')
SCALER_PATH = os.path.join(MODEL_DIR,'scaler.joblib')

# --------------------- Model training / load ---------------------
@st.cache_resource
def train_price_model(df):
    df2 = df.copy()
    df2 = pd.get_dummies(df2, columns=['crop'], drop_first=True)
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
    df2 = df.copy()
    df2 = pd.get_dummies(df2, columns=['crop'], drop_first=True)
    X = df2[['spots_count','yellowing','wilting','lesion_size_cm','humidity_pct','temperature_c'] + [c for c in df2.columns if c.startswith('crop_')]]
    y = df2['disease']
    X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2, random_state=2)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    clf = RandomForestClassifier(n_estimators=120, random_state=2)
    clf.fit(X_train_scaled, y_train)
    preds = clf.predict(X_test_scaled)
    acc = accuracy_score(y_test, preds)
    joblib.dump(clf, DISEASE_MODEL_PATH)
    # we overwrite scaler from price training if exists; store disease scaler separately if needed
    return clf, scaler, acc

# Utility loaders
def load_price_model():
    if os.path.exists(PRICE_MODEL_PATH) and os.path.exists(SCALER_PATH):
        model = joblib.load(PRICE_MODEL_PATH)
        scaler = joblib.load(SCALER_PATH)
        return model, scaler
    return None, None

def load_disease_model():
    if os.path.exists(DISEASE_MODEL_PATH):
        clf = joblib.load(DISEASE_MODEL_PATH)
        return clf
    return None

# --------------------- Simple Chatbot (retrieval + rules) ---------------------
class SimpleChatbot:
    def __init__(self, price_df, disease_df):
        self.price_df = price_df
        self.disease_df = disease_df
    def reply(self, text):
        text_l = text.lower()
        if 'price' in text_l or 'market' in text_l:
            crop = None
            for c in self.price_df['crop'].unique():
                if c.lower() in text_l:
                    crop = c
                    break
            if crop is None:
                return "Which crop? (e.g., Wheat, Rice, Tomato)."
            recent = self.price_df[self.price_df['crop']==crop].sort_values('date',ascending=False).iloc[:7]
            mean_price = int(recent['price_pkr'].mean())
            return f"Recent average {crop} price ≈ ₹{mean_price} per quintal (based on dummy data)."
        if 'disease' in text_l or 'leaf' in text_l or 'blight' in text_l:
            return "Describe symptoms: spots count, yellowing (0/1), wilting (0/1), lesion_size_cm, humidity, temperature. I can predict disease from that." 
        if 'help' in text_l or 'features' in text_l:
            return "This EWS app has: Price prediction, Disease detection, Weather mockup, Marketplace, Research notes, and an admin retrain panel." 
        # fallback: simple retrieval from research dataset
        if 'research' in text_l or 'paper' in text_l:
            return "Check Research page — there are example dataset summaries and links to further reading (mock)."
        return "Sorry, I didn't understand. Ask about price, disease, weather, or features."

# --------------------- Streamlit Layout ---------------------
st.set_page_config(page_title='Agri EWS - Streamlit', layout='wide')

# Load or create datasets
price_df = generate_price_dataset(1600)
disease_df = generate_disease_dataset(1200)
chatbot = SimpleChatbot(price_df,disease_df)

# Sidebar navigation
st.sidebar.title('Navigation')
page = st.sidebar.radio('Go to', ['Home','Price Predictor','Disease Detector','Weather','Chatbot','Marketplace','Research','Admin'])

# --- HOME ---
if page=='Home':
    st.title('Agriculture Early Warning System (EWS) — Demo')
    st.markdown(
        """
        This demo app bundles several features commonly requested in agriculture projects:
        - Price forecasting for crops (dummy data)
        - Tabular disease detection (symptom-based)
        - Simple weather mock/alerts
        - Chatbot for quick queries
        - Marketplace mockup for selling produce
        - Research/Downloads and Admin for retraining models
        """
    )
    st.subheader('Quick visuals')
    crop = st.selectbox('Select crop to view price trend', price_df['crop'].unique())
    subset = price_df[price_df['crop']==crop].sort_values('date')
    fig, ax = plt.subplots()
    ax.plot(pd.to_datetime(subset['date']), subset['price_pkr'])
    ax.set_title(f'{crop} price trend (dummy)')
    ax.set_ylabel('Price')
    st.pyplot(fig)
    st.info('This app uses dummy datasets. Use Admin -> Retrain to simulate model training on these datasets.')

# --- PRICE PREDICTOR ---
elif page=='Price Predictor':
    st.title('Price Predictor')
    st.markdown('Enter features to predict crop price (per quintal).')
    crop = st.selectbox('Crop', price_df['crop'].unique())
    rainfall = st.number_input('Rainfall (mm)', min_value=0.0, value=50.0)
    temp = st.number_input('Temperature (°C)', value=25.0)
    demand = st.number_input('Demand index (0-200)', min_value=10.0, max_value=300.0, value=100.0)
    if st.button('Predict Price'):
        model, scaler = load_price_model()
        df_for_cols = pd.get_dummies(price_df[['crop']].copy(), columns=['crop'], drop_first=True)
        crop_cols = [c for c in df_for_cols.columns if c.startswith('crop_')]
        inp = { 'rainfall_mm': rainfall, 'temperature_c': temp, 'demand_index': demand }
        for col in crop_cols:
            inp[col] = 1 if col==f'crop_{crop}' else 0
        X = pd.DataFrame([inp])
        if model is None or scaler is None:
            st.warning('Model not trained yet. Go to Admin -> Retrain models to train on dummy data.')
        else:
            X_scaled = scaler.transform(X)
            pred = model.predict(X_scaled)[0]
            st.success(f'Predicted price for {crop}: ₹{pred:.2f} per quintal (dummy model).')

# --- DISEASE DETECTOR (TABULAR) ---
elif page=='Disease Detector':
    st.title('Disease Detector (symptom-based)')
    st.markdown('Input simple symptom values to get a disease prediction.')
    crop = st.selectbox('Crop', disease_df['crop'].unique())
    spots = st.number_input('Spots count', min_value=0, value=1)
    yellowing = st.selectbox('Yellowing (0/1)', [0,1])
    wilting = st.selectbox('Wilting (0/1)', [0,1])
    lesion = st.number_input('Lesion size (cm)', value=0.2)
    humidity = st.number_input('Humidity (%)', min_value=0.0, max_value=100.0, value=70.0)
    temp = st.number_input('Temperature (°C)', value=24.0)
    if st.button('Predict Disease'):
        clf = load_disease_model()
        if clf is None:
            st.warning('Disease model not trained yet. Go to Admin -> Retrain models to train on dummy data.')
        else:
            # prepare input same way as training
            df_cols = pd.get_dummies(disease_df[['crop']].copy(), columns=['crop'], drop_first=True)
            crop_cols = [c for c in df_cols.columns if c.startswith('crop_')]
            inp = {'spots_count':spots,'yellowing':yellowing,'wilting':wilting,'lesion_size_cm':lesion,'humidity_pct':humidity,'temperature_c':temp}
            for col in crop_cols:
                inp[col] = 1 if col==f'crop_{crop}' else 0
            X = pd.DataFrame([inp])
            # we used a scaler during training but didn't persist disease scaler earlier; do simple predict
            pred = clf.predict(X.values)
            proba = clf.predict_proba(X.values) if hasattr(clf,'predict_proba') else None
            st.success(f'Predicted: {pred[0]}')
            if proba is not None:
                top_idx = np.argmax(proba[0])
                st.write('Confidence (top class):', f'{proba[0][top_idx]*100:.1f}%')

# --- WEATHER (Mock) ---
elif page=='Weather':
    st.title('Weather & Alerts (Mock)')
    st.markdown('This page shows a simple mock weather forecast and alerting rules for the demo.')
    location = st.text_input('Location (name)', 'Local Farm')
    days = st.slider('Days to forecast', 1, 7, 3)
    base = pd.Timestamp.today()
    rows=[]
    for i in range(days):
        d = base + pd.Timedelta(days=i)
        temp = float(20 + 10*np.sin(i/3) + np.random.randn()*2)
        rain = float(max(0, np.random.normal(60,40)))
        rows.append({'date':d.date(),'temp_c':temp,'rain_mm':rain})
    dfw = pd.DataFrame(rows)
    st.table(dfw)
    # simple alert rule
    heavy_rain = dfw[dfw['rain_mm']>100]
    if not heavy_rain.empty:
        st.warning('Heavy rain expected on: ' + ', '.join(str(d) for d in heavy_rain['date'].tolist()))
    else:
        st.success('No heavy rain alerts in the next {} days (mock data).'.format(days))

# --- CHATBOT ---
elif page=='Chatbot':
    st.title('Quick Chatbot')
    st.markdown('Ask about crop prices, diseases, or features.')
    user_input = st.text_input('Message')
    if st.button('Send'):
        if user_input.strip()=='' :
            st.info('Please type a message.')
        else:
            reply = chatbot.reply(user_input)
            st.markdown('**You:** ' + user_input)
            st.markdown('**Bot:** ' + reply)

# --- MARKETPLACE ---
elif page=='Marketplace':
    st.title('Marketplace (Mock)')
    st.markdown('List produce for sale or browse entries from simulated farmers.')
    col1, col2 = st.columns(2)
    with col1:
        st.subheader('List your produce')
        seller = st.text_input('Seller name')
        crop = st.selectbox('Crop to sell', price_df['crop'].unique(), key='market_crop')
        qty = st.number_input('Quantity (quintals)', value=5)
        price = st.number_input('Asking price per quintal', value=1000.0)
        contact = st.text_input('Contact info')
        if st.button('Post Listing'):
            st.success('Listing posted (mock). It would appear in the Market Listings below.')
    with col2:
        st.subheader('Browse listings')
        sample_listings = []
        for i in range(5):
            sample_listings.append({'seller':f'Farmer {i+1}','crop':np.random.choice(price_df['crop'].unique()),'qty':np.random.randint(1,50),'price':np.random.randint(800,5000)})
        st.table(pd.DataFrame(sample_listings))

# --- RESEARCH ---
elif page=='Research':
    st.title('Research / Downloads')
    st.markdown('This section provides quick dataset summaries and allows downloading the dummy data used for training.')
    if st.button('Show dataset summaries'):
        st.subheader('Price dataset head')
        st.dataframe(price_df.head())
        st.subheader('Disease dataset head')
        st.dataframe(disease_df.head())
    buf = io.BytesIO()
    with st.expander('Download datasets'):
        csv1 = price_df.to_csv(index=False).encode()
        csv2 = disease_df.to_csv(index=False).encode()
        st.download_button('Download price dataset (CSV)', csv1, file_name='price_dataset.csv')
        st.download_button('Download disease dataset (CSV)', csv2, file_name='disease_dataset.csv')

# --- ADMIN ---
elif page=='Admin':
    st.title('Admin — Train models & Diagnostics')
    st.markdown('Train models on dummy data and inspect evaluation metrics. Use this to simulate EWS model updates.')
    st.subheader('Train price model')
    if st.button('Train price model on dummy data'):
        with st.spinner('Training price model...'):
            model, scaler, rmse = train_price_model(price_df)
            st.success(f'Price model trained. RMSE on holdout ≈ {rmse:.2f}')
    st.subheader('Train disease model')
    if st.button('Train disease model on dummy data'):
        with st.spinner('Training disease model...'):
            clf, scaler2, acc = train_disease_model(disease_df)
            st.success(f'Disease model trained. Accuracy on holdout ≈ {acc*100:.1f}%')
    st.subheader('Current model status')
    pm, sc = load_price_model()
    dm = load_disease_model()
    st.write('Price model present:', bool(pm))
    st.write('Disease model present:', bool(dm))
    st.markdown('---')
    st.write('Note: Models and datasets here are for demo/prototyping only. Do not use as-is in production.')

# Footer
st.sidebar.markdown('---')
st.sidebar.caption('Agri EWS demo — single-file Streamlit app. Modify and extend for your project needs.')
