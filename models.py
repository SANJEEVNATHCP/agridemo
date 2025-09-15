import tensorflow as tf
import joblib
import pandas as pd
import numpy as np
import os
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import OneHotEncoder

DISEASE_MODEL_PATH = "plant_disease_mobilenetv2.keras"
CLASS_NAMES_PATH = "class_names.txt"
PRICE_MODEL_PATH = "crop_price_model.pkl"
COLUMNS_PATH = "model_columns.pkl"

def create_dummy_disease_model():
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(224,224,3)),
        tf.keras.layers.Conv2D(8, (3,3), activation="relu"),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(2, activation="softmax")
    ])
    model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
    model.save(DISEASE_MODEL_PATH)
    with open(CLASS_NAMES_PATH,"w") as f:
        f.write("healthy\n")
        f.write("diseased\n")

def create_dummy_price_model():
    data = pd.DataFrame({
        "crop":["wheat","rice","maize","cotton"]*10,
        "temp_max":np.random.randint(25,40,40),
        "temp_min":np.random.randint(15,25,40),
        "rainfall":np.random.randint(50,200,40),
        "month":np.random.randint(1,12,40),
        "demand":np.random.randint(1,10,40),
        "price":np.random.randint(1000,3000,40)
    })

    X = data[["crop","temp_max","temp_min","rainfall","month","demand"]]
    y = data["price"]

    enc = OneHotEncoder(handle_unknown="ignore")
    X_enc = enc.fit_transform(X[["crop"]]).toarray()
    X_final = np.concatenate([X_enc, X.drop("crop",axis=1)], axis=1)

    model = LinearRegression()
    model.fit(X_final,y)

    joblib.dump((model,enc), PRICE_MODEL_PATH)
    joblib.dump(X.columns.tolist(), COLUMNS_PATH)

def load_models():
    # disease model
    if not os.path.exists(DISEASE_MODEL_PATH):
        create_dummy_disease_model()
    disease_model = tf.keras.models.load_model(DISEASE_MODEL_PATH)
    with open(CLASS_NAMES_PATH,"r") as f:
        DISEASE_CLASSES = [line.strip() for line in f]

    # price model
    if not os.path.exists(PRICE_MODEL_PATH) or not os.path.exists(COLUMNS_PATH):
        create_dummy_price_model()
    model, enc = joblib.load(PRICE_MODEL_PATH)
    model_columns = joblib.load(COLUMNS_PATH)

    class PriceModelWrapper:
        def _init_(self, model, enc):
            self.model = model
            self.enc = enc
        def predict(self, df):
            X_enc = self.enc.transform(df[["crop"]]).toarray()
            X_final = np.concatenate([X_enc, df.drop("crop",axis=1)], axis=1)
            return self.model.predict(X_final)

    return disease_model, DISEASE_CLASSES, PriceModelWrapper(model,enc), model_columns

def recommend_for_farmer(crop):
    recs = {
        "wheat":"Use nitrogen fertilizer, consider rice as rotation.",
        "rice":"Use phosphate-rich fertilizer, consider maize as rotation.",
        "maize":"Balanced NPK recommended, cotton works well after maize.",
        "cotton":"Potassium-rich fertilizer, rotate with wheat."
    }
    return recs.get(crop,"Try soil testing for personalized advice.")