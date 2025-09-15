import os, random
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
import joblib
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# ======================
# Generate Crop Price Dataset
# ======================
print("🔹 Generating dummy crop price dataset...")
crops = ["Wheat", "Rice", "Maize", "Sugarcane", "Cotton"]
data = []
for i in range(500):
    crop = random.choice(crops)
    temp_max = np.random.uniform(20, 40)
    temp_min = np.random.uniform(5, 25)
    rainfall = np.random.uniform(0, 200)
    month = random.randint(1, 12)
    demand = np.random.randint(50, 200)
    base_price = {"Wheat":2200,"Rice":2500,"Maize":1800,"Sugarcane":300,"Cotton":6000}[crop]
    price = base_price + (0.5 * rainfall) - (temp_max - temp_min) * 10 + demand
    data.append([crop, temp_max, temp_min, rainfall, month, demand, round(price, 2)])
df = pd.DataFrame(data, columns=["crop","temp_max","temp_min","rainfall","month","demand","price"])
df.to_csv("crop_prices_dataset.csv", index=False)
print("✅ crop_prices_dataset.csv saved")

# ======================
# Train Crop Price Model
# ======================
print("🔹 Training crop price model...")
X = df[["crop","temp_max","temp_min","rainfall","month","demand"]]
y = df["price"]
X = pd.get_dummies(X)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
price_model = RandomForestRegressor(n_estimators=50, random_state=42)
price_model.fit(X_train, y_train)
joblib.dump(price_model, "crop_price_model.pkl")
joblib.dump(X.columns, "model_columns.pkl")
print("✅ crop_price_model.pkl & model_columns.pkl saved")

# ======================
# Generate Dummy Disease Images
# ======================
print("🔹 Generating dummy plant disease dataset...")
os.makedirs("disease_data/healthy", exist_ok=True)
os.makedirs("disease_data/diseased", exist_ok=True)
for i in range(50):
    img = Image.new("RGB",(224,224),(34,139,34))
    img.save(f"disease_data/healthy/leaf_{i}.png")
    img = Image.new("RGB",(224,224),(34,139,34))
    draw = ImageDraw.Draw(img)
    for _ in range(10):
        x,y=random.randint(0,200),random.randint(0,200)
        r=random.randint(5,20)
        draw.ellipse((x,y,x+r,y+r),fill=(139,69,19))
    img.save(f"disease_data/diseased/leaf_{i}.png")
print("✅ Dummy disease dataset created")

# ======================
# Train Disease Detection Model
# ======================
print("🔹 Training disease detection model...")
IMG_SIZE=(224,224)
datagen = ImageDataGenerator(validation_split=0.2,rescale=1./255)
train_gen = datagen.flow_from_directory("disease_data",target_size=IMG_SIZE,batch_size=8,subset="training")
val_gen   = datagen.flow_from_directory("disease_data",target_size=IMG_SIZE,batch_size=8,subset="validation")
base_model=tf.keras.applications.MobileNetV2(weights="imagenet",include_top=False,input_shape=(224,224,3))
base_model.trainable=False
x=tf.keras.layers.GlobalAveragePooling2D()(base_model.output)
output=tf.keras.layers.Dense(len(train_gen.class_indices),activation="softmax")(x)
disease_model=tf.keras.Model(inputs=base_model.input,outputs=output)
disease_model.compile(optimizer="adam",loss="categorical_crossentropy",metrics=["accuracy"])
disease_model.fit(train_gen,validation_data=val_gen,epochs=2)
disease_model.save("plant_disease_mobilenetv2.h5")
with open("class_names.txt","w") as f:
    for c in train_gen.class_indices: f.write(f"{c}\n")
print("✅ plant_disease_mobilenetv2.h5 & class_names.txt saved")
print("\n🎉 All required files are ready!")