import tensorflow as tf

model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(224,224,3)),
    tf.keras.layers.Conv2D(8,(3,3),activation="relu"),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(2,activation="softmax")
])
model.compile(optimizer="adam",loss="categorical_crossentropy",metrics=["accuracy"])
model.save("plant_disease_mobilenetv2.keras")

with open("class_names.txt","w") as f:
    f.write("healthy\n")
    f.write("diseased\n")

print("✅ Dummy model + class_names.txt created")