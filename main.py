from tensorflow import keras
import numpy as np
from google.colab import files
from tensorflow.keras.preprocessing import image

print("Upload model file (.h5)")
uploaded_model = files.upload()

model_name = list(uploaded_model.keys())[0]
model = keras.models.load_model(model_name)

plants = [
"Aloe",
"Corn",
"Guava",
"Hibiscus",
"Lemon",
"Mango",
"Neem",
"Rice",
"Tomato",
"Tulsi"
]

print("Upload leaf image")
uploaded_image = files.upload()

img_name = list(uploaded_image.keys())[0]

img = image.load_img(
img_name,
target_size=(224, 224)
)

img_array = image.img_to_array(img)
img_array = np.expand_dims(img_array, axis=0)

prediction = model.predict(img_array)

index = np.argmax(prediction)
confidence = np.max(prediction) * 100

print("Predicted Plant:", plants[index])
print("Confidence:", round(confidence, 2), "%")
