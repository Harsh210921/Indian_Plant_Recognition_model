import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt
from google.colab import drive

#Connect Google Drive

drive.mount('/content/drive')

print("Loading Training Data...")

#Load training dataset

train_dataset = keras.utils.image_dataset_from_directory(
directory='/content/drive/MyDrive/common_plants/TRAIN',
image_size=(224, 224),
batch_size=32
)

print("Loading Validation Data...")

#Load validation dataset

valid_dataset = keras.utils.image_dataset_from_directory(
directory='/content/drive/MyDrive/common_plants/VALID',
image_size=(224, 224),
batch_size=32
)

#Load MobileNetV2 model with pre-trained ImageNet weights

base_model = keras.applications.MobileNetV2(
input_shape=(224, 224, 3),
include_top=False,
weights='imagenet'
)

#Freeze MobileNetV2 layers

base_model.trainable = False

#Convert extracted features into a single vector

x = keras.layers.GlobalAveragePooling2D()(base_model.output)

#Output layer for 10 plant classes

final_predictions = keras.layers.Dense(
units=10,
activation='softmax'
)(x)

#Create final model

model = keras.Model(
inputs=base_model.input,
outputs=final_predictions
)

#Configure training settings

model.compile(
optimizer='adam',
loss='sparse_categorical_crossentropy',
metrics=['accuracy']
)

print("Starting Training...")

#Train the model

training_history = model.fit(
train_dataset,
validation_data=valid_dataset,
epochs=10
)

print("Loading Test Data...")

#Load test dataset

test_dataset = keras.utils.image_dataset_from_directory(
directory='/content/drive/MyDrive/common_plants/TEST',
image_size=(224, 224),
batch_size=32
)

print("Evaluating Model...")

#Check model performance on test data

final_loss, final_accuracy = model.evaluate(test_dataset)

print("Official Final Accuracy:", final_accuracy)

#Save trained model

model.save('/content/drive/MyDrive/plant_master2_model.h5')

print("Model saved successfully!")
