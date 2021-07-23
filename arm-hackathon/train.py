import numpy as np
import os

import cv2
import matplotlib.pyplot as plt 

import tensorflow as tf
import tensorflow.keras
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.applications import MobileNet
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.layers import Dense, Conv2D , MaxPool2D , Flatten , Dropout , GlobalAveragePooling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# datasets acquired from https://github.com/DeepQuestAI/Fire-Smoke-Dataset/blob/master/
TRAIN_DATASET = "FIRE-SMOKE-DATASET/Train"
VAL_DATASET = "FIRE-SMOKE-DATASET/Test"

IMG_SIZE = 224

# machine learning workflow built based on https://www.analyticsvidhya.com/blog/2020/10/create-image-classification-model-python-keras/
classes = ["Fire", "Neutral", "Smoke"]
def get_data_dir(data_dir):
    """Get data directory based on each label"""
    data = [] 
    for class_ in classes: 
        class_label = classes.index(class_)
        for img in os.listdir(os.path.join(data_dir, label)):
            img_tensor = cv2.imread(os.path.join(path, img))
            img_tensor = img_tensor[...,::-1]
            img_tensor = cv2.resize(img_tensor, (IMG_SIZE, IMG_SIZE))
            data.append([img_tensor, class_labels])
    return np.array(data)


def extract_features(data):
    """Separate features and labels into separate arrays"""
    features = [feature for feature, _ in data]
    labels = [label for _, label in data]
    return features, labels


def normalize_data(arr):
    """Normalize the data"""
    arr /= 255.0
    arr.reshape(-1, IMG_SIZE, IMG_SIZE, 1)

train = get_data_dir(TRAIN_DATASET)
val = get_data_dir(VAL_DATASET)

train_data, train_labels = extract_features(train)
val_data, val_labels = extract_features(val)

normalize_data(train_data)
normalize_data(val_data)

# convert labels 
train_labels = to_categorical(np.array(train_labels), 3)
val_labels = to_categorical(np.array(val_labels), 3)

print("data normalized and preprocessed!")

datagen = ImageDataGenerator()
datagen.fit(train_data)

# fine-tune MobileNet model based on https://towardsdatascience.com/transfer-learning-using-mobilenet-and-keras-c75daf7ff299
base_model = tensorflow.keras.applications.MobileNetV2(input_shape=(224, 224, 3), include_top=False, weights="imagenet")
base_model.trainable = False
model = tensorflow.keras.Sequential([base_model,
                                 tensorflow.keras.layers.GlobalAveragePooling2D(),
                                 tensorflow.keras.layers.Dropout(0.1),
                                 tensorflow.keras.layers.Dense(3, activation="softmax")                                     
                                ])

model.compile(optimizer=tensorflow.keras.optimizers.Adam(learning_rate=0.0001), loss=tensorflow.keras.losses.CategoricalCrossentropy(), metrics=['accuracy'])
model.fit(train_data, train_labels, batch_size=1, epochs=10, validation_data=(val_data, val_labels))

# save model
model.save("wildfire_detection_model.h5")