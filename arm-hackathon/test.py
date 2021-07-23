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

IMG_SIZE = 224
IMG_NAME = "smoke.jpg"

# machine learning workflow built based on https://www.analyticsvidhya.com/blog/2020/10/create-image-classification-model-python-keras/

def load_img(img):
    """Load input image from given img name"""
    data = []
    img_tensor = cv2.imread(os.path.join("samples", img))
    img_tensor = img_tensor[...,::-1]
    img_tensor = cv2.resize(img_tensor, (IMG_SIZE, IMG_SIZE))
    data.append([img_tensor])

    return np.array(data)

train = load_img(IMG_NAME)

x_train = []
for feature in train:
    x_train.append(feature)
x_train = np.array(x_train) / 255
x_train.reshape(-1, IMG_SIZE, IMG_SIZE, 1)

# reconstruct model
model = tensorflow.keras.models.load_model("wildfire_detection_model.h5")

# output prediction
pred = model.predict(x_train[0])
labels = ["Fire", "Neutral", "Smoke"]
out = labels[list(pred[0]).index(max(pred[0]))]
print("#"*100)
print(out.upper())
print("#"*100)