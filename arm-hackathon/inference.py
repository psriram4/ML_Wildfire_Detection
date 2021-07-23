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

import contentful
from PIL import Image
import requests
import urllib


IMG_SIZE = 224

def get_img(url):
    # data = []
    url_response = urllib.request.urlopen(url)
    img_array = np.array(bytearray(url_response.read()), dtype=np.float32)
    print("img_arr shape: ", img_array.shape)
    img = cv2.imdecode(img_array, -1)
    print("img shape: ", img.shape)
    img_arr = img[...,::-1]
    resized_arr = cv2.resize(img_arr, (img_size, img_size)) # Reshaping images to preferred size
    # data.append([resized_arr])
    print("resized arr shape: ", resized_arr.shape)
    return [resized_arr]

# use Contentful API to retrieve image
access_token = "FXQohn0Fz1qMBVNeHrsuTJQGPMuQfGFPdFFWYOB8mKA"
space_id = "x26sj0rr7e8n"
client = contentful.Client(space_id, access_token)
entries = client.entries()
assets = client.assets()

train = []
print("Entries: ")
for asset in assets:
    print(asset.url())
    url = "http:"+asset.url()
    train.append(get_img(url))

train = np.array(train)

x_train = []
for feature in train:
    x_train.append(feature)
x_train = np.array(x_train) / 255
x_train.reshape(-1, img_size, img_size, 1)

# reconstruct model
model = tensorflow.keras.models.load_model("wildfire_detection_model.h5")

# output prediction
pred = model.predict(x_train[0])
labels = ["Fire", "Neutral", "Smoke"]
out = labels[list(pred[0]).index(max(pred[0]))]
print("#"*100)
print(out.upper())
print("#"*100)