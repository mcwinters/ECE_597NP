# https://stackabuse.com/courses/practical-deep-learning-for-computer-vision-with-python/lessons/image-classification-with-transfer-learning-creating-cutting-edge-cnn-models/
# ^^^^^^ source for help ^^^^^^
import urllib.request
import string

import matplotlib.pyplot as plt
import numpy as np
import tensorflow
from PIL import ImageEnhance
from tensorflow import keras
from tensorflow.keras.preprocessing import image

# Public domain image
url = "https://upload.wikimedia.org/wikipedia/commons/0/02/Black_bear_large.jpg"
urllib.request.urlretrieve(url, "bear.jpg")

# Load image and resize (doesn't keep aspect ratio)
img = image.load_img("bear.jpg", target_size=(224, 224))
img_rotated_45 = image.load_img("bear.jpg", target_size=(224, 224)).rotate(45)
img_rotated_90 = image.load_img("bear.jpg", target_size=(224, 224)).rotate(90)
img_darkened = ImageEnhance.Brightness(img).enhance(0.5)
img_brightened = ImageEnhance.Brightness(img).enhance(1.5)
img.save('image.jpg')
img_rotated_45.save('image_rotated_45.jpg')
img_rotated_90.save('image_rotated_90.jpg')
img_brightened.save('image_brightened.jpg')
img_darkened.save('image_darkened.jpg')
# Turn to array of shape (224, 224, 3)
img = image.img_to_array(img)
img_rotated_45 = image.img_to_array(img_rotated_45)
img_rotated_90 = image.img_to_array(img_rotated_90)
img_darkened = image.img_to_array(img_darkened)
img_brightened = image.img_to_array(img_brightened)
# Expand array into (1, 224, 224, 3)
img = np.expand_dims(img, 0)
img_rotated_45 = np.expand_dims(img_rotated_45, 0)
img_rotated_90 = np.expand_dims(img_rotated_90, 0)
img_darkened = np.expand_dims(img_darkened, 0)
img_brightened = np.expand_dims(img_brightened, 0)
# Preprocess for models that have specific preprocess_input() function
# img_preprocessed = preprocess_input(img)

# Load model and run prediction
effnet = keras.applications.EfficientNetB0(weights="imagenet", include_top=True)
pred = effnet.predict(img)
pred_rotated_45 = effnet.predict(img_rotated_45)
pred_rotated_90 = effnet.predict(img_rotated_90)
pred_darkened = effnet.predict(img_darkened)
pred_brightened = effnet.predict(img_brightened)
decoded_original = tensorflow.keras.applications.imagenet_utils.decode_predictions(pred)
decoded_rotated_45 = tensorflow.keras.applications.imagenet_utils.decode_predictions(pred_rotated_45)
decoded_rotated_90 = tensorflow.keras.applications.imagenet_utils.decode_predictions(pred_rotated_90)
decoded_darkened = tensorflow.keras.applications.imagenet_utils.decode_predictions(pred_darkened)
decoded_brightened = tensorflow.keras.applications.imagenet_utils.decode_predictions(pred_brightened)

image_data_original = {}
image_data_rotated_45 = {}
image_data_rotated_90 = {}
image_data_darkened = {}
image_data_brightened = {}


def plot_original():
    for row in decoded_original:
        for _, species, confidence in row:
            image_data_original[string.capwords(species.replace("_", " "))] = confidence

    figure = plt.figure(figsize=(10, 5))
    species = image_data_original.keys()
    confidences = image_data_original.values()
    plt.bar(species, confidences)
    plt.xlabel("Species")
    plt.ylabel("Confidences")
    plt.title("Confidences of Original Image")
    plt.ylim(0, 1)
    plt.show()


def plot_rotated_45():
    for row in decoded_rotated_45:
        for _, species, confidence in row:
            image_data_rotated_45[string.capwords(species.replace("_", " "))] = confidence

    figure = plt.figure(figsize=(10, 5))
    species = image_data_rotated_45.keys()
    confidences = image_data_rotated_45.values()
    plt.bar(species, confidences)
    plt.xlabel("Species")
    plt.ylabel("Confidences")
    plt.title("Confidences of Rotated Image (45 Degrees)")
    plt.ylim(0, 1)
    plt.show()


def plot_rotated_90():
    for row in decoded_rotated_90:
        for _, species, confidence in row:
            image_data_rotated_90[string.capwords(species.replace("_", " "))] = confidence

    figure = plt.figure(figsize=(10, 5))
    species = image_data_rotated_90.keys()
    confidences = image_data_rotated_90.values()
    plt.bar(species, confidences)
    plt.xlabel("Species")
    plt.ylabel("Confidences")
    plt.title("Confidences of Rotated Image (90 Degrees)")
    plt.ylim(0, 1)
    plt.show()


def plot_darkened():
    for row in decoded_darkened:
        for _, species, confidence in row:
            image_data_darkened[string.capwords(species.replace("_", " "))] = confidence

    figure = plt.figure(figsize=(10, 5))
    species = image_data_darkened.keys()
    confidences = image_data_darkened.values()
    plt.bar(species, confidences)
    plt.xlabel("Species")
    plt.ylabel("Confidences")
    plt.title("Confidences of Darkened Image")
    plt.ylim(0, 1)
    plt.show()


def plot_brightened():
    for row in decoded_brightened:
        for _, species, confidence in row:
            image_data_brightened[string.capwords(species.replace("_", " "))] = confidence

    figure = plt.figure(figsize=(10, 5))
    species = image_data_brightened.keys()
    confidences = image_data_brightened.values()
    plt.bar(species, confidences)
    plt.xlabel("Species")
    plt.ylabel("Confidences")
    plt.title("Confidences of Brightened Image")
    plt.ylim(0, 1)
    plt.show()


plot_original()
plot_rotated_45()
plot_rotated_90()
plot_darkened()
plot_brightened()
