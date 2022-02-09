from keras.applications.vgg16 import VGG16
from tensorflow.keras.preprocessing import image
from PIL import Image
from tensorflow.keras.applications.vgg16 import preprocess_input, decode_predictions
import numpy as np


# get the pre-trained model
model = VGG16(weights='imagenet')

# model summary
# print(model.summary())

# loading a sample image
img_path = "input/dog.jpg"

img = image.load_img(img_path, color_mode='rgb', target_size=(224, 224))

# ===== Resizing to fit into VGGNet ======

# converts a PIL image to 3D Num Array
x = image.img_to_array(img)
# Adding the 4th dimension, for number of images
x = np.expand_dims(x, axis=0)

# ==== Using the pre-trained model for prediction =====
x = preprocess_input(x)
features = model.predict(x)
p = decode_predictions(features)

print(p)
