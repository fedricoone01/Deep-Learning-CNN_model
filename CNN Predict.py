# -*- coding: utf-8 -*-
"""
Created on Tue Jun  7 14:54:29 2022

@author: Fede
"""

# Import libraries
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from keras.models import load_model
from tensorflow.keras.utils import load_img, img_to_array

# Load trained model
length, height = 150, 150
model = './modelo/modelo.h5'
weights_model = './modelo/pesos.h5'
cnn = load_model(model)
cnn.load_weights(weights_model)

def predict(file):
    x = load_img(file, target_size=(length, height))
    x = img_to_array(x)
    x = np.expand_dims(x, axis=0)
    array = cnn.predict(x)
    print(array)
    result = array[0]
    print(result)
    answer = np.argmax(result)
    print(answer)
    if answer == 0:
        print("pred: Perro")
    elif answer == 1:
        print("pred: Gato")
    elif answer == 2:
        print("pred: Gorila")

# Select test image
image = r"C:\Users\fede_\OneDrive\Escritorio\Python FINANZAS\test_set\test_set\cats\cat.4154.jpg"
img = mpimg.imread(image)
imgplot = plt.imshow(img)
plt.show()

# Prediction
predict(image)


