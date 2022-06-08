# -*- coding: utf-8 -*-
"""
Created on Tue Jun  7 12:03:31 2022

@author: Fede
"""
# Libraries
import os
import matplotlib.pyplot as plt
from tensorflow.python.keras import backend as K
import tensorflow as tf
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dropout, Flatten, Dense
from tensorflow.python.keras.layers import  Convolution2D, MaxPooling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator

K.clear_session()

# Load dataset
data_training = r'C:\Users\fede_\OneDrive\Escritorio\Python FINANZAS\training_set'
data_testing = r'C:\Users\fede_\OneDrive\Escritorio\Python FINANZAS\test_set'

# Parameters
epochs=10
length, height = 150, 150
batch_size = 32
steps = 100
validation_steps = 64
filtersConv1 = 64
filtersConv2 = 128
filter_size1 = (3, 3)
filter_size2 = (2, 2)
pool_size = (2, 2)
classes = 2
lr = 0.0008

# Pre-processing our images
training_datagen = ImageDataGenerator(
        rescale=1. / 255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1. / 255)

training_generator = training_datagen.flow_from_directory(
        data_training,
        target_size=(length, height),
        batch_size=batch_size,
        class_mode='categorical')

testing_generator = test_datagen.flow_from_directory(
        data_testing,
        target_size=(length, height),
        batch_size=batch_size,
        class_mode='categorical')

# Create CNN
cnn = Sequential()
cnn.add(Convolution2D(filtersConv1, filter_size1, padding ="same", input_shape=(length, height, 3), activation='relu'))
cnn.add(MaxPooling2D(pool_size=pool_size))

cnn.add(Convolution2D(filtersConv2, filter_size2, padding ="same"))
cnn.add(MaxPooling2D(pool_size=pool_size))

cnn.add(Flatten())
cnn.add(Dense(256, activation='relu'))
cnn.add(Dropout(0.3))
cnn.add(Dense(classes, activation='sigmoid')) 

# Compile model
cnn.compile(loss='binary_crossentropy',
        optimizer="adadelta",
        metrics=['accuracy'])

# Train the model
history = cnn.fit_generator(
        training_generator,
        steps_per_epoch=steps,
        epochs=epochs,
        validation_data=testing_generator,
        validation_steps=validation_steps)

# Summarize history for accuracy
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

# Summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

# Save the model and heir weights
target_dir = './modelo/'
if not os.path.exists(target_dir):
    os.mkdir(target_dir)
cnn.save('./modelo/modelo.h5')
cnn.save_weights('./modelo/pesos.h5')










