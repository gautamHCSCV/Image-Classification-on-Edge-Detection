import tensorflow as tf
import os
import warnings
warnings.filterwarnings("ignore")
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import Input, Dense, Conv2D, MaxPool2D, Flatten
from tensorflow.keras.models import Model
print(tf.__version__)

path = '../Images/cifar-100-images/CIFAR100/'

train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)
test_datagen = ImageDataGenerator(rescale=1./255)
train_generator = train_datagen.flow_from_directory(
        path + 'TRAIN',
        target_size=(150, 150),
        batch_size=32)
validation_generator = test_datagen.flow_from_directory(
        path + 'TEST',
        target_size=(150, 150),
        batch_size=32)


# Sobel in y-direction

inp = Input(shape = (150,150,3))
sobel = tf.image.sobel_edges(inp)
x = Conv2D(16,(3,3), activation = 'relu')(sobel[:, :, :, :, 0])
x = MaxPool2D(2)(x)
x = Conv2D(32,(3,3), activation = 'relu')(x)
x = MaxPool2D(3)(x)
x = Flatten()(x)
out = Dense(100, activation = 'softmax')(x)

model = Model(inputs = inp, outputs = out)
print(model.summary())
model.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])
model.fit(
        train_generator,
        epochs=15,
        validation_data=validation_generator)


# Sobel in x-direction

inp = Input(shape = (150,150,3))
sobel = tf.image.sobel_edges(inp)
x = Conv2D(16,(3,3), activation = 'relu')(sobel[:, :, :, :, 1])
x = MaxPool2D(2)(x)
x = Conv2D(32,(3,3), activation = 'relu')(x)
x = MaxPool2D(3)(x)
x = Flatten()(x)
out = Dense(100, activation = 'softmax')(x)

model = Model(inputs = inp, outputs = out)
model.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])
model.fit(
        train_generator,
        epochs=15,
        validation_data=validation_generator)


# Mobilenet V2
inp = Input(shape = (150,150,3))
sobel = tf.image.sobel_edges(inp)
x = tf.keras.applications.mobilenet_v2.MobileNetV2(input_shape=(150,150,3),include_top = False)(sobel[:,:,:,:,0])
x = Flatten()(x)
out = Dense(100, activation = 'softmax')(x)

model = Model(inputs = inp, outputs = out)
print(model.summary())

model.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])
model.fit(
        train_generator,
        epochs=45,
        validation_data=validation_generator)

print(model.evaluate(validation_generator))