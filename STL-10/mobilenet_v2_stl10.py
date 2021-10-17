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
from tensorflow.keras.applications import MobileNet
from tensorflow.keras.applications.mobilenet import preprocess_input
print(tf.__version__)

path = 'img/'
train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)
test_datagen = ImageDataGenerator(rescale=1./255)
train_generator = train_datagen.flow_from_directory(
        path + 'TRAIN',
        target_size=(96, 96),
        batch_size=32)
validation_generator = test_datagen.flow_from_directory(
        path + 'TEST',
        target_size=(96, 96),
        batch_size=32)

inp = Input(shape = (96,96,3))
gray = tf.image.rgb_to_grayscale(inp)
sobel = tf.image.sobel_edges(gray)
out = MobileNet(input_shape=(96,96, 1), alpha=1., weights=None, classes=10)(sobel[:,:,:,:,0])

model = Model(inputs = inp, outputs = out)
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
print(model.summary())

lr_decayed_fn = tf.keras.optimizers.schedules.CosineDecay(1e-3, 15)
callback = tf.keras.callbacks.LearningRateScheduler(lr_decayed_fn)

print(model.fit(train_generator, epochs=70, validation_data=validation_generator, callbacks = [callback], verbose = 1))

print(model.evaluate(validation_generator))

