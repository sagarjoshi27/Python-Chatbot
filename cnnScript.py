# -*- coding: utf-8 -*-
"""
Created on Tue Dec 10 10:05:16 2019

@author: HP
"""

# Importing Libraries
from __future__ import absolute_import, division, print_function, unicode_literals

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import tensorflow.keras.backend as K
import matplotlib.pyplot as plt
import pandas as pd
import sklearn as sk
import numpy as np
import random

import warnings
warnings.filterwarnings("ignore")

# Collecting and Splitting the Dataset
#Augmentation Configuration used for training
train_datagen = ImageDataGenerator(rescale = 1./255,
                                   shear_range = 0.2,
                                   zoom_range = 0.2,
                                   horizontal_flip = True)

#Augmentation configuration used for testing, also rescale images
test_datagen = ImageDataGenerator(rescale = 1./255)

#Read images found in sub-folders of data/train and then data/test
training_set = train_datagen.flow_from_directory('data/train',
                                                 target_size = (60, 60),
                                                 batch_size = 18,
                                                 class_mode = 'categorical')

test_set = test_datagen.flow_from_directory('data/test',
                                            target_size = (60, 60),
                                            batch_size = 18,
                                            class_mode = 'categorical')

# Building the CNN
model = Sequential()

#Adding Three Convolution Layers
model.add(Conv2D(32,(3,3), input_shape = (60,60,3)))
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Conv2D(64,(3,3)))
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.25))

model.add(Conv2D(64,(3,3)))
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.25))

#Hidden layers
model.add(Flatten())
model.add(Dense(128))
model.add(Activation("relu"))

model.add(Dense(128))
model.add(Activation("relu"))

#Output Layer
model.add(Dense(4))
model.add(Activation("softmax"))

model.summary()

# Compile the model
model.compile(loss="categorical_crossentropy",
              optimizer="adam", metrics=["accuracy"])

model.fit_generator(
    training_set,
    steps_per_epoch = 500,
    epochs = 50,
    validation_data = test_set,
    validation_steps = 100,
    verbose = 2)

model.save('sport_identifier.h5')
