import os
import cv2
from keras import Sequential
from keras.layers import Conv2D, Activation, Flatten, Dense
from keras.optimizers import Adam

import numpy as np
import matplotlib.pyplot as plt

image = cv2.imread('./sample.png')
image = cv2.resize(image, None, fx=0.7, fy=0.7, interpolation=cv2.INTER_AREA)
image = np.array(image)
input_shape = image.shape
print(input_shape)

model = Sequential()
model.add(Conv2D(32, kernel_size=8, strides=4, input_shape=input_shape, activation='relu'))
model.add(Conv2D(64, kernel_size=4, strides=2, activation='relu'))
model.add(Conv2D(64, kernel_size=3, strides=1, activation='relu'))
model.add(Flatten())
model.add(Dense(256, activation='relu'))
model.add(Dense(3, activation='linear')) # Output Layer
model.compile(loss='mae', optimizer=Adam(lr=0.001))
print(model.summary())

if os.path.isfile("weight_backup.h5"):
    model.load_weights("weight_backup.h5")

from keras import backend as K


inp = model.input                                           # input placeholder
outputs = [layer.output for layer in model.layers]          # all layer outputs
functor = K.function([inp]+ [K.learning_phase()], outputs ) # evaluation function

# Testing
test = np.random.random(input_shape)[np.newaxis,...]
layer_outs = functor([test, 1.])
print (layer_outs.shape)

# plt.imshow(layer_outs, interpolation='nearest')

# with a Sequential model
# get_3rd_layer_output = K.function([model.layers[0].input],
#                                   [model.layers[1].output])
#
# image = np.expand_dims(image, axis=0)
# layer_output = get_3rd_layer_output([image])[0]
#
# layer_output = np.squeeze(layer_output)
#
# print(layer_output[:][:][0].shape)
#
# plt.imshow(layer_output)