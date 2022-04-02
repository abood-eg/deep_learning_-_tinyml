from tensorflow.keras.layers import Conv2D, Dense, MaxPooling2D, Flatten, Dropout
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder
from logging import exception
from os import listdir
import numpy as np
import pandas as pd
import os
import tensorflow_model_optimization as tfmot
from tensorflow.keras.models import Sequential

source = '/home/abdo_khattab/Documents/work/train-data/'
test_source= '/home/abdo_khattab/Documents/work/test-data/' 
folders = listdir(source)
test_folders=listdir(test_source)
labels = []
inputs = []
test_inputs=[]
test_labels=[]

for direct in folders:
    for file in listdir(f"{source}{direct}"):

        labels.append(file[:-1])
        with open(f"{source}{direct}/{file}", "r") as f:

            inputs += [line.split(",") for line in [line.strip()for line in f.read().split("\n") if line]]



for direct in folders:
    for file in listdir(f"{test_source}{direct}"):

        test_labels.append(file[:-1])
        with open(f"{source}{direct}/{file}", "r") as f:

            test_inputs += [line.split(",") for line in [line.strip()for line in f.read().split("\n") if line]]



data = np.array(inputs, dtype=np.float32)
data = data.reshape(12, 5, 11, 1)
test_data=np.array(test_inputs,dtype=np.float32)
test_data = test_data.reshape(12, 5, 11, 1)
print(data.shape)
print(labels)
print(test_labels)
lab_encoder = LabelEncoder()
labels = lab_encoder.fit_transform(labels)
test_labels=lab_encoder.fit_transform(test_labels)

# define a CNN model using conv1d which is proven to be verry efficent in time serie data 

model = Sequential()

quantized_model=tfmot.quantization.keras.quantize_model


model.add(Conv2D(32,(2,2), activation='relu',
          padding='valid', input_shape=data[0].shape))
model.add(MaxPooling2D(3, (2, 2)))
model.add(Flatten())

model.add(Dense(32, activation='relu'))
model.add(Dense(7, activation='softmax'))

model=quantized_model(model)

model.summary()
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
model.fit(data, np.array(labels), epochs=110, shuffle=True, batch_size=11)


scores = model.evaluate(test_data, np.array(test_labels), verbose=1, batch_size=11)


print('Accurracy: {}'.format(scores[1]))
