from tensorflow.keras.layers import Conv1D, Dense, MaxPooling2D, Flatten, Dropout
from tensorflow.keras.models import Sequential
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder
from logging import exception
from os import listdir
import numpy as np
import pandas as pd
import os 

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

source = '/home/abdo_khattab/Documents/work/data/'
folders = listdir(source)
labels = []
inputs = []

for direct in folders:
    for file in listdir(f"{source}{direct}"):

        labels.append(file[:-1])
        with open(f"{source}{direct}/{file}", "r") as f:

            inputs += [line.split(",") for line in [line.replace(" ", "") for line in f.read().split("\n") if line]]


data = np.array(inputs, dtype=np.float32)
data = data.reshape(12, 5, 11, 1)

print(data.shape)
print(labels)

lab_encoder = LabelEncoder()
labels = lab_encoder.fit_transform(labels)

# define model for simple BI-LSTM + DNN based binary classifier

model = Sequential()

model.add(Conv1D(32, 1, activation='relu',
          padding='causal', input_shape=data[0].shape))
model.add(MaxPooling2D(3, (2, 2)))
model.add(Dropout(.02))
model.add(Flatten())

model.add(Dense(32, activation='relu'))
model.add(Dense(7, activation='softmax'))

model.summary()
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
model.fit(data, np.array(labels), epochs=400, batch_size=5)


scores = model.evaluate(data, np.array(labels), verbose=1, batch_size=5)


print('Accurracy: {}'.format(scores[1]))
