from tensorflow.keras.layers import Conv2D, Dense, MaxPooling2D, LSTM, Flatten, Activation, Dropout
from tensorflow.keras.models import Sequential
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder
from logging import exception
from os import listdir
import numpy as np
import pandas as pd
source = '/home/abdo_khattab/Documents/work/data/'
folders=listdir(source)
labels = []
inputs = []
for direct in folders:
    for file in listdir(source+direct):
    
        labels.append(file)
        with open(source+direct+'/'+file, "r") as f:
            file = f.read()
            file = file.strip()
            file = file.replace(' ', '')
            lines = file.splitlines()
            for i in lines:
                lines[lines.index(i)] = i.split(',')
            inputs.append(lines)
data = np.array(inputs, dtype=np.float32)
data = data.reshape(7, 20, 11, 1)
print(data.shape)
print(labels)
lenc = LabelEncoder()
labels = lenc.fit_transform(labels)
labels = np.array(labels)

# define model for simple BI-LSTM + DNN based binary classifier

model = Sequential()

model.add(Conv2D(16, (2, 2), activation='relu', input_shape=data[0].shape))
model.add(Conv2D(32, (2, 2), activation='relu'))
model.add(Flatten())
model.add(Dense(32, activation='relu'))
model.add(Dense(7, activation='softmax'))
model.summary()
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.fit(data, np.array(labels), epochs=400, batch_size=5)
