
## importing the important liberaries that we are to use in this program
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


## creating a list to store the input data and the test data in it 

## same goes for labels

inputs = []
labels = []
test_inputs=[]
test_labels=[]

#create a function tha navigate in my system to get the data from files on operating system

#this function takes in the soure folder and two lists to store the inputs and the labels

def collect_data(source,inputs,labels):
    folders = listdir(source)
    for direct in folders:
        for file in listdir(f"{source}{direct}"):

            labels.append(file[:-1])
        
            with open(f"{source}{direct}/{file}", "r") as f:

                inputs += [line.split(",") for line in [line.strip()for line in f.read().split("\n") if line]]

    return inputs,labels


#calling the function collect data
collect_data(source,inputs,labels)

# converting the list of input data to numpy array 
data = np.array(inputs, dtype=np.float32)
## reshape the data so it can fit on the convolution layer of the model
data = data.reshape(12, 5, 11, 1)

print(data.shape)
print(labels)
## encode the labels using label ecoder which convert the words into numbers to fit in the neural network
lab_encoder = LabelEncoder()
labels = lab_encoder.fit_transform(labels)


# define a CNN model using conv2d which is proven to be good for our data

model = Sequential()
## this is called pre-aware quantization 
## this simple line of code is helping our model to know that it is going to be quantized then the accuracy won't drop 
## it leads to a better model 

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

## now let's collect the test data by calling the fuction collect data 

## now we pass the folder that contains the test data and two lists for the test data and labels
collect_data(test_source,test_inputs,test_labels)
## convertion the list to a numpy array
## reshape the test data
test_data = np.array(inputs, dtype=np.float32)
test_data = test_data.reshape(12, 5, 11, 1)

lab_encoder = LabelEncoder()
test_labels = lab_encoder.fit_transform(test_labels)


### now to evaluate the model pass the test data to the model to evaluate the accuracy 
### and see if the model is over fitting 
scores = model.evaluate(test_data, np.array(test_labels), verbose=1, batch_size=11)
print('Accurracy: {}'.format(scores[1]))
