'''
this program aims to train a model that takes in sensor reading that represent a hand gestures 
then classify each gesture and give them a unique label

'''
# importing the important liberaries that we are to use in this program
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 
from tensorflow.keras.layers import Conv2D, Dense, MaxPooling2D, Flatten, Dropout
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder 
from os import listdir
import numpy as np
import tensorflow_model_optimization as tfmot
from tensorflow.keras.models import Sequential


# set the pathes to the folder wich contain the train and test data


# creating a list to store the input data and the test data in it
# same goes for labels
test_source ='data/test-data/'
train_source='data/train-data/'

test_data=[]
test_labels=[]
test_inputs=[]
testnewlables=[]
testnewlist=[]

newlist=[]
labels=[]
inputs=[]
newlables=[]

no_of_lines_per_reading=5
# create a function tha navigate in my system to get the data from files on operating system

# this function takes in the soure folder and two lists to store the inputs and the labels


def collect_data(source, inputs,labels):
    for file_name in listdir(source):
        
        with open(f'{source}{file_name}', "r") as f:
            file = f.read()
            file = file.strip()
            file = file.replace(' ', '')
            lines = file.splitlines()
            for ind,line in enumerate(lines):
                lines[ind] = line.split(',')
            inputs.append(lines)
            labels.append(file_name)
    
def collect_datav2(inputs,List,labels,newlabels):
    # number of files = length of the inputs list
    for j in range(len(inputs)):    
        for i in range (0,int(len(inputs[0])),no_of_lines_per_reading):
            List.append(inputs[j][i:i+no_of_lines_per_reading])
            newlabels.append(labels[j])

collect_data(train_source,inputs,labels)
collect_datav2(inputs,newlist,labels,newlables)

collect_data(test_source, test_inputs,test_labels)
collect_datav2(test_inputs,testnewlist,labels,testnewlables)
## this function convert each of the input data and the labels into a numpy array 
def data_conversion(newlist,newlabels):
    data = np.array(newlist, dtype=np.float32)
# reshape the data so it can fit on the convolution layer of the model
    data = data.reshape(12, 5, 11, 1)
# encode the labels using label ecoder which convert the words into numbers to fit in the neural network
    lab_encoder = LabelEncoder()
    newlabels = lab_encoder.fit_transform(newlabels)
 
    return data,newlabels

inputs,labels=data_conversion(newlist,newlables)
# define a CNN model using conv2d which is proven to be good for our data
model = Sequential()
# this is called pre-aware quantization
# this simple line of code is helping our model to know that it is going to be quantized then the accuracy won't drop
# it leads to a better model

quantized_model = tfmot.quantization.keras.quantize_model


model.add(Conv2D(32, (2, 2), activation='relu',
          padding='valid', input_shape=inputs[0].shape))
model.add(MaxPooling2D(3, (2, 2)))
model.add(Dropout(.02))
model.add(Flatten())

model.add(Dense(32, activation='relu'))
model.add(Dense(7, activation='softmax'))

model = quantized_model(model)

model.summary()
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
model.fit(inputs,np.array(labels,dtype=np.float32), epochs=110, shuffle=True, batch_size=11)

# now let's collect the test data by calling the fuction collect data

# now we pass the folder that contains the test data and two lists for the test data and labels
# convertion the list to a numpy array
# reshape the test data
test_inputs,test_labels=data_conversion(testnewlist,testnewlables)
# now to evaluate the model pass the test data to the model to evaluate the accuracy
# and see if the model is over fitting
scores = model.evaluate(test_inputs,test_labels, verbose=1, batch_size=11)
print('Accurracy: {}'.format(scores[1]))

