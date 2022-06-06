'''
this program aims to train a model that takes in sensor reading that represent a hand gestures 
then classify each gesture and give them a unique label

'''
# importing the important liberaries that we are to use in this program
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 
from sklearn.preprocessing import LabelEncoder 
from os import listdir
import numpy as np


# set the pathes to the folder wich contain the train and test data


# creating a list to store the input data and the test data in it
# same goes for labels
test_source ='data/test-data/'
train_source='data/train-data/'
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

## this function convert each of the input data and the labels into a numpy array 
def data_conversion(newlist,newlabels):
    data = np.array(newlist, dtype=np.float32)
# reshape the data so it can fit on the convolution layer of the model
    data = data.reshape(12, 5, 11, 1)
# encode the labels using label ecoder which convert the words into numbers to fit in the neural network
    lab_encoder = LabelEncoder()
    newlabels = lab_encoder.fit_transform(newlabels)
 
    return data,newlabels
