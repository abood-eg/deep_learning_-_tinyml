'''
this program aims to train a model that takes in sensor reading that represent a hand gestures 
then classify each gesture and give them a unique label

'''
# importing the important liberaries that we are to use in this program
from sklearn.preprocessing import LabelEncoder 
from os import listdir
import numpy as np


# set the pathes to the folder wich contain the train and test data


# creating a list to store the input data and the test data in it
# same goes for labels
test_source ='data/test-data/'
train_source='data/train-data/'
no_of_lines_per_reading=4



# create a function tha navigate in my system to get the data from files on operating system

# this function takes in the soure folder and two lists to store the inputs and the labels


def collect_data(source, inputs,labels,gesture,words):
    for file_name in listdir(source):
        
        with open(f'{source}{file_name}', "r") as f:
            file = f.read()
            file = file.strip()
            file = file.replace(' ', '')
            lines = file.splitlines()
            for ind,line in enumerate(lines):
                lines[ind] = line.split(',')
            inputs.append(lines)
            labels.append(file_name[:-4])
    for file_no in range(len(inputs)):    
        for line_no in range (0,int(len(inputs[0])),no_of_lines_per_reading):
            gesture.append(inputs[file_no][line_no:line_no+no_of_lines_per_reading])
            words.append(labels[file_no])

## this function convert each of the input data and the labels into a numpy array 
def data_conversion(gestures,words):
    gestures = np.array(gestures, dtype=np.float32)
    
# reshape the data so it can fit on the convolution layer of the model
    gestures = gestures.reshape(15,4,11,1)
# encode the labels using label ecoder which convert the words into numbers to fit in the neural network
    lab_encoder = LabelEncoder()
    words = lab_encoder.fit_transform(words)
    return gestures,words
