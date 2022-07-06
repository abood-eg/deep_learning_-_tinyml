import talky_model
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import numpy as np
import os 
import pandas as pd
import csv
source='data/train-data/'
gestures=[]
words=[]
inputs=[]
labels=[]


talky_model.collect_data(source, inputs, labels, gestures, words)

gestures,words=talky_model.data_conversion(gestures, words)
gestures=gestures.reshape(15,44)
   

xtrain,xtest,ytrain,ytest=train_test_split(gestures, words,test_size=.4 ,shuffle=True)

logreg=LogisticRegression()
logreg.fit(xtrain,ytrain)
ypred=logreg.predict(xtest)

from sklearn.metrics import confusion_matrix

cm=confusion_matrix(ytest, ypred)
print(cm)