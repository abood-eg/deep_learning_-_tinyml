import talky_model

import training

test_source ='data/test-data/'

test_labels=[]
test_inputs=[]
test_words=[]
test_gestures=[]



# now let's collect the test data by calling the fuction collect data
talky_model.collect_data(test_source, test_inputs,test_labels,test_gestures,test_words)

# now we pass the folder that contains the test data and two lists for the test data and labels
# convertion the list to a numpy array
# reshape the test data
test_gestures,test_words=talky_model.data_conversion(test_gestures,test_words)
# now to evaluate the model pass the test data to the model to evaluate the accuracy
# and see if the model is over fitting
scores = training.model.evaluate(test_gestures,test_words, verbose=1, batch_size=11)
print('Accurracy: {}'.format(scores[1]))


from sklearn.metrics import confusion_matrix

cm = confusion_matrix(training.words, test_words)
print(cm)

