import talky_model
import numpy as np
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
from tensorflow.keras.layers import  Dense, Dropout, Flatten, MaxPooling2D, Conv2D
import tensorflow_model_optimization as tfmot
from tensorflow.keras.models import Sequential
import matplotlib.pyplot as plt

train_source='data/train-data/'
gestures=[]
labels=[]
inputs=[]
words=[]


talky_model.collect_data(train_source,inputs,labels,gestures,words)



gestures,words=talky_model.data_conversion(gestures,words)

model = Sequential()
# this is called pre-aware quantization
# this simple line of code is helping our model to know that it is going to be quantized then the accuracy won't drop
# it leads to a better model

quantized_model = tfmot.quantization.keras.quantize_model

model.add(Conv2D(32, (2, 2), activation='relu',
          padding='valid', input_shape=gestures[0].shape))
model.add(MaxPooling2D(3, (2, 2)))
model.add(Dropout(.02))
model.add(Flatten())


#model.add(Dense(32, activation='relu',input_shape=gestures[0].shape))
#model.add(Dropout(.02))
model.add(Dense(16, activation='relu'))
model.add(Dropout(.02))
model.add(Dense(3, activation='softmax'))

model = quantized_model(model)
 
model.summary()
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
model.fit(gestures,np.array(words,dtype=np.float32), epochs=110,
                                 shuffle=True, batch_size=11)



converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations=[tf.lite.Optimize.DEFAULT]
tflite_model=converter.convert()
with open('lite_model.tflite','wb')as litemodel:
    litemodel.write(tflite_model)


print(os.path.getsize('lite_model.tflite'))


# from sklearn.metrics import confusion_matrix
# y_pred=model.predict(gestures)
# y_pred = np.argmax(y_pred, axis=1)
# cm=confusion_matrix(words, y_pred)
# print(cm)
# print(y_pred)



# from tinymlgen import port
# c_code = port(model, pretty_print=True)
# print(c_code)