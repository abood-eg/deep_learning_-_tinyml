import talky_model
import numpy as np
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 
from tensorflow.keras.layers import Conv2D, Dense, MaxPooling2D, Flatten, Dropout
import tensorflow as tf
import tensorflow_model_optimization as tfmot
from tensorflow.keras.models import Sequential

train_source='data/train-data/'
newlist=[]
labels=[]
inputs=[]
newlables=[]





talky_model.collect_data(train_source,inputs,labels)
talky_model.collect_datav2(inputs,newlist,labels,newlables)



inputs,labels=talky_model.data_conversion(newlist,newlables)



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
model.fit(inputs,np.array(labels,dtype=np.float32), epochs=110,
                                 shuffle=True, batch_size=11)
