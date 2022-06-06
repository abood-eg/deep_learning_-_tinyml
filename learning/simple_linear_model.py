import tensorflow as tf 
import numpy as np
from numpy import array
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import matplotlib.pyplot as plt

X=np.array([0,1,2,3,4,5,6,7,8,9])

y=np.array([0,2,4,6,8,10,12,14,16,18])

model = Sequential()

model.add(Dense(units=1,input_shape=[1]))

model.compile(optimizer=tf.keras.optimizers.Adam(0.01),
              loss='mean_squared_error', metrics=['accuracy'])
history = model.fit(X, np.array(y), epochs=400, batch_size=1)

plt.xlabel=('Epoch Number')
plt.ylabel=('Loss Magnitude')
plt.plot(history.history['loss'])

print(model.predict([-40]))