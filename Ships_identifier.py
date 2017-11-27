import json
import numpy as np
from matplotlib import pyplot as plt
from keras.utils.np_utils import to_categorical
import keras
from keras.models import Sequential
from keras.layers.core import Flatten, Dense, Dropout, Lambda
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.optimizers import Adam, SGD

f = open(r'D:/Datascience_lessons/raw data/shipsnet.7z/train/shipsnet.json')

#f = open(r'/train/shipsnet.json')
dataset = json.load(f)
f.close()

data = np.array(dataset['data']).astype('float32')
labels = np.array(dataset['labels']).astype('float32')
print(data.shape)
x = data / 255.
print(data.shape)

x = x.reshape([-1, 3, 80, 80]).transpose([0,2,3,1])

n_train = len(labels)

np.random.seed(seed=315)

percentage = 0.8


training_size = int(percentage*n_train)
mask=np.random.permutation(np.arange(n_train))[:training_size]

x_train, y_train = x [mask], labels[mask]
x_val, val_y = np.delete(x, mask,0), np.delete(labels, mask,0)
print(len(x_train))
print('***')
print(len(x_val))


train_y = to_categorical(y_train, num_classes=2)
y_val = to_categorical(val_y, num_classes=2)

num_classes  = 2
input_shape = (80, 80, 3)

model = Sequential()
model.add(Conv2D(32, kernel_size=(5, 5), strides=(1, 1),
                 activation='relu',
                 input_shape=input_shape))

model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
model.add(Conv2D(64, (5, 5), activation='relu'))
model.add(Dropout(0.35))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.2))
model.add(Flatten())
model.add(Dense(1000, activation='relu'))
model.add(Dropout(0.3))

model.add(Dense(num_classes, activation='softmax'))
model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.SGD(lr=0.01),
              metrics=['accuracy'])

model.fit(x_train, train_y, batch_size=32, epochs=20, validation_data=[x_val,y_val],shuffle=True)
model.save('Ships_CNN1.h5')


