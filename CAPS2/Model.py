from __future__ import print_function
import numpy as np
from mnist import MNIST
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D


data_set = MNIST('../MNIST/')

data_set.load_training()

train_img = data_set.train_images
train_img = np.array(train_img).reshape((-1, 28, 28, 1))/255.0

train_labels = data_set.train_labels
train_labels = np.array(train_labels)

data_set.load_testing()

test_img = data_set.test_images
test_img = np.array(test_img).reshape((-1, 28, 28, 1))/255.0

test_labels = data_set.test_labels
test_labels = np.array(test_labels)


model = Sequential()
model.add(Conv2D(32, (5, 5), padding='same', activation='relu', input_shape=(28, 28, 1)))
model.add(Conv2D(32, (3, 3), padding='same', activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(64, (3, 3), padding='same', activation='relu'))
model.add(Conv2D(64, (3, 3), padding='valid', activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(128, (3, 3), padding='valid', activation='relu'))
model.add(Flatten())
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(128, activation='relu'))
model.add(Dense(10, activation='softmax'))
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
h = model.fit(train_img, train_labels, batch_size=512, epochs=3, validation_split=0.1)

score = model.evaluate(test_img, test_labels, verbose=0)
print(score)


model.save('trained_model.h5')




