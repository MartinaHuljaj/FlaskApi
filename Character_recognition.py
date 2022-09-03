import tensorflow as tf
import numpy as np
import random
from tensorflow import keras
from emnist import extract_training_samples
images, labels = extract_training_samples('byclass')
from keras import Sequential
from keras.layers import BatchNormalization, Dropout, Flatten, Dense
from keras.layers.convolutional import Conv2D, MaxPooling2D

print(images.shape)
print(labels.shape)
train_data =(np.expand_dims(images, axis=-1)/255.).astype(np.float32)
train_labels =(labels).astype(np.int64)

# length=len(images)
# split_index=int(0.9*length)
# train_data=images[:split_index]
# train_labels=images[:split_index]
# test_data=images[split_index:]
# test_labels=images[split_index:]

VALIDATION_SPLIT = 0.2
# mnist = tf.keras.datasets.mnist
# (train_images, train_labels), (test_images, test_labels) = mnist.load_data()
# # učitavamo podatke
# train_data =(np.expand_dims(train_images, axis=-1)/255.).astype(np.float32)
# train_labels =(train_labels).astype(np.int64)
# test_data =(np.expand_dims(test_images, axis=-1)/255.).astype(np.float32)
# test_labels =(test_labels).astype(np.int64)

def create_model():
    model = Sequential()

    model.add(Conv2D(24, (3, 3), input_shape=(28, 28, 1), activation="relu"))
    model.add(MaxPooling2D(2, 2))
    model.add(Conv2D(36, (3, 3), activation="relu"))
    model.add(MaxPooling2D(2, 2))
    model.add(Flatten())

    model.add(Dense(256, activation="relu"))
    model.add(Dense(62, activation="softmax"))

    model.compile(loss="sparse_categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
    model.summary()
    return model

model=create_model()
model.fit(train_data, train_labels, epochs=50, validation_split=VALIDATION_SPLIT)
model.save('char_model')

