# %%
from array import array
from cgi import test
from encodings import utf_8
from pickletools import uint8
from sre_parse import WHITESPACE
from tensorflow import keras
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import os
from keras import Sequential
from keras import layers
from keras.layers import StringLookup
import cv2
from PIL import Image
from keras.utils import np_utils
import pickle
from sklearn.model_selection import GridSearchCV
from scikeras.wrappers import KerasClassifier

# %%
#ucitavanje rijeci
words=[]
text_file=open(f"data/words.txt","r").readlines()

#izbacivanje komentara i rijeci koje mozda imaju gresku
for line in text_file:
    if line[0]=="#":
        continue
    if line.split(" ")[1] !="err":
        words.append(line)
    

#np.random.shuffle(words)
print(len(words))
#np.random.shuffle(words)


characters=[]
character_file=open(f"data/class.txt","r").readlines()

#izbacivanje komentara i rijeci koje mozda imaju gresku
for line in character_file:
  characters.append(line[:len(line)-1])
    
print(len(characters))
print(characters)

# %%
#dobivanje pathova i labela za svaku sliku

def get_paths_and_labels(data):
    paths=[]
    labels=[]
    for line in data:
        image_name=line.split(" ")[0]
        first_level=image_name.split("-")[0]
        second_level=image_name.split("-")[0] + "-" + image_name.split("-")[1]
        path="data/words/"+first_level+"/"+second_level+"/"+image_name+".png"
        paths.append(path)
        label=line.split(" ")[8]
        labels.append(label)


    return paths, labels


word_paths, word_labels=get_paths_and_labels(words)
print("Done with getting paths and labels")

# %%
#poboljsanje kvaliteta slika
def image_processing(img_paths):
    
    for path in img_paths:
        print(path)
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        pxmin = np.min(img)
        pxmax = np.max(img)
        imgContrast = (img - pxmin) / (pxmax - pxmin) * 255
        kernel=np.ones((1,1),np.uint8)

        imgMorph = cv2.erode(imgContrast, kernel, iterations = 1)

        # write
        cv2.imwrite(path, imgMorph)

#dobivanje univerzalne velicine slike

def distortion_free_resize(images):
    w=256
    h=64
    # for image_path in images:
    #     image=cv2.imread(image_path)
    #     img_h=image.shape[0]
    #     img_w=image.shape[1]
    #     if img_h > h:
    #         h = img_h
    #     if img_w > w:
    #         w = img_w
    # print("width "+str(w))
    # print("height"+str(h))
    

    for image_path in images:
        image=Image.open(image_path)
        img_w,img_h=image.size
        padding_top_bottom=int((h-img_h)/2)
        padding_left_right=int((w-img_w)/2)
        if(padding_top_bottom<0):
            padding_top_bottom=0
        if(padding_left_right<0):
            padding_left_right=0
        padding_img = Image.new(mode="RGB",size= (w, h),color= (255, 255, 255))
        padding_img.paste(image, (padding_left_right, padding_top_bottom))
        padding_img.save(image_path)

# image_processing(word_paths)
# print("Done with contrast enhancment")

#distortion_free_resize(word_paths)
print("Done with resizing")


# %%
#podjela podataka u treniranje, validacija i testiranje
length=len(word_paths)
split_index=int(0.9*length)
train_data=word_paths[:split_index]
train_labels=word_labels[:split_index]
test_data=word_paths[split_index:]
test_labels=word_labels[split_index:]

# %%
#uklanjanje znaka za novi red
def get_vocabulary(labels):
    labels_cleaned=[]
    characters = set()
    max_len = 0
    #print(labels)
    print("--------------------------------------------------")
    for label in labels:
        label = label.split(" ")[-1].strip()

        max_len = max(max_len, len(label))
        labels_cleaned.append(label) 
    #print(labels_cleaned)
    return labels_cleaned,max_len

train_labels,max_len1=get_vocabulary(train_labels)
test_labels,max_len2=get_vocabulary(test_labels)
max_len=max(max_len1,max_len2)


# %%
#dobivanje svih znakova iz rijeci

char_to_num = StringLookup(vocabulary=list(characters), mask_token=None)
print(char_to_num)

# %%
#normalizacija slika i vektoriziranje oznaka
def saving_images(image_paths):
    images=[]
    for path in image_paths:
        image = tf.io.read_file(path)
        image = tf.image.decode_png(image, 1)
        images.append(image)
    images=np.array(images)
    images=images.astype('float32') / 255.0
    return images

train_data=saving_images(train_data)
test_data=saving_images(test_data)

print("Spremljene slike")
def vectorize_labels(labels):
    new_labels=[]
    for label in labels:
        label = char_to_num(tf.strings.unicode_split(label, input_encoding="UTF-8"))
        length = tf.shape(label)[0]
        pad_amount = max_len - length
        label = tf.pad(label, paddings=[[0, pad_amount]], constant_values=99)
        new_labels.append(label)
    return new_labels

    

train_labels=vectorize_labels(train_labels)
test_labels=vectorize_labels(test_labels)

train_data=np.array(train_data)
test_data=np.array(test_data)
train_labels=np.array(train_labels)
test_labels=np.array(test_labels)
characters=np.array(characters)
# print(test_labels)
# print('--------------------------------------------------------------------------------------------------')
# print(test_data[0])

batch_size = 16
nb_classes =3
nb_epochs = 5
img_channel = 3
nb_filters = 32
nb_pool = 2
nb_conv = 3

print("printamo oblike")
print(train_data.shape)
print(train_labels.shape)
# %%

from keras import backend as K
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.layers import Input, Dense, Activation, Flatten
from keras.layers import Reshape, Lambda
from keras.models import Model, load_model
from keras.layers import GRU
from keras.optimizers import Adam
from keras.utils.data_utils import get_file
from keras.preprocessing import image
from keras.layers import add, concatenate
import keras.callbacks
# %%
# def ctc_lambda_func(args):
#     y_pred, labels, input_length, label_length = args
#     # the 2 is critical here since the first couple outputs of the RNN
#     # tend to be garbage:
#     y_pred = y_pred[:, 2:, :]
#     return K.ctc_batch_cost(labels, y_pred, input_length, label_length)

def CTCLoss(y_true, y_pred):
    # Compute the training-time loss value
    batch_len = tf.cast(tf.shape(y_true)[0], dtype="int64")
    input_length = tf.cast(tf.shape(y_pred)[1], dtype="int64")
    label_length = tf.cast(tf.shape(y_true)[1], dtype="int64")

    input_length = input_length * tf.ones(shape=(batch_len, 1), dtype="int64")
    label_length = label_length * tf.ones(shape=(batch_len, 1), dtype="int64")

    loss = keras.backend.ctc_batch_cost(y_true, y_pred, input_length, label_length)
    return loss

# def calculate_edit_distance(labels, predictions):
#     # Get a single batch and convert its labels to sparse tensors.
#     saprse_labels = tf.cast(tf.sparse.from_dense(labels), dtype=tf.int64)

#     # Make predictions and convert them to sparse tensors.
#     input_len = np.ones(predictions.shape[0]) * predictions.shape[1]
#     predictions_decoded = keras.backend.ctc_decode(
#         predictions, input_length=input_len, greedy=True
#     )[0][0][:, :max_len]
#     sparse_predictions = tf.cast(
#         tf.sparse.from_dense(predictions_decoded), dtype=tf.int64
#     )

#     # Compute individual edit distances and average them out.
#     edit_distances = tf.edit_distance(
#         sparse_predictions, saprse_labels, normalize=False
#     )
#     return tf.reduce_mean(edit_distances)

# class EditDistanceCallback(keras.callbacks.Callback):
#     def __init__(self, pred_model):
#         super().__init__()
#         self.prediction_model = pred_model

#     def on_epoch_end(self, epoch, logs=None):
#         edit_distances = []

#         for i in range(len(test_data)):
#             labels = test_labels[i]
#             predictions = self.prediction_model.predict(test_data[i])
#             edit_distances.append(calculate_edit_distance(labels, predictions).numpy())

#         print(
#             f"Mean edit distance for epoch {epoch + 1}: {np.mean(edit_distances):.4f}"
#         )

def createModel():
    model=Sequential([
        Input(name="input", shape=(64,256,1), dtype='float32'),
        Conv2D(64, (3,3), padding='same',
                    activation='relu', kernel_initializer='he_normal',
                    name='conv1'),
        MaxPooling2D(pool_size=(2, 2), name='max1'),
        Conv2D(64, (3,3), padding='same',
                    activation='relu', kernel_initializer='he_normal',
                    name='conv2'),
        MaxPooling2D(pool_size=(2, 2), name='max2'),
        Reshape(target_shape=(256 // (2 ** 2), (64 // (2 ** 2)) * 64), name='reshape'),
        Dense(64, activation='relu', name='dense1'),
        GRU(512, return_sequences=True, kernel_initializer='he_normal', name='gru1'),
        GRU(512, return_sequences=True, go_backwards=True, kernel_initializer='he_normal', name='gru1_b'),
        GRU(512, return_sequences=True, kernel_initializer='he_normal', name='gru2'),
        GRU(512, return_sequences=True, go_backwards=True, kernel_initializer='he_normal', name='gru2_b'),
        Dense(90, kernel_initializer='he_normal', activation="softmax",
                    name='dense2')

    ])

    adam = Adam(learning_rate=0.0000001)
    
    model.compile(loss=CTCLoss, optimizer=adam)
    model.summary()

    return model

# #edit_distance_callback = EditDistanceCallback(prediction_model)
# param_grid = dict(batch_size=[4,8,16,32,64,80,128,256], epochs=[20,30,40,50,60,70,80,100,120])
# model = KerasClassifier(model=createModel, verbose=0)
# grid = GridSearchCV(estimator=model, param_grid=param_grid, n_jobs=-1, cv=3, scoring='accuracy')
# # %%
# grid_result=grid.fit(train_data,train_labels,validation_data=(test_data,test_labels))
# # %%
# print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
# #filename='done_model2.sav'
# grid.save("my_model")


#za istestirat fukcije
model=createModel()
model.fit(train_data,train_labels,validation_data=(test_data,test_labels), epochs=3)
model.evaluate(test_data,test_labels)

model.save('my_test_model')