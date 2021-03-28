#!/usr/bin/env python

import tensorflow as tf
from keras import backend as K
import os
import random
import numpy as np

 
from tqdm import tqdm 

from skimage.io import imread, imshow
from skimage.transform import resize
import matplotlib.pyplot as plt

seed = 42
np.random.seed = seed



IMG_WIDTH = 128
IMG_HEIGHT = 128
IMG_CHANNELS = 3

TRAIN_PATH = 'stage1_train/'
TEST_PATH = 'stage1_test/'

train_ids = next(os.walk(TRAIN_PATH))[1]
test_ids = next(os.walk(TEST_PATH))[1]

X_train = np.zeros((len(train_ids), IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS), dtype=np.uint8)
Y_train = np.zeros((len(train_ids), IMG_HEIGHT, IMG_WIDTH, 1), dtype=np.bool)
X_test = np.zeros((len(test_ids), IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS), dtype=np.uint8)
Y_test = np.zeros((len(test_ids), IMG_HEIGHT, IMG_WIDTH, 1), dtype=np.bool)

# resizing the input data into 128*128 image pixel

print('Resizing training images and masks')
for n, id_ in tqdm(enumerate(train_ids), total=len(train_ids)):   
    path = TRAIN_PATH + id_
    img = imread(path + '/images/' + id_ + '.png')[:,:,:IMG_CHANNELS]  
    img = resize(img, (IMG_HEIGHT, IMG_WIDTH), mode='constant', preserve_range=True)
    X_train[n] = img  #Fill empty X_train with values from img
    mask = np.zeros((IMG_HEIGHT, IMG_WIDTH, 1), dtype=np.bool)
    for mask_file in next(os.walk(path + '/masks/'))[2]:
        mask_ = imread(path + '/masks/' + mask_file)
        mask_ = np.expand_dims(resize(mask_, (IMG_HEIGHT, IMG_WIDTH), mode='constant',  
                                      preserve_range=True), axis=-1)
        mask = np.maximum(mask, mask_)  
            
    Y_train[n] = mask   

# test images
sizes_test = []
print('Resizing test images and masks for testing') 
for n, id_ in tqdm(enumerate(test_ids), total=len(test_ids)):
    path = TEST_PATH + id_
    img = imread(path + '/images/' + id_ + '.png')[:,:,:IMG_CHANNELS]
    sizes_test.append([img.shape[0], img.shape[1]])
    img = resize(img, (IMG_HEIGHT, IMG_WIDTH), mode='constant', preserve_range=True)
    X_test[n] = img
    mask = np.zeros((IMG_HEIGHT, IMG_WIDTH, 1), dtype=np.bool)
    for mask_file in next(os.walk(path + '/masks/'))[2]:
        mask_ = imread(path + '/masks/' + mask_file)
        mask_ = np.expand_dims(resize(mask_, (IMG_HEIGHT, IMG_WIDTH), mode='constant',  
                                      preserve_range=True), axis=-1)
        mask = np.maximum(mask, mask_)  
            
    Y_test[n] = mask  

print('Resizing images... Done!')

# Testing to see the input images
image_x = random.randint(0, len(train_ids))
imshow(X_train[image_x])
plt.show()
imshow(np.squeeze(Y_train[image_x]))
plt.show()



# Define IoU metric
def iou_coef(y_true, y_pred, smooth=1):
    intersection = K.sum(K.abs(y_true * y_pred), axis=-1)
    union = K.sum(y_true,-1) + K.sum(y_pred,-1) - intersection
    return (intersection+ smooth) / (union+smooth)        

def iou_coef_loss(y_true, y_pred):
    return -iou_coef(y_true, y_pred)



# Convolutional Neural Network
import keras
from keras.models import Model
from keras.layers import Input
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers import Dropout
from keras.layers import Conv2DTranspose
from keras.layers import concatenate
from keras.layers.convolutional import Conv2D
from keras.layers.pooling import MaxPooling2D



# Full U-net model functions

def build_model(input_layer, start_neurons):
    s = keras.layers.Lambda(lambda x: x / 255)(input_layer)

    conv1 = Conv2D(start_neurons * 1, (3, 3), activation="relu", padding="same")(s)
    conv1 = Conv2D(start_neurons * 1, (3, 3), activation="relu", padding="same")(conv1)
    pool1 = MaxPooling2D((2, 2))(conv1)
#     pool1 = Dropout(0.25)(pool1)

    conv2 = Conv2D(start_neurons * 2, (3, 3), activation="relu", padding="same")(pool1)
    conv2 = Conv2D(start_neurons * 2, (3, 3), activation="relu", padding="same")(conv2)
    pool2 = MaxPooling2D((2, 2))(conv2)

    conv3 = Conv2D(start_neurons * 4, (3, 3), activation="relu", padding="same")(pool2)
    conv3 = Conv2D(start_neurons * 4, (3, 3), activation="relu", padding="same")(conv3)
    pool3 = MaxPooling2D((2, 2))(conv3)
    
    conv4 = Conv2D(start_neurons * 8, (3, 3), activation="relu", padding="same")(pool3)
    conv4 = Conv2D(start_neurons * 8, (3, 3), activation="relu", padding="same")(conv4)
    pool4 = MaxPooling2D((2, 2))(conv4)
    
    # Middle
    convm = Conv2D(start_neurons * 16, (3, 3), activation="relu", padding="same")(pool4)
    convm = Conv2D(start_neurons * 16, (3, 3), activation="relu", padding="same")(convm)

    deconv4 = Conv2DTranspose(start_neurons * 8, (2, 2), strides=(2, 2), padding="same")(convm)
    uconv4 = concatenate([deconv4, conv4])
    uconv4 = Conv2D(start_neurons * 8, (3, 3), activation="relu", padding="same")(uconv4)
    uconv4 = Conv2D(start_neurons * 8, (3, 3), activation="relu", padding="same")(uconv4)
    
    deconv3 = Conv2DTranspose(start_neurons * 4, (2, 2), strides=(2, 2), padding="same")(uconv4)
    uconv3 = concatenate([deconv3, conv3])
    uconv3 = Conv2D(start_neurons * 4, (3, 3), activation="relu", padding="same")(uconv3)
    uconv3 = Conv2D(start_neurons * 4, (3, 3), activation="relu", padding="same")(uconv3)
    
    deconv2 = Conv2DTranspose(start_neurons * 2, (2, 2), strides=(2, 2), padding="same")(uconv3)
    uconv2 = concatenate([deconv2, conv2])
    uconv2 = Conv2D(start_neurons * 2, (3, 3), activation="relu", padding="same")(uconv2)
    uconv2 = Conv2D(start_neurons * 2, (3, 3), activation="relu", padding="same")(uconv2)

    deconv1 = Conv2DTranspose(start_neurons * 1, (2, 2), strides=(2, 2), padding="same")(uconv2)
    uconv1 = concatenate([deconv1, conv1])
    uconv1 = Conv2D(start_neurons * 1, (3, 3), activation="relu", padding="same")(uconv1)
    uconv1 = Conv2D(start_neurons * 1, (3, 3), activation="relu", padding="same")(uconv1)
    
    output_layer = Conv2D(1, (1,1), padding="same", activation="sigmoid")(uconv1)

    return output_layer


input_layer1  = keras.layers.Input(shape=(128,128,3))
output_layer = build_model(input_layer1, 16)

model = Model(input = input_layer1, output = output_layer)

model.summary()

model.compile(loss='binary_crossentropy', 
              optimizer=keras.optimizers.SGD(lr=0.001),
            #   metrics=[iou_coef])
              metrics=['accuracy'])



from collections import namedtuple

nb_classes =2                   # Binary classifications
Dataset = namedtuple('Dataset',['X','Y','y','nb_classes'])

y_train = Y_train               #not necessary but to match up with the previous MINST code 
y_test = Y_test                 #not necessary but to match up with the previous MINST code 

trn = Dataset(X_train, Y_train, y_train, nb_classes)
tst = Dataset(X_test , Y_test, y_test, nb_classes)

cfg = {}
cfg['SGD_BATCHSIZE']    = 16
cfg['SGD_LEARNINGRATE'] = 0.001
cfg['NUM_EPOCHS']       = 1    #For testing
cfg['FULL_MI']          = True


cfg['ACTIVATION'] = 'relu'   #for U-net
# cfg['ACTIVATION'] = 'tanh'

# How many hidden neurons to put into each of the layers
cfg['LAYER_DIMS'] = [128,64,32,16,8,16,32,64,10000] 
ARCH_NAME =  '-'.join(map(str,cfg['LAYER_DIMS']))

# Where to save activation and weights data
cfg['SAVE_DIR'] = 'rawdata/' + cfg['ACTIVATION'] + '_' + ARCH_NAME 




import loggingreporter            # Saving activations 

def do_report(epoch):
    # Only log activity for some epochs.  Mainly this is to make things run faster.
    if epoch < 20:       # Log for all first 20 epochs
        return True
    elif epoch < 100:    # Then for every 5th epoch
        return (epoch % 5 == 0)
    elif epoch < 200:    # Then every 10th
        return (epoch % 10 == 0)
    else:                # Then every 100th
        return (epoch % 100 == 0)

    
reporter = loggingreporter.LoggingReporter(cfg=cfg, 
                                          trn=trn, 
                                          tst=tst, 
                                          do_save_func=do_report)


r = model.fit(X_train, Y_train, verbose  = 2, 
              batch_size = 16, epochs = cfg['NUM_EPOCHS'] ,
              validation_data=(tst.X, tst.Y),
              callbacks  = [reporter,])
#                 callbacks=callbacks)





