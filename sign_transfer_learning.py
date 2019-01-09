#!/usr/bin/env python
# coding: utf-8

# In[90]:


import cv2                 # working with, mainly resizing, images
import numpy as np         # dealing with arrays
import os                  # dealing with directories
from random import shuffle # mixing up or currently ordered data that might lead our network astray in training.
from tqdm import tqdm      

TRAIN_DIR = '/home/riya/Documents/sample_Signature/training'
TEST_DIR = '/home/riya/Documents/sample_Signature/testing'
IMG_SIZE = 224


# In[91]:


def label_img(img):
    image_label = img.split('.')
    name = image_label[0]
    # checking if the signature is forged or genuine
    #                            
    
    if name[4:7] == name[-3:]:
        return 1
  
    elif name[4:7] != name[-3:]:
        return 0


# In[92]:


def person_img(img):
    label = img.split('.')
    k = label[0]
    return [k[-3:]]


# In[93]:


def create_features(DIR):
    features = []
    labels = []
    for img in tqdm(os.listdir(DIR)):
        label = label_img(img)
        person = person_img(img)
       
        path = os.path.join(DIR,img)
        img = cv2.imread(path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (IMG_SIZE,IMG_SIZE))
        features.append([img])
        labels.append(np.array(label))
    return features


# In[2]:


train_features  = np.array(create_features(TRAIN_DIR))
test_features  = np.array(create_features(TEST_DIR))
print test_features.shape
train_features_reshaped = train_features.reshape(-1,224,224,3)
test_features_reshaped = test_features.reshape(-1,224,224,3)
print test_features_reshaped.shape
print train_features_reshaped.shape

train_features_reshaped = train_features_reshaped.astype('float32')
test_features_reshaped = test_features_reshaped.astype('float32')
train_features_reshaped = train_features_reshaped / 255.
test_features_reshaped = test_features_reshaped / 255.
print train_features_reshaped


# In[69]:


def create_labels(DIR):
    labels = []
    for img in tqdm(os.listdir(DIR)):
        label = label_img(img)
        labels.append(np.array(label))
    return labels


# In[77]:


train_labels = np.array(create_labels(TRAIN_DIR))
test_labels = np.array(create_labels(TEST_DIR))
print test_labels.shape


# In[100]:
import numpy as np
from keras.utils import to_categorical


# In[101]:


train_labels_one_hot = to_categorical(train_labels)
print train_labels_one_hot


# In[102]:


test_labels_one_hot = to_categorical(test_labels)
print test_labels_one_hot.shape



#model

from keras.models import Sequential, Model, load_model
from keras import applications
from keras import optimizers
from keras.layers import Dropout, Flatten, Dense

img_rows, img_cols, img_channel = 224, 224, 3

base_model = applications.VGG16(weights='imagenet', include_top=False, input_shape=(img_rows, img_cols, img_channel))
base_model.summary()

add_model = Sequential()
add_model.add(Flatten(input_shape=base_model.output_shape[1:]))
add_model.add(Dense(256, activation='relu'))
add_model.add(Dense(2, activation='sigmoid'))

model = Model(inputs=base_model.input, outputs=add_model(base_model.output))
model.compile(loss='binary_crossentropy', optimizer=optimizers.SGD(lr=1e-4, momentum=0.9),
              metrics=['accuracy'])

model.summary()

batch_size = 32
epochs = 20

history = model.fit(train_features_reshaped, train_labels_one_hot, batch_size=batch_size,epochs=epochs,verbose=1,validation_data=(test_features_reshaped, test_labels_one_hot))








