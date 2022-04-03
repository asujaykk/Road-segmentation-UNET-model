# -*- coding: utf-8 -*-

from tensorflow import keras
import tensorflow as tf
import numpy as np
from IPython.display import Image, display
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import PIL
from PIL import ImageOps
from PIL import Image
import os
import cv2
import time


model_in_size = (None, 160, 160, 3)
img_size=(160, 160)
out_size=(600,400)
model_path = "models/pretrained_models/road_segmentation_160_160.h5"




model = keras.models.load_model(model_path)
#model.compile(optimizer="rmsprop", loss="sparse_categorical_crossentropy")


def get_mask(in_img,model_size,outsize):
    """Quick utility to display a model's prediction."""
    img= cv2.resize(in_img, model_size)
    data=np.expand_dims(img, axis=0)
    #print(data.shape)
    #print("s:"+str(time.time()))
    val_pred = model.predict(data)
    #print("e:"+str(time.time()))
    mask = np.argmax(val_pred[0], axis=-1)
    mask = np.expand_dims(mask, axis=-1)
    #print(mask.shape)
    #print("done")
    img = PIL.ImageOps.autocontrast(keras.preprocessing.image.array_to_img(mask))
    img=img_to_array(img)
    #display(img)
    return cv2.resize(in_img,outsize), cv2.resize(img,outsize)

path="/media/akhil_kk/DATA_DRIVE/data_sets/VID_20220129_181919.mp4"
cap=cv2.VideoCapture(path)
count=0
fskp=8

while True:
    count=count+fskp
    #cap.set(cv2.CAP_PROP_POS_FRAMES, count)
    ret,frame=cap.read()
    if ret:    
        # Display mask predicted by our model
        img,mask=get_mask(frame,img_size,out_size)  # Note that the model only sees inputs at 160x160.
        #print(mask.dtype)
        (B, G, R) = cv2.split(img)
        #print(R.dtype)
        #R=mask.astype(np.uint8)
        
        #merged = cv2.merge([B, G, R])
        cv2.imshow('Input frame',img)
        cv2.imshow('predicted mask',mask)
        cv2.imshow('Final output ',cv2.merge([B, G, np.bitwise_xor(R,mask.astype(np.uint8))]))
    cv2.waitKey(1)
    
