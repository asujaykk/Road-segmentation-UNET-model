
# -*- coding: utf-8 -*-

from tensorflow import keras
import tensorflow as tf
import numpy as np
import PIL
import os
import cv2
import time
from PIL import ImageOps
from tensorflow.keras.preprocessing.image import load_img, img_to_array

import onnxruntime
import tf2onnx
import onnxruntime as rt
import argparse


parser = argparse.ArgumentParser(description = 'Run inference model on mp4 videos')
parser.add_argument('--src',metavar='string',type=str,required = True,help='Path to input mp4 video')
parser.add_argument('--model',metavar='String',type=str,default='models/onnx_models/road_seg_160_160.onnx',help='The location of onnx model to be used')
args = parser.parse_args()


model_in_size = (None, 160, 160, 3)
img_size=(160, 160)
out_size=(600,400)
onnx_model_path = args.model       #"onnx_models/road_seg.onnx"

path=args.src                     #"/media/akhil_kk/DATA_DRIVE/data_sets/VID_20220129_181919.mp4"
cap=cv2.VideoCapture(path)
count=0
fskp=8



providers = ['CPUExecutionProvider']
model = rt.InferenceSession(onnx_model_path, providers=providers)

def get_mask(in_img,model_size,outsize):
    """Quick utility to display a model's prediction."""
    img= cv2.resize(in_img, model_size)
    img=img.astype('float32')
    data=np.expand_dims(img, axis=0)
    #print("s:"+str(time.time()))
    val_pred=model.run(None, {"input": data})
    #print("e:"+str(time.time()))
    mask = np.argmax(val_pred[0], axis=-1)
    mask = np.expand_dims(mask, axis=-1)
    img = PIL.ImageOps.autocontrast(keras.preprocessing.image.array_to_img(mask[0]))
    img=img_to_array(img)
    return cv2.resize(in_img,outsize), cv2.resize(img,outsize)



while True:
    #count=count+fskp
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
        cv2.imshow('Input frame onn',img)
        cv2.imshow('predicted mask onn',mask)
        cv2.imshow('Final output onn',cv2.merge([B, G, np.bitwise_or(R,mask.astype(np.uint8))]))
    cv2.waitKey(1)
    
