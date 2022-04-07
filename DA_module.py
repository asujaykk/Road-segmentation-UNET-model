#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 10 18:23:06 2021

@author: akhil_kk
"""

import cv2
import tensorflow as tf
from tensorflow.keras.preprocessing.image import img_to_array
import random
import os
import glob


import argparse

#parse the commandline parameters

parser = argparse.ArgumentParser(description = 'Perform data augmentation ')
parser.add_argument('--input',metavar='string',type=str,required = True,help='Path to extracted kitti data set root folder')
parser.add_argument('--factor',metavar='int',type=int,default=70, help='number of times the base images to be replicated (integer)')
args = parser.parse_args()



"""
Default location of images and masks inside the kitti dataset folder are derived
"""
img_loc=args.input+"/default/image_2/"
mask_loc=args.input+"/default/instance/"


"""
Setting the target image location
here target and source location are same
"""
t_img_loc=img_loc       
t_mask_loc=mask_loc    

  
# prnt total number of images in 'img_loc' directory
img_count=len(glob.glob1(img_loc,"*.jpg"))
print(img_count)

            
def data_augmentation(times,base_c):
    t_c=times*base_c   # t_c= total number of images after augmentation
    i=base_c           # starting file number of first processed image/mask
    j=0                # staring file number of original image/mask
    while i<t_c:       # iterate and process until total file count reached      
        """
        The random number will be used to perform different operation on each image
        the random number "r_num" can have values from 0-4
        The operation performed for each value of 'r_num' is provided below
        0 : flip the image horizontally
        1 : random contrast
        2 : random saturation
        3 : random hue
        4 : random brightness
        """    
        r_num=random.randint(0,4)   
        
       # location of input image and mask for each iteration
        
        img_path=img_loc+str(j)+".jpg"
        mask_path=mask_loc+str(j)+".png"
        
        # location of output image and mask for each iteration
        t_img_path=t_img_loc+str(i)+".jpg"
        t_mask_path=t_mask_loc+str(i)+".png"
        
        """
        # load the image and mask
        # mask file reading and writing is not working as expected with tf.keras.utils
        # Because the mask written back is converted to binary image 
        # which means the pixel values of the mask are modified by the operation
        # So to solve the problem easly, I'm using cv2.imread and cv2.imwrite for mask processing, later this will be solved once I could find the root cause ;) 
        """
        
        img=tf.keras.utils.load_img(img_path)       
        mask = cv2.imread(mask_path)               
        #mask=tf.keras.utils.load_img(mask_path)  
        
        #convert the PIL image object to numpy array
        img = img_to_array(img)
        #mask = img_to_array(mask)
         
         
        #perform different image processing based on 'r_num' value  
        if r_num==0:   # flip the image (both mask and image need to be flipped)     
           img = tf.image.flip_left_right(img)
           mask = cv2.flip(mask,1)
           #mask=tf.image.flip_left_right(mask)
        
        
        elif r_num==1:  # image contrast will be changed and no change required in the mask
            img=tf.image.random_contrast(img, 0.5, 2.0)
        
        
        elif r_num==2:  # image saturation will be changed and no change required in the mask
            img = tf.image.random_saturation(img, 0.75, 1.25)       
        
        
        elif r_num==3:  # image hue will be changed and no change required in the mask
            img = tf.image.random_hue(img, 0.1)
        
        
        elif r_num==4:  # image brightness will be changed and no change required in the mask
            img = tf.image.random_brightness(img, 2.0)
        
        
        #save processed image and mask            
        tf.keras.utils.save_img(t_img_path, img)
        #tf.keras.utils.save_img(t_mask_path, mask)
        cv2.imwrite(t_mask_path,mask)
        
        j+=1
        i+=1
        if j==base_c:   # repeat the process on input images
           j=0


data_augmentation(args.factor,img_count)

