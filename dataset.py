
import os
import numpy as np

from tensorflow.keras.preprocessing.image import load_img
from tensorflow import keras

"""
## Prepare `Sequence` class to load & vectorize batches of data
"""

class seg_dataset(keras.utils.Sequence):
    """Helper to iterate over the data (as Numpy arrays)."""

    def __init__(self, batch_size, img_size, input_img_paths, target_img_paths):
        self.batch_size = batch_size
        self.img_size = img_size
        self.input_img_paths = input_img_paths
        self.target_img_paths = target_img_paths

    def __len__(self):
        return len(self.target_img_paths) // self.batch_size

    def __getitem__(self, idx):
        """Returns tuple (input, target) correspond to batch #idx."""
        i = idx * self.batch_size
        batch_input_img_paths = self.input_img_paths[i : i + self.batch_size]
        batch_target_img_paths = self.target_img_paths[i : i + self.batch_size]
        x = np.zeros((self.batch_size,) + self.img_size + (3,), dtype="float32")
        for j, path in enumerate(batch_input_img_paths):
            img = load_img(path, target_size=self.img_size)
            x[j] = img
        y = np.zeros((self.batch_size,) + self.img_size + (1,), dtype="uint8")
        for j, path in enumerate(batch_target_img_paths):
            img = load_img(path, target_size=self.img_size, color_mode="grayscale")
            y[j] = np.expand_dims(img, 2)
            # Ground truth labels are 1, 2, 3. Subtract one to make them 0, 1, 2:
            #y[j] =y[j]
        return x, y


# 

def get_img_path_list(input_img_dir,input_mask_dir):
        	
       input_img_paths = sorted(
           [
               os.path.join(input_img_dir, fname)
               for fname in os.listdir(input_img_dir)
               if fname.endswith(".jpg")
           ]
        )
 
       target_img_paths = sorted(
           [
               os.path.join(input_mask_dir, fname)
               for fname in os.listdir(input_mask_dir)
               if fname.endswith(".png") and not fname.startswith(".")
           ]
       )
       print("Number of samples:", len(input_img_paths))
       return input_img_paths, target_img_paths
        

"""
## Set aside a validation split
"""
import random
def split_pathlist(input_img_paths,target_img_paths,val_samples=1000,seed=1337):

   # Split our img paths into a training and a validation set
   random.Random(seed).shuffle(input_img_paths)
   random.Random(seed).shuffle(target_img_paths)
   train_input_img_paths = input_img_paths[:-val_samples]
   train_target_img_paths = target_img_paths[:-val_samples]
   val_input_img_paths = input_img_paths[-val_samples:]
   val_target_img_paths = target_img_paths[-val_samples:]
   
   return train_input_img_paths,train_target_img_paths,val_input_img_paths,val_target_img_paths



	    
	 
