
from tensorflow import keras

from model import get_model 
import dataset
from dataset import seg_dataset
import argparse

#parse the commandline parameters

parser = argparse.ArgumentParser(description = 'Train the unet model with kitti dataset')
parser.add_argument('--dataset',metavar='string',type=str,required = True,help='Path to extracted kitti data set folder')
parser.add_argument('--output',metavar='String',type=str,required = True,help='The location where the keras model will be saved as .h5 file')
parser.add_argument('--batch',metavar='Integer',type=int,default=4,help='Batch size of dataset for training and evaluation(Note: default value is 32. Reduce the batch size if your PC resources are limited)')
parser.add_argument('--epoch',metavar='Integer',type=int,default=15,help='Epoch for training and evaluation, default value is 15')

args = parser.parse_args()




#img_loc="/media/akhil_kk/DATA_DRIVE/data_sets/road_seg/road_seg_kitti/default/image_2/"
#mask_loc="/media/akhil_kk/DATA_DRIVE/data_sets/road_seg/road_seg_kitti/default/instance/"
#model_path = "models/pretrained_models/road_segmentation_160_160.h5"


"""
Reading label file to get number of classes
"""

dataset_loc=args.dataset
label_file=open(args.dataset+"/label_colors.txt",'r')
label_content= label_file.readlines()
label_count=len(label_content)


"""
Setting the parameters for the UNET model
"""
input_img_dir = dataset_loc+"/default/image_2/"
input_mask_dir = dataset_loc+"/default/instance/"
img_size = (160, 160)
num_classes = label_count
batch_size = args.batch
epochs = args.epoch
model_path =args.output+"/road_segmentation_160_160_test.h5"




input_img_paths, target_img_paths=dataset.get_img_path_list(input_img_dir,input_mask_dir)


"""
## Prepare `Sequence` class to load & vectorize batches of data
"""


# Free up RAM in case the model definition cells were run multiple times
keras.backend.clear_session()


# Build model
model = get_model(img_size, num_classes)
model.summary()




"""
## Set aside a validation split
"""

# Split our img paths into a training and a validation set
train_input_img_paths,train_target_img_paths,val_input_img_paths,val_target_img_paths=dataset.split_pathlist(input_img_paths,target_img_paths)

# Instantiate data Sequences for training and validation split
train_gen = seg_dataset(batch_size, img_size, train_input_img_paths, train_target_img_paths)

val_gen = seg_dataset(batch_size, img_size, val_input_img_paths, val_target_img_paths)



"""
## Train the model
"""

# Configure the model for training.
# We use the "sparse" version of categorical_crossentropy
# because our target data is integers.
model.compile(optimizer="rmsprop", loss="sparse_categorical_crossentropy")

callbacks = [
    keras.callbacks.ModelCheckpoint(model_path, save_best_only=True)
]

# Train the model, doing validation at the end of each epoch.

model.fit(train_gen, epochs=epochs, validation_data=val_gen, callbacks=callbacks)

    
