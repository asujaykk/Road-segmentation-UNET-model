
from tensorflow import keras

from model import get_model 
import dataset
from dataset import seg_dataset

img_loc="/media/akhil_kk/DATA_DRIVE/data_sets/road_seg/road_seg_kitti/default/image_2/"
mask_loc="/media/akhil_kk/DATA_DRIVE/data_sets/road_seg/road_seg_kitti/default/instance/"

input_img_dir = img_loc
input_mask_dir = mask_loc
img_size = (160, 160)
num_classes = 3
batch_size = 4
epochs = 10

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

# Instantiate data Sequences for each split
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
    keras.callbacks.ModelCheckpoint("road_seg_400_608.h5", save_best_only=True)
]

# Train the model, doing validation at the end of each epoch.

model.fit(train_gen, epochs=epochs, validation_data=val_gen, callbacks=callbacks)

if __name__ == '__main__':
    
