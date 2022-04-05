# Road-segmentation-UNET-model
This project aims to locate and segment the road area from an image typically taken from the frontal camera of a vehicle
Road segmentation is an crucial step in ADAS system of a vehicle for various tasks. 

# Approach
1. Data preperation
2. Model/algorithm
3. Framework to be used for model realization
4. Train the model
5. Evaluate the model
6. Issues observed
7. Performance improvement
8. Next steps

# 1.Data preparation

## Data collection  
 The data needed for this task is a collection of images that are taken from the frontal camera of a car/vehicle.
 I used my mobile phone camera to capture continous vedio while im driving. The camera was mounted at the centre of front wind shield.
 I collected 28 hours of such video during my trips at different districts in kerala and karnataka. 
 The data include different types of road (highway, road with lane mark and without lane mark, mud road, oneway, multi lane road)
 
 ## Data annotation
  The next difficult step was to annotate the frames from these videos for image segmentation model. 
  As you know this is a bit time consuming task compared to bounding box drawing in case of object detection task.
  Here the segmentation mask need to be drawn on each frame.
  I used Intel's CVAT tool for annotating my data set for image segmentation
  The data annotation was performed for 600 images/frames (these frames are randomly taken from different videos) for 7 classes.
  
  ## Data set format
  The annotated data in CVAT tool can be downloaded in different format based on your need. In my case i choose 'Kitti' data set format since it will contain the direct mask image, which is needed for unet model trainig.


  # 2. Model/algorithm
   Unet architecture is choosen for this task since the training cost for this model is less compared to FCNN also can achive comparable accuracy with respect to FCNN.


   # 3. Framework to be used for model realization
   Initially the plan was to use pytorch library to realize the model. But since i already used pytorch library for image classification and speech recognition task, this time i thought to use keras and tensorflow for this project so that i will get familiarise with this libraries.
 
 
   # 4.Train the model
   I choose google colab to train the UNET model for (160 x 160) input size, and an augmented dataset of 7000 images. The training took around 20 minutes in colab with GPU (NVIDIA tesla k80 24gb gpu) backend support. We choose 32 batch size and 15 epochs for faster training. The trained model was saved in my google drive for inferenecing.
   
   The same model was trained in my personal laptop (i3 processor with 4gb Ram) without GPU support and the training took almost 10 hours to complete for 4 batch size and 15 epochs.
   
   
  
  
 
 
