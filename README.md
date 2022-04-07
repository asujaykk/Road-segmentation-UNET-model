# Road-segmentation-UNET-model
This project aims to locate and segment the road area from an image typically taken from the frontal camera of a vehicle
Road segmentation is an crucial step in ADAS system of a vehicle for various tasks. 

# Approach
1. Data preperation.
2. Model/algorithm.
3. Framework to be used for model realization.
4. Train the model.
5. Inference the model.
6. Performance improvement.
7. Next steps.

# 1.Data preparation

## Data collection  
 The data required for this task is a collection of images taken from the front camera of a car / vehicle. I used my mobile phone camera to capture continuous video while I was driving. The camera was mounted at the center of the front windshield.
 I collected 28 hours of such videos during my travels in various districts of Kerala and Karnataka.
 Data includes different types of roads (highway,road with lane marking, road without lane mark, mud road, one way, multi lane road) including junctions, turnings and elevations.

 ## Data annotation
  The next difficult step was to annotate the frames from these videos for image segmentation model. 
  As you know this is a bit time consuming task compared to bounding box drawing in case of object detection task.
  Here the segmentation mask need to be drawn on each frame.
  I used Intel's CVAT tool for annotating my data set for this task.
  The data annotation was performed over 600 images/frames (these frames are randomly taken from different videos) for 7 classes. 
  Subset of above big data set is used in this project, and this subset only include 100 annotated images with two class (road , non-road).
  
  The class label for each pixel will be as follows
        1.Non road area : The pixel value of this class in the mask frame will be 0
        2.Road area     : The pixel value of this class in the mask frame will be 2

  A sample image from the data set looks like below                                
  <img src="https://user-images.githubusercontent.com/78997596/161812129-56097ac0-7673-421d-9b49-64e5a692ea4e.jpg" width="300" height="200">

  
  The RGB road mask of the above image is as follows
  
  <img src="https://user-images.githubusercontent.com/78997596/161812224-33cb17a6-40f0-472b-8e37-1cf469f104e9.png" width="300" height="200">
  
  ## Data set format
  The annotated data in CVAT tool can be downloaded in different format based on your need. In my case i choose 'Kitti' format since it contains the direct mask image, which is needed for unet model trainig.

  ## Data augmentation
  Data augmentation is a common practice in machine learning if the data set is very small or the variety of data is less in the data set. Basically data augmentation is nothing but introducing some changes in input data without loosing the major features of it. With this technique you can create a large set of data from the limited original input data. 
  In this project we only have 100 raw images for training, and which is very small for a data hungry model like unet. 
  So we are going to expand 100 image data set to 7000 image data set with data augmentation.
  In this project the DA module use the following  operation
    1. Random brightness : The brightness level of the input image will be changed randomly within limit.
    2. Random saturation : The saturation level of the input image will be changed randomly within limit.
    3. Random hue : The hue level of the input image will be changed randomly within limit.
    4. Random contrast : The contrast level of the input image will be changed randomly within limit.
    5. Horizontal flip : The input image will be flipped along y axis, 
    
Note: The mask will not be modified in 1-4 operation, instead the original mask will be replicated. But in case of 5th operation both input image and mask need to be flipped to retain the spatial information of image and mask.

  ## Prepare the dataset
  Clone this repo with following git command to your working directory
  
~~~
  git clone https://github.com/asujaykk/Road-segmentation-UNET-model.git
~~~
  Extarct the data set available in the data/data_set folder to data_set/data_temp_folder 
~~~
  unzip data/data_set folder/road_seg_kitti.zip –d data_set/data_temp_folder
~~~
  Expand the data set by executing the following data augmentation script
~~~ 
 python3 DA_module.py --input data_set/data_temp_folder/road_seg_kitti
~~~
        
  
  # 2. Model/algorithm
   Unet architecture is choosen for this task since the training cost for this model is less compared to FCNN.


   # 3. Framework to be used for model realization
   Initially the plan was to use pytorch library to realize the model. But since i already used pytorch library for image classification and speech recognition task, this time i thought to use keras and tensorflow for this project so that i will get familiarise with this libraries.
 
 
   # 4.Train the model
   I choose google colab to train the UNET model for (160 x 160) input size, and an augmented dataset of 7000 images. The training took around 20 minutes in colab with GPU (NVIDIA tesla k80 24gb gpu) backend support. We choose 32 batch size and 15 epochs for faster training. The trained model was saved in my google drive for inferenecing.
   
   The same model was trained in my personal laptop (i3 processor with 4gb Ram) without GPU support and the training took almost 10 hours to complete for 4 batch size and 15 epochs.
   
   # 5.Inference the model
   The model can be used to predict the road area from new images/Videos, and we have an Inference pipeline created to predict the road area from an input mp4 Video/IP cam videos.
   
   1. The inference pipeline support mp4 video as input and the input image will be resized to (600 x 400) and the predicted mask will be also at (600 x 400 ) size.
   2. The infrenec model by default shows input image (color image), predicted road mask (binary image) , Final output (mask impossed on red layer of input image)
  
  
 # 6. Performance improvement.
 The model inference took 265 to 340 ms to process one input image, which is pretty slow for a real world application. 
 So we decided to convert the kers model in to onnx model, for better performance and deployement.
 
 The generated onnx model provided better performance on low end machines 
 in my pC the onnx model inference took 75 to 90 ms to process one frame which was acceptable for a real world application. 
 

![Screencast 2022-1649225143098](https://user-images.githubusercontent.com/78997596/161906906-9ec9989e-9617-4500-adef-e1d40c03c75c.gif)
