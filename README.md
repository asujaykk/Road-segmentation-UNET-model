# Road-segmentation with U-Net model

## Image segmentation
Image segmentation is the process of classifying each pixel in an image to a particular class, At a top level image segmentation identify a region in an image which belongs to a particular object type. Unlike object detection task where the model can only predict the bounding box region in which the objects are present, the segmentation model can precisly extract the object/region boundaries based on the object/region shape.

This  project aims to locate and segment the road region from a picture normally taken from the front camera of a vehicle using image segmentation. Road segmentation is a critical step in Advanced Driver Assistance System (ADAS) for a variety of tasks, such as extracting the driveable area, path planning,
lane change detection etc.
In this section we will focus only on separating the main road region from an image.

# Prerequisite
You need to have the following libraries installed on your computer before starting the project.
1. tensorflow   2.7.0
2. opencv   4.5.4.60
3. keras   2.7.0
4. numpy   1.21.4
5. Python  3.6

If you want to create and run the onnx model, you need the following packages as well.
1. onnxruntime  1.11.0
2. onnx    1.11.0
3. tf2onnx  1.11.0

# Steps involved
1. Data preparation.
2. Model/algorithm selection.
3. Framework to be used for model realization.
4. Train the model.
5. Inferencing keras model.
6. Keras to ONNX conversion.
7. Inferencing the onnx model.
8. Next steps.

# 1.Data preparation

## Data collection  
 The data required for this task is a collection of images taken from the front camera of a car / vehicle. I used my mobile phone camera to continuously record video while I was driving. The camera is mounted in the center of the front windshield.
 I collected 28 hours of such videos during my travels in various districts of Kerala and Karnataka.
 Those videos includes different types of roads scenarios (highway,road with lane marking, road without lane mark, mud road, one way, multi lane road), junctions, curves and elevations.

 ## Data annotation
  The next difficult step is to annotate the frames from these videos for image segmentation model. Data annotation is the process of categorizing and labeling the raw data in such a way that it can be fed to an ai model for training. In segmentation task, each pixel of an image need to be labeled. 
As you know this is a bit time consuming task compared to bounding box drawing in case of object detection task.

Instead of manually labeling each pixel in an image, this is achived by categorizing a region (a polygon) in the image with a label. So we need to draw the polygon through the boundaries of a region in the image. In this project the polygon will be drawn through the road region boundaries. And the pixels inside the polygon will be labelled as 'road region' and the pixels outside of this polygon will be labeled as 'non-road region'. 

Basically the annotation for image segmentation is creating a mask image (with the same size of input image) corresponds to each input image. And the mask pixel values are the class label value for the corresponding pixel in the input image.

I used Intel's CVAT tool for this task.The data annotation was performed over 600 images/frames (these frames are randomly taken from different videos) for 7 classes. 
  
  
  ## Data set format
  The interpreted data in the CVAT tool can be downloaded in different formats as per your requirement. In this project I chose the 'Kitti' format because it contains the direct mask image required for U-Net model training.
  
  The 'kitti data set' folder contains the following files and folder structure.
  1. label_colors.txt: This file contain the class label information.                             
  2. default/image_2 : This directory contains all input images (x.jpg files).
  3. default/instance: This directory contains target mask image of each input image (x.png files). 
  4. Default/semantic_rgb: This directory contains the mask image in RGB format (x.png files).

A subset of my 7 class data set is used in this project, and this subset only include 100 annotated images with two class (road , non-road).

  The class label for each pixel is as follows
     
   1. Non road area : The pixel value of this class in the mask frame will be 0
   2. Road area     : The pixel value of this class in the mask frame will be 2

  A sample image from the data set looks like below                                
  <img src="https://user-images.githubusercontent.com/78997596/161812129-56097ac0-7673-421d-9b49-64e5a692ea4e.jpg" width="300" height="200">

  
  The annotated RGB mask of the above image is as follows.
  
  <img src="https://user-images.githubusercontent.com/78997596/161812224-33cb17a6-40f0-472b-8e37-1cf469f104e9.png" width="300" height="200">
  
  ## Data augmentation
  Data augmentation is a common practice in machine learning if the data set is too small or the diversity of data in the data set is low. Basically data augmentation is nothing but introducing some changes in the input data without losing key features. With this technique you can create a large set of data from the limited original input data. 
  In this project we only have 100 raw images, and which is very less for a training task. 
  So we are going to expand 100 image to 7000 image with data augmentation.
  In this project the DA(Data Augmentation.) module use the following  operations,
   
   1. Random brightness : The brightness level of the input image will be changed randomly within a limit.
   2. Random saturation : The saturation level of the input image will be changed randomly within a limit.
   3. Random hue : The hue level of the input image will be changed randomly within a limit.
   4. Random contrast : The contrast level of the input image will be changed randomly within a limit.
   5. Horizontal flip : The input image will be flipped along y axis.
    
Note: The mask will not be modified in 1-4 operation, instead the original mask will be replicated. But in case of 5th operation both input image and mask need to be flipped to retain the strutural relationship between image and mask.

  ## Prepare the dataset
  Clone this repo to your working directory with the following git command.
  
~~~
  git clone https://github.com/asujaykk/Road-segmentation-UNET-model.git
~~~
  Extarct the data set 'road_seg_kitti.zip' available in the 'data/data_set folder' to 'data_set/data_temp_folder' with the following command.
~~~
  unzip data/data_set folder/road_seg_kitti.zip –d data_set/data_temp_folder/road_seg_kitti
~~~

  Expand the data set  70 times by executing the following data augmentation script (The script will take few minute to process).
~~~ 
 python3 DA_module.py --input data_set/data_temp_folder/road_seg_kitti --factor 70
~~~
The command line parameters expected by 'DA_module.py' is explained below.
1. --input  : This parameter should be a path to data set directory, in this case it is 'data_set/data_temp_folder/road_seg_kitti'
2. --factor : This parameter should be a number (default is 70), and the script will replicate the input image and mask by this factor.  
  
  Note: Please do not run the 'DA_module.py' on the data set twice, because the script will take the current image count in 'default/image_2' and then replicate it 70 times. So if you run 'DA_module.py' twice then the image will be expanded to '70*70' times.
   
  # 2. Model/algorithm selection.
   U-Net architecture is chosen for this task since the training cost for this model is less, and this model gave good benchmark over different data sets for different tasks. For more information with respect to U-Net architecture please check this link https://paperswithcode.com/method/u-net.

   # 3. Framework to be used for model realization
   Initially the plan was to use pytorch library to realize the model. But since i already used pytorch library for image classification and speech recognition task, this time i thought to use keras and tensorflow so that i can explore these libraries in detail.
 The model realization is available in 'model.py' file.
 
   # 4.Train the model
  Once you have the model architecture and data ready, then the next step is to train your model with the prepared data set and generate an optimized model. This trained model will be used for your future prediction.
  Model training require rich hardwares for faster training, especially GPU's(graphical Processing Unit) with cuda cores. If your PC configuration is not great then you have to wait hours to get the result.
  As a developer the best option would be to use online platform like kaggle notebook , google colab or AWS-EC2, because they provide best GPU backend support (Go for paid version if you need un-interrupted training for a long time with more RAM and GPU memory) and rich memory for processing complex models on huge data set.
  
   I chose google colab to train the U-Net model for (160 x 160) input size, and an augmented dataset of 7000 images. I chose 32 batch size and 15 epochs for faster training. The trained model was saved in my google drive for inferenecing.
  
  ## Training time on different Machines.
  1.  Google colab with GPU (NVIDIA tesla k80 24gb gpu) backend support : 20 minute  (32 batch size , 15 epoch)
  2.  Personal PC, i3 processor with 4gb Ram without GPU support :  23 hours (8 batch size , 15 epoch)
  3.  Personal PC, i7 processor with 16gb Ram with NVIDIA 6gb GPU support :  1.5 hours (32 batch size , 15 epoch)
  4.  Jetson nano developement board, ARM processor, 4gb Ram, GPU with 128 cuda cores :  15 hours (2 batch size , 15 epoch)
   
 You can use the script "model_train.py" for training the U-Net model over the data set that we created before.
 ~~~
    python3 model_train.py --input data_set/data_temp_folder/road_seg_kitti --output models/pretrained_models --batch 4 --epoch 15
 ~~~
  The command line parameters expected by 'model_train.py' is explained below.
  1. --input  : Path to the root of dataset directory
  2. --output : Path where the output model will be saved, the default name of this model will be "road_segmentation_160_160_test.h5"
  3. --batch  : The batch size of data to be used for training and evaluation (default value is 4)
  4. --epoch  : The epoch for training and evaluation (default value is 15)
  
  If you are facing 'OOM'(out of memory) error from tensorflow while training , then please reduce the batch size to a small number. You can increase the batch size upto 32 based on available computer resources.
 
 Once the training completed successfully,then you can find your trained model "road_segmentation_160_160_test.h5" under  "models/pretrained_models" directory ;)
  
   # 5.Inferencing keras model.
   Model inferencing is the process of using the model for prediction from new images.
   The model can be used to predict the road region from new images/Videos, and we have an Inference pipeline created to predict the road region from an input mp4 Video/IP cam videos.
      
   1. The inference pipeline support mp4 video as input and the input image will be resized to (600 x 400) and the predicted mask will be also at (600 x 400 ) size.
   2. The infrenec model by default shows input image (color image), predicted road mask (binary image) , Final output (mask impossed on red layer of input image) in seperate windows as shown below.
  ![Screenshot from 2022-04-09 12-55-31](https://user-images.githubusercontent.com/78997596/162561628-b3afaa92-3fd6-4700-95c9-8b6c12099b89.png) 
 
 We have a few pretrained models under 'models/pretrained_models' folder for testing. And they can be tested with the following inference script.
 ~~~
 python3 inference.py --src <path_to_mp4_video>  --model models/pretrained_models/road_segmentation_160_160.h5
 ~~~
 The command line parameters expected by 'inference.py' is explained below.
 1. --src : Path to input video file (recommended mp4 format).
 2. --model : Path to keras model to be used for inferencing (default : 'models/pretrained_models/road_segmentation_160_160.h5').
 
 If you want to test the model that you created before, then please change the '--model' parameter to 'models/pretrained_models/road_segmentation_160_160_test.h5'. and run the 'inference.py'.

 
 # 6.Keras to ONNX conversion.
 The keras model inference took 265 to 340 ms to process one input image, which is pretty slow for real world application. 
 Thus for better performance and deployement we decided to convert the kers model in to onnx model(Open Neural Network Exchange).
 If you want to know why we are converting keras model to onnx, then please check this link :https://pythonsimplified.com/onnx-for-model-interoperability-faster-inference/.
 
 There is already a script available in this repo for converting keras model to onnx model.
 You can execute the below command to convert keras model to onnx model.
 ~~~
 python3 generate_onnx.py --input models/pretrained_models/road_segmentation_160_160.h5  --output models/onnx_models/road_seg_160_160.onnx --temp models/saved_model/road_seg
 ~~~
 The command line parameters expected by 'inference.py' is explained below.
 1. --input : Path to input keras model (.h5 file).
 2. --output : Path to output onnx file to be created.
 3. --temp  : A directory to save intermediate files created by the script.


If you want to convert the keras model that you created before to onnx model then please run below command.
 ~~~
 python3 generate_onnx.py --input models/pretrained_models/road_segmentation_160_160_test.h5  --output models/onnx_models/road_seg_160_160_test.onnx --temp models/saved_model/road_seg_test
 ~~~

 The generated onnx model provided better performance on low end machines. 
 The onnx model inference took 75 to 90 ms to process one frame on a low end machine , which is acceptable for real world application. 
 
 # 5.Inferencing onnx model.
To inference the onnx model please use the 'inference_onnx.py' like below.

~~~
python3 inference_onnx.py --src <path_to_mp4 video> --model models/onnx_models/road_seg_160_160.onnx
~~~
The command line parameters expected by 'inference_onnx.py' is explained below.
 1. --src : Path to input video file (recommended mp4 format). 
 2. --model : Path to onnx model to be used for inferencing (default : models/onnx_models/road_seg_160_160.onnx).


 If you want to test the model that you created before, then please change the '--model' parameter to 'models/onnx_models/road_seg_160_160_test.onnx' and run the 'inference_onnx.py'.

The inference video output is given below. In which,
1. The left frame represent the input image.
2. Middle frame represent the mask predicted by the model (the white part is the road area).
3. The right frame represent the final output, in which the mask is imposed in the RED layer of the input image (as a result the road will be highlighted in RED colour).

![Screencast 2022-1649225143098](https://user-images.githubusercontent.com/78997596/161906906-9ec9989e-9617-4500-adef-e1d40c03c75c.gif)

# Next steps
1. Will be updated soon :)

