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
 The data required for this task is a collection of images taken from the front camera of a car / vehicle. I used my mobile phone camera to capture continuous video while I was driving. The camera was mounted in the center of the front windshield.
 I collected 28 hours of such videos during my travels in various districts of Kerala and Karnataka.
 Data includes different types of roads (highway, lane road, lane mark road, mud road, one way, multi lane road)
 ![0](https://user-images.githubusercontent.com/78997596/161808366-48a3e413-073a-4784-90fd-b1de44b7456c.jpg)
![0](https://user-images.githubusercontent.com/78997596/161808448-0e722968-e25e-434b-84f3-630e16dbade5.png)

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
   
   # 5.Inference the model
   The model can be used to predict the road area from new images/Videos, and we have an Inference pipeline created to predict the road area from an input mp4 Video/IP cam videos.
   
   1. The inference pipeline support mp4 video as input and the input image will be resized to (600 x 400) and the predicted mask will be also at (600 x 400 ) size.
   2. The infrenec model by default shows input image (color image), predicted road mask (binary image) , Final output (mask impossed on red layer of input image)
  
  
 # 6. Performance improvement.
 The model inference took 265 to 340 ms to process one input image, which is pretty slow for a real world application. 
 So we decided to convert the kers model in to onnx model, for better performance and deployement.
 
 The generated onnx model provided better performance on low end machines 
 in my pC the onnx model inference took 75 to 90 ms to process one frame which was acceptable for a real world application. 
 
