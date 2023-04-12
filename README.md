# EVA-8_Phase-1_Assignment-12
This is the assignment of 12th session in Phase-1 of EVA-8 from TSAI

## Introduction

### Objective
Objective of this assignment is to work with Yolo model. Below are the sub objectives:
1. Run inference on pre-trained Yolov3 model using OpenCV and use your own image holding objects from COCO dataset.
2. Pre-trained Yolov3 on custom dataset containing 4 classes and total of 200 images. Annotate this 200 images as well.
3. Run this Yolov3 model on a youtube downloaded video containing any of the four classes and upload this video back to youtube and share the link.
4. share the repo with all implementation, notebook with code and results images.
4. **Bonus-part**: Use the same custom dataset of 200 images and train a Yolov4 model.

### Getting started
- All notebooks are self contained means opening it on colab and running each cell should re-create the same results although the models are trained localy.
- There are cells in the notebook which is modifying some code line as some of the repo's are old and need some modification.
### Repository 
Compelete assignment implementation is divided into three parts both in README as well as in the repository folder.

#### Part-1
- OpenCV based inference implementation of Yolov3.
- Generated image on custom image holding COCO object.

#### Part-2
- Annotate custom collacted data of 4 classes.
- Train a Yolov3 model on that.

#### Part-3
- Download youtube video containing any of four classes.
- generate frames using FFMPG and infer over the frame using custom trained model.
- upload this video back to youtube and share the link.

## Dataset representation
As mentioned above, here 200 images of 4 custom classes are collected. This four classes are -

 **cybertruck**, **motherboard**, **dji_drone**, **courage**.
<br>
Below is the image showing the data we used for this classes.
![Alt text](Part_2-training_on_custom_dataset/dataset.png?raw=true "model architecture")
This dataset are also added as a zip file in this repo in `Part_2-training_on_custom_dataset/deveshcustom_512.zip`

## About YOLO (You Only Look Once)
YOLO (You Only Look Once) is a real-time object detection model that uses a single neural network to predict the bounding boxes and class probabilities of objects in an image. Unlike traditional object detection algorithms that perform region proposals and classification separately, YOLO divides the input image into a grid of cells and predicts the bounding boxes and class probabilities directly from the grid cells. This makes YOLO extremely fast and accurate, with the ability to process up to 60 frames per second on a GPU. YOLO has become a popular model for real-time object detection tasks, such as autonomous driving, surveillance, and robotics.
<br>
Below image shows a architecture of famous Yolov3 model.
![Alt text](Part_2-training_on_custom_dataset/yolov3_arch.png?raw=true "model architecture")

## PART-1: OpenCV based YOLOv3 inference
notebook `part_1-OpenCV_Yolo/Yolo_with_OpenCV.ipynb` contains code in which we first download the weights and config file of Yolov3 and then load the model and convert the image into blob. Run the inference on the image parse the output to get the bounding boxes and then finally overlay it on the input image.
### Image output from OpenCV inference
<img src="Part_1-OpenCV_Yolo/image_out.jpg" width="300" height="600" title="output image">

## PART-2: Training YOLOV3 on custom dataset
Part-2 is the interesting part of the assignment where we train 
### Plot showing 16 output images
<img src="Part_2-training_on_custom_dataset/16_output.png">

## PART-3: Inference on Youtube video
### Class - Courage (the cowardly dog)
[![IMAGE ALT TEXT](https://img.youtube.com/vi/mxIH-kjL918/0.jpg)](https://www.youtube.com/watch?v=Vnumdu73oUI)

### Class - Cybertruck (truck from the future)
[![IMAGE ALT TEXT](https://img.youtube.com/vi/J2U9Hmmpqhc/0.jpg)](https://www.youtube.com/watch?v=lrXfjzat3po)


## PART-Bonus: Training YOLOV4 model on custom dataset

## Conclusion