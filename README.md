<h1>VISION BEYOND LIMITS</h1>

<p align="center"><img src ="https://user-images.githubusercontent.com/84843295/145776554-d2d61b46-f9ca-4f91-a75e-ed2dafe7170d.png" /></p>

We implemented a multi-class classification approach for disaster assessment from the given data set of post earthquake satellite imagery. The idea was to classify the different damage types in an image.

## Table Of Content
* [Introduction](#introduction)
* [File-Structure](#file-structure)
* [Pre-Requisites](#pre-requisites)
* [Approach](#approach)
  * [Convolutional Neural Networks](#convolutional-neural-networks)
  * [Multi-Class classification](#multi-class-classification)



## Introduction
We need to extract data from the post earthquake imagery training set and classify buildings on the basis of how damaged they are.
The buildings are classified into 5 types .i.e:
* Destroyed buildings
* Major Damaged buildings
* Minor Damaged buildings                                         
* No damaged buildings
* Unclassified
<p align="center"><img src ="https://user-images.githubusercontent.com/84843295/145777282-f9d16fd6-6abb-420a-9f55-61ae038cdf4e.png" /></p>
We further need to assign their respective colours to the classified objects in the image. This data is then to be trained in our network and should give us the classification with colour-codes as given. This will require the concepts of deep learning and computer visions to be thoroughly applied.

## File Structure
```
📦Vision-Beyond-Limits-main
 ┣ 📂Mask-20211120T183800Z-001
 ┃ ┗ 📂Mask
 ┃ ┃ ┣ 📜mexico-earthquake_00000001_post_disaster.png
 ┃ ┃ ┣ 📜mexico-earthquake_00000002_post_disaster.png
 ┃ ┃ ┣ 📜mexico-earthquake_00000003_post_disaster.png
 ┃ ┃ ┣ 📜mexico-earthquake_00000004_post_disaster.png
 ┃ ┃ ┣ 📜mexico-earthquake_00000007_post_disaster.png
 ┃ ┃ ┗ ...
 ┣ 📂Output_classified_images
 ┃ ┣ 📜Output_1.png
 ┃ ┣ 📜Output_10.png
 ┃ ┣ 📜Output_11.png
 ┃ ┣ 📜Output_12.png
 ┃ ┗ ...
 ┣ 📂vbl_data
 ┃ ┣ 📂augmented_data
 ┃ ┃ ┣ 📂data_180
 ┃ ┃ ┃ ┣ 📂images_180
 ┃ ┃ ┃ ┃ ┣ 📜mexico-earthquake_00000001_post_disaster.png_180.png
 ┃ ┃ ┃ ┃ ┣ 📜mexico-earthquake_00000002_post_disaster.png_180.png
 ┃ ┃ ┃ ┃ ┣ 📜mexico-earthquake_00000003_post_disaster.png_180.png
 ┃ ┃ ┃ ┃ ┣ 📜mexico-earthquake_00000004_post_disaster.png_180.png
 ┃ ┃ ┃ ┃ ┣ 📜mexico-earthquake_00000007_post_disaster.png_180.png
 ┃ ┃ ┃ ┃ ┗ ...
 ┃ ┃ ┃ ┗ 📂masks_180
 ┃ ┃ ┃ ┃ ┣ 📜mexico-earthquake_00000001_post_disaster.png_180.png
 ┃ ┃ ┃ ┃ ┣ 📜mexico-earthquake_00000002_post_disaster.png_180.png
 ┃ ┃ ┃ ┃ ┣ 📜mexico-earthquake_00000003_post_disaster.png_180.png
 ┃ ┃ ┃ ┃ ┣ 📜mexico-earthquake_00000004_post_disaster.png_180.png
 ┃ ┃ ┃ ┃ ┣ 📜mexico-earthquake_00000007_post_disaster.png_180.png
 ┃ ┃ ┃ ┃ ┗ ...
 ┃ ┃ ┣ 📂data_270
 ┃ ┃ ┃ ┣ 📂images_270
 ┃ ┃ ┃ ┃ ┣ 📜mexico-earthquake_00000001_post_disaster.png_270.png
 ┃ ┃ ┃ ┃ ┣ 📜mexico-earthquake_00000002_post_disaster.png_270.png
 ┃ ┃ ┃ ┃ ┣ 📜mexico-earthquake_00000003_post_disaster.png_270.png
 ┃ ┃ ┃ ┃ ┣ 📜mexico-earthquake_00000004_post_disaster.png_270.png
 ┃ ┃ ┃ ┃ ┣ 📜mexico-earthquake_00000007_post_disaster.png_270.png
 ┃ ┃ ┃ ┃ ┗ ...
 ┃ ┃ ┃ ┗ 📂masks_270
 ┃ ┃ ┃ ┃ ┣ 📜mexico-earthquake_00000001_post_disaster.png_270.png
 ┃ ┃ ┃ ┃ ┣ 📜mexico-earthquake_00000002_post_disaster.png_270.png
 ┃ ┃ ┃ ┃ ┣ 📜mexico-earthquake_00000003_post_disaster.png_270.png
 ┃ ┃ ┃ ┃ ┣ 📜mexico-earthquake_00000004_post_disaster.png_270.png
 ┃ ┃ ┃ ┃ ┣ 📜mexico-earthquake_00000007_post_disaster.png_270.png
 ┃ ┃ ┃ ┃ ┗ ...
 ┃ ┃ ┗ 📂data_90
 ┃ ┃ ┃ ┣ 📂images_90
 ┃ ┃ ┃ ┃ ┣ 📜mexico-earthquake_00000001_post_disaster.png_90.png
 ┃ ┃ ┃ ┃ ┣ 📜mexico-earthquake_00000002_post_disaster.png_90.png
 ┃ ┃ ┃ ┃ ┣ 📜mexico-earthquake_00000003_post_disaster.png_90.png
 ┃ ┃ ┃ ┃ ┣ 📜mexico-earthquake_00000004_post_disaster.png_90.png
 ┃ ┃ ┃ ┃ ┣ 📜mexico-earthquake_00000007_post_disaster.png_90.png
 ┃ ┃ ┃ ┃ ┗ ...
 ┃ ┃ ┃ ┗ 📂masks_90
 ┃ ┃ ┃ ┃ ┣ 📜mexico-earthquake_00000001_post_disaster.png_90.png
 ┃ ┃ ┃ ┃ ┣ 📜mexico-earthquake_00000002_post_disaster.png_90.png
 ┃ ┃ ┃ ┃ ┣ 📜mexico-earthquake_00000003_post_disaster.png_90.png
 ┃ ┃ ┃ ┃ ┣ 📜mexico-earthquake_00000004_post_disaster.png_90.png
 ┃ ┃ ┃ ┃ ┣ 📜mexico-earthquake_00000007_post_disaster.png_90.png
 ┃ ┃ ┃ ┃ ┗ ...
 ┃ ┣ 📂orginal_images
 ┃ ┃ ┣ 📜mexico-earthquake_00000001_post_disaster.png
 ┃ ┃ ┣ 📜mexico-earthquake_00000002_post_disaster.png
 ┃ ┃ ┣ 📜mexico-earthquake_00000003_post_disaster.png
 ┃ ┃ ┣ 📜mexico-earthquake_00000004_post_disaster.png
 ┃ ┃ ┣ 📜mexico-earthquake_00000007_post_disaster.png
 ┃ ┃ ┗ ...
 ┃ ┗ 📂original_mask
 ┃ ┃ ┣ 📜mexico-earthquake_00000001_post_disaster.png
 ┃ ┃ ┣ 📜mexico-earthquake_00000002_post_disaster.png
 ┃ ┃ ┣ 📜mexico-earthquake_00000003_post_disaster.png
 ┃ ┃ ┣ 📜mexico-earthquake_00000004_post_disaster.png
 ┃ ┃ ┣ 📜mexico-earthquake_00000007_post_disaster.png
 ┃ ┃ ┗ ...
 ┣ 📜augment.py
 ┣ 📜LICENSE
 ┣ 📜masking.py
 ┣ 📜Model.ipynb
 ┣ 📜README.md
 ┗ 📜README.txt

```



## Pre-Requisites
The following modules or packages/environment are required for running the code
* Python v3 si required 
  version: python==3.7.12
* Libraries like Keras and tensorflow are also required for machine learning algorithms
  ```
  pip install keras==2.7.0
  ```
  ```
  pip install tensorflow==2.7.0
  ```
* Numpy and OpenCV are 2 very important libraries for image processing
  ```
  pip install numpy==1.16.5
  pip install opencv==4.1.2
  ```
* Matplotlib is used to plot graphs of accuracy and loss
  ```
  pip install matplotlib==3.2.2
  ```
* Sklearn library is used for encoding the images and providing the classes with labels
  ```
  pip install sklearn==1.0.1
  ```

## Approach
Good performance of deep learning algorithms is limited to the size of data available, and the network structure is considered. One of the most critical challenges for using a deep learning method for monitoring the buildings damaged in the disaster is that the training images of damaged targets are usually not very much. So models that can give considerably high accuracy compared to that of a regular model on a small dataset needed to be chosen.

## Convolutional Neural Networks

<p align="center"><img src = "https://user-images.githubusercontent.com/84843295/145997095-9b0b54ae-1153-4a17-948d-bea7ea077849.png" /></p>
Convolutional neural networks are distinguished from other neural networks by their superior performance with image, speech, or audio signal inputs. They have three main types of layers, which are:

* Convolutional layer
* Pooling layer
* Fully-connected (FC) layer

The convolutional layer is the first layer of a convolutional network. While convolutional layers can be followed by additional convolutional layers or pooling layers, the fully-connected layer is the final layer. With each layer, the CNN increases in its complexity, identifying greater portions of the image. Earlier layers focus on simple features, such as colors and edges. As the image data progresses through the layers of the CNN, it starts to recognize larger elements or shapes of the object until it finally identifies the intended object.
We have used <b>U-Net</b> model which is convolutional network architecture for fast and precise segmentation of images.
<h3>U-Net</h3>
<p align="center"><img src = "https://user-images.githubusercontent.com/84843295/145995165-ee2b07b5-55d1-406b-92cd-6786dfefa05e.png" /></p>

## Multi-Class Classification
When we solve a classification problem having only two class labels, then it becomes easy for us to filter the data, apply any classification algorithm, train the model with filtered data, and predict the outcomes. But when we have more than two class instances in input train data, then it might get complex to analyze the data, train the model, and predict relatively accurate results. To handle these multiple class instances, we use multi-class classification.
Multi-class classification is the classification technique that allows us to categorize the test data into multiple class labels present in trained data as a model prediction.There are mainly two types of multi-class classification techniques:-

* One vs. All (one-vs-rest)
* One vs. One

<p align="center"><img src = "https://user-images.githubusercontent.com/84843295/145996084-7eb04eb1-9e4c-4b4e-880f-73cc8c7ab03b.png" /></p>

<p align="center"><img src = "https://user-images.githubusercontent.com/84843295/146252435-cf5904d0-e76d-4d93-aeca-b5f590d31769.png" /></p>


## Team Members
* [Neel Shah](https://github.com/Neel-Shah-29)		        
* [Dhruv Kunjadiya](https://github.com/Dhruv454000)           
* [Arnav Zutshi](https://github.com/shahpratham)
* [Pratham Shah](https://github.com/shahpratham)

## Mentors
* [Aman Chhaparia](https://github.com/amanchhaparia)
* [Prathamesh Tagore](https://github.com/meshtag)
* [Mann D0shi](https://github.com/MannDoshi)
