<h1>VISION BEYOND LIMITS</h1>

<p align="center"><img src ="https://user-images.githubusercontent.com/84843295/145776554-d2d61b46-f9ca-4f91-a75e-ed2dafe7170d.png" /></p>

We implemented a multi-class classification approach for disaster assessment from the given data set of post earthquake satellite imagery. The idea was to classify the different damage types in an image.

## Table Of Content
* [Introduction](#Introduction)
* [Approach](#Approach)


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

## Approach
Good performance of deep learning algorithms is limited to the size of data available, and the network structure is considered. One of the most critical challenges for using a deep learning method for monitoring the buildings damaged in the disaster is that the training images of damaged targets are usually not very much. So models that can give considerably high accuracy compared to that of a regular model on a small dataset needed to be chosen.

## Convolutional Neural Networks (CNN)

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

## Team Members
* NEEL SHAH		        
* DHRUV KUNJADIYA           
* ARNAV ZUTSHI
* PRATHAM SHAH

## Mentors
* Aman Chhaparia
* Prathamesh Tagore
* Mann Shah
