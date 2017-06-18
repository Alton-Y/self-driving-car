This repository includes all of the projects I have finished while working on the Udacity Self-Driving Car NanoDegree. Topics covered by the NanoDegree include deep learning, computer vision, sensor fusion, controllers, and vehicle kinematics. Click on the links to find details about each project.

[![Udacity - Self-Driving Car NanoDegree](https://s3.amazonaws.com/udacity-sdc/github/shield-carnd.svg)](http://www.udacity.com/drive)

[T1_P1]: ./markdown/T1_P1.jpg
[T1_P2]: ./markdown/T1_P2.jpg
[T1_P3]: ./markdown/T1_P3.jpg
[T1_P4]: ./markdown/T1_P4.jpg
[T1_P5]: ./markdown/T1_P5.jpg


## **Finding Lane Lines on the Road**
![T1_P1]
### [/CarND-Term1-P1-LaneLines](https://github.com/Alton-Y/self-driving-car/tree/master/CarND-Term1-P1-LaneLines)
#### In this project, I wrote code to identify lane lines on the road in images and a video stream. 
* Make a pipeline that finds lane lines on the road
* Use computer vision technique such as color selection, Canny edge detection, and Hough transform to find lane lines


## **Build a Traffic Sign Recognition Program**
![T1_P2]
### [/CarND-Term1-P2-Traffic-Sign-Classifier-Project](https://github.com/Alton-Y/self-driving-car/tree/master/CarND-Term1-P2-Traffic-Sign-Classifier-Project)
#### In this project, I used deep neural networks and convolutional neural networks to classify traffic signs using the German Traffic Sign Dataset.
* Load the data set
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images

## **Behavioral Cloning**
![T1_P3]
### [/CarND-Term1-P2-Traffic-Sign-Classifier-Project](https://github.com/Alton-Y/self-driving-car/tree/master/CarND-Term1-P3-Behavioral-Cloning)
#### In this project, I used convolution neural network to train a model to redicts steering angles from images in order to control a self-driving car in the simulator without leaving the track.
* Use the simulator to collect data on good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road


## **Advanced Lane Finding**
![T1_P4]
### [/CarND-Term1-P4-Advanced-Lane-Lines](https://github.com/Alton-Y/self-driving-car/tree/master/CarND-Term1-P4-Advanced-Lane-Lines)
#### In this project, I wrote a software pipeline to identify the lane boundaries in a video from a front-facing camera on a car.

* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* Apply a distortion correction to raw images
* Use color transforms, gradients, etc., to create a thresholded binary image
* Apply a perspective transform to rectify binary image ("birds-eye view")
* Detect lane pixels and fit to find the lane boundary
* Determine the curvature of the lane and vehicle position with respect to center
* Warp the detected lane boundaries back onto the original image
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position


## **Vehicle Detection**
![T1_P5]
### [/CarND-Term1-P5-Vehicle-Detection](https://github.com/Alton-Y/self-driving-car/tree/master/CarND-Term1-P5-Vehicle-Detection)
#### In this project, I wrote a software pipeline to identify vehicles in a video from a front-facing camera on a car using color and gradient features and a support vector machine classifier.

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier
* Apply a color transform and append binned color features, as well as histograms of color, to your HOG feature vector
* Normalize your features and randomize a selection for training and testing
* Implement a sliding-window technique and use your trained classifier to search for vehicles in images
* Run the pipeline on a video stream and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles
* Estimate a bounding box for vehicles detected
