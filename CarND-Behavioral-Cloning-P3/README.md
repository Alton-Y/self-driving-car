# **Behavioral Cloning** 

## Writeup Template

### You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

---

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data on good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./model.png "Model Visualization"


## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
### Required Files

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* Reader.md summarizing the results
* run1.mp4 and run2.mp4 are video recordings of the vehicle driving autonomously around the track

### Quality of Code

#### 1. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

#### 2. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

My model consists of a convolution neural network with series of convolution 2D layers and fully connected layers (model.py line 71-90).

My model has a cropping layer (model.py line 74) to remove part of the top and bottom of each frame to reduce the data input size. This also helps training the model faster with higher precision. 

The model includes RELU layers to introduce nonlinearity, and the data is normalized in the model using a Keras lambda layer (model.py line 73). 

#### 2. Attempts to reduce overfitting in the model

The model contains dropout layers in order to reduce overfitting (model.py lines 84,87). 

The model was trained and validated on different data sets to ensure that the model was not overfitting. All four sets of training data were loaded and concatenated in code line 4-13. The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

#### 3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually (model.py line 92).

#### 4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used a combination of center lane driving (both directions on the test circuit), recovering from the left and right sides of the road.

There are several spots along the track require extra data to reinforce the autonomous performance such as crossing the bridge, tight turns, as well as the spot where the paved road intersects with the dirt road.

For details about how I created the training data, see the next section. 



### Model Architecture and Training Strategy

#### 1. Solution Design Approach


My first step was to use a convolution neural network model similar to the NVIDIA CNN architecture from this paper:
https://images.nvidia.com/content/tegra/automotive/images/2016/solutions/pdf/end-to-end-dl-using-px.pdf

In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set. I found that my first model had a low mean squared error on the training set but a high mean squared error on the validation set. This implied that the model was overfitting. 

To combat the overfitting, I modified the model so that it has zero mean and equal variance across the data set. This is a common practice to set up a well-conditioned data set for machine learning to ensure numerical stability. (Line 73)s

The next step was to run the simulator to see how well the car was driving around track one. There were a few spots where the vehicle fell off the track such as crossing the bridge, tight turns, as well as the spot where the paved road intersects with the dirt road. To improve the driving behavior in these cases, I reinforced the learning data by creating more data by driving through these spots over and over again.

Also because of the correction data for when the car was leaning to the edge of the track were mostly corrected with large steering angles. This makes the model to over correct and could not provide a smooth ride. This behavior was corrected by capturing data with minor off-center behavior with less aggressive steering angles.

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road at various speeds (9 km/h to 30 km/h)

#### 2. Final Model Architecture

The final model architecture (model.py lines 71-92) consisted of a convolution neural network with the following layers and layer sizes and here is a visualization of the architecture from Keras:

![image1]

####3. Creation of the Training Set & Training Process

To capture good driving behavior, I first recorded two laps on track one (both directions) using center lane driving. Here is an example image of center lane driving:
[run1]: ./writeup/run1.jpg "Run 1"
[run2]: ./writeup/run2.jpg "Run 2"

##### Normal Direction
![run1]
##### Reversed Direction
![run2]


I then recorded the vehicle recovering from the left side and right sides of the road back to center so that the vehicle would learn to correct from off-center behaviors. These images show what a recovery looks like:


[run3a]: ./writeup/run3a.jpg 
[run3b]: ./writeup/run3b.jpg
[run3c]: ./writeup/run3c.jpg

![run3a]
![run3b]
![run3c]


After the correction training, the model still underperformed at tight turns and across the bridge. Therefore some reinforcement training was done on those specific spots.

[bridge]: ./writeup/bridge.jpg
[dirt]: ./writeup/dirt.jpg
[tight]: ./writeup/tight.jpg
##### Bridge
![bridge]

##### Dirt Road Intersection
![dirt]

##### Tight Bend
![tight]

At the end, to smooth out the oscillation likely caused by over-correcting off-center driving, more data were gathered to correct off-center driving with smaller steering angle.

##### Gentle Steering Correction
[gentle]: ./writeup/gentle.jpg
![gentle] 



After the collection process, I had 9444 frames across 4 training set folders. Each of these frame produces three frames from three different camera angles (Left, Center, and Right). To produce more unbiased data, each frame is horizontally flipped to create augmented data to produce better training result with reversed steering angles. Therefore there are totally 9444x3x2 = 56664 data points. 

The correction factor of steering angle is applied to non-center camera angles. The value applied is 0.22 rad and it was worked out from iterative process. Since there is a optimum steering angle physically depends on the offset distance of the cameras. However this cannot be found using trigonometry or any sort of theoretical method without sufficient data such as the offset distance of the cameras. Therefore the correction factor was iterated through a trial and error process.

[left]: ./writeup/left.jpg
[center]: ./writeup/center.jpg
[right]: ./writeup/right.jpg
[flipped]: ./writeup/center_flipped.jpg

##### Left Canera Angle (Steering angle + 0.22 rad)
![left]

##### Center Canera Angle (Steering angle unchanged)
![center]

##### Right Canera Angle (Steering angle - 0.22 rad)
![right]

##### Horizontal Flipped Image (Steering angle x -1)
![flipped] 




I finally randomly shuffled the data set and put 20% of the data into a validation set using sklearn library (Line 17). 

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was Z as evidenced by 10. I used an adam optimizer so that manually training the learning rate wasn't necessary. The model weights were saved as model_weights.h5 in case further training is necessary.

Since there are 56664 data points and each of them contains a 320x160 RGB image. The memory needed to train the model would exceed the hardware limitation. Therefore a fit generater was used to run the fit process batch-by-batch in parallel to ensure training efficiency.  


### Simulation
The trained model was run in the driving simulator using drive.py.
SInce the speed of the car is controlled by a PID controller in drive.py. One can select the driving speed and start from relatively slow speed. At the begining, all simulations were run at 9 km/h to slow down the process in order to spot the flaws in the trained model. Once the model has completed the track. The model was tested in faster running speed and results were recorded as run1.mp4 and run2.mp4.

