## Project: Build a Traffic Sign Recognition Program
[![Udacity - Self-Driving Car NanoDegree](https://s3.amazonaws.com/udacity-sdc/github/shield-carnd.svg)](http://www.udacity.com/drive)

**Traffic Sign Recognition**

**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)
[image12]: ./writeup/1_2.png "Randomized Sample"
[image122]: ./writeup/1_2_2.png "Label Distribution"
[image123]: ./writeup/1_2_3.png "Extra Data"
[rgb_norm]: ./writeup/rgb_norm.png
[bw_norm]: ./writeup/bw_norm.png
[norm_10]: ./writeup/norm10.png
[hist2]: ./writeup/hist2.png 
[hist3]: ./writeup/hist3.png 
[extra1]: ./writeup/extra1.png
[extra2]: ./writeup/extra2.png 
[streetview1]: ./writeup/streetview1.png 
[streetview2]: ./writeup/streetview2.png 
[visualize]: ./writeup/visualize.png 

## Rubric Points
---
### Dataset Exploration
#### 1. Dataset Summary
The dataset provided by Udacity was imported into IPython using "pickle" and sepatated into training, validation, and testing set. 
Numpy library was used to calculate the summary statisctis of the imported dataset:
* The size of training set is 34799
* The size of test set is 12630
* The shape of a traffic sign image is (32, 32, 3)
* The number of unique classes/labels in the data set is 43

#### 2. Exploratory Visualization
In code cell 3, a randomized sample of the dataset of each unique classes was shown as below.
![image12]

To learn the calss label distribution across the training data set, a histrogram was plotted using matplotlib.
![image122]


### Design and Test a Model Architecture
#### 1. Preprocessing
The code for preprocessing is contained in the code cell 12-14 of the IPython notebook.

As a first step, the images were normalized in all R,G,B channels separately. The idea is to map values which are between 0-255, remove empty values from either end, and map the remaining values to between -1 and +1. This provides the neural network with data that have zero mean and small variance whihc can improve the training performance.

After normalization, the colour infomation is taken out by taking the mean of the normalized RGB values. This effectively converted the images into greyscale. This allows the images to be read by the neural network as 2D data.

The image below shows one of the training sign being normalized in RGB channel in histrograms and the greyscale result. The high constrast result allows the neural network training to be more effective.
![rgb_norm]
![bw_norm]

The same processes were applied to the entire dataset including all training, validation, and testing data. A random sample of 10 road signs are presented below include the original images, histograms and result of the normalized images.
![norm_10]

#### 2.Model Architecture
The original dataset provided contains 34799 images of 43 types of road signs for network training purposes. As shown in previous plot, the distribution of signs across the training set is not uniformly distributed. That means some signs would appear in network training more often than the ones with fewer training images. To solve this problem, extra training data needs to be generated in order to evenly distribut the chances of each sign appearing in the training set. Moreover, to better train the neural network, the extra data has to be argumented so that they are not repeated. Graph below shows the number of extra augmented data needed to be generated.
![hist2]

After adding the extra data, the distrubtion of signs across the training set would become uniform as shown below.
![hist3]

In code cells 5-6, extra sign images were generated from the existing pool and argumented randomly using cv2. 
![extra1]

Each image is randomly selected from the original training set, randomly scaled, stretched, and rotated to simulate signs being seen from different angles. A sample of the extra training data results are shown below.
![extra2]



#### 3. Model Training

My final model is based off of lenet-5 with modificationed. My goal was to create a relatively light weight network with is quick to trian. It consisted of the following layers:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x1 Greyscale image   					| 
| Convolution 5x5     	| 1x1 stride, outputs 28x28x6 					|
| RELU					|												|
| Pooling	   		   	| 2x2 stride, outputs 14x14x6	 				|
| Convolution 5x5	    | 1x1 stride, outputs 10x10x16     				|
| RELU 					| 		      									|
| Pooling				| 2x2 stride, outputs 5x5x16					|
| Flatten				| outputs 400									|
| Fully connected		| outputs 120									|
| RELU 					| 		      									|
| Dropout 				| 		      									|
| Fully connected		| outputs 84									|
| RELU 					| 		      									|
| Dropout 				| 		      									|
| Fully connected		| outputs 43									|


#### 4. Solution Approach
The model was trained at rate of 0.001, epochs fo 50, batch size of 2048, and dropout rate of 50%.

My final model results were:
* training set accuracy of 0.991
* validation set accuracy of 0.957
* test set accuracy of 0.927

The network is base off of lenet-5 with two extra dropout layers. The lenet-5 network is already well known and proven to work in image sets similar to the road sign set. Originally the lenet would perform just shy of 93% accuracy on the validation set. However, after adding two dropout layers after the fully connected layers, the validation set accuracy has been improved to 95.7% within 17 epochs. 



### Test a Model on New Images

#### Acquiring New Images

A sample of 8 German road signs were aquired from Google Street View service. The quality of the images match average images we find in the training set with similar lighting and sharpness. 

![streetview1]



#### Performance on New Images

The model performs very well on the new image set as it correctly classified all 8 of the iamges provided.

| Image			        |     Prediction	        | 
|:---------------------:|:-------------------------:| 
| Priority road      	| Priority road  			| 
| Road work    			| Road work 				|
| Keep right			| Keep right				|
| Speed limit (30km/h)  | Speed limit (30km/h)		|
| No passing			| No passing      			|
| No entry				| No entry      			|
| Wild animals crossing	| Wild animals crossing     |



#### Model Certainty - Softmax Probabilities
The top five softmax probabilities of the predictions on the captured images are outputted as below. Note the top five softmax probabilities often have similar shape as the correct classification. For example, the road work sign has a red border triangular shape and the next four top probabilities choices also do. Similar trend can be observed on the 30kmh speed limit sign. All top five choices are speed limit signs which are extremely similar to the correct answer. This shows the working of the trained model.

![streetview2]


### Visualize the Neural Network's State with Test Images
The trained network seems to be able to pick up the diagonal lines. These angled lines are prominent features which appear on many road signs. To identify the combination of diagonal lines can help the network to classify one sign over the others. Another feature can be shown which show the distribution of light area against the dark area of the sign. This can tell the symbol appears on the sign and they can be classified accordingly.

![visualize]

