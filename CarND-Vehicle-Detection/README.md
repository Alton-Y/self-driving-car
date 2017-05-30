# Vehicle Detection
[![Udacity - Self-Driving Car NanoDegree](https://s3.amazonaws.com/udacity-sdc/github/shield-carnd.svg)](http://www.udacity.com/drive)


In this project, your goal is to write a software pipeline to detect vehicles in a video (start with the test_video.mp4 and later implement on full project_video.mp4), but the main output or product we want you to create is a detailed writeup of the project.  Check out the [writeup template](https://github.com/udacity/CarND-Vehicle-Detection/blob/master/writeup_template.md) for this project and use it as a starting point for creating your own writeup.  

---

The goals / steps of this project are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier
* Optionally, you can also apply a color transform and append binned color features, as well as histograms of color, to your HOG feature vector. 
* Note: for those first two steps don't forget to normalize your features and randomize a selection for training and testing.
* Implement a sliding-window technique and use your trained classifier to search for vehicles in images.
* Run your pipeline on a video stream (start with the test_video.mp4 and later implement on full project_video.mp4) and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.

Here are links to the labeled data for [vehicle](https://s3.amazonaws.com/udacity-sdc/Vehicle_Tracking/vehicles.zip) and [non-vehicle](https://s3.amazonaws.com/udacity-sdc/Vehicle_Tracking/non-vehicles.zip) examples to train your classifier.  These example images come from a combination of the [GTI vehicle image database](http://www.gti.ssr.upm.es/data/Vehicle_database.html), the [KITTI vision benchmark suite](http://www.cvlibs.net/datasets/kitti/), and examples extracted from the project video itself.   You are welcome and encouraged to take advantage of the recently released [Udacity labeled dataset](https://github.com/udacity/self-driving-car/tree/master/annotations) to augment your training data.  

---

**Vehicle Detection Project**

The goals / steps of this project are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier
* Optionally, you can also apply a color transform and append binned color features, as well as histograms of color, to your HOG feature vector. 
* Note: for those first two steps don't forget to normalize your features and randomize a selection for training and testing.
* Implement a sliding-window technique and use your trained classifier to search for vehicles in images.
* Run your pipeline on a video stream (start with the test_video.mp4 and later implement on full project_video.mp4) and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.


[image1]: ./examples/car_not_car.png



## [Rubric](https://review.udacity.com/#!/rubrics/513/view) Points
###Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---
###Writeup / README

####1. Provide a Writeup / README that includes all the rubric points and how you addressed each one.  You can submit your writeup as markdown or pdf.  [Here](https://github.com/udacity/CarND-Vehicle-Detection/blob/master/writeup_template.md) is a template writeup for this project you can use as a guide and a starting point.  

You're reading it!

###Histogram of Oriented Gradients (HOG)

####1. Explain how (and identify where in your code) you extracted HOG features from the training images.

I started by reading in all the `vehicle` and `non-vehicle` images.  Here is an example of one of each of the `vehicle` and `non-vehicle` classes:

These code are located in P5_explore.ipynb before being implemented into the main pipeline. First the histograms were plotted to inspect the RGB channel distribution of a sample vehicle classified image.

[sample]: ./writeup/sample.png
[hist]: ./writeup/hist.png
[spatially]: ./writeup/spatially.png

![sample]
![hist]

Similarly, the spatially binned features were plotted of the sample image.

![spatially]

After checking the sample of the dataset. The entire data set was loaded into the code using the following method:

`import glob`

`cars = glob.glob('dataset/vehicles/**/*.png')`

`notcars = glob.glob('dataset/non-vehicles/**/*.png')`

[carnotcar]: ./writeup/carnotcar.png
![carnotcar]

8792 cars and 8968 non-cars images were loaded in total.

I then explored the HOG method to identify the vehicle shape from the images using the following parameters:

`orient = 9`

`pix_per_cell = 8`

`cell_per_block = 2`

Test HOG result is shown below. As you can see the visualization shows the HOG has sucessifully picked up the shape of the car and I am going to use this set of parameter for the HOG pipeline.

[hog_sample]: ./writeup/hog_sample.png
![hog_sample]

[norm]: ./writeup/norm.png
![norm]

####2. Explain how you settled on your final choice of HOG parameters.

I tried various combinations of parameters and turns out this set gives the most consistant result with relatively short amount of computing time.

####3. Describe how (and identify where in your code) you trained a classifier using your selected HOG features (and color features if you used them).

The next major tuning is to select the best colour feature to classify car images. The color spaced I have tried included RGB, HSV, HLS, YCrCb. Within each method, the model was trained with each single color channel and all the channels across training set of 5000. At the end, YCrCb color space and all channels give the best training result of 98.44% with 16000 of car and not-car images. 

|Color Space|  Channel | Accuracy  |
|-------|-----|-----|
| RGB   | 0   |0.968|
| RGB   | 1   |0.981|
| RGB   | 2   |0.979|
| RGB   | ALL |0.971|
| HSV   | 0   |0.994|
| HSV   | 1   |0.994|
| HSV   | 2   |0.994|
| HSV   | ALL |0.994|
| HLS   | 0   |0.999|
| HLS   | 1   |0.993|
| HLS   | 2   |0.995|
| HLS   | ALL |0.990|
| YCrCb | 0   |0.992|
| YCrCb | 1   |0.987|
| YCrCb | 2   |0.983|
| YCrCb | ALL |0.998|





###Sliding Window Search

####1. Describe how (and identify where in your code) you implemented a sliding window search.  How did you decide what scales to search and how much to overlap windows?

Bit of preliminary testing with the test image set provide some insight ot how to set up the sliding window search. In short, three ranges (close, mid, long) were setup to scan three regions with different scales. They all have overlap of 50% to balance speed and accuracy. Images below show the three sliding window search and their parameters.


[scale]: ./writeup/scale.png
![scale]

Here is the close range sliding window search applied on test images.
[closerange]: ./writeup/closerange.png
![closerange]


####2. Show some examples of test images to demonstrate how your pipeline is working.  What did you do to optimize the performance of your classifier?

Ultimately I searched on two scales using YCrCb 3-channel HOG features plus spatially binned color and histograms of color in the feature vector, which provided a nice result.  Here are all three slider window searches applied onto one test frame.

[close]: ./writeup/close.png
[mid]: ./writeup/mid.png
[far]: ./writeup/far.png

![close]
![mid]
![far]

To combat false flag and minimize noise, all three results are added up and presented in a heatmap.

[heat]: ./writeup/heat.png
![heat]

With the threshold setting set to 4, false flagged result is eliminated and the position of the cars can be identified.

[label]: ./writeup/label.png
![label]

 
---

### Video Implementation

####1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (somewhat wobbly or unstable bounding boxes are ok as long as you are identifying the vehicles most of the time with minimal false positives.)
Here's a link to my video result: [Project\_video_output.mp4](./project_video_output.mp4)


####2. Describe how (and identify where in your code) you implemented some kind of filter for false positives and some method for combining overlapping bounding boxes.

I recorded the positions of positive detections in each frame of the video.  From the positive detections I created a heatmap and then thresholded that map to identify vehicle positions.  I then used `scipy.ndimage.measurements.label()` to identify individual blobs in the heatmap.  I then assumed each blob corresponded to a vehicle.  I constructed bounding boxes to cover the area of each blob detected.  

Here's an example result showing the heatmap from a series of frames of video, the result of `scipy.ndimage.measurements.label()` and the bounding boxes then overlaid on the last frame of video:

### Here are six frames and their corresponding heatmaps:

[heats]: ./writeup/heats.png
![heats]

### Here is the output of `scipy.ndimage.measurements.label()` on the integrated heatmap from all six frames:
[result]: ./writeup/result.png
![result]




---

###Discussion

####1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

The result is still not smooth enough. It should be corrected by higher overlap in window search to have more accuracy position of the cars. Also interframe smoothing can also be implemented to smooth out the result and reduce chances of false positive. Most of the false positve happens on the side of the road. With the previous lane detection, we can use the infomation to detect vehicles only on the road. This could prevent false positive from happening on situration like a car appearing on a roadside billboard. 
Larger training set can provide better classification results across different coditions and motorcycles. 

---

