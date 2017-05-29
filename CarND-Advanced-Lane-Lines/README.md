## Advanced Lane Finding
[![Udacity - Self-Driving Car NanoDegree](https://s3.amazonaws.com/udacity-sdc/github/shield-carnd.svg)](http://www.udacity.com/drive)


In this project, your goal is to write a software pipeline to identify the lane boundaries in a video, but the main output or product we want you to create is a detailed writeup of the project.  Check out the [writeup template](https://github.com/udacity/CarND-Advanced-Lane-Lines/blob/master/writeup_template.md) for this project and use it as a starting point for creating your own writeup.  

---

The goals / steps of this project are the following:

* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* Apply a distortion correction to raw images.
* Use color transforms, gradients, etc., to create a thresholded binary image.
* Apply a perspective transform to rectify binary image ("birds-eye view").
* Detect lane pixels and fit to find the lane boundary.
* Determine the curvature of the lane and vehicle position with respect to center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

The images for camera calibration are stored in the folder called `camera_cal`.  The images in `test_images` are for testing your pipeline on single frames.  If you want to extract more test images from the videos, you can simply use an image writing method like `cv2.imwrite()`, i.e., you can read the video in frame by frame as usual, and for frames you want to save for later you can write to an image file.  

To help the reviewer examine your work, please save examples of the output from each stage of your pipeline in the folder called `ouput_images`, and include a description in your writeup for the project of what each image shows.    The video called `project_video.mp4` is the video your pipeline should work well on.  

The `challenge_video.mp4` video is an extra (and optional) challenge for you if you want to test your pipeline under somewhat trickier conditions.  The `harder_challenge.mp4` video is another optional challenge and is brutal!

If you're feeling ambitious (again, totally optional though), don't stop there!  We encourage you to go out and take video of your own, calibrate your camera and show us how you would implement this project from scratch!

---

**Advanced Lane Finding Project**

The goals / steps of this project are the following:

* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* Apply a distortion correction to raw images.
* Use color transforms, gradients, etc., to create a thresholded binary image.
* Apply a perspective transform to rectify binary image ("birds-eye view").
* Detect lane pixels and fit to find the lane boundary.
* Determine the curvature of the lane and vehicle position with respect to center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

[//]: # (Image References)


[video1]: ./project_video.mp4 "Video"
[calibration]: ./writeup/calibration.png

## [Rubric](https://review.udacity.com/#!/rubrics/571/view) Points

### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---

### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one.  You can submit your writeup as markdown or pdf.  [Here](https://github.com/udacity/CarND-Advanced-Lane-Lines/blob/master/writeup_template.md) is a template writeup for this project you can use as a guide and a starting point.  



### Camera Calibration

#### 1. Briefly state how you computed the camera matrix and distortion coefficients. Provide an example of a distortion corrected calibration image.

The code for this step is contained in the first code cell of the IPython notebook located in the file called `Camera Calibration.ipynb`).

I start by preparing "object points", which will be the (x, y, z) coordinates of the chessboard corners in the world. Here I am assuming the chessboard is fixed on the (x, y) plane at z=0, such that the object points are the same for each calibration image. 

This is done using cv2 library `cv2.findChessboardCorners` by running it through a folder of images named `camera_cal`. At the end of the process I end up with calibration data stored as `objpoints` and `imgpoints` which store the 3D points in real world space and 2D points in image plane respectively. These will then used later to correct each video frame to correct the optical distortions. 

Image below shows result from using code in `Camera Correction Test.ipynb` where one of the checkerboard images was undistorted using the function `cal_undistort` in code cell 2. As you can see the checker pattern has been straighten out and all the curve edges have been corrected.'

The camera correction data was stored using pickle in file `camera.p`.

![calibration]



### Pipeline (single images)

[all_edges]: ./writeup/all_edges.png
[all_fit]: ./writeup/all_fit.png
[all_load]: ./writeup/all_load.png
[all_top1]: ./writeup/all_top1.png
[all_top2]: ./writeup/all_top2.png
[all_undistorted]: ./writeup/all_undistorted.png
[calibration]: ./writeup/calibration.png
[single_edges]: ./writeup/single_edges.png
[single_complete]: ./writeup/single_complete.png
[single_fit]: ./writeup/single_fit.png
[single_load]: ./writeup/single_load.png
[single_top2]: ./writeup/single_top2.png
[single_undistorted]: ./writeup/single_undistorted.png
[radius]: ./writeup/radius.png


#### 1. Provide an example of a distortion-corrected image.

To demonstrate this step, I will describe how I apply the distortion correction to one of the test images like this one:
![single_load]
Along with other seven test images located in `test_images` folder.
![all_load]


All frames are distortion corrected using methods shown in `Camera Correction Test.ipynb`. Here are the results.
![single_undistorted]
![all_undistorted]


#### 2. Describe how (and identify where in your code) you used color transforms, gradients or other methods to create a thresholded binary image.  Provide an example of a binary image result.

I used a combination of color and gradient thresholds to generate a binary image (thresholding steps in function `colgrad()` in `P4.ipynb`).  Here's an example of my output applied to all test images.

![all_edges]

#### 3. Describe how (and identify where in your code) you performed a perspective transform and provide an example of a transformed image.

The code for my perspective transform includes a function called `perspectiveCorrect(img, src, dst)`, which appears in code cell 6 in the file `P4.ipynb`.  The `perspectiveCorrect` function takes as inputs an image (`img`), as well as source (`src`) and destination (`dst`) points.  I chose the hardcode the source and destination points in the following manner:

```python
src = np.float32([[622,430],[660,430],[1145,720],[175,720]])
dst = np.float32([[280,-1000],[1000,-1000],[1000,720],[280,720]])
```

This resulted in the following source and destination points:

| Source        | Destination   | 
|:-------------:|:-------------:| 
| 622, 430      | 280, -1000    | 
| 660, 430      | 1000, -1000   |
| 1145, 720     | 1000, 720     |
| 175, 720      | 280, 720      |

I verified that my perspective transform was working as expected by drawing the `src` and `dst` points onto a test image and its warped counterpart to verify that the lines appear parallel in the warped image.

![all_top1]

After verifying the color images. The same prespective transformation was applied to the results from edge detection as follows:

![all_top2]

After the transformation, the results appeared to be excluded most of the extra features such as side or the road and trees. Therefore no extra masking was needed for lane detection.


#### 4. Describe how (and identify where in your code) you identified lane-line pixels and fit their positions with a polynomial?

A technique called sliding window was used in function `slidingWindows()` in code cell 8. This is the first step to identify the left and right lanes of the road before any curve fitting. This method uses an iterative process starting from the bottom of the frame. Once a line is detected based on a contrast spike in the image, the search moves upwards by shifting the searching window (shown in green boxes).  If a lane is present, the algorithm moves up another step until the entire frame was scanned. This process is time consumer therefore it is only done on image which no previous line detected. Once a lane is detected, the process repeats only on close proximity away from the existing curve fit result. 

The lane was detected as shown in red and blue. 

Function `lineFit` in code cell 9 fits a curve based on the sliding window result using a second order polynomial curve fit with numPy function `np.polyfit`. The curve fit results are drawn as yellow lines shown below.

![single_top2]
![single_fit]

#### 5. Describe how (and identify where in your code) you calculated the radius of curvature of the lane and the position of the vehicle with respect to center.

The radius of curvature of the lanes were calculated using resource: http://www.intmath.com/applications-differentiation/8-radius-curvature.php

![radius]

The method was implemented in function named `findRadius` which converts the algeberic polyfit results into estimated curvature of the road. 


#### 6. Provide an example image of your result plotted back down onto the road such that the lane area is identified clearly.

The curve fit lane detection results were compared and reprojected to the camera frame using `greenBox` in `P4.ipynb`. 

![all_fit]

Finally the calculated curvature and offset distance were included into the final output.
![single_complete]



### Pipeline (video)

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (wobbly lines are ok but no catastrophic failures that would cause the car to drive off the road!).

Here's a link to my video result: [Project\_video_output.mp4](./project_video_output.mp4)

---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

Currently the pipeline is still highly dependent on the driving condition of where the video was captured. More generalized work has to be done on the edge detection to ensure robustness across different road surface and brightness of the video. 

Also in the future, the pipeline should include a check to compare the curvature of the road with previous frames. When a sharp jump is detected, it could be caused by false/bad lane detection or other noise in the image has affect the result. If that was found, the result should be rejected and perhaps include some weighting algorithm to smooth out the results by averaging with previous frames.




