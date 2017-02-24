#**Finding Lane Lines on the Road** 

[//]: # (Image References)

[image1]: ./markdown_images/figure_1.png "Grayscale"
[image2]: ./markdown_images/figure_2.png "Canny"
[image3]: ./markdown_images/figure_3.png "Hough"
[image4]: ./markdown_images/figure_4.png "Linear"
[image5]: ./markdown_images/figure_5.png "Result"
---

###1. Edge Detection Pipeline

* At first, the image is converted to grayscale. Specifically, the blue channel of the RGB input image is extracted. The reason behind is to enhance the contrast of the yellow line against the road surface. As shown in the figure below, the blue channel grayscale provides the best contrast for edge detection as both the yellow and white lanes stand out. 
![RGB Channel Comparison][image1]

* Gaussian blur is applied to the grayscale image to remove noise in the image. This lower the sensitivity of the edge detection and advoid the algrithum to pick up unnessesary noise. Image is then masked as shown as the white region which indicate the road surface for lane detection to operate. This advoid the script from processing details from the sky or on the side of the road. 
![Canny Edge and Mask][image2]

* The technique  of Hough transform is applied to the image. This method used in computer vision can identify lines by voting procedure. By tuning the parameters to the correct settings, features  that resemble as road lanes are extracted as shown in red lines. 
![Hough Transformation][image3]

* As shown in previous image, unwanted lines such as the car's hood has been capture by the image analysis. This can be taken out by further adjusting the parameters. However this may cause over-fitting. To keep the lane detection algorithm as generalized for different situration as possible, over-fitting should be prevented. To remove the unwanted lines, a filter calculates the slope of each line and remove lines which are close to horizontal. In this case, the slope threshold was set to be 20 degrees. By calculating the slope of each line can also classify which line belongs to which lane. The lines which belong to the left lane will have positive slope while on the right should have negative slope.
To connect and extrpolate the lane, the detected lines should be processed. Once the lines have been classified into left and right lane, linear regression is applied to each lane to clean up the data. 
![Linear Fit][image4]

* A more robust RANSAC: RANdom SAmple Consensus algorithm from scikit-learn was used. This can detect and remove outliers from the data and produce cleaner results.
http://scikit-learn.org/stable/modules/linear_model.html#ransac-random-sample-consensus 
To further enhance the result, length of the detected lines can be used as weighted values in linear fit algorithm. This provides more information to the line fitting to weight more heavily on longer lines and less on shorter results which may contain more noise. 
Figure below shows one of the case where a more robust linear fitting algorithm helps producing better lane detection result.

* Finally, the detected lanes were plotted on top of the input image to create the final result.
![Result][image5]



###2. Identify potential shortcomings with your current pipeline

One potential shortcoming would be what would happen when the brightness of the road changes. As the time of day or road surface changes, the algorithm is not robust enough to adjust the parameters to adopt to different road surface brightness. This would lead to misjudging the lane position and cause errors. 

Also if other extra road markings appear within the lane (e.g. indicating freeway intersection or lane merge) or construction zone (e.g. temporary road markings), this current setup would possibly break apart and not able to detect the lanes.


###3. Suggest possible improvements to your pipeline

A possible improvement would be to analysis more footages to account for possible shortcomings. As a general rule of machine learning, more data leads to better model and predictions. More robust models would allow better lane detections. For example, mulitple footages aiming at different angles from the car can be combined and analysed at the same time. Time could possibly reduce blind spots and correct for lens distortions. 



