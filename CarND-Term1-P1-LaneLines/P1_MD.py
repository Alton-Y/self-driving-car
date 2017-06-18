import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import cv2
import math
from sklearn import linear_model
from moviepy.editor import VideoFileClip  

# solidWhiteRight
# solidYellowLeft
# challenge
clip2 = VideoFileClip("challenge.mp4")
# frame = clip2.get_frame(30/25)
frame = clip2.get_frame(30/25)
image = frame 
    
b,g,r = cv2.split(image)
image_gs = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
image_gs_b = cv2.cvtColor(b,cv2.COLOR_GRAY2RGB)
image_gs_g = cv2.cvtColor(g,cv2.COLOR_GRAY2RGB)
image_gs_r = cv2.cvtColor(r,cv2.COLOR_GRAY2RGB)
    
fig = plt.figure(1,figsize=(12,8))  
plt.clf()
subplot(221)
plt.title("Grayscale")
plt.imshow(image_gs, cmap='gray')
subplot(222)
plt.title("Red Channel")
plt.imshow(image_gs_r)
subplot(223)
plt.title("Green Channel")
plt.imshow(image_gs_g)
subplot(224)
plt.title("Blue Channel")
plt.imshow(image_gs_b)

#
# Apply Gaussian Noise kernel
kernel_size = 7
image_blur = gaussian_blur(image_gs_b, kernel_size)
    
# Apply Canny transform
low_threshold = 50
high_threshold = 150
image_canny = canny(image_blur, low_threshold, high_threshold) 

# Mask
image_mask = np.zeros_like(image_canny)   
ignore_mask_color = 255
imshape = image.shape
vertices = np.array([[(imshape[1]*0.12,imshape[0]*0.98),(imshape[1]*0.40, imshape[0]*0.63), (imshape[1]*0.60, imshape[0]*0.63), (imshape[1]*0.95,imshape[0]*0.98)]], dtype=np.int32);
cv2.fillPoly(image_mask, vertices, ignore_mask_color)
image_masked = cv2.bitwise_and(image_canny, image_mask)

fig = plt.figure(2,figsize=(12,8))  
plt.clf()
subplot(211)
plt.title("Canny Edge")
plt.imshow(image_canny, cmap='gray')
subplot(212)
plt.title("Mask Region")
plt.imshow(image_mask, cmap='gray')


#
# Hough Transform
rho = 1
theta = np.pi/180
threshold = 30
min_line_len = 30
max_line_gap = 40

# image_hough = hough_lines(image_masked, rho, theta, threshold, min_line_len, max_line_gap)
img = image_masked #temp
lines = cv2.HoughLinesP(img, rho, theta, threshold, np.array([]), minLineLength=min_line_len, maxLineGap=max_line_gap)
# NOTE: format of lines:
# array([[X1, Y1, X2, Y2]],
#       [[X1, Y1, X2, Y2]], dtype=int32)

# Calculate slope of each hough line
# NOTE: No. of found hough lines = length of variable lines
# Slicing the X1,Y1,X2,Y2 coordinates from lines for ele.wise calc
X1 = lines[:,:,0]
Y1 = -lines[:,:,1];
X2 = lines[:,:,2];
Y2 = -lines[:,:,3];
# calculate slope 
m = rad2deg((Y2-Y1)/(X2-X1))
l = ((Y2-Y1)**2+(X2-X1)**2)**0.5

# NOTE: points shall be separted to left and right lane
# determined by the valeu of the slope

# NOTE: Remove hough line results with extreme slopes
# left_lane (+ve slope), right_lane (-ve slope)
slope_threshold = 25; # ignore hough lines which are +/- this value
left_lane = lines[m > slope_threshold]
right_lane = lines[m < -slope_threshold]


# rearrange left_lane and right_lane matrices
left_lane_X = hstack((left_lane[:,0],left_lane[:,2]))
left_lane_Y = hstack((left_lane[:,1],left_lane[:,3]))
right_lane_X = hstack((right_lane[:,0],right_lane[:,2]))
right_lane_Y = hstack((right_lane[:,1],right_lane[:,3]))


line_img = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)
draw_lines(line_img, lines, [255,0,0], 5)
image_hough = line_img # return
    
    
fig = plt.figure(3,figsize=(12,8))  
plt.clf()
subplot(211)
plt.title("Remaining Edges")
plt.imshow(image_masked, cmap='gray')
subplot(212)
plt.title("Mask Region")
plt.imshow(image_hough)



# NOTE: linear fit lanes
# Because ultimately we want to plot the x position of the lane by inputing y location. So we can draw the lines between certain vertical locations within the frame. 
# Therefore we have to invert the regression model by setting y-coordinates as input and x-coordinates as output
# apply .reshape(-1, 1) to funtion input to allow model fit


# Error Handling: if no lines were found, make them zeros
if left_lane_X.size == 0:
    left_lane_X_fit = array([0,0])
    left_lane_Y_fit = array([0,0])
    print('\nLeft Lane Not Found')
else:   
    # Hough line length for line fitting weight
    left_length = hstack((l[m>slope_threshold],l[m>slope_threshold]))
    left_lane_reg = linear_model.RANSACRegressor()
    left_lane_reg.fit(left_lane_Y.reshape(-1, 1), left_lane_X, left_length)
    # Feed test points back to linear regression models
    # aka find the X intercept
    left_lane_Y_fit = array([imshape[0]*0.63,imshape[0]])
    left_lane_X_fit = left_lane_reg.predict(left_lane_Y_fit.reshape(2,1))
    

    
if right_lane_X.size == 0:
    right_lane_X_fit = array([0,0])
    right_lane_Y_fit = array([0,0])
    print('\nRight Lane Not Found')
else:
    # Hough line length for line fitting weight
    right_length = hstack((l[m<-slope_threshold],l[m<-slope_threshold]))
    right_lane_reg = linear_model.RANSACRegressor()
    right_lane_reg.fit(right_lane_Y.reshape(-1, 1), right_lane_X, right_length)
    # Feed test points back to linear regression models
    # aka find the X intercept
    right_lane_Y_fit = array([imshape[0]*0.63,imshape[0]])
    right_lane_X_fit = right_lane_reg.predict(right_lane_Y_fit.reshape(2,1))


    # Draw lines
    fit_lines = array([[[left_lane_X_fit[0],left_lane_Y_fit[0],left_lane_X_fit[1],left_lane_Y_fit[1]],[right_lane_X_fit[0],right_lane_Y_fit[0],right_lane_X_fit[1],right_lane_Y_fit[1]]]], dtype=int32)
    draw_lines(image_hough, fit_lines, [0,255,0], 2)

    # fig = plt.figure(2)


  
    
    # Overlay Hough Transform Results
    # result = weighted_img(image_hough, image)
    
    # Results with fit lanes only
    image_fitline = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)
    draw_lines(image_fitline, fit_lines, [255,0,0], 10)
    result = weighted_img(image_fitline, image,1,1,0)
    
    # rgb_canny = cv2.cvtColor(image_masked,cv2.COLOR_GRAY2RGB)
    # result = weighted_img(rgb_canny,image_hough)
        
    # # # # # # TODO:
    # fig = plt.figure(1,figsize=(10,5))  
    # plt.clf()
    # subplot(221)
    # plt.imshow(result) 
    # subplot(222)
    # plt.imshow(image_gs,cmap='Greys_r')    
    # plt.imshow(image_masked,cmap='Greys_r')   
    # subplot(223)
    # plt.imshow(line_img) 
    # # plt.imshow(image_mask,cmap='Greys_r')   
    # subplot(224)
    #  # plt.imshow(fit_lines,cmap='Greys_r')   
    # plt.plot(left_lane_X_fit,-left_lane_Y_fit)
    # plt.plot(right_lane_X_fit,-right_lane_Y_fit)
    # plt.scatter(left_lane_X, -left_lane_Y,  s=left_length, cmap='viridis')
    # plt.scatter(right_lane_X, -right_lane_Y,  color='blue')
    # plt.xlim((0, image.shape[1]))
    # plt.ylim((-image.shape[0], 0))
    # plt.xticks()
    # plt.yticks()
    # plt.show()    
    # 
    return result



fig = plt.figure(4,figsize=(12,8))  
plt.clf()
subplot(211)
plt.title("Hough Transofrm Result")
plt.imshow(image_hough)

subplot(212)
plt.title("Linear Fit")
plt.scatter(left_lane_X, -left_lane_Y, color='red')
plt.scatter(right_lane_X, -right_lane_Y,  color='blue')
plt.plot(left_lane_X_fit,-left_lane_Y_fit)
plt.plot(right_lane_X_fit,-right_lane_Y_fit)
plt.xlim((0, image.shape[1]))
plt.ylim((-image.shape[0], 0))
plt.xticks()
plt.yticks()
plt.legend(['Linear Fit (Left)','Linear Fit (Right)','Detected Points (Left)','Detected Points (Right)'])
plt.show()  
















