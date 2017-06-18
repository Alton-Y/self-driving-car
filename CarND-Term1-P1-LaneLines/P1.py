#importing some useful packages
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import cv2
import math
from sklearn import linear_model

def grayscale(img):
    """Applies the Grayscale transform
    This will return an image with only one color channel
    but NOTE: to see the returned image as grayscale
    (assuming your grayscaled image is called 'gray')
    you should call plt.imshow(gray, cmap='gray')"""
    
    b,g,r = cv2.split(img)
    image_gs = cv2.cvtColor(b,cv2.COLOR_GRAY2RGB)
    return image_gs
    # Or use BGR2GRAY if you read an image with cv2.imread()
#     return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
def canny(img, low_threshold, high_threshold):
    """Applies the Canny transform"""
    return cv2.Canny(img, low_threshold, high_threshold)

def gaussian_blur(img, kernel_size):
    """Applies a Gaussian Noise kernel"""
    return cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)

def region_of_interest(img, vertices):
    """
    Applies an image mask.
    
    Only keeps the region of the image defined by the polygon
    formed from `vertices`. The rest of the image is set to black.
    """
    #defining a blank mask to start with
    mask = np.zeros_like(img)   
    
    #defining a 3 channel or 1 channel color to fill the mask with depending on the input image
    if len(img.shape) > 2:
        channel_count = img.shape[2]  # i.e. 3 or 4 depending on your image
        ignore_mask_color = (255,) * channel_count
    else:
        ignore_mask_color = 255
        
    #filling pixels inside the polygon defined by "vertices" with the fill color    
    cv2.fillPoly(mask, vertices, ignore_mask_color)
    
    #returning the image only where mask pixels are nonzero
    masked_image = cv2.bitwise_and(img, mask)
    return masked_image

# TODO: detect the brightness of the road
def road_brightness(image,X,Y,width,height):
    image_crop = image[int(Y):int(Y+height),int(X):int(X+width)]
    brightness = image_crop.mean()
    return brightness
    



def draw_lines(img, lines, color=[255, 0, 0], thickness=2):
    """
    NOTE: this is the function you might want to use as a starting point once you want to 
    average/extrapolate the line segments you detect to map out the full
    extent of the lane (going from the result shown in raw-lines-example.mp4
    to that shown in P1_example.mp4).  
    
    Think about things like separating line segments by their 
    slope ((y2-y1)/(x2-x1)) to decide which segments are part of the left
    line vs. the right line.  Then, you can average the position of each of 
    the lines and extrapolate to the top and bottom of the lane.
    
    This function draws `lines` with `color` and `thickness`.    
    Lines are drawn on the image inplace (mutates the image).
    If you want to make the lines semi-transparent, think about combining
    this function with the weighted_img() function below
    """
    for line in lines:
        for x1,y1,x2,y2 in line:
            cv2.line(img, (x1, y1), (x2, y2), color, thickness)

# def hough_lines(img, rho, theta, threshold, min_line_len, max_line_gap):
#     """
#     `img` should be the output of a Canny transform.
#         
#     Returns an image with hough lines drawn.
#     """
#     lines = cv2.HoughLinesP(img, rho, theta, threshold, np.array([]), minLineLength=min_line_len, maxLineGap=max_line_gap)
#     line_img = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)
#     draw_lines(line_img, lines)
#     return line_img

# Python 3 has support for cool math symbols.

def weighted_img(img, initial_img, α=0.8, β=1., λ=0.):
    """
    `img` is the output of the hough_lines(), An image with lines drawn on it.
    Should be a blank image (all black) with lines drawn on it.
    
    `initial_img` should be the image before any processing.
    
    The result image is computed as follows:
    
    initial_img * α + img * β + λ
    NOTE: initial_img and img must be the same shape!
    """
    return cv2.addWeighted(initial_img, α, img, β, λ)
    
    
    
import os
os.listdir("test_images/")




    
    
def process_image(image):
    
    brightness = road_brightness(image,image.shape[1]*0.45,image.shape[0]*0.7,image.shape[1]*0.15,image.shape[0]*0.2)
    
    # img_output = frame + 95 - int(brightness)
    # print(brightness)
    # print(road_brightness(img_output,frame.shape[1]*0.45,frame.shape[0]*0.7,frame.shape[1]*0.15,frame.shape[0]*0.2))
    
    
    # Apply grayscale
    image_gs = grayscale(image) 
    # print(image_gs)
    # image_gs = image_gs + 95 - int(brightness)
    
    # print(road_brightness(image_gs,frame.shape[1]*0.45,frame.shape[0]*0.7,frame.shape[1]*0.15,frame.shape[0]*0.2))
    
    # Apply Gaussian Noise kernel
    kernel_size = 7
    image_blur = gaussian_blur(image_gs, kernel_size)
    
    # Apply Canny transform
    low_threshold = 50
    high_threshold = 150
    image_canny = canny(image_blur, low_threshold, high_threshold) 
    
    # Mask
    image_mask = np.zeros_like(image_canny)   
    ignore_mask_color = 255
    imshape = image.shape
    vertices = np.array([[(imshape[1]*0.12,imshape[0]*0.98),(imshape[1]*0.40, imshape[0]*0.63), (imshape[1]*0.60, imshape[0]*0.63), (imshape[1]*0.95,imshape[0]*0.98)]], dtype=np.int32);
    # vertices = mask_vertices

    cv2.fillPoly(image_mask, vertices, ignore_mask_color)
    image_masked = cv2.bitwise_and(image_canny, image_mask)

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


# Tuning
from moviepy.editor import VideoFileClip  

# solidWhiteRight
# solidYellowLeft
# challenge
clip2 = VideoFileClip("challenge.mp4")
frame = clip2.get_frame(96/25)
process_image(frame)    

# TODO: Build your pipeline that will draw lane lines on the test_images
## then save them to the test_images directory.

# # # # creating a for loop to read all images within "test_images/" directory
# # # im_dir = "test_images/" # define image folder path
# # # im_list = os.listdir(im_dir)
# # # print(im_list)
# # # im_list = im_list[5:6]
# # # i = 1;
# # # for im_name in im_list: # for loop
# # #     image = mpimg.imread(im_dir+im_name) # read image from (im_dir + im_name)
# # #     
# # #     
# # #     result = process_image(image)
# # #     
# # #     print('Image Path:', im_dir+im_name)
# # #     # fig = plt.figure(i,figsize=(10,5))
# # #     # plt.imshow(result)
# # #     i = i+1
# # #     
        
    
    # fig = plt.figure(figsize=(10,10))
    # plt.subplot(321)
    # plt.title('Raw Image Input')
    # plt.imshow(image)

    #   plt.subplot(322)
    # plt.title('Grayscale Filter')
    # plt.imshow(image_gs, cmap='gray')
    # 
    # plt.subplot(323)
    # plt.title('Canny Edge Detection')
    # plt.imshow(image_canny,cmap='Greys_r')
    # 
    # plt.subplot(324)
    # plt.imshow(image_masked,cmap='Greys_r')
    # 
    # plt.subplot(325)
    # plt.imshow(image_hough,cmap='Greys_r')
    # 
    # plt.subplot(326)
    # plt.imshow(result)
    # 
    # plt.show()
    

# # # # # ## Test on Videos
# # # # # # Import everything needed to edit/save/watch video clips
from moviepy.editor import VideoFileClip
# # from IPython.display import HTML
# mask_normal = np.array([[[115,540],[432,340],[508,340],[912,540]]])
white_output = 'white.mp4'
clip1 = VideoFileClip("solidWhiteRight.mp4")
white_clip = clip1.fl_image(process_image) #NOTE: this function expects color images!!
# %time white_clip.write_videofile(white_output, audio=False)
white_clip.write_videofile(white_output, audio=False)
    
#
# mask_normal = np.array([[[115,540],[432,340],[508,340],[912,540]]])
yellow_output = 'yellow.mp4'
clip2 = VideoFileClip('solidYellowLeft.mp4')
yellow_clip = clip2.fl_image(process_image) 
yellow_clip.write_videofile(yellow_output, audio=False)
    
    
#
# mask_extra = np.array([[[153,684],[576,453],[678,453],[1216,633]]])
challenge_output = 'extra.mp4'
clip2 = VideoFileClip('challenge.mp4')
challenge_clip = clip2.fl_image(process_image)
challenge_clip.write_videofile(challenge_output, audio=False)
#     
##


# img_yuv = cv2.cvtColor(frame, cv2.COLOR_BGR2YUV)
# img_yuv[:,:,0] = cv2.equalizeHist(img_yuv[:,:,0])
# img_output = cv2.cvtColor(img_yuv, cv2.COLOR_YUV2BGR)
# frame = mpimg.imread(im_dir+'solidWhiteCurve.jpg')

# brightness = road_brightness(frame,frame.shape[1]*0.45,frame.shape[0]*0.7,frame.shape[1]*0.15,frame.shape[0]*0.2)
# print(int(brightness))
# img_output = frame + 95 - int(brightness)
# # print(brightness)
# print(road_brightness(img_output,frame.shape[1]*0.45,frame.shape[0]*0.7,frame.shape[1]*0.15,frame.shape[0]*0.2))

# # b,g,r = cv2.split(frame)
# # singlechannel = cv2.cvtColor(b,cv2.COLOR_GRAY2RGB)
# # 
# # fig = plt.figure(1,figsize=(10,5))  
# # plt.imshow(singlechannel) 
# # #
# # # # 
# # fig = plt.figure(2,figsize=(10,5))   
# # plt.imshow(grayscale(singlechannel),cmap='gray')
# # # #
# # # # 
# # fig = plt.figure(3,figsize=(10,5))   
# # result = process_image(singlechannel)
# # plt.imshow(result) 
# # # # print(brightness)

    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
