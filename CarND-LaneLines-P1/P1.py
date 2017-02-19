#importing some useful packages
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import cv2

import math

def grayscale(img):
    """Applies the Grayscale transform
    This will return an image with only one color channel
    but NOTE: to see the returned image as grayscale
    (assuming your grayscaled image is called 'gray')
    you should call plt.imshow(gray, cmap='gray')"""
    return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
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

def hough_lines(img, rho, theta, threshold, min_line_len, max_line_gap):
    """
    `img` should be the output of a Canny transform.
        
    Returns an image with hough lines drawn.
    """
    lines = cv2.HoughLinesP(img, rho, theta, threshold, np.array([]), minLineLength=min_line_len, maxLineGap=max_line_gap)
    line_img = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)
    draw_lines(line_img, lines)
    return line_img

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



# TODO: Build your pipeline that will draw lane lines on the test_images
# then save them to the test_images directory.


def line_detect(image):
    # Apply grayscale
    image_gs = grayscale(image) 
    
    # Apply Gaussian Noise kernel
    kernel_size = 5
    image_blur = gaussian_blur(image_gs, kernel_size)
    
    # Apply Canny transform
    low_threshold = 50
    high_threshold = 120
    image_canny = canny(image_blur, low_threshold, high_threshold) 
    
    # Mask
    image_mask = np.zeros_like(image_canny)   
    ignore_mask_color = 255   
    imshape = image.shape
    vertices = np.array([[(0,imshape[0]),(imshape[1]*0.45, imshape[0]*0.6), (imshape[1]*0.55, imshape[0]*0.6), (imshape[1],imshape[0])]], dtype=np.int32)
    cv2.fillPoly(image_mask, vertices, ignore_mask_color)
    image_masked = cv2.bitwise_and(image_canny, image_mask)

    # Hough Transform
    rho = 2
    theta = np.pi/180
    threshold = 30
    min_line_len = 100
    max_line_gap = 75
    image_hough = hough_lines(image_masked, rho, theta, threshold, min_line_len, max_line_gap)
    
    # Overlay Hough Transform Results
    result = weighted_img(image_hough, image)
    
    return result




# creating a for loop to read all images within "test_images/" directory
im_dir = "test_images/" # define image folder path
im_list = os.listdir(im_dir)
# im_list = im_list[1:2]
for im_name in im_list: # for loop
    image = mpimg.imread(im_dir+im_name) # read image from (im_dir + im_name)
    result = line_detect(image)

    
    
    print('Image Path:', im_dir+im_name)
    print('Dimesions:', image.shape)   
    fig = plt.figure(figsize=(15,8))
    plt.imshow(result)
    
        
    
#     fig = plt.figure(figsize=(20,18))
#     plt.subplot(321)
#     plt.title('Raw Image Input')
#     plt.imshow(image)

#     plt.subplot(322)
#     plt.title('Grayscale Filter')
#     plt.imshow(image_gs, cmap='gray')
    
#     plt.subplot(323)
#     plt.title('Canny Edge Detection')
#     plt.imshow(image_canny,cmap='Greys_r')
    
#     plt.subplot(324)
#     plt.imshow(image_masked,cmap='Greys_r')
    
#     plt.subplot(325)
#     plt.imshow(image_hough,cmap='Greys_r')
    
#     plt.subplot(326)
#     plt.imshow(result)
    
    plt.show()

    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
