""" Part 1: Apply the averaging, Sobel, Laplacian,
and median filters to an image, show the original
image and the filtered images.

Once again, the openCV documentation was of
enormous help to me.

OpenCV has built in cv2.blur() (averaging), cv2.Laplacian(), and cv2.medianBlur() functions, 
but I wanted to try implementing them myself, using only the filter2D() function:

https://docs.opencv.org/2.4/modules/imgproc/doc/filtering.html
"""

import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt

workingDirectory = './'
saveDirectory = './'

# Read and resize the image
img = cv.imread(workingDirectory + 'duke.jpg', 1)
# My cat, "Duke" will be our model
W = 512
height, width, depth = img.shape
imgScale = W/width
newX, newY = img.shape[1]*imgScale, img.shape[0]*imgScale
img = cv.resize(img, (int(newX), int(newY)))
cv.imwrite(saveDirectory + 'duke_resized.jpg', img)
# create a grayscale version
img_gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

# averaging filter
# https://docs.opencv.org/3.1.0/d4/d13/tutorial_py_filtering.html
avg_kernel = np.ones((5,5), np.float32)/25
avg = cv.filter2D(img, -1, avg_kernel)

# Sobel filter 
# Apply a Gaussian blur to the grayscale version to reduce noise
gaus = cv.GaussianBlur(img_gray,(5, 5),0)

# https://docs.opencv.org/2.4/doc/tutorials/imgproc/imgtrans/sobel_derivatives/sobel_derivatives.html
sobel_kernel_x = np.matrix('-1, 0, 1; -2, 0, 2; -1, 0, 1')
sobel_kernel_y = np.matrix('-1, -2, -1; 0, 0, 0; 1, 2, 1')
sobelx = cv.filter2D(gaus, -1, sobel_kernel_x)
sobely = cv.filter2D(gaus, -1, sobel_kernel_y)
sobel = np.add((sobelx * 0.5), (sobely * 0.5))

# Laplacian filter
laplacian_kernel = np.matrix('0, 1, 0; 1, -4, 1; 0, 1, 0')
laplacian = cv.filter2D(img, -1, laplacian_kernel)

# Median filter
"""
Well, I tried doing this the hard way, it works, but takes a couple minutes to complete,
maybe I'll try fixing it later, but for now, I'll just use the built-in OpenCV function.
med = img_gray
# Loop through the image, the lazy way, ignoring edges
for r in range(2, img.shape[0]-1):
    for c in range(2, img.shape[1]-1):
        median_filter = np.matrix([[img_gray[r-1][c-1], img_gray[r-1][c], img_gray[r-1][c+1]], 
                                  [img_gray[r][c-1], img_gray[r][c], img_gray[r][c+1]],
                                  [img_gray[r+1][c-1], img_gray[r+1][c], img_gray[r+1][c+1]]])

        # Numpy's documentation says that calling np.median() on a matrix is SUPPOSED to compute
        # the median along a flattened version of the matrix, and return a single value, but. . . 
        # it isn't actually doing that, so my workaround is to just take the median along both 
        # axes: https://docs.scipy.org/doc/numpy/reference/generated/numpy.median.html#numpy.median
        median = np.median(np.median(median_filter, axis=0), axis=1)
        med[r][c] = median
"""
med = cv.medianBlur(img,5)

# show the images
titles = ['Original', 'Averaging', 'Sobel', 'Laplacian', 'Median'] 
images = [img, avg, sobel, laplacian, med]

plt.figure(num='Part 1')
for i in range(len(images)):
    plt.subplot(2,3,i+1),plt.imshow(images[i], cmap = 'gray')
    plt.title(titles[i])
    plt.xticks([]), plt.yticks([])

# plt.show()

# Part 2: Find the boundary of the pupil
# Read the image in grayscale
eye = cv.imread('iris.bmp', 0)

# Apply a laplacian filter
# eye = cv.filter2D(eye, -1, laplacian_kernel)

# Create the vertical and horizontal kernels
sobel_x_kernel = np.matrix([[-1, 0, 1],
                            [-2, 0, 2],
                            [-1, 0, 1]])

sobel_y_kernel = np.matrix([[-1, -2, -1],
                            [0, 0, 0],
                            [1, 2, 1]])

sobel_x = cv.filter2D(eye, -1, sobel_x_kernel)
sobel_y = cv.filter2D(eye, -1, sobel_y_kernel)
eye_sobel = np.add(sobel_x, sobel_y)

# Create a custom kernel, since we know 15 <= pupil_radius <= 25, we'll use a ring
# kernel, with the smaller circle having raidus 15 and the larger circle having
# radius 25.
# start by creating an empty matrix of size (big_radius*2)+1 x (big_radius*2)+1
small_radius = 15
big_radius = 25
eye_kernel = np.zeros(((big_radius*2)+1, (big_radius*2)+1))
center = (big_radius, big_radius)
# Set each pixel to a 1 if its distance from the center is greater than the small radius
# AND less than the big radius
for r in range(0, eye_kernel.shape[0]):
    for c in range(0, eye_kernel.shape[1]):
        # Use the distance formula to determine if this pixel should be part of our ring
        distance = np.sqrt((center[0] - c)**2 + (center[1] - r)**2)
        if distance > small_radius and distance < big_radius:
            eye_kernel[r, c] = 1

pupil_center = cv.filter2D(eye_sobel, -1, eye_kernel)

cv.imshow('eye', eye)
cv.imshow('sobelx', sobel_x)
cv.imshow('sobely', sobel_y)
cv.imshow('sobel', eye_sobel)
cv.imshow('eye_kernel', eye_kernel)
cv.imshow('pupil_center', pupil_center)
cv.waitKey(0)
cv.destroyAllWindows()
