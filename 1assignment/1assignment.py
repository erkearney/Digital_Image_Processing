# Eric Kearney -- Aug. 26 2018 -- Digital Image Processing, Assignment 1
# I want to try using OpenCV for this class. If it gets too difficult I will switch to MatLab
# The OpenCV tutorials were invaluable to me for this assignment:
# https://docs.opencv.org/3.0-beta/doc/py_tutorials/py_gui/py_image_display/py_image_display.html#py-display-image

# Images will display one at a time, press any key to show the next image, 
# once the final image is displayed, press any key to close all windows.

import cv2 as cv
import numpy as np

# Load the image
workingDirectory = './'        # The directory where you put the original 'Portrait.jpg' image.
saveDirectory = './'            # The directory where you want to save the files this program creates
img = cv.imread(workingDirectory + 'Portrait.jpg', 1)

# This image size is actually 2584x3446, so we'll need to resize it
# Manivannan Murugavel's Medium post helped me: 
# https://medium.com/@manivannan_data/resize-image-using-opencv-python-d2cdbbc480f0
W = 512
height, width, depth = img.shape
imgScale = W/width
newX, newY = img.shape[1]*imgScale, img.shape[0]*imgScale
img = cv.resize(img, (int(newX), int(newY)))

# Display the image
cv.imshow('original', img)
# Save the resized image and close the display after any keyboard press
cv.waitKey(0)
cv.imwrite(saveDirectory + 'PortraitResized.jpg', img)

# Convert the image to grayscale
img = cv.imread(workingDirectory + 'PortraitResized.jpg', 0)
cv.imshow('gray', img)
cv.waitKey(0)
cv.imwrite(saveDirectory + 'ProtraitGrayScale.jpg', img)

# Display regions of interest

# Box, Abid Rahman K's stackoverflow answer helped me here:
# https://stackoverflow.com/questions/11492214/opencv-via-python-is-there-a-fast-way-to-zero-pixels-outside-a-set-of-rectangle
boxWidth = 160
boxHeight = 200
boxR = 170         # The starting row of the box
boxC = 65          # The starting column of the box

# First, create a blank image of the same size
boxImg = np.zeros(img.shape, np.uint8)
# Then, copy everything within the box from the orginial image to the new image
boxImg[boxC:boxC+boxHeight, boxR:boxR+boxWidth] = img[boxC:boxC+boxHeight, boxR:boxR+boxWidth]
cv.imshow('box', boxImg)
cv.waitKey(0)
cv.imwrite(saveDirectory + 'PortraitBox.jpg', boxImg)

# Circle
center = (250, 160)
radius = 100
cv.circle(img, center, radius, (0,0,255), 1)
# Loop over every pixel in the image
# -- Yikes, I'm sure there's a more effecient way to do this!
for r in range(img.shape[0]):
    for c in range(img.shape[1]):
        # Use the distance formula to determine if the this pixel is 'in' the circle
        distance = np.sqrt((center[0] - c)**2 + (center[1] - r)**2)
        if distance > radius:
            img[r, c] = 0

cv.imshow('circle', img)
cv.waitKey(0)
cv.imwrite(saveDirectory + 'PortraitCircle.jpg', img)
cv.destroyAllWindows()
