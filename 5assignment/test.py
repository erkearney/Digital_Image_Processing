import cv2
import numpy as np

# load the image
img = cv2.imread("portrait.jpg", 1)
# This image size is actually 2584x3446, so we'll need to resize it
W = 512
height, width, depth = img.shape
imgScale = W/width
newX, newY = img.shape[1]*imgScale, img.shape[0]*imgScale
image = cv2.resize(img, (int(newX), int(newY)))

hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
lower = np.array([0, 48, 80], dtype = 'uint8')
upper = np.array([20, 255, 255], dtype = 'uint8')
mask = cv2.inRange(hsv, lower, upper)

# find the colors within the specified boundaries and apply
# the mask
output = cv2.bitwise_and(image, image, mask=mask)

ret,thresh = cv2.threshold(mask, 40, 255, 0)
im2,contours,hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

if len(contours) != 0:
    #find the biggest area
    c = max(contours, key = cv2.contourArea)

    x,y,w,h = cv2.boundingRect(c)

for r in range(0, output.shape[0]):
    for c in range(0, output.shape[1]):
        if(c < x or c > x+w):
            output[r, c] = 0
        elif(r < y or r > y + h):
            output[r, c] = 0

# show the images
cv2.imshow("Result", np.hstack([image, output]))

cv2.waitKey(0)
