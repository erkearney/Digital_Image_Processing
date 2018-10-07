import cv2 as cv
import numpy as np

from matplotlib import pyplot as plt

img = cv.imread('./iris.bmp', -1)

sobelH = np.matrix([[-1, -2, -1], 
                    [0, 0, 0], 
                    [1, 2, 1]])

sobelV = np.matrix([[-1, 0, 1],
                    [-2, 0, 2],
                    [-1, 0, 1]])

img2 = cv.filter2D(img, -1, sobelH)
img3 = cv.filter2D(img, -1, sobelV)
img4 = np.power(np.divide(np.add(np.square(img2), np.square(img3)), 2), 1/2)

cv.imshow('sobelH', img2)
cv.imshow('sobelV', img3)
cv.imshow('sobel', img4)
cv.waitKey(0)
cv.destroyAllWindows()
