import cv2 as cv
import numpy as np

img = cv.imread('example_03.jpeg', 0)
# Normalize the input image
img = np.divide(img, 255)

noise = 0.10 * np.random.rand(img.shape[0], img.shape[1]) - 0.05

noisy_image = np.add(img, noise)

cv.imshow('original', img)
cv.imshow('noise', noise)
cv.imshow('noisy_image', noisy_image)
cv.waitKey(0)
cv.destroyAllWindows()
