import cv2 as cv
import numpy as np

img = cv.imread('./example_03.jpeg')
gaus = cv.GaussianBlur(img,(5, 5),0)
double_gaus = cv.GaussianBlur(gaus,(5,5),0)
difference = np.subtract(gaus, double_gaus)
alpha = 2
sharpened = gaus + alpha * difference

cv.imshow('original', img)
cv.imshow('gaus', gaus)
cv.imshow('double_gaus', double_gaus)
cv.imshow('difference', difference)
cv.imshow('sharpened', sharpened)
cv.waitKey(0)
cv.destroyAllWindows()
