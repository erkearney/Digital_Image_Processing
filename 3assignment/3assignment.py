import cv2 as cv
import numpy as np

img = cv.imread('face_dark.bmp')
X1 = (img[:][:][1]+img[:][:][2]+img[:][:][3])/3
cv.imshow('original', img)



cv.waitKey(0)
cv.destroyAllWindows()
