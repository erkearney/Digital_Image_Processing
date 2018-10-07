import cv2 as cv
import numpy as np
import argparse

ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True,
    help="path to input image")
args = vars(ap.parse_args())

img = cv.imread(args["image"])

cv.imshow('Original', img)

def detect_edge(image):
    kernel = np.matrix([[0.6, 0, 0.6], [0, -2, 0], [0.4, 0, 0.4]])
    return cv.filter2D(image, -1, kernel)

img = detect_edge(img)

cv.imshow('filtered', img)
cv.waitKey(0)
cv.destroyAllWindows()
