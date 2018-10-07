# Adrian Rosebrock's article on pyimagesearch helped me immensely here:
# https://www.pyimagesearch.com/2015/10/05/opencv-gamma-correction/
import cv2 as cv
import numpy as np
import argparse

ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True,
    help="path to input image")
args = vars(ap.parse_args())

original = cv.imread(args["image"])

def adjust_gamma(image, gamma=1.0):
    # build a lookup table (i.e. dictionary) 
    # mapping the pixel valus [0, 255] to their 
    # adjusted gamma values
    invGamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** invGamma) * 255 
	for i in np.arange(0, 256)]).astype("uint8")

    # apply gamma correction using the lookup table
    return cv.LUT(image, table)

# loop over various values of gamma
for gamma in np.arange(0.0, 3.5, 0.5):
# ignore when gamma is 1
    if gamma == 1:
        continue

    # apply gamma correction and show the images
    gamma = gamma if gamma > 0 else 0.1
    adjusted = adjust_gamma(original, gamma=gamma)
    cv.putText(adjusted, "g={}".format(gamma), (10, 30),
        cv.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 3)
    cv.imshow("Images", np.hstack([original, adjusted]))
    cv.waitKey(0)
