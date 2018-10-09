"""
Adrian Rosebrock's article "Skin Detection: A Step-by-Step Example
using Python and OpenCV helped me immensely with this project.

https://www.pyimagesearch.com/2014/08/18/skin-detection-step-step-example-using-python-opencv/

This project is, in part, an implementation of the technique(s) 
discussed in "A survey of skin-color modeling and detetction methods",
by P. Kakumanu, S. Makrogiannis, and N. Bourbakis, published in
"Pattern Recognition", volume #40, issue #3, pages 1106-1122 (2007).

https://www.sciencedirect.com/science/article/pii/S0031320306002767
"""

import cv2
import numpy as np
import argparse
from pathlib import Path
from matplotlib import pyplot as plt

default_image = "./face_dark.bmp"
parser = argparse.ArgumentParser()
parser.add_argument("-i", "--image", help="path to the image")
args = parser.parse_args()

def validate_image():
    if args.image:
        if Path(args.image).is_file():
            img = cv2.imread(args.image)
        else:
            print("File {} not found".format(args.image))
            exit()
    else:
        if Path(default_image).is_file():
            img = cv2.imread(default_image)
        else:
            print("No image supplied, and the default image, {} " \
                    "was not found".format(default_image))
            exit()
    return img

def create_mask(image):
    # TODO, add histogram equalization or something
    # For now, just using hard coded values
    lower = np.array([0, 48, 80], dtype = "uint8")
    upper = np.array([20, 255, 222], dtype = "uint8")
    # Convert the image to HSV
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    # Calculuate the histogram of the luminance of this image
    hist = cv2.calcHist([hsv], [2], None, [256], [0,256])
    plt.plot(hist)
    plt.show()
    # filter out values outside of our hardcoded range
    filtered = cv2.inRange(hsv, lower, upper)
    # perform a gaussian blur on the mask to get rid of noise
    gaus = cv2.GaussianBlur(filtered, (3, 3), 0)
    return gaus

def detect_skin(image):
    img = image.copy()
    mask = create_mask(img)
    for r in range(0, img.shape[0]):
        for c in range(0, img.shape[1]):
            if mask[r][c] == 0:
                img[r][c] = 0

    return img

def main():
    img = validate_image()
    mask = create_mask(img)
    skin = detect_skin(img)

    cv2.imshow('original', img)
    cv2.imshow('mask', mask)
    cv2.imshow('skin', skin)


    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
