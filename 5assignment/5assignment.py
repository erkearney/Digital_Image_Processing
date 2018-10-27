'''
Face detection, written by Eric Kearney on October 27th, 2018 This program 
detects faces by first performing a skin detection, then, selects the largest 
contour, which will often be the face.

I say often because it's obvious that this approach will fail if the person
in the image is dressed in such a way that more skin is exposed.
'''
import cv2
import numpy as np
import argparse
from pathlib import Path
from matplotlib import pyplot as plt

default_image = './portrait.jpg'
parser = argparse.ArgumentParser()
parser.add_argument('-i', '--image', help='path to the image')
args = parser.parse_args()

def validate_image():
    if args.image:
        if Path(args.image).is_file():
            img = cv2.imread(args.image)
        else:
            print('File {} not found'.format(args.image))
            exit()
    else:
        if Path(default_image).is_file():
            img = cv2.imread(default_image)
            # This image size is actually 2584x3446, so we'll need to resize it
            W = 512
            height, width, depth = img.shape
            imgScale = W/width
            newX, newY = img.shape[1]*imgScale, img.shape[0]*imgScale
            img = cv2.resize(img, (int(newX), int(newY)))
        else:
            print('No image supplied, and the default image, {} ' \
                    'was not found'.format(default_image))
            exit()
    return img

def detect_skin(image):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    # These values work pretty well for typical human skin
    lower = np.array([0, 48, 80], dtype = 'uint8')
    upper = np.array([20, 255, 255], dtype = 'uint8')
    return cv2.inRange(hsv, lower, upper)

def find_largest_contour(image):
    _, contours, _ = cv2.findContours(image.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    largest = max(contours, key = cv2.contourArea)
    mask = np.zeros(img.shape[:2], dtype = 'uint8')
    #https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_imgproc/py_contours/py_contours_begin/py_contours_begin.html#contours-getting-started
    cv2.drawContours(mask, [largest], -1, (255,255,255), -1)
    return mask

if __name__ == '__main__':
    img = validate_image()
    cv2.imshow('original', img)
    mask = detect_skin(img)
    cv2.imshow('mask', mask)
    # Remove non-skin pixels
    skin = cv2.bitwise_and(img, img, mask=mask)
    cv2.imshow('skin', skin)
    face_mask = find_largest_contour(mask)
    cv2.imshow('face_mask', face_mask)
    face = cv2.bitwise_and(skin, skin, mask=face_mask)
    cv2.imshow('face', face)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
