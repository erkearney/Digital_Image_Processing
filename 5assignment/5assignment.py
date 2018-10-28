'''
Face detection, written by Eric Kearney on October 27th, 2018 This script 
detects faces by first performing a skin detection, then, selects the largest 
contour, which will often be the face.

I say often because it's obvious that this approach will fail if the person
in the image is dressed in such a way that more skin is exposed.

I included a couple more images, 'chris_hemsworth_pass.jpg shows another 
example of this script working. Image source:
https://vancouversun.com/entertainment/celebrity/chris-hemsworth-talks-style-and-that-thor-costume

'chris_hemsworth_fail.jpg' shows this script failing, the image subject's
arm is exposed, to the point where it becomes the largest skin contour. 
Image source:
https://ew.com/movies/2018/01/16/chris-hemsworth-thor-contract-avengers-4/

This script also assumes the input image has only one face in it.

Use: python3 5assignment.py --image <path to your image>
If no image argument is supplied, this script will look for my photo, 
'portrait.jpg' within the same directory.
'''
import cv2
import numpy as np
import argparse
from pathlib import Path
from matplotlib import pyplot as plt

default_image = './portrait.jpg'
parser = argparse.ArgumentParser()
parser.add_argument('-i', '--image', help='path to the image')
parser.add_argument('-s', '--steps', action='store_true', 
                    help='show the face detection step-by-step')
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
        else:
            print('No image supplied, and the default image, {} ' \
                    'was not found'.format(default_image))
            exit()

    # Resize the image
    W = 512
    height, width, depth = img.shape
    imgScale = W/width
    newX, newY = img.shape[1]*imgScale, img.shape[0]*imgScale
    img = cv2.resize(img, (int(newX), int(newY)))
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

def dilate_then_erode(image):
    img = image.copy()
    # Use the '+' kernel
    kernel = np.matrix([[0,1,0],
                        [1,1,1],
                        [0,1,0]], dtype = 'uint8')
    img = cv2.dilate(img,kernel,iterations=1)
    img = cv2.erode(img,kernel,iterations=1)
    return img

if __name__ == '__main__':
    img = validate_image()
    mask = detect_skin(img)
    # Remove non-skin pixels
    skin = cv2.bitwise_and(img, img, mask=mask)
    face_mask = find_largest_contour(mask)
    face_mask_2 = dilate_then_erode(face_mask)
    face = cv2.bitwise_and(img, img, mask=face_mask_2)

    if args.steps:
        cv2.imshow('original', img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        cv2.imshow('detected skin', mask)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        cv2.imshow('non-skin removed', skin)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        cv2.imshow('largest contour', face_mask)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        cv2.imshow('after dilation and erosion', face_mask_2)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        cv2.imshow('final result', face)
    else:
        cv2.imshow('Result', np.hstack([img, face]))
        cv2.imwrite('Results.jpg', np.hstack([img, face]))
        
    cv2.waitKey(0)
    cv2.destroyAllWindows()
