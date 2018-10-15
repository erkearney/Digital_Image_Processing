'''
Written by Eric Kearney on 10/14/2018. This program detects skin on an input
image and returns an image with every pixel that does not have detected skin
set to 0. By default, This program uses the 'face_dark.bmp" image, include it
in the same directory as this program, or supply your own image using
3assignment,py --image <path_to_image>

This is my first 'serious submission', it does what it's supposed to, but. . .
badly. I wasn't able to find a way to work with histograms directly in 
opencv, which made the automatic thresholding VERY difficult. I truly hope
I'll be able to take another look at this project at another time, for now,
there is some seriously sketchy code in here, see the report for details.
'''

import cv2
import numpy as np
import argparse
from pathlib import Path
from matplotlib import pyplot as plt
from scipy import stats

default_image = './face_dark.bmp'
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
        else:
            print('No image supplied, and the default image, {} ' \
                    'was not found'.format(default_image))
            exit()
    return img

def create_mask(image):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    lower = np.array([0, 48, 80], dtype = 'uint8')
    luminance_thresh = find_luminance_local_min(hsv)
    upper = np.array([20, 255, luminance_thresh], dtype = 'uint8')
    return cv2.inRange(hsv, lower, upper)

def find_luminance_local_min(hsv):
    # Find and return the local minima between the two largest local maxima
    # of the luminance component of the image, or at least that's what we're
    # supposed to do here. For now, I've used the mode to find the two
    # most frequently occuring values (aka the two highest peaks of the 
    # histogram), and simply took the average of those two values, hoping
    # there'd be a trough between them. To improve this, it would be necessary
    # to make sure the two peaks aren't too close to each other. In any case,
    # this entire approach is probably completely wrong.
    # Get just the luminance component of the image, flatten it to a 1D 
    # array
    lum = hsv[:,:,2]
    lum = lum.ravel()
    # Find the first peak of the histogram by taking the mode of the array.
    mode1 = stats.mode(lum)
    # Remove all values in that array that are less than the peak, then find
    # the next biggest peak
    lum = lum[lum > mode1[0]]
    mode2 = stats.mode(lum)
    lum = lum[lum < mode2[0]]
    return mode1[0] / 2 + mode2[0] / 2

def filter_skin(image, mask):
    # Blur the mask to smooth the result and help get rid of the noise
    gaus = cv2.GaussianBlur(mask, (7,7), 0)
    return cv2.bitwise_and(image, image, mask = gaus)

def main():
    img = validate_image()
    mask = create_mask(img)
    result = filter_skin(img, mask)

    cv2.imshow('original', img)
    cv2.imshow('mask', mask)
    cv2.imwrite('./mask.bmp', mask)
    cv2.imshow('result', result)
    cv2.imwrite('./result.bmp', result)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()    
