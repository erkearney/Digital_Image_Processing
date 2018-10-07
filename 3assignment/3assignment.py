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

def main():
    img = validate_image()
    img_blue = img.copy()
    img_blue[:,:,1] = 0
    img_blue[:,:,2] = 0

    img_green = img[:,:,1]
    img_red = img[:,:,2]





    cv2.imshow('original', img)
    cv2.imshow('blue', img_blue)
    cv2.imshow('green', img_green)
    cv2.imshow('red', img_red)


    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
