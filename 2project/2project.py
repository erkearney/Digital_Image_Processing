# https://www.pyimagesearch.com/2018/07/16/opencv-saliency-detection/
# This is a simple image saliency tutorial written in Python by 
# Adrian Rosebrock, with some small modifications made by me.
# https://docs.opencv.org/3.4.2/d8/d65/group__saliency.html
# OpenCVs' saliency module doesn't come packaged by default, it's in 
# the 'contrib' module: https://pypi.org/project/opencv-contrib-python/

import cv2
import numpy as np
import argparse
from pathlib import Path

parser = argparse.ArgumentParser(
    description="Visual Saliency model built with OpenCV"
)
parser.add_argument("-i", "--image", help="path to the input image")
args = parser.parse_args()

def main():
    # Basic setup
    input_image = Path("./test_image.bmp")
    if args.image:
        input_image = Path(args.image)
    print("Using {} as the input image".format(input_image))
    if not input_image.is_file():
        print("ERROR: {} not found".format(input_image))
        exit(1)
    else:
        image = cv2.imread(str(input_image))

    # Initialize OpenCV's static fine grained saliency detector and compare
    # the saliency map
    saliency = cv2.saliency.StaticSaliencyFineGrained_create()
    (success, saliency_map) = saliency.computeSaliency(image)

    # cv2.threshold() wants its input to be Uint8.
    saliency_map = np.uint8(saliency_map * 255)
    thresh_map = cv2.threshold(saliency_map, 0, 255, 
        cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]

    # Show the images
    cv2.imshow("Input", image)
    cv2.imshow("Output", saliency_map)
    cv2.imshow("Tresh", thresh_map)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    exit(0)


if __name__ == "__main__":
    main()
