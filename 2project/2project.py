# https://www.pyimagesearch.com/2018/07/16/opencv-saliency-detection/
# This is a simple image saliency tutorial written in Python by 
# Adrian Rosebrock, with some small modifications made by me.
# The test images were provided by Rosebrock
# https://docs.opencv.org/3.4.2/d8/d65/group__saliency.html
# OpenCVs' saliency module doesn't come packaged by default, it's in 
# the 'contrib' module: https://pypi.org/project/opencv-contrib-python/

import cv2
import numpy as np
import argparse
from pathlib import Path
# skimage has built-in SSIM, so we'll just use that
from skimage.measure import compare_ssim as ssim

parser = argparse.ArgumentParser(
    description="Visual Saliency model built with OpenCV"
)
parser.add_argument("-i", "--image", help="path to the input image")
parser.add_argument("-d", "--distorted", help="path to the distorted image")
args = parser.parse_args()

def saliency(image):
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

    return thresh_map

def calc_quality(original, distorted):
    # Measure the quality of images using MSE, PSNR, SSIM, and my own metric
    # https://docs.scipy.org/doc/numpy-1.15.0/reference/routines.math.html
    MSE = (np.square(original - distorted)).mean(axis=None)
    L = 255.0
    if MSE == 0.0:
        PSNR = 0.0
    else:
        PSNR = 10 * np.log10(L / np.sqrt(MSE))

    SSIM = ssim(original, distorted, multichannel=True)

    print("MSE: {}, PSNR: {}, SSIM: {}".format(MSE, PSNR, SSIM))

def calc_TSS(original_thresh, distorted_thresh):
    # This is the metric of my creation, which I call Thresholded Saliency
    # Similarity. First, we calculate the saliency of the original image and 
    # the distorted the image, then we threshold both results to a binary 
    # image, then perform an exclusive OR over the two binary images, and sum 
    # the number of 1s that result. In other words, we count the number of 
    # pixels that are different between the two binary images. The larger TSS
    # is, the worse the distortion.
    TSS = np.sum(np.bitwise_xor(original_thresh, distorted_thresh))
    # Normalize TSS by dividing by the size of the thresholded images and the 
    # maximum pixel value, 255 in this case
    TSS /= (original_thresh.size * 255)
    print("TSS = {}".format(TSS))

def main():
    # Validate the input images exist and then read them
    input_image = Path("./test_image.bmp")
    if args.image:
        input_image = Path(args.image)
    print("Using {} as the input image".format(input_image))
    if not input_image.is_file():
        print("ERROR: {} not found".format(input_image))
        exit(1)
    else:
        image = cv2.imread(str(input_image), 0)

    distorted_image = Path("./distorted_image.bmp")
    if args.distorted:
        distorted_image = Path(args.distorted)
    print("Using {} as the distorted image".format(distorted_image))
    if not distorted_image.is_file():
        print("ERROR: {} not found".format(distorted_image))
        exit(1)
    else:
        distorted = cv2.imread(str(distorted_image), 0)

    if image.shape != distorted.shape:
        print("""ERROR: The input image has shape {}, the distorted image has
                 shape {}, they must match!""".format(image.shape, 
                                                      distorted.shape))
        exit(1)


    # Calculate and show the saliency and thresholded saliency, return the 
    # thresholded version
    original_thresh = saliency(image)
    distorted_thresh = saliency(distorted)
    
    calc_quality(image, distorted)
    calc_TSS(original_thresh, distorted_thresh)

    exit(0)


if __name__ == "__main__":
    main()
