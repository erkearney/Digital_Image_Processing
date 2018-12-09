# Calculate the convext hull of fixed input points, using the Jarvis 
# march algorithm: 
# https://en.wikipedia.org/wiki/Gift_wrapping_algorithm#Pseudocode
import cv2
import numpy as np
import sys

def wrap_gift():
    if(len(sys.argv)) < 2:
        file_path = "./test.png"
    else:
        file_path = sys.argv[1]

    print("Using {} as the input image".format(file_path))

    im = cv2.imread(file_path)
    
    cv2.imshow("input image", im)
    




    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    wrap_gift()
