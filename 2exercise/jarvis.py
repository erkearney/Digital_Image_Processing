""" Calualte the convex hull of fixed input points and classify random input
points by their location as inside or outside of the given convex hull.

This program is an implementation of the Jarvis march algorithm:
https://en.wikipedia.org/wiki/Gift_wrapping_algorithm#Pseudocode """

import cv2
import sys
import numpy as np

def read_image():
    """ Get the input image. """
    if(len(sys.argv)) < 2:
        # Default image
        image_path = "./test.png"
    else:
        image_path = sys.argv[1]

    print("Using {} as the input image".format(image_path))
    img = cv2.imread(image_path, 0)
    if(img.all() != None):
        cv2.imshow("input image", img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        return img
    else:
        print("ERROR: {} not found, exiting.".format(image_path))
        exit(1)

def create_image():
    """ Create the supplied image. """
    img = np.matrix([[0,0,0,0,0,0,0,0,0,0],
                     [0,0,0,0,0,0,0,0,0,0],
                     [0,0,0,0,0,0,0,0,0,0],
                     [0,0,0,1,1,1,0,0,0,0],
                     [0,0,0,1,1,1,0,0,0,0],
                     [0,0,0,1,1,1,1,1,0,0],
                     [0,0,1,0,0,0,0,0,0,0],
                     [0,0,0,0,0,0,0,0,0,0],
                     [0,0,0,0,0,0,0,0,0,0],
                     [0,0,0,0,0,0,0,0,0,0]])

    return img

def cvt_to_points(image):
    """ Converts a binary image to an array of points """
    points = []
    cols = image.shape[0]
    rows = image.shape[1]
    for c in range(cols):
        for r in range(rows):
            if(image[c][r] > 0):
                points.append((c,r))
    return points

def jarvis(points):
    """ Calculates the convex hull of points. """
    vertexs = []
    # point_on_hull starts as the leftmost point in points
    point_on_hull = min(points, key = lambda point: point[1])



def main():
    #img = create_image()
    img = read_image()
    print(img)
    points = cvt_to_points(img)
    jarvis(points)

if __name__ == "__main__":
    main()
