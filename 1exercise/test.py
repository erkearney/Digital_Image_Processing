import numpy as np
import cv2 as cv

img = cv.imread('./serveimage.jpeg')

for r in range(0, img.shape[0]):
    for c in range(0, img.shape[1]):
        if img[r][c][1] > 100:
            img[r][c][1] = 0
