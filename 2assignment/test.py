import cv2 as cv
import numpy as np

img = cv.imread('iris.bmp', -1)
img = np.add(img, 100)
#img = img.astype(int)

for r in range(0, 5):
    for c in range(0, 5):
        print(img[r][c])






img2 = np.full((25, 25), 50)
for r in range(0, 5):
    for c in range(0, 5):
        print(img2[r][c])


cv.imshow('img', img)
cv.imshow('img2', img2)
cv.waitKey(0)
cv.destroyAllWindows()
