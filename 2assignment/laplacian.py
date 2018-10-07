import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt

img = cv.imread('iris.bmp', 0)

laplacian_kernel = np.matrix([[0, 1, 0], [1, -4, 1], [0, 1, 0]])
laplacian_img = cv.filter2D(img, -1, laplacian_kernel)

sobel_x_kernel = np.matrix([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])
sobel_y_kernel = np.matrix([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])

img_sobel_x = cv.filter2D(laplacian_img, -1, sobel_x_kernel)
img_sobel_y = cv.filter2D(laplacian_img, -1, sobel_y_kernel)

titles = ['original', 'laplacian', 'sobelx', 'sobely']
images = [img, laplacian_img, img_sobel_x, img_sobel_y]

plt.figure(num='Part 1')
for i in range(len(images)):
    plt.subplot(2,3,i+1),plt.imshow(images[i], cmap = 'gray')
    plt.title(titles[i])
    plt.xticks([]), plt.yticks([])

plt.show()
