#https://stackoverflow.com/questions/22937589/how-to-add-noise-gaussian-salt-and-pepper-etc-to-image-in-python-with-opencv
import cv2
import numpy as np

def create_image():
    img = np.full((256, 256), 0.7)
    # There is no exact center, I will use the top-right corner 
    center = (128, 128)
    # Use the distance formula to determine how far each pixel is from 
    # the center
    radius = 80
    for r in range(0, img.shape[0]):
        for c in range(0, img.shape[1]):
            distance = np.sqrt((center[0] - c)**2 + (center[1] - r)**2)
            if distance > radius:
                img[r, c] = 0.4

    return img

def gaussian_noise(image, mean, var):
    row,col= image.shape
    sigma = var**0.5
    gauss = np.random.normal(mean,sigma,(row,col))
    gauss = gauss.reshape(row,col)
    return image + gauss

def salt_pepper(image, s_vs_p, amount):
    row,col = image.shape
    num_salt = np.ceil(amount * image.size * s_vs_p)
    out = np.copy(image)
    # Salt mode
    coords = [np.random.randint(0, i - 1, int(num_salt)) for i in image.shape]
    out[coords] = 1
    # Pepper mode
    num_pepper = np.ceil(amount * image.size * (1 - s_vs_p))
    coords = [np.random.randint(0, i - 1, int(num_pepper))
        for i in image.shape]
    out[coords] = 0
    return out

def uniform_noise(image, variance):
    img = image.copy()
    noise = np.random.uniform(0-variance, 0+variance, img.size)
    noise = np.reshape(noise, (img.shape[0], img.shape[1]))
    return img + img * noise

def main():
    img = create_image()
    gaus = gaussian_noise(img, 0, 0.1)
    s_and_p = salt_pepper(img, 0.5, 0.004)
    uni = uniform_noise(img, 0.05)

    cv2.imshow('img', img)
    cv2.imshow('gaus', gaus)
    cv2.imshow('s&p', s_and_p)
    cv2.imshow('uni', uni)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
