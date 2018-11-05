import cv2
import numpy as np
import os
import sys

# Create data matrix from a list of images
def create_data_matrix(images):
    print("Creating data matrix",end=" ... ")
    """
    images: The image dataset

    The size of the data matrix is
    (w * h, numImages)

    where,

    w = width of each image in the dataset
    h = height of each image in the dataset
    """

    num_images = len(images)
    size = images[0].shape
    data = np.zeros((num_images, size[0] * size[1] * size[2]), dtype=np.float32)
    for i in range(0, num_images):
        image = images[i].flatten()
        data[i,:] = image

    print("DONE")
    return data

# Read images from the directory
def read_images(path):
    """
    path: The path to the image dataset
    """
    images = []
    for filename in os.listdir(path):
        if filename.endswith(".bmp"):
            # Add to array of images
            image_path = os.path.join(path, filename)
            img = cv2.imread(image_path)

            # Make sure the image loaded properly
            if img is None:
                print("image: {} not read properly".format(imagePath))
            else:
                # Convert the image to floating point
                img = np.float32(img)/255.0
                # Add the image to the list
                images.append(img)
                # Flip image
                img_flip = cv2.flip(img, 1)
                # Append flipped image
                images.append(img_flip)

    # The images array contains each image, and the flipped version
    # of the image, so we need to divide by 2 here
    num_images = len(images) / 2
    # Exit if no image was found
    if num_images == 0:
        print("No images found")
        sys.exit(1)

    print(str(num_images) + " files read.")
    return images

# Add the weighted eigen faces to the mean face
def create_new_face(*args):
    # Start with the mean image
    output = average_face

    # Add the eigen faces with the weights
    for i in range(0, NUM_EIGEN_FACES):
        """
        OpenCV does not allow slider values to be negative.
        So we use weight = sliderValue - MAX_SLIDER_VALUE / 2
        """
        slider_values.append(cv2.getTrackbarPos("Weight" + str(i), "Trackbars"))
        weight = slider_values[i] - MAX_SLIDER_VALUE/2
        output = np.add(output, eigen_faces[i] * weight)

    # Display the Result at 2x size
    output = cv2.resize(output, (0,0), fx=2, fy=2)
    cv2.imshow("Result", output)

def reset_slider_values(*args):
    for i in range(0, NUM_EIGEN_FACES):
        cv2.setTrackbarPos("Weight", + str(i), "Trackbars", MAX_SLIDER_VALUE/2)
    create_new_face()

if __name__ == "__main__":
    # Number of EigenFaces
    NUM_EIGEN_FACES = 10

    # Maximum weight
    MAX_SLIDER_VALUE = 255

    # Directory containing images
    dir_name = "LargeDataSet/enrolling"

    # Read images
    images = read_images(dir_name)

    # Size of images
    size = images[0].shape

    # Create data matrix for PCA
    data = create_data_matrix(images)

    # Comput the eigenvectors from the stack of images created
    print("Calculating PCA ", end=" ... ")
    mean, eigen_vectors, = cv2.PCACompute(data, mean=None, maxComponents=NUM_EIGEN_FACES)
    print("DONE")

    average_face = mean.reshape(size)

    eigen_faces = []

    for eigen_vector in eigen_vectors:
        eigen_face = eigen_vector.reshape(size)
        eigen_faces.append(eigen_face)

    # Create window for displaying the mean face
    cv2.namedWindow("Result", cv2.WINDOW_AUTOSIZE)

    # Display the result at 2x size
    output = cv2.resize(average_face, (0,0), fx=2, fy=2)
    cv2.imshow("Result", output)

    # Create Window for trackbars
    cv2.namedWindow("Trackbars", cv2.WINDOW_AUTOSIZE)

    slider_values = []

    # Create Trackbars
    for i in range(0, NUM_EIGEN_FACES):
        slider_values.append(MAX_SLIDER_VALUE/2)
        cv2.createTrackbar("Weight" + str(i), "Trackbars", int(MAX_SLIDER_VALUE/2), int(MAX_SLIDER_VALUE), create_new_face)

    # You can reset the sliders by clicking on the mean image
    cv2.setMouseCallback("Result", reset_slider_values)

    print("""Usage:
    Change the weights using the sliders
    Click on the result window to reset sliders
    Hit ESC to terminate program.""")

    cv2.waitKey(0)
    cv2.destroyAllWindows()

