#https://www.learnopencv.com/principal-component-analysis/
import cv2
import numpy as np
import os

NUM_TRAINING = 20
TRAINING_DATASET = "./LargeDataSet/enrolling"

class Person:
    def __init__(self, path, id_num):
        self.__id_num = id_num
        self.__path = path
        self.__images = []
        self.find_images()

    def find_images(self):
        # Append a 0 to id_num if it's less than 10, because we need 07, not 7
        if int(self.__id_num) < 10:
            identifier = "ID" + "0" + str(self.__id_num)
        else:
            identifier = "ID" + str(self.__id_num)

        # Add all images of the form "ID<id_num>_XXX.bmp
        for filename in os.listdir(self.__path):
            if identifier in filename:
                image_path = os.path.join(self.__path, filename)
                img = cv2.imread(image_path, 0)

                # Make sure the image loaded properly
                if img is None:
                    print("image: {} not read properly".format(imagePath))
                else:
                    # Convert the image to floating point
                    img = np.float32(img)/255.0
                    # Add the image to the list
                    self.__images.append(img)
                    #print("{} loaded".format(image_path))

        self.__num_images = len(self.__images)

    def compute_mean(self):
        num_images = len(self.__images)
        size = self.__images[0].shape
        # data will hold all the flattened face images
        data = np.zeros((num_images, size[0] * size[1]), dtype=np.float32)  
        for i in range(self.__num_images):
            # For each training image, change to 1 column vector*
            image = self.__images[i].flatten()
            # * numpy.matrix.flatten() actually converts the matrix to a
            # single row, there doesn't appear to be a way to convert a 
            # matrix to a vector
            data[i,:] = image
        mean = np.mean(data, axis=0)
        return mean

    def get_size(self):
        return self.__images[0].size

    def get_shape(self):
        return self.__images[0].shape

def main():
    # Create a Person object for each person in the training dataset
    people = [Person(TRAINING_DATASET, str(i)) for i in range(NUM_TRAINING)]
    image_size = people[0].get_size()
    image_shape = people[0].get_shape()
    data = np.zeros((5, image_size), dtype=np.float32)
    mean, eigen_vectors = cv2.PCACompute(data, mean=None, maxComponents=NUM_TRAINING)
    print(len(eigen_vectors))
    for eigen_vector in eigen_vectors:
        face = eigen_vector.reshape(image_shape)
        print(face)
    '''
    # Create a matrix of the average vectors for each person. This matrix will
    # be of size NUM_TRAINING X size of each image
    averages = np.zeros((NUM_TRAINING, image_size), dtype=np.float32)
    for i in range(len(people)):
        averages[i,:] = people[i].compute_mean()
    # Compute the eigen vectors
    mean, eigen_vectors = cv2.PCACompute(averages, mean=None, maxComponents=NUM_TRAINING)
    # Compute the eigen faces
    eigen_faces = [eigen_vector.reshape(image_size) for eigen_vector in eigen_vectors]
    face = eigen_faces[0].reshape(image_shape)
    cv2.imshow("face", face)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    # Show the eigen faces
    for eigen_face in eigen_faces:
        face = eigen_face.reshape(image_size)
        cv2.imshow("Eigen face", face)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    '''

if __name__ == "__main__":
    main()
