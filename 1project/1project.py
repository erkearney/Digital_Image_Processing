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
                img = cv2.imread(image_path)

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

    def get_size(self):
        return self.__images[0].size

    def compute_mean(self):
        num_images = len(self.__images)
        size = self.__images[0].shape
        # data will hold all the flattened face images
        data = np.zeros((num_images, size[0] * size[1] * size[2]), dtype=np.float32)  
        for i in range(self.__num_images):
            # For each training image, change to 1 column vector*
            image = self.__images[i].flatten()
            # * numpy.matrix.flatten() actually converts the matrix to a
            # single row, there doesn't appear to be a way to convert a 
            # matrix to a vector
            data[i,:] = image
        mean = np.mean(data, axis=0)
        return mean

    def compute_deviation(self, Xi, Me):
        # Returns Ai = Xi - Me
        return Xi - Me

    def compute_eigen_vectors(self, Ai):
        # Returns the eigen vectors of A' * A
        A_transpose = np.transpose(Ai)
        print(A_transpose)

def main():
    # Calculate the mean vector, Me, for all persons in the system
    people = []
    # Use the first person to setup the mean vector
    person = Person(TRAINING_DATASET, 0)
    Me = person.compute_mean()
    people.append(person)
    # Create a person object for each person in the training set
    for i in range(1, NUM_TRAINING):
        person = Person(TRAINING_DATASET, str(i))
        # Add each persons' average vector to the mean vector
        Me = np.add(Me, person.compute_mean())
        people.append(person)

    Me = np.divide(Me, NUM_TRAINING)

    # Let Ai = Xi - Me
    A = np.zeros((NUM_TRAINING, people[0].get_size()), dtype=np.float32)
    for i in range(NUM_TRAINING):
        Ai = people[i].compute_deviation(people[i].compute_mean(), Me)
        A[i,:] = Ai

    # Calculate the eigen vecotrs of A' * A and store it as P2
    #https://docs.opencv.org/3.0-beta/modules/core/doc/operations_on_arrays.html?highlight=pca#cv2.PCACompute
    #https://docs.scipy.org/doc/numpy-1.15.1/reference/generated/numpy.matmul.html
    print("Computing eigenvectors...")
    #mean, P2 = cv2.PCACompute(np.matmul(A.T, A), None)
    #print(P2)

if __name__ == "__main__":
    main()
