"""
Face recognition using eigen faces and PCA. Completed by Eric Kearney on
11/15/2018, as part of Dr. Jiang's CS390S -- Digital Image Processing
class. Resources the helped me complete this project include the numpy 
documentation: https://docs.scipy.org/doc/numpy-1.15.1/reference/index.html
and 3Blue1Brown's Essence of Linear Algebra series on YouTube:
https://www.youtube.com/playlist?list=PLZHQObOWTQDPD3MizzM2xVFitgF8hE_ab.

This project closely follows the steps described in the slides provided in
class by Dr. Jiang. Each step from the slides is sepearted using a:
# --------------------------------------------------------------------
line.
"""
import cv2
import numpy as np
from matplotlib import pyplot as plt
import os
from pathlib import Path
import argparse

# NUM_TRAINING_PEOPLE is the number of people to use in the training dataset
# NUM_TESTING_PEOPLE is the number of people to use in the testing dataset
# NUM_TRAINING_IMAGES is the number of images to use for each person in the training dataset
NUM_TRAINING_PEOPLE = 20
NUM_TESTING_PEOPLE = 10
NUM_TRAINING_IMAGES = 5
DEFAULT_TRAINING_DATASET = "./LargeDataSet/enrolling"
DEFAULT_TESTING_DATASET = "./LargeDataSet/testing"

parser = argparse.ArgumentParser()
parser.add_argument('-tr', "--training", help="path to the training dataset")
parser.add_argument('-te', '--testing', help="path to the testing dataset")
args = parser.parse_args()

class Person:
    def __init__(self, path, id_num, num_images):
        self.__id_num = id_num
        self.__path = path
        self.__images = []
        self.__find_images(num_images)

    def __find_images(self, num_images):
        # Append a 0 to id_num if it's less than 10, because we need 07, not 7
        if int(self.__id_num) < NUM_TESTING_PEOPLE:
            identifier = "ID" + "0" + str(self.__id_num)
        else:
            identifier = "ID" + str(self.__id_num)

        images_loaded = 0
        # Add a number of images equal to num_images of the form 
        # "ID<id_num>_XXX.bmp"
        for filename in os.listdir(self.__path):
            # End this loop once we've loaded the specified number of images
            if images_loaded >= num_images:
                break

            if identifier in filename:
                image_path = os.path.join(self.__path, filename)
                img = cv2.imread(image_path, 0)

                # Make sure the image loaded properly
                if img is None:
                    print("image: {} not read properly".format(imagePath))
                else:
                    img = np.uint16(img)
                    # Add the image to the list
                    self.__images.append(img)
                    #print("{} loaded".format(image_path))
                    images_loaded += 1

        self.__num_images = len(self.__images)

    def get_id_num(self):
        return self.__id_num

    def get_images(self):
        return self.__images

    def get_size(self):
        return self.__images[0].size

    def get_shape(self):
        return self.__images[0].shape

    def get_num_images(self):
        return self.__num_images

    def compute_mean(self):
        num_images = self.get_num_images()
        size = self.get_size()
        # Data will hold the flattened images
        data = np.zeros((num_images, size), dtype=np.float32)
        #data = np.zeros((size, num_images), dtype=np.float32)
        for i in range(self.__num_images):
            # 1. For each training image, change to 1 column vector*
            image = (self.__images[i].flatten()).T
            # * numpy.matrix.flatten() actually converts the matrix to an
            # array, so we need to take the transpose of the flattened image
            # to get our vector
            data[i,:] = image
        # 2. For each person, calculate the average vector Xi
        Xi = np.mean(data, axis=0)
        return Xi

    def compute_deviation(self, Xi, Me):
        # Returns Ai = Xi - Me
        return Xi - Me

def main():
    # Setup
    if args.training:
        if Path(args.training).is_dir():
            print("Using {} as the path to the training dataset".format(args.training))
            TRAINING_DATASET = args.training
        else:
            print("ERROR: The training dataset could not be found in {}".format(args.training))
    else:
        print("Using the default training dataset")
        if Path(DEFAULT_TRAINING_DATASET).is_dir():
            TRAINING_DATASET = DEFAULT_TRAINING_DATASET
        else:
            print("ERROR: The default training dataset could not be found in {}".format(DEFAULT_TRAINING_DATASET))
            exit(1)

    if args.testing:
        if Path(args.testing).is_dir():
            print("Using {} as the path to the testing dataset".format(args.testing))
            TESTING_DATASET = args.testing
        else:
            print("ERROR: The testing path could not be found in {}".format(args.testing))
    else:
        print("Using default testing dataset")
        if Path(DEFAULT_TESTING_DATASET).is_dir():
            TESTING_DATASET = DEFAULT_TESTING_DATASET
        else:
            print("ERROR: The default testing dataset could be found in {}".format(DEFAULT_TESTING_DATASET))
            exit(1)
    ##########################################################################
    # Training
    # 3. Calculate the mean vector, Me, for all persons in the system
    # The following line uses list comprenhenion to create an array of Person
    # objects, using the Person(path, id_num, num_images) constructor
    people = [Person(TRAINING_DATASET, str(i), NUM_TRAINING_IMAGES) for i in range(NUM_TRAINING_PEOPLE)]
    size = people[0].get_size()
    # Use the first person to setup the mean vector
    Me = people[0].compute_mean()
    # Then add every other persons' mean vector to Me
    for i in range(1, len(people)):
        Me = np.add(Me, people[i].compute_mean())
    # Finally, divide the mean vector by the number of people
    Me = np.divide(Me, NUM_TRAINING_PEOPLE)
    # ------------------------------------------------------------------------
    # 4. Let Ai = Xi - Me
    # Create a matrix of size NUM_TRAINING_PEOPLE x image_size, fill it with 
    # Ai = Xi - Me
    A = np.zeros((size, NUM_TRAINING_PEOPLE), dtype=np.float32)
    for i in range(NUM_TRAINING_PEOPLE):
        Ai = people[i].compute_deviation(people[i].compute_mean(), Me)
        A[:,i] = Ai
    # ------------------------------------------------------------------------
    # 5. Calculate the eigen vectors of A' * A and store it as P2
    # https://docs.scipy.org/doc/numpy-1.15.1/reference/generated/numpy.linalg.eigh.html#numpy.linalg.eigh
    ATA = np.matmul(A.T, A)
    eigen_values, P2 = np.linalg.eigh(ATA)
    # ------------------------------------------------------------------------
    # 6. Calculate the weight of the training data projected into eigen space
    # with wt_A = P2*(A'*A)
    # 6a. Calculate the eigen vectors of A*A' as P = A*P2
    P = np.matmul(A, P2)
    # 6b. Calculate the weight of the training data projected into eigensapce
    # wt_A = P'*A
    wt_A = np.matmul(P.T,A)
    plt.plot(wt_A[0,:])
    plt.xticks(np.arange(0, (NUM_TRAINING_PEOPLE), step = 2))
    plt.title("Weights for the first person");
    plt.show()

    eigen_vectors = P.T
    # ------------------------------------------------------------------------
    # Show average face minus the mean
    """
    shape = people[0].get_shape()
    average_faces = A.T
    for i in range(average_faces.shape[0]):
        face = np.reshape(average_faces[i], shape)
        #norm_value = 1 / (np.amax(face))
        #face *= norm_value
        face = cv2.resize(face, (0,0), fx=5, fy=5)
        cv2.imshow("face", face)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    """
    # Show eigen faces
    """
    for i in range(eigen_vectors.shape[0]):
        face = eigen_vectors[i].reshape(shape)
        cv2.imshow("face {}".format(str(i)), face)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        cv2.imwrite("eigen_face_{}.bmp".format(str(i)), face)
        print("Face number {}\n{}".format(str(i), face))
    """

    ##########################################################################
    # Face recognition
    start_num = 5
    num_testing = 10
    # This creates an array of 20 zeros
    CMC_Array = [0] * NUM_TRAINING_PEOPLE
    # Create a Person object with 1 image for each person we want to test with
    people_test = [Person(TESTING_DATASET, str(i), 1) for i in range(start_num, (start_num + num_testing))]
    for person in people_test:
        #print("Testing person {}".format(person.get_id_num()))
        # 1. For the input image Im, change to 1 column vector: Y
        Y = person.get_images()[0].flatten()
        # --------------------------------------------------------------------
        # 2. Calculate B = Y - Me
        B = Y - Me
        # --------------------------------------------------------------------
        # 3. Calculate the weight of the input data projected into eigenspace
        # wt_B=P'*b
        wt_B = np.matmul(P.T, B)
        # --------------------------------------------------------------------
        # 4. Calculate the euclidean distance for the input image:
        # eud(i) = sqrt(sum((wt_B-wt_A(:,i)).^2));
        eud_dist = np.zeros(NUM_TRAINING_PEOPLE)
        for i in range(NUM_TRAINING_PEOPLE):
            eud_dist[i] = np.sqrt(np.sum(np.square(np.subtract(wt_B, wt_A[:,i]))))
        index_sorted = np.argsort(eud_dist)
        #print(index_sorted)
        for i in range(len(index_sorted)):
            if int(index_sorted[i]) == int(person.get_id_num()):
                #print("Guess number {} was correct for person number {}".format(i, person.get_id_num()))
                CMC_Array[i] += 1
                break
        # --------------------------------------------------------------------
    # Show results
    total = 0
    for i in range(len(CMC_Array)):
        total += CMC_Array[i]
        # Normalize
        CMC_Array[i] = total/num_testing
    #print("CMC_Array: {}".format(CMC_Array))
    plt.plot(CMC_Array)
    plt.title("CMC")
    plt.axis([0, NUM_TRAINING_PEOPLE, 0, 1])
    plt.xlabel("Guess")
    plt.ylabel("Cumulative Accuracy")
    plt.xticks(np.arange(0, (NUM_TRAINING_PEOPLE+1), step = 2))
    plt.show()
    # --------------------------------------------------------------------

if __name__ == "__main__":
    main()
