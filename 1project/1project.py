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
                    img = np.float32(img)/255
                    # Add the image to the list
                    self.__images.append(img)
                    #print("{} loaded".format(image_path))

        self.__num_images = len(self.__images)

    def get_id_num(self):
        return self.__id_num

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
    # 3. Calculate the mean vector, Me, for all persons in the system
    people = [Person(TRAINING_DATASET, str(i)) for i in range(NUM_TRAINING)]
    size = people[0].get_size()
    # Use the first person to setup the mean vector
    Me = people[0].compute_mean()
    # Then add every other persons' mean vector to Me
    for i in range(1, len(people)):
        Me = np.add(Me, people[i].compute_mean())
    # Finally, divide the mean vector by the number of people
    Me = np.divide(Me, NUM_TRAINING)

    # 4. Let Ai = Xi - Me
    # Create a matrix of size NUM_TRAINING x image_size, fill it with 
    # Ai = Xi - Me
    A = np.zeros((size, NUM_TRAINING), dtype=np.float32)
    for i in range(NUM_TRAINING):
        Ai = people[i].compute_deviation(people[i].compute_mean(), Me)
        A[:,i] = Ai

    ATA = np.matmul(A.T, A)
    # 5. Calculate the eigen vectors of A' * A and store it as P2
    # https://docs.scipy.org/doc/numpy-1.15.1/reference/generated/numpy.linalg.eig.html
    eigen_values, P2 = np.linalg.eig(ATA)

    # 6. Calculate the weight of the training data projected into eigen space
    # with wt_A = P2*(A'*A)
    # 6a. Calculate the eigen vectors of A*A' as P = A*P2
    P = np.matmul(A, P2)
    # 6b. Calculate the weight of the training data projected into eigensapce
    # wt_A = P'*A
    wt_A = np.matmul(P.T,A)

    shape = people[0].get_shape()
    '''
    # Show average face minus the mean
    average_faces = A.T
    for i in range(average_faces.shape[0]):
        face = np.reshape(average_faces[i], shape)
        #norm_value = 1 / (np.amax(face))
        #face *= norm_value
        face = cv2.resize(face, (0,0), fx=5, fy=5)
        cv2.imshow("face", face)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    '''

    # Show eigen faces
    eigen_vectors = P.T
    for i in range(eigen_vectors.shape[0]):
        face = eigen_vectors[i].reshape(shape)
        #norm_value = 1 / (np.amax(face))
        #face *= norm_value
        face += 0.5
        cv2.imwrite("eigen_face_{}.bmp".format(str(i)), face)
        face = cv2.resize(face, (0,0), fx=5, fy=5)
        print(face)
        cv2.imshow("eigen face", face)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
