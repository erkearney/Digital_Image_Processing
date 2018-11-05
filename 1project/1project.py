#https://www.learnopencv.com/principal-component-analysis/
import cv2
import numpy as np
import os

class Person:
    def __init__(self, path, id_num):
        self.__id_num = id_num
        self.__path = path
        self.__images = []
        self.find_images()
        self.feature_extraction()

    def find_images(self):
        # Append a 0 to id_num if it's less than 10, because we need 07, not 7
        if int(self.__id_num) < 10:
            identifier = "ID" + "0" + self.__id_num
        else:
            identifier = "ID" + self.__id_num

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

    def feature_extraction(self):
        num_images = len(self.__images)
        size = self.__images[0].shape
        # data will hold all the flattened face images
        data = np.zeros((num_images, size[0] * size[1] * size[2]), dtype=np.float32)  
        for i in range(self.__num_images):
            # For each training image, change to 1 column vector
            image = self.__images[i].flatten()
            data[i,:] = image

        # Compute the eigenvectors using Principle Component Analysis
        mean, eigenvectors = cv2.PCACompute(data, mean=None, maxComponents=10)
        print(mean)
        average_face = mean.reshape(size)
        cv2.imshow("average", average_face)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

def main():
    people = []
    for i in range(20):
        person = Person("./LargeDataSet/enrolling", str(i))
        people.append(person)

if __name__ == "__main__":
    main()
