import numpy as np
import cv2
import copy
import matplotlib.pyplot as plt
class OtsuThresholder:
    def __init__(self):
        self. total_mean = None
        self.total_variance = None
        self.threshold_val = None
        self.img = self.n_rows = self.n_cols = None
        self.class_mean_arr = np.zeros((2, ))
        self.class_count = np.zeros((2, ))
        self.total_pixels = None

    '''
    This calculates the class means based on the current threshold value
    '''
    def calcClassMean(self):
        eps = 1e-8
        self.class_count[0] = self.img[self.img <= self.threshold_val].size
        self.class_count[1] = self.img[self.img > self.threshold_val].size
        self.class_mean_arr[0] = np.sum(self.img[self.img <= self.threshold_val])/(self.class_count[0] + eps)
        self.class_mean_arr[1] = np.sum(self.img[self.img > self.threshold_val])/(self.class_count[1] + eps)

    '''
    Finds the between class variance which we are trying to maximize
    '''
    def calcBetweenClassVariance(self):
        return np.round(np.multiply(np.square(self.class_mean_arr-self.total_mean), self.class_count/self.total_pixels).sum(), 3)

    '''
    Binarize the image based on threshold value  
    '''
    def getBinaryImage(self):
        binary_image = copy.deepcopy(self.img)
        binary_image[self.img <= self.threshold_val] = 0
        binary_image[self.img > self.threshold_val] = 255
        return binary_image.reshape(self.n_rows, self.n_cols)

    def plotHistogram(self):
        (val, count) = np.unique(self.img, return_counts=True)
        plt.bar(val, count)
        plt.show()

    '''
    1. Initialize random threshold value
    2. Find class Means
    3. Find Between Class Variance
    4. Iterate over all possible values which the pixel take and find the threshold
       value which maximizes the between class variance
    5. Binarize Image
    '''
    def thresholdImage(self, img):
        self.n_rows, self.n_cols = img.shape
        self.img = copy.deepcopy(img.reshape((self.n_rows * self.n_cols, )))
        self.total_pixels = self.n_rows * self.n_cols
        self.threshold_val = np.random.choice(self.img)
        self.total_mean = np.mean(self.img)
        self.calcClassMean()
        best_threshold_val = self.threshold_val
        max_between_class_variance = self.calcBetweenClassVariance()
        count = 0
        for test_threshold_val in np.unique(self.img).tolist():
            print('At iteration number ' + str(count))
            self.threshold_val = test_threshold_val
            self.calcClassMean()
            curr_between_class_variance = self.calcBetweenClassVariance()
            if curr_between_class_variance > max_between_class_variance:
                print(curr_between_class_variance, max_between_class_variance)
                max_between_class_variance = curr_between_class_variance
                best_threshold_val = test_threshold_val
            count = count + 1
        self.threshold_val = best_threshold_val
        return self.getBinaryImage()

img_fname = 'palmleaf1.pgm'
img_name = img_fname.split('.')[0]


img = cv2.imread(img_fname, 0)
o = OtsuThresholder()
binary_img = o.thresholdImage(img)
print(o.threshold_val)
o.plotHistogram()
cv2.imwrite(img_name+'_binary.png', binary_img)
