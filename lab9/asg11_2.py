import numpy as np
import cv2
import copy


class KMeansCluster:

    def __init__(self, number_of_clusters=3, MAX_ITERATIONS=5, random_initialization_flag=True, *centroid):
        self.number_of_clusters = number_of_clusters
        self.MAX_ITERATIONS = MAX_ITERATIONS
        self.random_initialization_flag = random_initialization_flag
        self.img = self.n_rows = self.n_cols = self.num_channels = None
        self.current_cluster_id = None
        self.previous_cluster_id = None
        self.segment_image = None
        if not random_initialization_flag:
            if len(centroid) != number_of_clusters:
                print('Please give correct number of cluster centroids')
                exit()
            else:
                # Each Column Represents the centroid Val
                self.centroid = np.array([centroid_val for centroid_val in centroid]).T


    '''
    This cost function finds sum of square distances between pixel and their centroids 
    '''
    def objectiveCostFunction(self):
        return np.round(np.sum(np.linalg.norm(self.img - self.segment_image)**2), 3)

    '''
    This sets up 3*N(num_clusters) centroids randomly
    '''
    def setupUpRandomCentroids(self):
        self.centroid = np.random.randint(0, 255, (3, self.number_of_clusters))

    '''
    1. For N samples find distance from each of the m clusters and assign it to the cluster centroid
        at a minimum distance from it
    2. Recompute the cluster centers
    3. Recompute cluster groups and check if all of them again fall into the same clusters 
    4. If yes end, If no go back to step 1, also end if Max iterations are reached
    '''

    def cluster(self, img):
        self.n_rows, self.n_cols, self.num_channels = img.shape
        self.img = img.reshape(self.n_rows * self.n_cols, self.num_channels).T
        if self.random_initialization_flag:
            self.setupUpRandomCentroids()
        self.findCorrectCluster()
        self.previous_cluster_id = copy.deepcopy(self.current_cluster_id)
        self.updateCentroids()
        current_iteration = 1

        while (current_iteration < self.MAX_ITERATIONS):
            self.findCorrectCluster()
            if (np.all(np.equal(self.current_cluster_id, self.previous_cluster_id))):
                print('Halting Search Found Stable Centroids at ' + str(current_iteration) + ' iteration')
                break
            self.previous_cluster_id = copy.deepcopy(self.current_cluster_id)
            self.updateCentroids()
            current_iteration = current_iteration + 1
        return self.segmentImage()

    '''
    Assign the RGB value of the cluster center to each pixel belonging to the cluster
    '''

    def segmentImage(self):
        self.segment_image = copy.deepcopy(self.img)
        for k in range(self.number_of_clusters):
            kth_cluster_centroid = self.centroid[:, [k]]
            self.segment_image[:, self.current_cluster_id == k] = kth_cluster_centroid
        return self.segment_image.T.reshape(self.n_rows, self.n_cols, self.num_channels)

    '''
    This is the greedy implementation of clustering, this function will call the cluster function N times
    and will select the cluster group which minimizes the objective cost function
    '''

    def greedyCluster(self, img, number_of_iterations):
        self.random_initialization_flag = True
        best_segment_img = self.cluster(img)
        worst_segment_img = copy.deepcopy(best_segment_img)
        best_cluster_cost = worst_cluster_cost = self.objectiveCostFunction()
        curr_iteration = 0
        while curr_iteration < number_of_iterations:
            print()
            print('Current iteration '+ str(curr_iteration))
            curr_segment_img = self.cluster(img)
            curr_cost = self.objectiveCostFunction()
            if curr_cost < best_cluster_cost:
                best_segment_img = curr_segment_img
                best_cluster_cost = curr_cost
            if curr_cost > worst_cluster_cost:
                worst_segment_img = curr_segment_img
                worst_cluster_cost = curr_cost
            curr_iteration = curr_iteration + 1
        print('Worst Case Cost = ' + str(worst_cluster_cost))
        print('Best Case Cost = ' + str(best_cluster_cost))
        return best_segment_img, worst_segment_img

    '''
    Assigns cluster id to each pixel based on which centroid the pixel is closest to
    '''

    def findCorrectCluster(self):
        euclid_dist_from_cluster = []
        for k in range(self.number_of_clusters):
            kth_cluster_centroid = self.centroid[:, [k]]
            euclid_dist_from_cluster.append(np.linalg.norm(self.img - kth_cluster_centroid, axis=0))
        self.current_cluster_id = np.argmin(np.array(euclid_dist_from_cluster), axis=0)

    '''
    Update centroids after cluster assignment, new centroids is just the mean of all the intensities
    within a cluster
    '''

    def updateCentroids(self):
        for k in range(self.number_of_clusters):
            new_centroid = np.mean(self.img[:, self.current_cluster_id == k], axis=1)
            if np.isnan(new_centroid[0]):
                new_centroid = np.random.randint(0, 255, (3,))
            self.centroid[:, k] = np.round(new_centroid)

'''
Find random and predefined clustering outputs
'''
k_random_centroid = KMeansCluster(3, 5, True)
k_predefined_centroid = KMeansCluster(3, 5, False, [255, 0, 0], [0, 0, 0], [255, 255, 255])

img_fname = 'flower.png'
img_name = img_fname.split('.')[0]


img = cv2.imread(img_fname)

segmented_img_random = k_random_centroid.cluster(copy.deepcopy(img))
segmented_img_predefined = k_predefined_centroid.cluster(img)

cv2.imwrite(img_name + '_segmented_random.png', segmented_img_random)
cv2.imwrite(img_name + '_segmented_predefined.png', segmented_img_predefined)


'''
Finds the best cluster option based on greedy clustering
'''
k = KMeansCluster(number_of_clusters=3)
best_segment_img, worst_segment_img = k.greedyCluster(img, 30)
cv2.imwrite(img_name + '_greedy_worst.png', worst_segment_img)
cv2.imwrite(img_name + '_greedy_best.png', best_segment_img)


print(np.unique(best_segment_img.reshape(k.n_rows * k.n_cols, k.num_channels).T, axis = 1))
# print(np.unique(worst_segment_img.reshape(k.n_rows * k.n_cols, k.num_channels).T, axis = 1))