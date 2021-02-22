import operator

import cv2
import numpy as np
import random as rnd
from scipy.linalg import null_space

'''
Takes images as input and outputs tuples of source and destination points with matching features
'''


class FourPointFeature:
    def __init__(self, random_index_list, consensus_val, test_homography_matrix):
        """

        :type random_index_list: list
        """
        self.test_homography_matrix = test_homography_matrix
        self.consensus_val = consensus_val
        self.random_index_list = random_index_list


def siftFeatureMatch(source_image, target_image):
    # Initiate SIFT detector
    sift = cv2.SIFT_create()

    # find the keypoints and descriptors with SIFT

    kp1, des1 = sift.detectAndCompute(img1, None)
    kp2, des2 = sift.detectAndCompute(img2, None)

    FLANN_INDEX_KDTREE = 0
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)

    flann = cv2.FlannBasedMatcher(index_params, search_params)

    matches = flann.knnMatch(des1, des2, k=2)

    # store all the good matches as per Lowe's ratio test.
    good = []
    for m, n in matches:
        if m.distance < 0.7 * n.distance:
            good.append(m)

    rc_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

    for point in dst_pts:
        # print(tuple(point[0].astype(int)))
        image = cv2.circle(img2, tuple(point[0]), 0, 0, 5)
    cv2.imwrite('source_feature_match.png', image)

    for point in rc_pts:
        # print(tuple(point[0]))
        image = cv2.circle(img1, tuple(point[0]), 0, 0, 5)
    cv2.imwrite('target_feature_match.png', image)
    # print(rc_pts.shape, dst_pts.shape)
    return rc_pts, dst_pts


'''
Selects random 4-point correspondences and returns the A matrix used to calculate homography
'''


def generateAMatrix(source_point_list, target_point_list):
    # print(source_point_list.shape)
    random_index_list = [rnd.randint(0, source_point_list.shape[0] - 1) for i in range(4)]
    # print(random_index_list)
    A_matrix = np.zeros([8, 9])

    for index, random_index in enumerate(random_index_list):
        # print('#######################')
        # print('source-----point', source_point_list[random_index])
        # print('target-----point', target_point_list[random_index])
        # print('************************')
        # print()

        A_matrix[2 * index, 0] = source_point_list[random_index, 0]
        A_matrix[2 * index, 1] = source_point_list[random_index, 1]
        A_matrix[2 * index, 2] = 1
        A_matrix[2 * index, 6] = -source_point_list[random_index, 0] * target_point_list[
            random_index, 0]
        A_matrix[2 * index, 7] = -source_point_list[random_index, 1] * target_point_list[
            random_index, 0]
        A_matrix[2 * index, 8] = -target_point_list[random_index, 0]

        A_matrix[2 * index + 1, 3] = -source_point_list[random_index, 0]
        A_matrix[2 * index + 1, 4] = -source_point_list[random_index, 1]
        A_matrix[2 * index + 1, 5] = -1
        A_matrix[2 * index + 1, 6] = source_point_list[random_index, 0] * \
                                     target_point_list[random_index, 1]
        A_matrix[2 * index + 1, 7] = source_point_list[random_index, 1] * \
                                     target_point_list[random_index, 1]
        A_matrix[2 * index + 1, 8] = target_point_list[random_index, 1]

    return A_matrix, random_index_list


def testHomography(distance_threshold, random_index_list, source_features, target_features, test_homography_matrix):
    count = 0
    total_test_points = source_features.shape[0]
    for index, source_target_feature_point in enumerate(zip(source_features, target_features)):
        if index in random_index_list:
            pass
        else:
            # source_homogeneous_point = np.ones([1,3])
            source_homogeneous_point = np.hstack((source_target_feature_point[0], np.array([[1]])))
            predicted_target_homogeneous_point = np.matmul(test_homography_matrix, source_homogeneous_point.T)
            # print(predicted_target_homogeneous_point.T, predicted_target_homogeneous_point[1])
            # print(predicted_target_homogeneous_point.T, source_target_feature_point[1])
            predicted_target_homogeneous_point = predicted_target_homogeneous_point / \
                                                 predicted_target_homogeneous_point[2, 0]
            # print()
            target_predicted = np.array([predicted_target_homogeneous_point.T[0, :2]])
            target_actual = source_target_feature_point[1]
            euclid_norm = np.linalg.norm(target_actual - target_predicted)
            # print(euclid_norm)
            if euclid_norm <= distance_threshold:
                count = count + 1
    # print(count/total_test_points)
    return count / total_test_points * 100 >= 80, count / total_test_points


def getHomographyMatrix(source_image, target_image, MAX_ITERATIONS=100):
    source_features, target_features = siftFeatureMatch(source_image, target_image)
    four_point_corresp_dict = {}
    iterations = 0
    while iterations <= MAX_ITERATIONS:
        test_A_matrix, random_index_list = generateAMatrix(source_features[:, -1, :], target_features[:, -1, :])
        test_homography_matrix = null_space(test_A_matrix)[:, 0].reshape([3, 3])
        is_valid_homography, accuracy = testHomography(10, random_index_list, source_features, target_features,
                                                       test_homography_matrix)
        four_point_corresp_dict[iterations] = FourPointFeature(random_index_list, accuracy, test_homography_matrix)
        iterations = iterations + 1
    iteration_index, four_point_corresp = \
        sorted(four_point_corresp_dict.items(), key=lambda value: value[1].consensus_val, reverse=True)[0]
    return four_point_corresp.test_homography_matrix


#
img1 = cv2.imread('img1.png', 0)
img2 = cv2.imread('img2.png', 0)
img3 = cv2.imread('img3.png', 0)


print(img2.shape, img3.shape, img1.shape)

canvas = np.zeros(([img1.shape[0], img1.shape[1]*3]))


homography_matrix_2_1 = getHomographyMatrix(img2, img1)
homography_matrix_2_3 = getHomographyMatrix(img2, img3)


print(homography_matrix_2_1)
print(homography_matrix_2_3)
# test_homography_matrix = homography_matrix
# random_index_list = four_point_corresp.random_index_list
# is_valid_homography, accuracy = testHomography(10)
#
# print(is_valid_homography, accuracy)
# print(homography_matrix)
# print(four_point_corresp)
