import operator

import cv2
import numpy as np
import random as rnd
from scipy.linalg import null_space


class FourPointFeature:
    def __init__(self, random_index_list, consensus_val, test_homography_matrix):
        """
        :type random_index_list: list
        """
        self.test_homography_matrix = test_homography_matrix
        self.consensus_val = consensus_val
        self.random_index_list = random_index_list


'''
Takes images as input and outputs tuples of source and destination points with matching features
'''


def bilinearTransform(source_pixel_location, source_image):
    pixel_row_pos = source_pixel_location[0]
    pixel_col_pos = source_pixel_location[1]
    n_rows, n_cols = source_image.shape
    if (not ((0 <= int(np.floor(pixel_row_pos)) < source_image.shape[0]) and
             (0 <= int(np.floor(pixel_col_pos)) < source_image.shape[1]))):
        return 0
    bounding_row_val = int(np.floor(pixel_row_pos))
    bounding_col_val = int(np.floor(pixel_col_pos))

    a = pixel_row_pos - bounding_row_val
    b = pixel_col_pos - bounding_col_val

    row_factor = 1 if bounding_row_val < (n_rows - 1) else -1
    col_factor = 1 if bounding_col_val < (n_cols - 1) else -1

    newPixelIntensity = (1 - a) * (1 - b) * source_image[bounding_row_val, bounding_col_val] + \
                        (1 - a) * b * source_image[bounding_row_val, bounding_col_val + col_factor] + \
                        a * (1 - b) * source_image[bounding_row_val + row_factor, bounding_col_val] + \
                        a * b * source_image[bounding_row_val + row_factor, bounding_col_val + col_factor]
    return np.round(newPixelIntensity, 0).astype(int)



'''
Selects random 4-point correspondences and returns the A matrix used to calculate homography
'''


def generateAMatrix(source_point_list, target_point_list):
    random_index_list = [rnd.randint(0, source_point_list.shape[0] - 1) for i in range(4)]
    A_matrix = np.zeros([8, 9])

    for index, random_index in enumerate(random_index_list):
        A_matrix[2 * index, 0] = source_point_list[random_index, 0]
        A_matrix[2 * index, 1] = source_point_list[random_index, 1]
        A_matrix[2 * index, 2] = 1
        A_matrix[2 * index, 6] = -source_point_list[random_index, 0] * target_point_list[random_index, 0]
        A_matrix[2 * index, 7] = -source_point_list[random_index, 1] * target_point_list[random_index, 0]
        A_matrix[2 * index, 8] = -target_point_list[random_index, 0]

        A_matrix[2 * index + 1, 3] = source_point_list[random_index, 0]
        A_matrix[2 * index + 1, 4] = source_point_list[random_index, 1]
        A_matrix[2 * index + 1, 5] = 1
        A_matrix[2 * index + 1, 6] = -source_point_list[random_index, 0] * target_point_list[random_index, 1]
        A_matrix[2 * index + 1, 7] = -source_point_list[random_index, 1] * target_point_list[random_index, 1]
        A_matrix[2 * index + 1, 8] = -target_point_list[random_index, 1]
    return A_matrix, random_index_list


def testHomography(distance_threshold, random_index_list, source_features, target_features, test_homography_matrix):
    count = 0
    total_test_points = source_features.shape[0]
    for index, source_target_feature_point in enumerate(zip(source_features, target_features)):
        if index in random_index_list:
            pass
        else:
            source_homogeneous_point = np.hstack((np.array([source_target_feature_point[0]]), np.array([[1]])))
            predicted_target_homogeneous_point = np.matmul(test_homography_matrix, source_homogeneous_point.T)
            predicted_target_homogeneous_point = predicted_target_homogeneous_point / \
                                                 predicted_target_homogeneous_point[2, 0]
            target_predicted = np.array([predicted_target_homogeneous_point.T[0, :2]])
            target_actual = source_target_feature_point[1]
            euclid_norm = np.linalg.norm(target_actual - target_predicted)
            if euclid_norm <= distance_threshold:
                count = count + 1
    return count / total_test_points * 100 >= 95, count / total_test_points


def siftFeatureMatchMatlab(correspondanceBetweenFlag):

    source_feature_file_name = 'source_features_' + correspondanceBetweenFlag + '.txt'
    target_feature_file_name = 'target_features_' + correspondanceBetweenFlag + '.txt'

    src = open(source_feature_file_name, 'r')
    trg = open(target_feature_file_name, 'r')

    src = src.readlines()
    trg = trg.readlines()

    src_features = [i.strip().split() for i in src]
    src_features = np.array([[float(location[0]), float(location[1])] for location in src_features])

    trg_features = [i.strip().split() for i in trg]
    trg_features = np.array([[float(location[0]), float(location[1])] for location in trg_features])
    return src_features, trg_features


def getHomographyMatrix(source_image, target_image, correspondanceBetweenFlag, MAX_ITERATIONS=100):
    source_features, target_features = siftFeatureMatchMatlab(correspondanceBetweenFlag)
    iterations = 0
    while iterations <= MAX_ITERATIONS:
        test_A_matrix, random_index_list = generateAMatrix(source_features, target_features)
        test_homography_matrix = null_space(test_A_matrix)[:, 0].reshape([3, 3])
        is_valid_homography, accuracy = testHomography(10, random_index_list, source_features, target_features,
                                                       test_homography_matrix)
        print(is_valid_homography, accuracy)
        if is_valid_homography:
            break
        MAX_ITERATIONS = MAX_ITERATIONS + 1
    return test_homography_matrix


def blendValues(intensity_from_img_1, intensity_from_img_2, intensity_from_img_3):
    if intensity_from_img_2 == 0:
        if intensity_from_img_1 == 0:
            return intensity_from_img_3
        elif intensity_from_img_3 == 0:
            return intensity_from_img_1
        else:
            return (intensity_from_img_1 + intensity_from_img_3) // 2
    else:
        return intensity_from_img_2


#
img1 = cv2.imread('img1.png', 0)
img2 = cv2.imread('img2.png', 0)
img3 = cv2.imread('img3.png', 0)

homography_matrix_2_1 = getHomographyMatrix(img2, img1, '12')
homography_matrix_2_3 = getHomographyMatrix(img2, img3, '23')
homography_matrix_2_2 = np.identity(3)

canvas_rows = int(img2.shape[0] * 2)
canvas_cols = int(img2.shape[1] * 2)

canvas_img = np.zeros([canvas_rows, canvas_cols])
canvas_pixel_location_homogeneous_array = np.hstack(
    (canvas_img, np.ones([canvas_img.shape[0], 1])))

row_offset = img2.shape[0] // 2
col_offset = img2.shape[1] // 2
for row_index in range(canvas_rows):
    for col_index in range(canvas_cols):
        current_pixel_location = np.array([[row_index - row_offset, col_index - col_offset, 1]])

        img1_location = np.matmul(homography_matrix_2_1, current_pixel_location.T).T
        img1_location = img1_location / np.array([img1_location[:, 2]]).T
        intensity_from_img_1 = bilinearTransform(img1_location[0], img1)

        img2_location = np.matmul(homography_matrix_2_2, current_pixel_location.T).T
        img2_location = img2_location / np.array([img2_location[:, 2]]).T
        intensity_from_img_2 = bilinearTransform(img2_location[0], img2)

        img3_location = np.matmul(homography_matrix_2_3, current_pixel_location.T).T
        img3_location = img3_location / np.array([img3_location[:, 2]]).T
        intensity_from_img_3 = bilinearTransform(img3_location[0], img3)
        canvas_img[row_index, col_index] = blendValues(intensity_from_img_1, intensity_from_img_2, intensity_from_img_3)
cv2.imwrite('canvas.png', canvas_img)
