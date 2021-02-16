import numpy as np
import cv2

class SourcePixel:
    def __init__(self, mappedSourceLocation, sourceImage):
        self.mappedSourceLocation = mappedSourceLocation
        self.sourceImage = sourceImage



def matchImageSize(source_img, target_image_shape):
    target_img = np.zeros(target_image_shape)
    n_rows, n_cols = source_img.shape
    target_img[:n_rows, :n_cols] = source_img
    return target_img


'''
@@Bilinear Transform Method
Given mapped source pixel location the function returns interpolated pixel intensities averaging over the 
nearest pixel locations, note for easier code, boundary pixels have been mapped to 0. Based on boundary location
+1 or -1 factor has been chosen to calculate the neighbourhood.
'''


def bilinearTransform(sourcePixelObj: SourcePixel):
    pixel_row_pos = sourcePixelObj.mappedSourceLocation[0]
    pixel_col_pos = sourcePixelObj.mappedSourceLocation[1]
    if (not ((0 <= int(pixel_row_pos) < sourcePixelObj.sourceImage.shape[0]) and
             (0 <= int(pixel_col_pos) < sourcePixelObj.sourceImage.shape[1]))):
        return 0
    bounding_row_val = int(np.floor(pixel_row_pos))
    bounding_col_val = int(np.floor(pixel_col_pos))
    a = pixel_row_pos - bounding_row_val
    b = pixel_col_pos - bounding_col_val

    row_factor = 1 if bounding_row_val < (n_rows - 1) else -1
    col_factor = 1 if bounding_col_val < (n_cols - 1) else -1

    newPixelIntensity = (1 - a) * (1 - b) * sourcePixelObj.sourceImage[bounding_row_val, bounding_col_val] + \
                        (1 - a) * b * sourcePixelObj.sourceImage[bounding_row_val, bounding_col_val + col_factor] + \
                        a * (1 - b) * sourcePixelObj.sourceImage[bounding_row_val + row_factor, bounding_col_val] + \
                        a * b * sourcePixelObj.sourceImage[bounding_row_val + row_factor, bounding_col_val + col_factor]

    return round(newPixelIntensity)


def returnPixelLocationArray(image_shape):
    n_rows = image_shape[0]
    n_cols = image_shape[1]
    return np.asarray([[i, j] for i in range(0, n_rows) for j in range(0, n_cols)])


source_img = cv2.imread('IMG1.png', 0)
target_img = cv2.imread('IMG2.png', 0)

target_matched_source_img = matchImageSize(source_img, target_img.shape)

# cv2.imwrite('temp.png', target_matched_source_img)

source_pixel_location_list = np.array([[29, 124], [157, 372]])
target_pixel_location_list = np.array([[93, 248], [328, 399]])

transformation_matrix = np.array([[source_pixel_location_list[0][0], source_pixel_location_list[0][1], 1, 0],
                                  [source_pixel_location_list[0][1], -source_pixel_location_list[0][0], 0, 1],
                                  [source_pixel_location_list[1][0], source_pixel_location_list[1][1], 1, 0],
                                  [source_pixel_location_list[1][1], -source_pixel_location_list[1][0], 0, 1]])

# print(transformation_matrix)

target_location_column_vector = np.array([[target_pixel_location_list[0][0], target_pixel_location_list[0][1],
                                           target_pixel_location_list[1][0], target_pixel_location_list[1][1]]]).T
print()
print(transformation_matrix)
print()
print(target_location_column_vector)

radian_to_degree = lambda radian: (radian * 180) / np.pi

# print(target_location_column_vector)
variable_vector = np.matmul(np.linalg.inv(transformation_matrix), target_location_column_vector)

t_x = variable_vector[2, 0]
t_y = variable_vector[3, 0]
cos_t = variable_vector[0, 0]
sin_t = variable_vector[1, 0]
# print(variable_vector[0,0], variable_vector[1,0], t_x, t_y)


# print(cv2.findHomography(source_pixel_location_list, target_pixel_location_list))


radian_rotation = np.arctan(variable_vector[1, 0] / variable_vector[0, 0])

degree_rotation = np.round(radian_to_degree(radian_rotation))

scaling_factor = variable_vector[1, 0] / np.sin(radian_rotation)

print(scaling_factor, degree_rotation, t_x, t_y)

rotation_matrix_test = np.array([[np.cos(radian_rotation), np.sin(radian_rotation)],
                                 [-np.sin(radian_rotation), np.cos(radian_rotation)]])

rotation_matrix = np.array([[cos_t, sin_t],
                            [-sin_t, cos_t]])

translate_matrix = np.array([[t_x], [t_y]])

target_coordinate = np.matmul(rotation_matrix, source_pixel_location_list[1].T) + translate_matrix.T

n_rows = target_img.shape[0]
n_cols = target_img.shape[1]

target_image = np.zeros([n_rows, n_cols])
target_pixel_location_array = returnPixelLocationArray(target_img.shape)

'''
This piece of code, takes each pixel location from the target image and based on the geometric transform applied
calculates the corresponding source co-ordinates, these source co-ordinates are then used to find the accurate pixel
intensities for the location using  bilinear interpolation of the neighbourhood pixels.
'''
for index, target_pixel_location in enumerate(target_pixel_location_array):
    # print('*********')
    # print(target_pixel_location)
    # print('*********')

    diff_col_vector = np.array([target_pixel_location]).T - translate_matrix
    # print(diff_col_vector)
    # print('------------')
    source_pixel_location = np.matmul(np.linalg.inv(rotation_matrix), diff_col_vector)
    # print(source_pixel_location)
    # print()
    mapped_intensity_val = bilinearTransform(SourcePixel(source_pixel_location.T[0], target_matched_source_img))
    target_image[
        int(target_pixel_location[0]), int(target_pixel_location[1])] = mapped_intensity_val

cv2.imwrite('temp.png', target_image)
diff_image = np.abs(target_img - target_image)
# diff_image[np.abs(target_img - target_image) <= 250] = 0
cv2.imwrite('temp_1.png', diff_image)


