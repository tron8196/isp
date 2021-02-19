import numpy as np
import cv2
import time

'''
Matches a given source images shape to the shape of the target image, assumption:- Target image shape > source image.
This is done by padding zeros
'''


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


def bilinearTransform(source_pixel_location):
    pixel_row_pos = source_pixel_location[0]
    pixel_col_pos = source_pixel_location[1]
    if (not ((0 <= int(np.floor(pixel_row_pos)) < target_matched_source_img.shape[0]) and
             (0 <= int(np.floor(pixel_col_pos)) < target_matched_source_img.shape[1]))):
        return 0
    bounding_row_val = int(np.floor(pixel_row_pos))
    bounding_col_val = int(np.floor(pixel_col_pos))

    a = pixel_row_pos - bounding_row_val
    b = pixel_col_pos - bounding_col_val

    row_factor = 1 if bounding_row_val < (n_rows - 1) else -1
    col_factor = 1 if bounding_col_val < (n_cols - 1) else -1

    newPixelIntensity = (1 - a) * (1 - b) * target_matched_source_img[bounding_row_val, bounding_col_val] + \
                        (1 - a) * b * target_matched_source_img[bounding_row_val, bounding_col_val + col_factor] + \
                        a * (1 - b) * target_matched_source_img[bounding_row_val + row_factor, bounding_col_val] + \
                        a * b * target_matched_source_img[bounding_row_val + row_factor, bounding_col_val + col_factor]

    return np.round(newPixelIntensity, 0).astype(int)


'''
Returns a 2-D ndarray of n_rows*n_cols centered on top left corner.
'''


def returnPixelLocationArray(image_shape):
    n_rows = image_shape[0]
    n_cols = image_shape[1]
    return np.asarray([[i, j] for i in range(0, n_rows) for j in range(0, n_cols)])


# Read the source and target image
source_img = cv2.imread('IMG1.png', 0)
target_img = cv2.imread('IMG2.png', 0)

'''
match the source image to target image, now all further transformations to be applied to source 
image will be applied on this
'''
target_matched_source_img = matchImageSize(source_img, target_img.shape)

# The two point correspondence
source_pixel_location_list = np.array([[29, 124], [157, 372]])
target_pixel_location_list = np.array([[93, 248], [328, 399]])

'''
Generate the 4*4 matrix as shown in the report to determine the unknowns
'''

transformation_matrix = np.array([[source_pixel_location_list[0][0], source_pixel_location_list[0][1], 1, 0],
                                  [source_pixel_location_list[0][1], -source_pixel_location_list[0][0], 0, 1],
                                  [source_pixel_location_list[1][0], source_pixel_location_list[1][1], 1, 0],
                                  [source_pixel_location_list[1][1], -source_pixel_location_list[1][0], 0, 1]])

target_location_column_vector = np.array([[target_pixel_location_list[0][0], target_pixel_location_list[0][1],
                                           target_pixel_location_list[1][0], target_pixel_location_list[1][1]]]).T

# This will give a 4*1 column vector giving values for amount of translation and rotation
variable_vector = np.matmul(np.linalg.inv(transformation_matrix), target_location_column_vector)

t_x = variable_vector[2, 0]
t_y = variable_vector[3, 0]
cos_t = variable_vector[0, 0]
sin_t = variable_vector[1, 0]

rotation_matrix = np.array([[cos_t, sin_t],
                            [-sin_t, cos_t]])

translate_matrix = np.array([[t_x], [t_y]])

n_rows = target_img.shape[0]
n_cols = target_img.shape[1]
'''
Now we will use the determined amount of rotation and translation to transform IMG1 so that we can match
the images taken from differing viewpoints.
'''
target_image = np.zeros([n_rows, n_cols])
target_pixel_location_array = returnPixelLocationArray(target_img.shape)

'''
using the equation shown in the report we generate corresponding source points to get the intensity values
to be put in the particular target location(Target to Source Mapping)
'''
diff_target_pixel_location_array = target_pixel_location_array - translate_matrix.T
source_pixel_location_array = np.matmul(np.linalg.inv(rotation_matrix), diff_target_pixel_location_array.T)

# We use Bilinear Interpolation technique to calculate the appropriate intensity value

target_image = np.apply_along_axis(bilinearTransform, 1, source_pixel_location_array.T).reshape(
    target_matched_source_img.shape)
cv2.imwrite('transformed_source.png', target_image)

'''
Now since we have manipulated IMG1 w.r.t the rotation and translation and now since the tow images correspond 
to the same view point a simple differnce will let us know if anythig has changed in the scene
'''
diff_image = np.abs(target_img - target_image)
cv2.imwrite('source_target_diff.png', diff_image)
