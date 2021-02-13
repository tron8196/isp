import numpy as np
import cv2

scale_factor = 1.3

# scale_factor = 0.8
source_img = cv2.imread('cells_scale.png', 0)


class SourcePixel:
    def __init__(self, mappedSourceLocation, sourceImage):
        self.mappedSourceLocation = mappedSourceLocation
        self.sourceImage = sourceImage


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


'''
This function returns a numpy array with origin at the center of the image
'''


def returnImageCenteredArray(image_shape):
    n_rows = image_shape[0]
    n_cols = image_shape[1]
    return np.asarray([[i, j] for i in range((-n_rows) // 2 + 1, n_rows // 2 + 1) for j in
                       range((-n_cols) // 2 + 1, n_cols // 2 + 1)])


''''
To form the pixel location matrix w.r.t the center of image for a given pixel location in X direction in the original
space say x we do the following x' = (x - N/2 + 1) to get the pixel locations with the origin being at the image center.
To go back to the original pixel locations with a given x = x' + N/2 -1.
Note: We will have to change N for x and y if rows and columns are different
'''


def remap_to_original_coord(pixel_coord, n_rows, n_cols):
    new_pixel_coord = np.zeros([1, 2])
    new_pixel_coord[0, 0] = pixel_coord[0, 0] + (n_rows // 2) - 1
    new_pixel_coord[0, 1] = pixel_coord[0, 1] + (n_cols // 2) - 1
    return new_pixel_coord


'''
Code Begins from here, commented wherever required
'''

'''
This code scales any given image by the scaling factor. the scaling is performed w.r.t the center of the image
'''

scaling_matrix = np.array([[scale_factor, 0], [0, scale_factor]])
scaling_matrix_inverse = np.linalg.inv(scaling_matrix)

target_pixel_location_array = returnImageCenteredArray(source_img.shape)

n_rows = source_img.shape[0]
n_cols = source_img.shape[1]

target_image = np.zeros([n_rows, n_cols])

for index, target_pixel_location in enumerate(target_pixel_location_array):
    mapped_source_pixel_location = np.matmul(scaling_matrix_inverse, np.asarray([target_pixel_location]).T)
    mapped_target_pixel_location = remap_to_original_coord(np.asarray([target_pixel_location]), n_rows, n_cols)
    mapped_intensity_val = bilinearTransform(
        SourcePixel(remap_to_original_coord(mapped_source_pixel_location.T, n_rows, n_cols)[0],
                    source_img))
    target_image[
        int(mapped_target_pixel_location[0, 0]), int(mapped_target_pixel_location[0, 1])] = mapped_intensity_val

cv2.imwrite('cells_scaled.png', target_image)
