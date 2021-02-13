import numpy as np
import cv2


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


def returnPixelLocationArray(image_shape):
    n_rows = image_shape[0]
    n_cols = image_shape[1]
    return np.asarray([[i, j] for i in range(0, n_rows) for j in range(0, n_cols)])


source_img = cv2.imread('pisa_rotated.png', 0)

target_pixel_location_array = returnPixelLocationArray(source_img.shape)

t_x = 5.84
t_y = 155

translate_matrix = np.array([[t_x], [t_y]])


n_rows = source_img.shape[0]
n_cols = source_img.shape[1]

target_image = np.zeros([n_rows, n_cols])

for index, target_pixel_location in enumerate(target_pixel_location_array):
    source_pixel_location = (np.asarray([target_pixel_location]).T - translate_matrix).T
    mapped_intensity_val = bilinearTransform(SourcePixel(source_pixel_location[0], source_img))
    target_image[
        int(target_pixel_location[0]), int(target_pixel_location[1])] = mapped_intensity_val

cv2.imwrite('lena_translated.png', target_image)