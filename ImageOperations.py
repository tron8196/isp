import numpy as np
import cv2
from Filters import getGaussianKernel


def bilinearTransform(source_pixel_location, source_image):
    pixel_row_pos = source_pixel_location[0]
    pixel_col_pos = source_pixel_location[1]
    n_rows, n_cols = source_image.shape
    if (not ((0 <= int(np.floor(pixel_row_pos)) < n_rows) and
             (0 <= int(np.floor(pixel_col_pos)) < n_cols))):
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
To Convolve one would require to pad the source image with zeros so that boundary pixels can be handled
Both source_img and kernel are 2D ndarray, kernel is a NxN matrix
'''


def getNeighbourhood(center_pixel_location, neighbourhood_size):
    half_neighbourhood_size = neighbourhood_size // 2
    neighbourhood_list = []
    for curr_row_index in range(center_pixel_location[0] - half_neighbourhood_size,
                                center_pixel_location[0] + half_neighbourhood_size + 1):
        for curr_col_index in range(center_pixel_location[1] - half_neighbourhood_size,
                                    center_pixel_location[1] + half_neighbourhood_size + 1):
            neighbourhood_list.append([curr_row_index, curr_col_index])
    return np.array(neighbourhood_list).reshape([neighbourhood_size, neighbourhood_size, 2])


def getCorrectKernelSize(std_dev):
    kernel_size = round(6*std_dev + 1)
    return kernel_size + 1 if kernel_size%2 == 0 else kernel_size





def convolveImageWithKernel(source_img, kernel):
    block_radius = kernel.shape[0]
    half_block_radius = block_radius // 2
    n_rows, n_cols = source_img.shape
    n_rows_to_pad = n_cols_to_pad = block_radius // 2
    padded_source_img = np.zeros([n_rows + n_rows_to_pad * 2, n_cols + n_cols_to_pad * 2])
    padded_source_img[block_radius // 2:n_rows + block_radius // 2,
    block_radius // 2:n_cols + block_radius // 2] = source_img
    # padded_source_img[n_rows_kernel: n_rows + 2*n_rows_kernel + 1, n_cols_kernel: n_cols + n_cols_kernel+1] = source_img
    smooth_source_image = np.zeros(padded_source_img.shape)

    '''
    Starting the convolution with the pixel that was at the origin in the original source image and ending 
    at the last pixel of the original image
    '''
    for curr_row_index in range(half_block_radius, n_rows + half_block_radius):
        for curr_col_index in range(half_block_radius, n_cols + half_block_radius):
            curr_pixel_window = padded_source_img[curr_row_index - n_rows_to_pad: curr_row_index + n_rows_to_pad + 1,
                                curr_col_index - n_cols_to_pad: curr_col_index + n_cols_to_pad + 1]
            modified_pixel_intensity = np.sum(np.multiply(curr_pixel_window, kernel))
            smooth_source_image[curr_row_index, curr_col_index] = modified_pixel_intensity
    return smooth_source_image[half_block_radius:n_rows + half_block_radius,
           half_block_radius:n_cols + half_block_radius]