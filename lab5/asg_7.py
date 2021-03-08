import numpy as np
import cv2


def getNormXGaussian(x, mu_x, std_x):
    return np.float((x - mu_x) / std_x) if std_x != 0 else 0


def getNormYGaussian(y, mu_y, std_y):
    return np.float((y - mu_y) / std_y) if std_y != 0 else 0


def getGaussianKernel(kernel_size, std_dev, weight=1):
    gaussianKernel = np.zeros([kernel_size, kernel_size])
    kernel_row_center = kernel_col_center = kernel_size // 2
    for curr_row_index in range(kernel_size):
        for curr_col_index in range(kernel_size):
            get_x_val_exp = ((getNormXGaussian(curr_row_index, kernel_row_center, std_dev)) ** 2) / 2
            get_y_val_exp = ((getNormYGaussian(curr_col_index, kernel_col_center, std_dev)) ** 2) / 2
            gaussianKernel[curr_row_index, curr_col_index] = weight * np.exp(-(get_x_val_exp + get_y_val_exp))
    return gaussianKernel / np.sum(gaussianKernel)


'''
Calculate an NxN 2D array containing std_deviation for each pixel location, now a kernel of (6*sigma+1)x(6*sigma+1) will
be used to convolve that patch.

algo

1. find the NxN sigma matrix
2. apd the original image




'''

img = cv2.imread('../lab4/Mandrill.png', 0)
n_rows, n_cols = img.shape

# y=image_maximums, specific to what's being asked from the fuction, x current location
# max rows that one can go upwards from current pixel location
max_up = lambda x: x if x > 0 else 0

# max rows that one can go upwards from current pixel location
max_down = lambda x, y: y - 1 - x if x < y - 1 else 0

# max rows that one can go upwards from current pixel location
max_left = lambda x: x if x > 0 else 0

# max rows that one can go upwards from current pixel location
max_right = lambda x, y: y - 1 - x if x < y - 1 else 0
#
# kernel_size = 5
# half_kernel_size = kernel_size // 2

std_dev = 1.6
def getCorrectKernelSize(std_dev):
    kernel_size = round(6 * std_dev + 1)
    return kernel_size + 1 if kernel_size % 2 == 0 else kernel_size




def getCurrentWindows(curr_row_index, curr_col_index):
    pass




def convolveImageWithKernel(source_img):
    std_dev = 1.2
    kernel_size = getCorrectKernelSize(std_dev)
    half_kernel_size = kernel_size // 2
    print(kernel_size)
    blurred_img = np.zeros(source_img.shape)
    n_rows, n_cols = source_img.shape
    adpative_kernel = np.zeros([kernel_size, kernel_size])
    for curr_row_index in range(n_rows):
        for curr_col_index in range(n_cols):
            # get std_dev, hence find kernel size
            current_image_window = np.zeros([kernel_size, kernel_size])
            full_size_kernel = getGaussianKernel(kernel_size, std_dev)
            max_left_val = max_left(curr_col_index)
            max_up_val = max_up(curr_row_index)
            max_right_val = max_right(curr_col_index, n_cols)
            max_down_val = max_down(curr_row_index, n_rows)

            min_col = 0
            min_row = 0
            max_col = kernel_size - 1
            max_row = kernel_size - 1

            if max_left_val < half_kernel_size:
                min_col = half_kernel_size - max_left_val

            if max_up_val < half_kernel_size:
                min_row = half_kernel_size - max_up_val

            if max_right_val < half_kernel_size:
                max_col = half_kernel_size + max_right_val

            if max_down_val < half_kernel_size:
                max_row = half_kernel_size + max_down_val

            current_image_window[min_row: max_row + 1, min_col: max_col + 1] \
                = img[curr_row_index - (half_kernel_size - min_row): curr_row_index + (max_row - half_kernel_size) + 1,
                  curr_col_index - (half_kernel_size - min_col): curr_col_index + (max_col - half_kernel_size) + 1]

            adpative_kernel[min_row: max_row + 1, min_col: max_col + 1] = full_size_kernel[min_row: max_row + 1,
                                                                          min_col: max_col + 1]

            adpative_kernel = adpative_kernel / np.sum(adpative_kernel)
            modified_pixel_intensity = np.sum(np.multiply(current_image_window, adpative_kernel))
            blurred_img[curr_row_index, curr_col_index] = modified_pixel_intensity
    return blurred_img


cv2.imwrite('blurred.png', convolveImageWithKernel(img))
