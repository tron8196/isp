import numpy as np
import cv2


def getNormXGaussian(x, mu_x, std_x):
    return np.float((x - mu_x) / std_x) if std_x != 0 else 0


def getNormYGaussian(y, mu_y, std_y):
    return np.float((y - mu_y) / std_y) if std_y != 0 else 0


def getGaussianKernel(kernelSize, std_x, std_y, weight=1):
    gaussianKernel = np.zeros([kernelSize, kernelSize])
    kernel_row_center = kernel_col_center = kernelSize // 2
    for curr_row_index in range(kernelSize):
        for curr_col_index in range(kernelSize):
            get_x_val_exp = ((getNormXGaussian(curr_row_index, kernel_row_center, std_x)) ** 2) / 2
            get_y_val_exp = ((getNormYGaussian(curr_col_index, kernel_col_center, std_y)) ** 2) / 2
            gaussianKernel[curr_row_index, curr_col_index] = weight * np.exp(-(get_x_val_exp + get_y_val_exp))
    return gaussianKernel / np.sum(gaussianKernel)


def getCorrectKernelSize(std_dev):
    kernel_size = round(6 * std_dev + 1)
    return kernel_size + 1 if kernel_size % 2 == 0 else kernel_size


'''
Starting the convolution with the pixel that was at the origin in the original source image and ending 
at the last pixel of the original image
'''


def convolveImageWithKernel(source_img, kernel):
    block_radius = kernel.shape[0]
    half_block_radius = block_radius // 2
    n_rows, n_cols = source_img.shape
    n_rows_to_pad = n_cols_to_pad = block_radius // 2
    padded_source_img = np.zeros([n_rows + n_rows_to_pad * 2, n_cols + n_cols_to_pad * 2])
    padded_source_img[block_radius // 2:n_rows + block_radius // 2,
    block_radius // 2:n_cols + block_radius // 2] = source_img
    smooth_source_image = np.zeros(padded_source_img.shape)
    for curr_row_index in range(half_block_radius, n_rows + half_block_radius):
        for curr_col_index in range(half_block_radius, n_cols + half_block_radius):
            curr_pixel_window = padded_source_img[curr_row_index - n_rows_to_pad: curr_row_index + n_rows_to_pad + 1,
                                curr_col_index - n_cols_to_pad: curr_col_index + n_cols_to_pad + 1]
            modified_pixel_intensity = np.sum(np.multiply(curr_pixel_window, kernel))
            smooth_source_image[curr_row_index, curr_col_index] = modified_pixel_intensity
    return smooth_source_image[half_block_radius:n_rows + half_block_radius,
           half_block_radius:n_cols + half_block_radius]


'''
Algorithm for Gaussian Smoothing

1. Generate a normalized Gaussian Kernel
2. pad the source image on boundary for convolution
3. Now because gaussian kernel is symmetric, Pixel by pixel averaging using the kernel, without inversion
4. Extract the image to remove the padded region 
'''
img = cv2.imread('Mandrill.png', 0)

for std_dev in [1.6, 1.2, 1, 0.6, 0.3, 0]:
    std_dev_x = std_dev_y = std_dev
    target_img = convolveImageWithKernel(img, getGaussianKernel(getCorrectKernelSize(std_dev), std_dev_x, std_dev_y))
    cv2.imwrite('Mandrill_gaussian_blur_'+str(std_dev)+'.png', target_img)
    if std_dev == 0:
        print('Checking if keeping std_deviation as zero brings any change to the image')
        print('Total Change = ', np.sum(img-target_img))