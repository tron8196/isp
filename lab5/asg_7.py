import numpy as np
import cv2
import matplotlib.pyplot as plt
'''
Calculate an NxN 2D array containing std_deviation for each pixel location, now a kernel of (6*sigma+1)x(6*sigma+1) will
be used to convolve that patch.
'''


'''
The below three functions calculate the Gaussian 2D RV for given mean and std deviation, they will be used to create the
gaussian kernel
'''
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
Because the kernel size will vary borders have to be handled carefully, the below function carefully forms a window
around a pixel such that we don't access pixel values outside the image array. Eg:- For window at pixel location with
deviation 0.5, based on kernel size (6*sigma+1)*(6*sigma+1) we will form a 5x5 kernel but because values to the top
and left of the pixel are not present we will consider only the 3*3 pixel locations, also because we will be getting
a gaussian kernel of 5x5 we will normalize it by dividing the sum of the 3x3 kernel 
'''

# y=image_maximums, specific to what's being asked from the function, x current location
# max rows that one can go upwards from current pixel location
max_up = lambda x: x if x > 0 else 0

# max rows that one can go downwards from current pixel location
max_down = lambda x, y: y - 1 - x if x < y - 1 else 0

# max rows that one can go left from current pixel location
max_left = lambda x: x if x > 0 else 0

# max rows that one can go right from current pixel location
max_right = lambda x, y: y - 1 - x if x < y - 1 else 0

def getCorrectKernelSize(std_dev):
    kernel_size = round(6 * std_dev + 1)
    return kernel_size + 1 if kernel_size % 2 == 0 else kernel_size




def getFilterAndImageWindows(source_img, curr_row_index, curr_col_index, std_dev):
    n_rows, n_cols = source_img.shape
    kernel_size = getCorrectKernelSize(std_dev)
    adaptive_kernel = np.zeros([kernel_size, kernel_size])
    half_kernel_size = kernel_size//2

    current_image_window = np.zeros([kernel_size, kernel_size])
    full_size_kernel = getGaussianKernel(kernel_size, std_dev)
    '''
    Find the min and max values that can be allowed to move from current pixel location and perform averaging
    operation only on that area, also the kernel now needs to be normalized with this subset neighbourhood    
    '''
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

    adaptive_kernel[min_row: max_row + 1, min_col: max_col + 1] = full_size_kernel[min_row: max_row + 1,
                                                                  min_col: max_col + 1]

    adaptive_kernel = adaptive_kernel / np.sum(adaptive_kernel)

    return current_image_window, adaptive_kernel


'''
This function receives a source image and a standard deviation array, this iterates over every pixel location in the
image and gets the specified std deviation value from the std deviation array
'''
def convolveImageWithAdaptiveKernel(source_img, std_dev_array):
    #get std_deciation and determine kernel value
    blurred_img = np.zeros(source_img.shape)
    n_rows, n_cols = source_img.shape
    # std_dev = 1.6
    # kernel_size = getCorrectKernelSize(std_dev)
    for curr_row_index in range(n_rows):
        for curr_col_index in range(n_cols):
            std_dev = std_dev_array[curr_row_index, curr_col_index]
            current_img_patch, current_filter_patch = getFilterAndImageWindows(source_img, curr_row_index,
                                                                               curr_col_index, std_dev)
            modified_pixel_intensity = np.sum(np.multiply(current_img_patch, current_filter_patch))
            blurred_img[curr_row_index, curr_col_index] = modified_pixel_intensity
    return blurred_img

'''
This function gets source image and a standard deviation value, which is applied globally to every pixel location
'''
def convolveImageWithFixedKernel(source_img, std_dev):
    #get std_deciation and determine kernel value
    blurred_img = np.zeros(source_img.shape)
    n_rows, n_cols = source_img.shape
    for curr_row_index in range(n_rows):
        for curr_col_index in range(n_cols):
            current_img_patch, current_filter_patch = getFilterAndImageWindows(source_img, curr_row_index,
                                                                               curr_col_index, std_dev)
            modified_pixel_intensity = np.sum(np.multiply(current_img_patch, current_filter_patch))
            blurred_img[curr_row_index, curr_col_index] = modified_pixel_intensity
    return blurred_img










'''
This is the implementation of the function to generate std deviation for every pixel location in the image
'''

def getStdDevArray(img_shape):
    n_rows, n_cols = img_shape
    std_dev_array = np.zeros(img_shape).astype(np.float)
    for curr_row_index in range(n_rows):
        for curr_col_index in range(n_cols):
            x_val = np.square(curr_row_index - n_rows/2)
            y_val = np.square(curr_col_index - n_cols/2)
            denominator = np.square(n_rows)
            val = np.round(2*np.exp(-(10.59*(x_val + y_val))/denominator), 5)
            std_dev_array[curr_row_index, curr_col_index] = val
    return std_dev_array



#Part 1
img = cv2.imread('Globe.png', 0)
std_dev_array = getStdDevArray(img.shape)
blurred_globe = convolveImageWithAdaptiveKernel(img, std_dev_array)
cv2.imwrite('blurred_globe.png', blurred_globe)


'''
PLot Showing the distribution of the standard deviation array
'''
fig = plt.figure(figsize=(6, 3.2))

ax = fig.add_subplot(111)
ax.set_title('colorMap')
plt.imshow(std_dev_array)
ax.set_aspect('equal')

cax = fig.add_axes([0.12, 0.1, 0.78, 0.8])
cax.get_xaxis().set_visible(False)
cax.get_yaxis().set_visible(False)
cax.patch.set_alpha(0)
cax.set_frame_on(False)
plt.colorbar(orientation='vertical')
plt.show()




#Part 2
img = cv2.imread('Nautilus.png', 0)
# std_dev_array = getStdDevArray(img.shape)
std_dev_array = np.ones(img.shape)
blurred_img_adaptive = convolveImageWithAdaptiveKernel(img, std_dev_array)
cv2.imwrite('blurred_img_adaptive.png', blurred_img_adaptive)
blurred_img_fixed = convolveImageWithFixedKernel(img, std_dev=1)
cv2.imwrite('blurred_img_fixed.png', blurred_img_fixed)
#
difference_img = np.absolute(blurred_img_adaptive - blurred_img_fixed)

print('Difference Image Stats')
print('Max Difference = ', np.max(difference_img))
print('Min Difference = ', np.min(difference_img))
print('Average Difference = ', np.average(difference_img))

