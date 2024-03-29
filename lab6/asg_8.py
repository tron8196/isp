import scipy.io
import numpy as np
import matplotlib.pyplot as plt

# y=image_maximums, specific to what's being asked from the function, x current location
# max rows that one can go upwards from current pixel location
max_up = lambda x: x if x > 0 else 0

# max rows that one can go downwards from current pixel location
max_down = lambda x, y: y - 1 - x if x < y - 1 else 0

# max rows that one can go left from current pixel location
max_left = lambda x: x if x > 0 else 0

# max rows that one can go right from current pixel location
max_right = lambda x, y: y - 1 - x if x < y - 1 else 0


class ShapeFromFocus:
    n_rows = n_cols = num_frames = 0
    frame_array = None
    ml_array = None
    sml_array = None

    def __init__(self, delta_d, neighbourhood=1):
        self.neighbourhood = neighbourhood
        self.delta_d = delta_d

    '''
    This functions loads the data from the .mat file and stores it in 3D array
    '''
    def loadFrameArray(self, matFilePath):
        mat = scipy.io.loadmat(matFilePath)
        self.num_frames = mat['numframes'][0, 0]
        self.n_rows, self.n_cols = mat['frame001'].shape
        self.frame_array = np.zeros([self.n_rows, self.n_cols, self.num_frames])
        for curr_frame_index in range(1, self.num_frames + 1):
            frame_var_str = 'frame'
            frame_var_str = frame_var_str + (3 - len(str(curr_frame_index))) * '0' + str(curr_frame_index)
            self.frame_array[:, :, curr_frame_index - 1] = mat[frame_var_str]

    '''
    Does the ML of adding absolute double derivatives in X and Y
    '''
    def loadMLArray(self):
        frame_array = self.frame_array
        pixel_val_to_right_of_curr_location = np.zeros(frame_array.shape)
        pixel_val_to_right_of_curr_location[:, :-1, :] = frame_array[:, 1:, :]

        pixel_val_to_left_of_curr_location = np.zeros(frame_array.shape)
        pixel_val_to_left_of_curr_location[:, 1:, :] = frame_array[:, :-1, :]

        pixel_val_above_curr_location = np.zeros(frame_array.shape)
        pixel_val_above_curr_location[1:, :, :] = frame_array[:-1, :, :]

        pixel_val_below_curr_location = np.zeros(frame_array.shape)
        pixel_val_below_curr_location[:-1, :, :] = frame_array[1:, :, :]

        horizontal_diff = pixel_val_to_right_of_curr_location + pixel_val_to_left_of_curr_location - 2 * frame_array
        vertical_diff = pixel_val_above_curr_location + pixel_val_below_curr_location - 2 * frame_array

        self.ml_array = np.absolute(horizontal_diff) + np.absolute(vertical_diff)

    '''
    Returns Image window around the current pixel location as determined by the neighbourhood    
    '''
    def getImageWindows(self, source_img, curr_row_index, curr_col_index, kernel_size):
        adaptive_kernel = np.zeros([kernel_size, kernel_size])
        half_kernel_size = kernel_size // 2

        current_image_window = np.zeros([kernel_size, kernel_size])
        '''
        Find the min and max values that can be allowed to move from current pixel location and perform averaging
        operation only on that area, also the kernel now needs to be normalized with this subset neighbourhood    
        '''
        max_left_val = max_left(curr_col_index)
        max_up_val = max_up(curr_row_index)
        max_right_val = max_right(curr_col_index, self.n_cols)
        max_down_val = max_down(curr_row_index, self.n_rows)

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
            = source_img[
              curr_row_index - (half_kernel_size - min_row): curr_row_index + (max_row - half_kernel_size) + 1,
              curr_col_index - (half_kernel_size - min_col): curr_col_index + (max_col - half_kernel_size) + 1]

        # print(current_image_window)
        return current_image_window

    '''
    Uses the ML array to sum up values in the neighbourhood
    '''
    def loadSMLArray(self):
        ml_array = self.ml_array
        self.sml_array = np.zeros(self.ml_array.shape)
        for frame_index in range(self.num_frames):
            current_img = ml_array[:, :, frame_index].astype(np.float)
            for row_index in range(self.n_rows):
                for col_index in range(self.n_cols):
                    region = self.getImageWindows(current_img, row_index, col_index, 2 * self.neighbourhood + 1)
                    self.sml_array[row_index, col_index, frame_index] = np.sum(region)


    '''
    Use the gaussian interpolation to calculate d_bar based on three values, m(maxima), m-1, and m+1
    '''
    def get_d_bar(self, f_m, f_m_plus_one, f_m_minus_one, d_m, d_m_plus_one, d_m_minus_one):
        numerator = (np.log(f_m) - np.log(f_m_plus_one))*(d_m**2 - d_m_minus_one**2) - (np.log(f_m) - np.log(f_m_minus_one))*(d_m**2 - d_m_plus_one**2)
        denominator = 2 * self.delta_d * (2 * np.log(f_m) - np.log(f_m_plus_one) - np.log(f_m_minus_one))

        return np.round(numerator/denominator, 3)

    '''
    The main calling function
    '''
    def getGaussianInterpolatedDepthMap(self, matFilePath):
        self.loadFrameArray(matFilePath)
        self.loadMLArray()
        self.loadSMLArray()
        depth_map = np.zeros((self.n_rows, self.n_cols))
        for curr_row in range(self.n_rows):
            for curr_col in range(self.n_cols):
                curr_pixel_sml_arr = self.sml_array[curr_row, curr_col, :]
                f_m = curr_pixel_sml_arr.max()
                d_m = curr_pixel_sml_arr.argmax()
                # print(f_m, d_m)
                f_m_minus_one = curr_pixel_sml_arr[d_m - 1 if d_m - 1 >= 0 else d_m + 1]
                f_m_plus_one = curr_pixel_sml_arr[d_m + 1 if d_m + 1 < 0 else d_m - 1]

                d_m_minus_one = d_m - 1
                d_m_plus_one = d_m + 1
                dd = self.delta_d
                d_bar = self.get_d_bar(f_m, f_m_plus_one, f_m_minus_one, d_m*dd, d_m_plus_one*dd, d_m_minus_one*dd)
                depth_map[curr_row, curr_col] = d_bar
        return depth_map


'''
Parameters
delta_d - The value at which successive frames were captured
neighbourhood - Size of the  neighbourhood to be taken into account to calculate the SML value at the pixel 
'''
a = ShapeFromFocus(50.50, neighbourhood=3)
gauss_depth_map = a.getGaussianInterpolatedDepthMap('stack.mat')


'''
The below code plots a 3D surface once the depth is determined at every pixel location
'''
n_rows, n_cols = gauss_depth_map.shape
x = range(n_rows)
y = range(n_cols)

X, Y = np.meshgrid(x, y)  # `plot_surface` expects `x` and `y` data to be 2D
hf = plt.figure()

ha = hf.add_subplot(111, projection='3d')
ha.plot_surface(X, Y, gauss_depth_map)

plt.show()
