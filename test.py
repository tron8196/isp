import numpy as np

img_array = np.zeros([50,50])

location_index = np.random.randint(0,10, 100).reshape([50,2])









# target_pixel_intensity_array = np.ones([source_pixel_location_array.shape[0], 1])
#     pixel_row_pos = source_pixel_location_array[:, 0]
#     pixel_col_pos = source_pixel_location_array[:, 1]
#     bounding_row_val = np.floor(pixel_row_pos).astype(int)
#     bounding_col_val = np.floor(pixel_col_pos).astype(int)
#     n_rows, n_cols = source_img.shape
#     bad_rows_to_delete_index = np.logical_or(source_pixel_location_array[:, 0]<0, source_pixel_location_array[:, 0] >= n_rows - 2)
#     bad_cols_to_delete_index = np.logical_or(source_pixel_location_array[:, 1]<0, source_pixel_location_array[:, 1] >= n_cols - 2)
#     data_points_to_be_deleted_index = np.logical_not(np.logical_or(bad_rows_to_delete_index, bad_cols_to_delete_index))
#     source_pixel_location_array = source_pixel_location_array[data_points_to_be_deleted_index]
#     target_pixel_intensity_array[data_points_to_be_deleted_index] = 0
#
#     pixel_row_pos = source_pixel_location_array[:, 0]
#     pixel_col_pos = source_pixel_location_array[:, 1]
#     bounding_row_val = np.floor(pixel_row_pos).astype(int)
#     bounding_col_val = np.floor(pixel_col_pos).astype(int)
#     a = pixel_row_pos - bounding_row_val
#     b = pixel_col_pos - bounding_col_val
#     print('----------------------------')
#     print(np.floor(pixel_row_pos))
#     linear_img = (1 - a) * (1 - b) * source_img[bounding_row_val, bounding_col_val] + \
#     (1 - a) * b * source_img[bounding_row_val, bounding_col_val + 1] + \
#     a * (1 - b) * source_img[bounding_row_val + 1, bounding_col_val] + \
#     a * b * source_img[bounding_row_val + 1, bounding_col_val + 1]
#
#     return linear_img.reshape(source_img.shape)
