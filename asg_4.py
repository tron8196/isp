import numpy as np
import cv2

source_img = cv2.imread('IMG1.png', 0)
target_img = cv2.imread('IMG2.png', 0)

source_pixel_location_list = np.array([[29,124],[157,372]])
target_pixel_location_list = np.array([[93, 248],[328, 399]])

transformation_matrix = np.array([[source_pixel_location_list[0][0], source_pixel_location_list[0][1], 1, 0],
          [source_pixel_location_list[0][1], -source_pixel_location_list[0][0], 0, 1],
          [source_pixel_location_list[1][0], source_pixel_location_list[1][1], 1, 0],
          [source_pixel_location_list[1][1], -source_pixel_location_list[1][0], 0, 1]])

# print(transformation_matrix)

target_location_column_vector = np.array([[target_pixel_location_list[0][0], target_pixel_location_list[0][1],
                                           target_pixel_location_list[1][0], target_pixel_location_list[1][1]]]).T


radian_to_degree = lambda radian: (radian*180)/np.pi

# print(target_location_column_vector)
variable_vector = np.matmul(np.linalg.inv(transformation_matrix), target_location_column_vector)


t_x = variable_vector[2,0]
t_y = variable_vector[3,0]

# print(variable_vector[0,0], variable_vector[1,0], t_x, t_y)


# print(cv2.findHomography(source_pixel_location_list, target_pixel_location_list))



radian_rotation =  np.arctan(variable_vector[1,0]/ variable_vector[0,0])


degree_rotation = np.round(radian_to_degree(radian_rotation))

scaling_factor = variable_vector[1,0]/np.sin(radian_rotation)

print(scaling_factor, degree_rotation, t_x, t_y)