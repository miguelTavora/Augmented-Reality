import cv2
import numpy as np
import glob


class Utils():

    def __init__(self):

        # loads the names of the images to be applyied
        self.__file_names = glob.glob('apply/*.jpg')

        # 2d images
        # key -> id, value -> img
        self.__id_img = {}
        # index to know which image is currently being used
        self.__index_img = 0

        # 3d objects
        # key -> id, value -> directory
        self.__id_cube = {}
        # index to know which image is currently being used
        self.__index_cube = 0
        self.__number_diffent_cubes = 3


    # used to get 2d images
    def get_id_image(self, id):


        if id in self.__id_img:
            return self.__id_img.get(id)

        else:
            img_selected = cv2.imread(self.__file_names[self.__index_img])
            self.__id_img[id] = img_selected
            self.__index_img = (self.__index_img + 1)%len(self.__file_names)
            return img_selected

    def get_cube_image(self, id, type_img):

        if id not in self.__id_cube:
            result = "apply/cube"+str(self.__index_cube)
            self.__id_cube[id] = result
            self.__index_cube = (self.__index_cube + 1)%self.__number_diffent_cubes
            
        return cv2.imread(self.__id_cube.get(id)+"/"+type_img+".png")

    def correct_corners(self, top_left, top_right, bottom_left, bottom_right):

        list_all = [top_left, top_right, bottom_left, bottom_right]

        list_corners_x = [top_left[0], top_right[0], bottom_left[0], bottom_right[0]]
        list_corners_y = [top_left[1], top_right[1], bottom_left[1], bottom_right[1]]


        # obtain the lowest, and second lowest index of the corner in the x value
        lowests_x =  sorted(list_corners_x, key = float)
        min_index_x = list_corners_x.index(lowests_x[0])
        min_index_x_2 = list_corners_x.index(lowests_x[1])

        # when the value is equal gets the same index, to prevent this gets the next value
        if lowests_x[0] == lowests_x[1]:
            min_index_x, min_index_x_2 = self.obtain_next_index_same_value(lowests_x[0], list_corners_x)

        max_index_x = list_corners_x.index(lowests_x[-1])
        max_index_x_2 = list_corners_x.index(lowests_x[-2])

        if lowests_x[-1] == lowests_x[-2]:
            max_index_x, max_index_x_2 = self.obtain_next_index_same_value(lowests_x[-1], list_corners_x)


        real_top_left = None
        real_bottom_left = None
        # based on the y value the lowest will be the top and highest the lowest
        if list_corners_y[min_index_x] < list_corners_y[min_index_x_2]:
            real_top_left = list_all[min_index_x]
            real_bottom_left = list_all[min_index_x_2]
        else:
            real_top_left = list_all[min_index_x_2]
            real_bottom_left = list_all[min_index_x]


        real_top_right = None
        real_bottom_right = None
        if list_corners_y[max_index_x] < list_corners_y[max_index_x_2]:
            real_top_right = list_all[max_index_x]
            real_bottom_right = list_all[max_index_x_2]

        else:
            real_top_right = list_all[max_index_x_2]
            real_bottom_right = list_all[max_index_x]

        return real_top_left, real_top_right, real_bottom_left, real_bottom_right


    def obtain_next_index_same_value(self, value, list_values):

        indexes = []

        for i in range(len(list_values)):
            if list_values[i] == value:
                indexes.append(i)

        return indexes[0], indexes[1]
