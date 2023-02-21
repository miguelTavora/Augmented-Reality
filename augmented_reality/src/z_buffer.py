import cv2
import numpy as np
import cv2.aruco as aruco

class ZBuffer():

    def points_world_to_camera(self, point, rvecs, tvecs):

        
        point_rotation = cv2.Rodrigues(rvecs[0].reshape(3, 1))[0].reshape(3, 3)
        # tvecs is the position of the point
        point_position = tvecs[0].reshape(3, 1)
        

        matrix_transformation = np.concatenate((point_rotation, point_position), axis = 1)
        matrix_transformation = np.vstack((matrix_transformation, [0,0,0,1])).reshape(4, 4)

        point = np.array([point[0], point[1], point[2], 1]).reshape(4, 1)

        camera_point = np.dot(matrix_transformation, point)

        return camera_point

    def draw_point_buffer(self, buffer, img, point, point_cam, color):

        # if its inside the image
        if point[0] >= 0 and point[0] < buffer.shape[0] and point[1] >= 0 and point[1] < buffer.shape[1]:
            if buffer[point[0], point[1]] == 0 or buffer[point[0], point[1]] < point_cam[2]:
                cv2.circle(img, point, 3, color, -1)