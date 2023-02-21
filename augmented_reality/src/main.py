import cv2
import numpy as np
import cv2.aruco as aruco
from camera_calibration import CameraCalibration
from augmented_reality import AugmentedReality

cam_calibration = CameraCalibration()


# obtain the 2d and 3d points of chess and calibrate the camera
# creating a npz file with all variables needed
objpoints, imgpoints = cam_calibration.obtain_points(False)
cam_calibration.calibrate_camera(objpoints, imgpoints)

augmented_reality = AugmentedReality()


video_cap = cv2.VideoCapture(0)

# create the buffer, to prevent always create
ret, frames = video_cap.read()
z_buffer = np.zeros((frames.shape[0], frames.shape[1], 1))


while True:
	# Capture frame-by-frame
	ret, frames = video_cap.read()

	dst = augmented_reality.undistort(frames)

	corners, ids, img_markers = augmented_reality.detect_markers(dst)

	if ids is not None:
		for i in range(len(ids)):

			#### DRAW 3D AXIS #####
			#augmented_reality.estimate_pose_marker(dst, corners, i)

			#### DRAW 2D IMAGE ON THE MARKER ####
			#top_left, top_right, bottom_left, bottom_right = augmented_reality.obtain_corners(dst, corners, i)
			#img_apply = augmented_reality.obtain_random_img(ids, i)
			#dst = augmented_reality.apply_img_to_world(dst, img_apply, ids, top_left, top_right, bottom_left, bottom_right, i)

			#### DRAW THE CUBE WITHOUT ANY IMAGE JUST THE SHAPE ####
			#dst = augmented_reality.draw_cube(dst, corners, i)

			### APPLY 3D OBJETS TO THE MARKER ####
			dst = augmented_reality.apply_3d_object(dst, corners, ids, i)

			### DRAW THE Z BUFFER POINTS ###
			#augmented_reality.draw_z_buffer_points(z_buffer, dst, corners, i)


	# Display the resulting frame
	cv2.imshow('Video', dst)
	cv2.imshow('Video2', img_markers)

	# press ESC to stop
	if cv2.waitKey(1) & 0xFF == ord('q'):
		break