import cv2
import numpy as np
from haar_cascade_classifier import HaarCascadeClassifier
import math 


class Normalizer():

	def normalize(self, img, faces):

		img_result = np.empty((0))

		for face in faces:
			(box_face, eyes_inside_face) = face

			# when detect eyes
			if eyes_inside_face != None:
				(x_left, y_left, x_right, y_right) = self.get_middle_of_eyes(eyes_inside_face)

				#self.__draw_line_eyes(img, x_left, y_left, x_right, y_right)

				# return the angle rotation of the face, orientation is True when rotation is to the left and False to the right
				angle_degres, orientation = self.calcule_rotation_face(x_left, y_left, x_right, y_right)

				img_rotated, rot_matrix = self.rotate_image_with_angle(img, angle_degres, orientation)

				rotated_left_eye = self.get_rotated_points(x_left, y_left, rot_matrix)
				rotated_right_eye = self.get_rotated_points(x_right, y_right, rot_matrix)
				#self.__draw_line_eyes(img_rotated, rotated_left_eye[0], rotated_left_eye[1], rotated_right_eye[0], rotated_right_eye[1])
				#cv2.imshow('Rotation', img_rotated)
				
				
				scale = self.obtain_scale(rotated_left_eye, rotated_right_eye)
				rescaled_img = self.resize_image(img_rotated, scale)
				#cv2.imshow('Scaled', rescaled_img)

				
				selected_img, img_exists = self.obtain_image_around_face(rescaled_img, rotated_left_eye, rotated_right_eye, scale)
				# it may go outside the image, so it need the if
				if img_exists: 
					#cv2.imshow('Selected', selected_img)
					img_result = selected_img

		return img_result
	# gets the middle point of the eyes to check the inclination of the line between eyes
	def get_middle_of_eyes(self, left_right_eye):

		# left eye
		(ex1, ey1, ew1, eh1) = left_right_eye[0]
		#right eye
		(ex2, ey2, ew2, eh2) = left_right_eye[1]

		middle_point_left_x = ex1+ int(ew1/2)
		middle_point_left_y = ey1+ int(eh1/2)

		middle_point_right_x = ex2+ int(ew2/2)
		middle_point_right_y = ey2+ int(eh2/2) 

		return (middle_point_left_x, middle_point_left_y, middle_point_right_x, middle_point_right_y)

	def calcule_rotation_face(self, x_left, y_left, x_right, y_right):

		dist = self.__calculate_distance_between_points(x_left, y_left, x_right, y_right)

		cat_opposite = abs(x_left - x_right)
		# sin(x) = cateto opposite / hypotenuse
		sin_x = cat_opposite / dist

		# angle = sin^-1 (sin(x))
		angle_degress = math.degrees(math.asin(sin_x))
		return angle_degress, y_left > y_right


	def __calculate_distance_between_points(self, x_left, y_left, x_right, y_right):

		# c = square root ( (xA - xB)**2 + (yA - yB)**2 )

		a2 = (x_left - x_right)**2
		b2 = (y_left - y_right)**2

		dist = math.sqrt((a2 + b2))
		return dist

	def rotate_image_with_angle(self, img, angle, orientation):

		half_width = int(img.shape[0]/2)
		half_height = int(img.shape[1]/2)
		center = (half_width, half_height)

		# new angle where True is to the left and False to the right
		new_angle = angle - 90 if orientation else 90 - angle

		# definition of the rotation matrix
		rot_matrix = cv2.getRotationMatrix2D(center, new_angle, 1.)

		# rotates the image with a certain angle, it subtracts 90 because is when
		# its perfetly aligned
		img_rotated = cv2.warpAffine(img, rot_matrix, (img.shape[1], img.shape[0]))
		#cv2.imshow('Rotation', img_rotated)

		return img_rotated, rot_matrix

	# given a point and a rotation matrix, returns the point on the new 
	def get_rotated_points(self, pos_x, pos_y, rot_matrix):

		return np.matmul(rot_matrix, np.array([pos_x, pos_y, 1])).astype(int)


	def obtain_scale(self, rotated_left, rotated_right):

		dist = self.__calculate_distance_between_points(rotated_left[0], rotated_left[1], rotated_right[0], rotated_right[1])

		# number said it is 31 - 16
		const_d = 16
		scale = const_d / dist

		return scale

	def resize_image(self, img, scale):

		width = int(img.shape[1] * scale)
		height = int(img.shape[0] * scale)

		img_resized = cv2.resize(img, (width, height), interpolation=cv2.INTER_AREA)

		# return image resized
		return img_resized

	def obtain_image_around_face(self, img, eye_left, eye_right, scale):

		scaled_left = eye_left * scale
		scaled_right = eye_right * scale
		#print(self.__calculate_distance_between_points(scaled_left[0], scaled_left[1], scaled_right[0], scaled_right[1]))

		# to put the only the face inside the rectangle
		start_face_x = int(scaled_left[0]) -16
		start_face_y = int(scaled_left[1]) -24
		end_face_x = start_face_x + 46
		end_face_y = start_face_y + 56



		img = img[start_face_y : end_face_y, start_face_x : end_face_x]

		# when the face goes out of the window
		img_exists = True
		if img.shape[0] < 56 or img.shape[1] < 46: 
			img_exists = False

		# converts to hray
		else:
			img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

		return img, img_exists



	# method draw the line between eyes
	def __draw_line_eyes(self, img, x_left, y_left, x_right, y_right):

		return cv2.line(img, (x_left, y_left),(x_right, y_right),(0,0,0), 3)
