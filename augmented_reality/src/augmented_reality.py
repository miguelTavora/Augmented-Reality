import cv2
import numpy as np
import cv2.aruco as aruco
from utils import Utils
from z_buffer import ZBuffer


class AugmentedReality():

	def __init__(self):

		data = np.load("calibration.npz")

		# obtain the values from the camera calibration
		self.__mtx = data['mtx']
		self.__dist = data['dist']
		self.__rvecs = data['rvecs']
		self.__tvecs = data['tvecs']

		self.__aruco_dim = aruco.Dictionary_get(cv2.aruco.DICT_6X6_250)

		video_cap = cv2.VideoCapture(0)
		ret, img = video_cap.read()
		h, w = img.shape[:2]

		self.__camera_mtx, roi = cv2.getOptimalNewCameraMatrix(self.__mtx, self.__dist, (w,h), 1, (w,h))
		self.__x, self.__y, self.__w, self.__h = roi

		self.__parameters = aruco.DetectorParameters_create()

		self.__utils = Utils()
		self.__z_buffer = ZBuffer()


	def undistort(self, frames):

		# undistort
		dst = cv2.undistort(frames, self.__mtx, self.__dist, None, self.__camera_mtx)

		# crop the image
		dst = dst[self.__y:self.__y + self.__h, self.__x:self.__x+self.__w]

		return dst

	def detect_markers(self, dst):

		corners, ids, rejectedImgPoints = aruco.detectMarkers(dst, self.__aruco_dim, parameters=self.__parameters)
		frame_markers = aruco.drawDetectedMarkers(dst.copy(), corners, ids)

		return corners, ids, frame_markers

	def obtain_corners(self, dst, corners, i):

		top_left     = (corners[i][0][0][0], corners[i][0][0][1])
		top_right    = (corners[i][0][1][0], corners[i][0][1][1])
		bottom_right = (corners[i][0][2][0], corners[i][0][2][1])
		bottom_left  = (corners[i][0][3][0], corners[i][0][3][1])

		tl, tr, bl, br = self.__utils.correct_corners(top_left, top_right, bottom_left, bottom_right)
		#print(tl, tr, bl, br)

		# draw corners
		cv2.circle(dst, (int(tl[0]), int(tl[1])), 5, (55, 226, 213), -1)
		cv2.circle(dst, (int(tr[0]), int(tr[1])), 5, (87, 65, 47), -1)
		cv2.circle(dst, (int(bl[0]), int(bl[1])), 5, (251, 203, 10), -1)
		cv2.circle(dst, (int(br[0]), int(br[1])), 5, (199, 10, 128), -1)

		return tl, tr, bl, br

	def estimate_pose_marker(self, dst, corners, i):

		self.__rvecs, self.__tvecs, trash = aruco.estimatePoseSingleMarkers(corners[i],  0.1, self.__mtx, self.__dist)
		aruco.drawAxis(dst, self.__mtx, self.__dist, self.__rvecs[0], self.__tvecs[0], 0.05)  # Draw Axis


	def obtain_random_img(self, ids, i):

		# obtain different images for different ids, but always the same during the time 
		apply_img = self.__utils.get_id_image(ids[i][0])
		
		return apply_img

	def obtain_cube_img(self, ids, i, type_img):

		apply_img = self.__utils.get_cube_image(ids[i][0], type_img)
		
		return apply_img


	def apply_img_to_world(self, dst, apply_img, ids, top_left, top_right, bottom_left, bottom_right, i):

		h_apply, w_apply = apply_img.shape[:2]

		pts1 = np.array([top_left, top_right, bottom_right, bottom_left])
		pts2 = np.float32([[0,0], [w_apply,0], [w_apply, h_apply], [0, h_apply]])
			
		matrix = cv2.findHomography(pts2, pts1)[0]
		img_out = cv2.warpPerspective(apply_img, matrix, (dst.shape[1], dst.shape[0]))
		cv2.fillConvexPoly(dst, pts1.astype(int), (0, 0, 0))
		dst = dst + img_out

		return dst



	def calculate_projected_points(self, dst, corners, i):

		axis = np.float32([[-0.05,-0.05,0], [-0.05,-0.05,0.1], [0.05,-0.05,0], [0.05,-0.05,0.1],
						[-0.05, 0.05, 0], [-0.05, 0.05, 0.1], [0.05, 0.05, 0], [0.05, 0.05, 0.1],
						[0.05, -0.05, 0.1], [0.05, 0.05, 0.1], [-0.05, 0.05, 0.1], [-0.05, -0.05, 0.1]]).reshape(-1, 3)

		
		rvec, tvec, _ = aruco.estimatePoseSingleMarkers(corners[i],  0.1, self.__mtx, self.__dist)

		img_pts, jac = cv2.projectPoints(axis, rvec, tvec, self.__mtx, self.__dist)


		pt1 = img_pts[0][0].ravel().astype(int)
		pt2 = img_pts[1][0].ravel().astype(int)
		pt3 = img_pts[2][0].ravel().astype(int)
		pt4 = img_pts[3][0].ravel().astype(int)

		
		pt12 = img_pts[4][0].ravel().astype(int)
		pt22 = img_pts[5][0].ravel().astype(int)
		pt32 = img_pts[6][0].ravel().astype(int)
		pt42 = img_pts[7][0].ravel().astype(int)


		pt13 = img_pts[8][0].ravel().astype(int)
		pt23 = img_pts[9][0].ravel().astype(int)
		pt33 = img_pts[10][0].ravel().astype(int)
		pt43 = img_pts[11][0].ravel().astype(int)
		# already got all the points that we need so we used the one already obtained

		return pt1,pt2,pt3,pt4,pt12,pt22,pt32,pt42,pt13,pt23,pt33,pt43


	def apply_3d_object(self, dst, corners, ids, i):

		pt1,pt2,pt3,pt4,pt12,pt22,pt32,pt42,pt13,pt23,pt33,pt43 = self.calculate_projected_points(dst, corners, i)


		img_front = self.obtain_cube_img(ids, i, "front")
		img_back = self.obtain_cube_img(ids, i, "back")
		img_right = self.obtain_cube_img(ids, i, "right")
		img_left = self.obtain_cube_img(ids, i, "left")
		img_top = self.obtain_cube_img(ids, i, "top")
		
		
		# when it cant render the block because of the points
		try:
			# when this happens dont need to render the front face
			if pt3[1] < pt32[1]:
				# back image
				dst = self.apply_img_to_world(dst, img_back, ids, pt1, pt2, pt3, pt4, i)

			else:
				# front image
				dst = self.apply_img_to_world(dst, img_front, ids, pt22, pt42, pt12, pt32, i)
		except:
			print("erro no primeiro front ou back")


		try:
			# to not overlapse and show the cube the way it should
			if pt3[1] < pt1[1]:
				# right image
				dst = self.apply_img_to_world(dst, img_right, ids, pt23, pt13, pt32, pt3, i)
					
				# left image
				dst = self.apply_img_to_world(dst, img_left, ids, pt33, pt43, pt12, pt1, i)

			
			else: 
				# left image
				dst = self.apply_img_to_world(dst, img_left, ids, pt33, pt43, pt12, pt1, i)

				# right image
				dst = self.apply_img_to_world(dst, img_right, ids, pt23, pt13, pt32, pt3, i)

		except:
			print("erro left ou right")
		

		try:
			if pt3[1] < pt32[1]:
				# front image
				dst = self.apply_img_to_world(dst, img_front, ids, pt22, pt42, pt12, pt32, i)

			else:
				# back image
				dst = self.apply_img_to_world(dst, img_back, ids, pt1, pt2, pt3, pt4, i)

		except:
			print("erro segundo front ou back")

		# top image
		dst = self.apply_img_to_world(dst, img_top, ids, pt23, pt33, pt13, pt43, i)

		return dst

	def draw_cube(self, dst, corners, i):

		pt1,pt2,pt3,pt4,pt12,pt22,pt32,pt42,pt13,pt23,pt33,pt43 = self.calculate_projected_points(dst, corners, i)

		 # front lines
		dst = cv2.line(dst, pt1, pt3, color=(255, 0, 0))
		dst = cv2.line(dst, pt1, pt2, color=(0, 255, 0))
		dst = cv2.line(dst, pt2, pt4, color=(255, 0, 255))
		dst = cv2.line(dst, pt3, pt4, color=(0, 0, 255))

		# top lines
		dst = cv2.line(dst, pt13, pt23, color=(255, 0, 0))
		dst = cv2.line(dst, pt23, pt33, color=(0, 255, 0))
		dst = cv2.line(dst, pt33, pt43, color=(255, 0, 255))

		# back lines
		dst = cv2.line(dst, pt23, pt32, color=(255, 0, 0))
		dst = cv2.line(dst, pt12, pt33, color=(255, 0, 0))


		# draw the corners of the cube
		cv2.circle(dst, pt33, 3, (0,0,255), -1)
		cv2.circle(dst, pt12, 3, (0,0,255), -1)
		cv2.circle(dst, pt23, 3, (0,0,255), -1)
		cv2.circle(dst, pt43, 3, (0,0,255), -1)
		cv2.circle(dst, pt13, 3, (0,0,255), -1)
		cv2.circle(dst, pt3, 3, (0,0,255), -1)
		cv2.circle(dst, pt1, 3, (0,0,255), -1)
		cv2.circle(dst, pt32, 3, (0,0,255), -1)

		return dst


	def draw_z_buffer_points(self, z_buffer, img, corners, i):

		axis = np.float32([[0.05, -0.05, 0], [0.05, 0.05, 0], [-0.05, 0.05, 0], [-0.05, -0.05, 0]]).reshape(-1, 3)

		axis_up = np.float32([[0.05, -0.05, 0.1], [0.05, 0.05, 0.1], [-0.05, 0.05, 0.1], [-0.05, -0.05, 0.1]]).reshape(-1, 3)

		rvec, tvec, _ = aruco.estimatePoseSingleMarkers(corners[i],  0.1, self.__mtx, self.__dist)

		# only used to z buffer
		pt1_cam = self.__z_buffer.points_world_to_camera(axis[0], self.__rvecs, self.__tvecs)
		pt2_cam = self.__z_buffer.points_world_to_camera(axis[1], self.__rvecs, self.__tvecs)
		pt3_cam = self.__z_buffer.points_world_to_camera(axis[2], self.__rvecs, self.__tvecs)
		pt4_cam = self.__z_buffer.points_world_to_camera(axis[3], self.__rvecs, self.__tvecs)

		pt5_cam = self.__z_buffer.points_world_to_camera(axis_up[0], self.__rvecs, self.__tvecs)
		pt6_cam = self.__z_buffer.points_world_to_camera(axis_up[1], self.__rvecs, self.__tvecs)
		pt7_cam = self.__z_buffer.points_world_to_camera(axis_up[2], self.__rvecs, self.__tvecs)
		pt8_cam = self.__z_buffer.points_world_to_camera(axis_up[3], self.__rvecs, self.__tvecs)

		
		# points to build the cube
		img_pts, jac = cv2.projectPoints(axis, rvec, tvec, self.__mtx, self.__dist)
		pt1 = img_pts[0][0].ravel().astype(int)
		pt2 = img_pts[1][0].ravel().astype(int)
		pt3 = img_pts[2][0].ravel().astype(int)
		pt4 = img_pts[3][0].ravel().astype(int)

		
		img_pts, jac = cv2.projectPoints(axis_up, rvec, tvec, self.__mtx, self.__dist)
		pt12 = img_pts[0][0].ravel().astype(int)
		pt22 = img_pts[1][0].ravel().astype(int)
		pt32 = img_pts[2][0].ravel().astype(int)
		pt42 = img_pts[3][0].ravel().astype(int)


		# draw the points
		self.__z_buffer.draw_point_buffer(z_buffer, img, pt1, pt1_cam, (0,0,255))
		self.__z_buffer.draw_point_buffer(z_buffer, img, pt2, pt2_cam, (0,0,255))
		self.__z_buffer.draw_point_buffer(z_buffer, img, pt3, pt3_cam, (0,0,255))
		self.__z_buffer.draw_point_buffer(z_buffer, img, pt4, pt4_cam, (0,0,255))

		self.__z_buffer.draw_point_buffer(z_buffer, img, pt12, pt5_cam, (0,255,255))
		self.__z_buffer.draw_point_buffer(z_buffer, img, pt22, pt6_cam, (0,255,255))
		self.__z_buffer.draw_point_buffer(z_buffer, img, pt32, pt7_cam, (0,255,255))
		self.__z_buffer.draw_point_buffer(z_buffer, img, pt42, pt8_cam, (0,255,255))