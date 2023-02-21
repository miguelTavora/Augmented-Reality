import cv2
import numpy as np


class VirtualObjecs():

	def __init__(self):

		img1 =  cv2.imread("objects/hair.jpg", 1)
		img2 =  cv2.imread("objects/hat2.jpg", 1)
		img3 =  cv2.imread("objects/hat.jpg", 1)

		self.__object = [img1, img3, img2]



	def normalize_object(self, box_face):

		x = int((box_face[3] - box_face[1])/ 1.5)

		# center the hair on the middle
		index_width = [box_face[1] - x + 30, box_face[1] + 30]

		index_height = [box_face[0] - 20, box_face[2] + 20]

		if index_width[0] > -1 and index_width[1] > -1 and index_height[0] > -1 and index_height[1] > -1:

			return (index_width, index_height)

		return None



	def show_object(self,img_complete, box_face, label):

		norm_obj = self.normalize_object(box_face)

		if norm_obj != None:

			index_width = norm_obj[0]
			index_height = norm_obj[1]

			img_head = img_complete[index_width[0] :index_width[1], index_height[0] : index_height[1]]
			#cv2.imshow("xs", img_head)
			
			result = self.apply_mask(img_head, self.__object[label[0]])
			img_complete[index_width[0] :index_width[1], index_height[0] : index_height[1]] = result
			cv2.imshow('Objects', img_complete)


	def apply_mask(self, img, img_obj):


		img_rescaled = cv2.resize(img_obj, (img.shape[1], img.shape[0]), interpolation = cv2.INTER_AREA)

		# obtain the mask of the image
		# first convert it to gray and apply threshold to the whites go to 0
		# in this case pixels with value higher than 250
		img_gray = cv2.cvtColor(img_rescaled, cv2.COLOR_BGR2GRAY)
		ret, mask = cv2.threshold(img_gray, 250, 255, cv2.THRESH_BINARY_INV)
		#cv2.imshow('mask',mask)

		# obtain the inverse of the mask
		mask_inv = cv2.bitwise_not(mask)
		#cv2.imshow('inv_mask',mask_inv)

		# obtain the part of the content from the image we want to add
		img2_clear = cv2.bitwise_and(img_rescaled, img_rescaled, mask = mask)
		#cv2.imshow('img2_clear', img2_clear)
		

		# apply the inverse mask to the image we want to add the object
		# black out the part where we will put the image
		img1_black = cv2.bitwise_and(img, img, mask = mask_inv)
		#cv2.imshow('img1_black',img1_black)
		

		# put the object on the real image
		result = cv2.add(img1_black, img2_clear)
		#cv2.imshow('img1_bla2ck', result)

		return result