import cv2
import numpy as np


class EigenFaces():

	def __init__(self):

		self.__W = 0
		self.__mean_face = None


	def fit(self, images, m):

		# m must be always <= N -1
		if m >= images.shape[0]:
			raise ValueError('The value of m must always be <= N -1 !!!')

		width = images.shape[1]
		height = images.shape[2]

		dataset = np.reshape(images, (images.shape[0], width * height))

		# calculate the mean of all the images
		mean = self.calculate_mean(dataset, width, height)
		self.__mean_face = mean

		# 
		A = dataset - mean
		# change the lines with the columns
		A = A.T

		R = np.dot(A.T, A)

		# definition of m here
		m = m

		eig_values, V = np.linalg.eig(R)


		# obtains the indexes of the highest values of eigen with size of m
		indexes = np.argsort(eig_values)[::-1][:m]

		biggest_values = []
		for value in V:
			biggest_values.append([value[index] for index in indexes])

		V = np.array(biggest_values)

		W = np.dot(A, V)

		W = W /  np.linalg.norm(W, axis = 0)
			
		# used to check if the calculations is correct
		#print(np.dot(W.T, W))

		self.__W = W

		return W


	def obtain_weights(self, image):

		image = np.reshape(image, (image.shape[0] * image.shape[1]))
		
		x_u = image - self.__mean_face

		y = np.dot(self.__W.T, x_u)
		#print(y.shape)
		#print(y)

		return y



	def calculate_mean(self, dataset, width, height):

		mean_face = np.mean(dataset, axis = 0)

		face = np.reshape(mean_face, (width, height))
		#cv2.imwrite("database/mean.jpg", face)
		#cv2.imshow("s", face)
		#cv2.waitKey(0)
		#cv2.destroyAllWindows()
		return mean_face


	def reconstruction(self, dataset, img):

		X = np.reshape(dataset, (dataset.shape[0], dataset.shape[1] * dataset.shape[2]))

		W = self.fit(dataset, 5)

		y = self.obtain_weights(img)

		result = (np.dot(W, y) + self.calculate_mean(X, dataset.shape[1], dataset.shape[2])) / 255


		result2 = np.reshape(result, (dataset.shape[1], dataset.shape[2]))

		#cv2.imwrite("reconstructed.jpg", result2)
		cv2.imshow("reconstructed", result2)
		#cv2.waitKey(0)
		#cv2.destroyAllWindows()

