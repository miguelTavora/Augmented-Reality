import cv2
import numpy as np
from eigen_faces import EigenFaces


class FisherFaces():

	# the X is the images as grayscale and numpy arrays
	# and y the true labels
	def __init__(self, X, y):

		self.__labels = np.unique(y)
		self.__y = y
		
		self.__width = X.shape[1]
		self.__height = X.shape[2]

		# convert the dataset to one line
		self.__X = np.reshape(X, (X.shape[0], self.__width * self.__height))

		self.__mean_face = None
		self.__W = None


	def fit(self, m = 5):

		all_mean = np.mean(self.__X, axis = 0)
		self.__mean_face = all_mean
		#img1 = np.reshape(all_mean.astype(np.uint8), (self.__width, self.__height))
		#cv2.imshow("all", img1)


		# obtains the means for all the true labels
		mean_labels = [np.mean(self.__X[self.__y == i], axis=0) for i in self.__labels]
		#cv2.imshow("mean1", np.reshape(mean_labels[0].astype(np.uint8), (self.__width, self.__height)))
		#cv2.imshow("mean2", np.reshape(mean_labels[1].astype(np.uint8), (self.__width, self.__height)))
		#cv2.waitKey(0)
		#cv2.destroyAllWindows()



		Sw = self.__calculate_sw(mean_labels)

		Sb = self.__calculate_sb(all_mean, mean_labels)

		Wpca = EigenFaces().fit( (np.reshape(self.__X, (self.__X.shape[0], self.__width, self.__height))), m)

		
		Sw2 = np.dot(np.dot(Wpca.T, Sw), Wpca)
		Sb2 = np.dot(np.dot(Wpca.T, Sb), Wpca)

		c_1 = np.dot(np.linalg.inv(Sw2), Sb2)


		eig_values, V = np.linalg.eig(c_1)


		# obtains the indexes of the highest values of eigen with size of m
		indexes = np.argsort(eig_values)[::-1][:m]

		biggest_values = []
		for value in V:
			biggest_values.append([value[index] for index in indexes])

		V = np.array(biggest_values)

		W = np.dot(Wpca, V)
		self.__W = W


	def obtain_weights(self, image):

		image = np.reshape(image, (image.shape[0] * image.shape[1]))


		x_u = image - self.__mean_face

		# it return imaginary values so it must convert to real
		y = np.real(np.dot(self.__W.T, x_u))
		#print(y.shape)
		#print(y)

		return y


	def __calculate_sw(self, means):

		Sw = np.zeros_like(self.__X.shape[0])


		for index in range(len(self.__X)):

			# E (x - ui) (x - ui).T # somatório da subtração de cada imagem menos a média da sua label multiplicado pela sua transposta 
			sub = np.reshape(self.__X[index], (self.__width * self.__height, 1)) - np.reshape(means[self.__y[index]], (self.__width * self.__height, 1))

			mult = np.dot(sub, sub.T)

			Sw = Sw + mult

		return Sw


	def __calculate_sb(self, all_mean, mean_labels):

		Sb = None

		for i in range(len(mean_labels)):
			# number of samples of each class
			ni = np.sum(self.__y == self.__labels[i])

			# E ni (ui - u) (ui - u)T # somatório do número de exemplos por classe vezes resultado das médias por class menos a média total multiplicado pela sua transposta
			mean_sub = np.reshape(mean_labels[i], (self.__width * self.__height, 1)) - np.reshape(all_mean, (self.__width * self.__height, 1))# .astype(np.uint8)

			sum_values = ni * np.dot(mean_sub, mean_sub.T)

			# first time it's zero, so it takes the first mean value
			Sb = Sb + sum_values if i != 0 else sum_values

		return Sb

