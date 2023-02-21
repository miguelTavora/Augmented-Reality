import cv2
import numpy as np
from eigen_faces import EigenFaces
from fisher_faces import FisherFaces
from sklearn.neighbors import KNeighborsClassifier

class Classifier():

	def __init__(self):

		# its used two classifiers to prevent fit both eigen and fisher
		# and to try both results at the same time
		self.__neigh_eigen = KNeighborsClassifier(n_neighbors=3)
		self.__neigh_fisher = KNeighborsClassifier(n_neighbors=3)
		self.__fisher = None
		self.__eigen = None

	def obtain_dataset(self):

		images = []
		y = []

		for i in range(23):
			# 0 at the end is to convert to gray-scale
			img = cv2.imread("database/dataset1/face"+str(i)+".jpg", 0)

			images.append(img)
			y.append(0)

		for j in range(1, 10):
			img = cv2.imread("database/dataset2/pedroMjorge"+str(j)+"Norm.png", 0)
			images.append(img)
			y.append(1)

		for k in range(1, 14):
			img = cv2.imread("database/dataset3/img"+str(k)+".jpg", 0)
			images.append(img)
			y.append(2)

		return np.array(images), y
		

	def fit_fisher(self, X, y, m = 5):

		fisher_faces = FisherFaces(X, y)
		fisher_faces.fit(m)
		self.__fisher = fisher_faces

		vectors = []
		for img in X:
			vectors.append(fisher_faces.obtain_weights(img))

		vectors = np.array(vectors)

		### gives the weights and labels to the classifier 
		self.__neigh_fisher.fit(vectors, y)


	def predict_fisher(self, image):

		vector = self.__fisher.obtain_weights(image)
		vector = vector[np.newaxis, :]

		label = self.__neigh_fisher.predict(vector)

		return label

	def fit_eigen(self, X, y, m = 5):

		eigen_faces = EigenFaces()
		eigen_faces.fit(X, m)
		self.__eigen = eigen_faces

		vectors = []
		for img in X:
			vectors.append(eigen_faces.obtain_weights(img))

		vectors = np.array(vectors)

		self.__neigh_eigen.fit(vectors, y)

	def predict_eigen(self, image):

		vector = self.__eigen.obtain_weights(image)
		vector = vector[np.newaxis, :]

		label = self.__neigh_eigen.predict(vector)

		return label