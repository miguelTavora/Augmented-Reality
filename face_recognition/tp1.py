import cv2
import numpy as np
from haar_cascade_classifier import HaarCascadeClassifier
from dnn import ConvolutionNeuralNetwork
from normalizer import Normalizer
from classifier import Classifier
from eigen_faces import EigenFaces
from virtual_objects import VirtualObjecs


# obtains the image from the camera
video_cap = cv2.VideoCapture(0)

haarClassifier = HaarCascadeClassifier()

cnn = ConvolutionNeuralNetwork()
# only with 80% of confidence show the recognition
#cnn = ConvolutionNeuralNetwork(confidence = 0.8)

normalizer = Normalizer()


classifier = Classifier()
# obtain the previous stored values
X, y = classifier.obtain_dataset()

# reconstruction of the image
eigen_faces = EigenFaces()
#eigen_faces.reconstruction(X, X[0])


# set the weights for the eigen faces
classifier.fit_eigen(X, y)

# set the weights for the fisher faces
classifier.fit_fisher(X, y)

virtual_objects = VirtualObjecs()

while True:
	# Capture frame-by-frame
	ret, frames = video_cap.read()

	### FOR CASCADE ####
	# obtains the image after classification
	#img = haarClassifier.detect_faces(frames)
	#img = haarClassifier.detect_eyes(img)
	
	#####  FOR CNN  ####
	#out_inf = cnn.image_inference(frames)
	#face_detection = cnn.classify(frames, out_inf)


	### CASCADE AND CNN ###
	out_inf = cnn.image_inference(frames)
	# uses dnn and cascade to detect eyes and get left and right eye
	img, face_eyes_detect = cnn.classify(frames, out_inf)
	# normalize the things
	norm = normalizer.normalize(img, face_eyes_detect)

	# shape (0, ) is when is not detected a image
	if norm.shape[0] != 0:
		# predict with label is the face
		label = classifier.predict_fisher(norm)
		print("fisher label:",label)
		
		#label2 = classifier.predict_eigen(norm)
		#print("eigen label:", label2)

		virtual_objects.show_object(frames, face_eyes_detect[0][0], label)
		


	# Display the resulting frame
	cv2.imshow('Video', frames)


	# press ESC to stop
	if cv2.waitKey(1) & 0xFF == ord('q'):
		break


video_cap.release()
cv2.destroyAllWindows()