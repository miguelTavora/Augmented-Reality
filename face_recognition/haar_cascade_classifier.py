import cv2
import os

class HaarCascadeClassifier():

	def __init__(self):

		# classifier for face and eye
		self.__face_cascade = cv2.CascadeClassifier(os.path.dirname(cv2.__file__)+"/data/haarcascade_frontalface_default.xml")
		self.__eye_cascade  = cv2.CascadeClassifier(os.path.dirname(cv2.__file__)+"/data/haarcascade_eye.xml")


		self.__scale_factor  = 1.1
		self.__min_neighbors = 2
		self.__min_size      = (30, 30)
		self.__flags         = cv2.CASCADE_SCALE_IMAGE
		self.__font = cv2.FONT_HERSHEY_SIMPLEX
		
		
	def detect_faces(self, frame):
		# the face detection only works with gray images
		gray_img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)


		# scaleFactor  -> Parameter specifying how much the image size is reduced at each image scale
		# minNeighbors -> Parameter specifying how many neighbors each candidate rectangle should have to retain it
		# minSize      -> Minimum possible object size. Objects smaller than that are ignored.
		# flags        -> Normaly its used CASCADE_SCALE_IMAGE to face detection
		faces = self.__face_cascade.detectMultiScale(
			gray_img,
			scaleFactor  = self.__scale_factor,
			minNeighbors = self.__min_neighbors,
			minSize      = self.__min_size,
			flags        = self.__flags
		)

		
		# draw a rectangle around the faces
		for (x, y, w, h) in faces:
			cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
			cv2.putText(frame,'Face',(x, y-5), self.__font, 1.5, (255,0,255), 2)



		return frame

	def detect_eyes(self, frame):

		# detect eyes on the image
		eyes = self.__eye_cascade.detectMultiScale(frame)

		# draw the retangle on the eyes
		for (ex,ey,ew,eh) in eyes:
			cv2.rectangle(frame,(ex,ey),(ex+ew,ey+eh),(0,0,255), 2)
			cv2.putText(frame,'Eye',(ex, ey-5), 1, 1, (255, 0, 0), 1)

		return frame

	def detect_eyes_inside_face(self, frame, box):

		# detect eyes on the image
		eyes = self.__eye_cascade.detectMultiScale(frame)
		# stores all the faces detected
		eyes_inside_face = []

		#print("tamaho lista olhos: "+str(len(eyes)))
		# draw the retangle on the eyes
		for (ex,ey,ew,eh) in eyes:
			# check if the eyes is inside the face 
			if ex >= box[0] and ey >= box[1] and ex+ew <= box[2] and ey+eh <= box[3]:
				#cv2.rectangle(frame,(ex,ey),(ex+ew,ey+eh),(0,0,255), 2)
				# get all the eyes detected to return
				eyes_inside_face.append((ex, ey, ew, eh))

		# get the correct eyes
		correct_eyes = self.choose_correct_eyes(eyes_inside_face)

		# it may not detect any eyes
		if correct_eyes != None:
			# draw the rectangle on the eyes
			#self.__draw_rect_eyes(frame, correct_eyes)

			# gets a list where position 0 is the left and 1 is the right eye
			left_right_eye = self.get_left_and_right_eye(correct_eyes)

			# draw a purple for left and red for the right eye
			#self.__draw_rect_left_right_eye(frame, left_right_eye)

			return left_right_eye

		return None


	def choose_correct_eyes(self, eyes_inside_face):

		# only detect the detect at least 2 eyes
		if eyes_inside_face != None and len(eyes_inside_face) > 1:
			# always returns the first 2 rectangles to detect the eyes
			# this is beacause it can detect more than two eyes inside the face
			# normally it is the first two rectangles that is the correct ones
			return eyes_inside_face[0:2]

	def get_left_and_right_eye(self, correct_eyes):

		(ex1,ey1,ew1,eh1) =  correct_eyes[0]
		(ex2,ey2,ew2,eh2) =  correct_eyes[1]

		# detects if the value of x of the first eye is bigger than the second
		# if it is means that the position of 0 of correct_eyes is the left and 1 is the right
		# otherwise means that the left is the index 1 and 0 the right
		left_right_eyes = correct_eyes if ex2 > ex1 else [correct_eyes[1], correct_eyes[0]]

		return left_right_eyes


	# method to debug and check if the eyes is correct
	def __draw_rect_eyes(self, frame, correct_eyes):

		#print(len(correct_eyes))
		for (ex,ey,ew,eh) in correct_eyes:
			cv2.rectangle(frame,(ex,ey),(ex+ew,ey+eh),(0,0,255), 2)

	# method to debug and check if the eyes is correct
	def __draw_rect_left_right_eye(self, frame, left_right_eye):

		# when its the left eye is purple, when is the right is red
		for index in range(len(left_right_eye)):
			color = (170, 51, 106) if index == 0 else (0,0,255)
			text =  "left" if index == 0 else "right"
			(ex,ey,ew,eh) = left_right_eye[index]
			cv2.rectangle(frame,(ex,ey),(ex+ew,ey+eh),color, 2)
			#cv2.putText(frame, text,(ex, ey-5), 1, 1, (255, 0, 0), 1)