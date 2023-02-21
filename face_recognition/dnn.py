import cv2
import numpy as np
from haar_cascade_classifier import HaarCascadeClassifier


class ConvolutionNeuralNetwork():

    def __init__(self, confidence = 0.5):
        self.__confidence = confidence

        # https://raw.githubusercontent.com/opencv/opencv/master/samples/dnn/face_detector/deploy.prototxt
        prototxt_path = "weights/deploy.prototxt.txt"
        # https://raw.githubusercontent.com/opencv/opencv_3rdparty/dnn_samples_face_detector_20180205_fp16/res10_300x300_ssd_iter_140000_fp16.caffemodel 
        model_path = "weights/res10_300x300_ssd_iter_140000_fp16.caffemodel"

        # load Caffe model
        self.__model = cv2.dnn.readNetFromCaffe(prototxt_path, model_path)
        # size of the text
        self.__font_scale = 1.0
        # classifier used for the eyes
        self.__haar_classifier = HaarCascadeClassifier()


    def image_inference(self, image):
        # preprocess the image: resize and performs mean subtraction
        blob = cv2.dnn.blobFromImage(image, 1.0, (300, 300), (104.0, 177.0, 123.0))

        # set the image into the input of the neural network
        self.__model.setInput(blob)

        # perform inference and get the result
        output = np.squeeze(self.__model.forward())

        return output

    def classify(self, image, output):

        # get width and height of the image
        h, w = image.shape[:2]

        face_and_eyes = []

        for i in range(0, output.shape[0]):
            # get the confidence
            confidence = output[i, 2]

            # if confidence is above the value specified in the construtor (default: 50%), then draw the surrounding box
            if confidence > self.__confidence:
                # get the surrounding box cordinates and upscale them to original image
                # and convert to integers
                box = (output[i, 3:7] * np.array([w, h, w, h])).astype(np.int64)

                # detects the eyes, only if it is inside the face
                eyes_inside_face = self.__haar_classifier.detect_eyes_inside_face(image, box)

                face_and_eyes.append((box, eyes_inside_face))    
                # draw the rectangle surrounding the face
                #cv2.rectangle(image, (box[0], box[1]), (box[2], box[3]), color=(255, 0, 255), thickness=2)
                # draw text as well
                #cv2.putText(image, f"{confidence*100:.2f}%", (box[0], box[1]-5), cv2.FONT_HERSHEY_SIMPLEX, self.__font_scale, (0, 255, 0), 1)



        return image, face_and_eyes