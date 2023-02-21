import cv2
import numpy as np
import glob


class CameraCalibration():

    def __init__(self):

        # Defining the dimensions of chessboard
        self.__CHECKERBOARD = (7, 7)
        # termination criteria
        self.__criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

        # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
        self.__objp = np.zeros((1, self.__CHECKERBOARD[0] * self.__CHECKERBOARD[1], 3), np.float32)
        self.__objp[0,:,:2] = np.mgrid[0:self.__CHECKERBOARD[0], 0:self.__CHECKERBOARD[1]].T.reshape(-1, 2)

        # Extracting path of individual image stored in a given directory
        self.__imgs_names = glob.glob('./calibration/*.jpg')
        self.__gray = None

    def obtain_points(self, show_imgs = True):

        # Creating vector to store vectors of 3D points for each checkerboard image
        objpoints = []
        # Creating vector to store vectors of 2D points for each checkerboard image
        imgpoints = []

        for fname in self.__imgs_names:
            img = cv2.imread(fname)
            self.__gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

            # Find the chess board corners
            # If desired number of corners are found in the image then ret = true
            ret, corners = cv2.findChessboardCorners(self.__gray, self.__CHECKERBOARD, cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_FAST_CHECK + cv2.CALIB_CB_NORMALIZE_IMAGE)
     
            
            # If desired number of corner are detected,
            # we refine the pixel coordinates and display
            # them on the images of checker board

            if ret == True:
                objpoints.append(self.__objp)
                # refining pixel coordinates for given 2d points.
                corners2 = cv2.cornerSubPix(self.__gray, corners, (11,11),(-1,-1), self.__criteria)
                imgpoints.append(corners2)
 
                # Draw and display the corners
                img = cv2.drawChessboardCorners(img, self.__CHECKERBOARD, corners2, ret)
            
            if show_imgs:
                cv2.imshow('img',img)
                cv2.waitKey(0)
            

        return objpoints, imgpoints


    def calibrate_camera(self, objpoints, imgpoints):

        # Performing camera calibration by
        # passing the value of known 3D points (objpoints)
        # and corresponding pixel coordinates of the
        # detected corners (imgpoints)

        ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, self.__gray.shape[::-1], None, None)
 
        """print("Camera matrix : \n", mtx)
        print("dist : \n", dist)
        print("rvecs : \n", rvecs)
        print("tvecs : \n", tvecs)"""

        np.savez('calibration.npz', mtx=mtx, dist=dist, rvecs=rvecs, tvecs=tvecs)
