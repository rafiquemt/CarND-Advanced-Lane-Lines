"""
   Project 4 - Advanced Lane Finding
   Tariq Rafique
"""
import glob
import os.path
import pickle

import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np

import cv2


"""
    The goals / steps of this project are the following:
    Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
    Apply a distortion correction to raw images.
    Use color transforms, gradients, etc., to create a thresholded binary image.
    Apply a perspective transform to rectify binary image ("birds-eye view").
    Detect lane pixels and fit to find the lane boundary.
    Determine the curvature of the lane and vehicle position with respect to center.
    Warp the detected lane boundaries back onto the original image.
    Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.
"""


def calibrate_camera(folderPath='camera_cal/calibration*.jpg'):
    """
    Return the calibration matrix to use for the calibrate_camera
    """
    objp = np.zeros((6 * 9, 3), np.float32)
    objp[:, :2] = np.mgrid[0:9, 0:6].T.reshape(-1, 2)

    # Arrays to store object points and image points from all the images.
    objpoints = []  # 3d points in real world space
    imgpoints = []  # 2d points in image plane.
    images = glob.glob(folderPath)

    for fname in images:
        img = cv2.imread(fname)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Find the chessboard corners
        ret, corners = cv2.findChessboardCorners(gray, (9, 6), None)

        # If found, add object points, image points
        if ret == True:
            objpoints.append(objp)
            imgpoints.append(corners)
        elif ret == False:
            print("****** FAILED to find chessboard in", fname)

    return cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)


def test_calibration(image, mtx, dist, folderPath='camera_cal'):
    img = cv2.imread(folderPath + '/' + image)
    undist = cv2.undistort(img, mtx, dist, None, mtx)
    cv2.imwrite('output_images/calibrated_' + image, undist)


def process_frame():
    pass


def main():
    pickle_file = 'cal_data/cal_data.p'
    print("Hello")
    if os.path.exists(pickle_file):
        print("loading pickle file for calibration", pickle_file)
        x = pickle.load(open(pickle_file, "rb"))
        mtx = x['mtx']
        dist = x['dist']
    else:
        x, mtx, dist, x, x = calibrate_camera()
        pickle.dump({'mtx': mtx, 'dist': dist}, open(pickle_file, 'wb'))
    test_calibration('calibration1.jpg', mtx, dist)
    test_calibration('calibration4.jpg', mtx, dist)
    test_calibration('calibration5.jpg', mtx, dist)


main()
