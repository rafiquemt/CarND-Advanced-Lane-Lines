"""
   Project 4 - Advanced Lane Finding
   Tariq Rafique
"""
# pylint: disable=I0011,E1101,C0111,C0103,C0301,W0611
import glob
import os.path
import pickle

import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
from moviepy.editor import VideoFileClip

import cv2


def calibrate_camera(folder_path='camera_cal/calibration*.jpg'):
    """
    Return the calibration matrix to use for the calibrate_camera
    """
    objp = np.zeros((6 * 9, 3), np.float32)
    objp[:, :2] = np.mgrid[0:9, 0:6].T.reshape(-1, 2)

    # Arrays to store object points and image points from all the images.
    objpoints = []  # 3d points in real world space
    imgpoints = []  # 2d points in image plane.
    images = glob.glob(folder_path)

    for fname in images:
        img = cv2.imread(fname)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Find the chessboard corners
        ret, corners = cv2.findChessboardCorners(gray, (9, 6), None)

        # If found, add object points, image points
        if ret is True:
            objpoints.append(objp)
            imgpoints.append(corners)
        elif ret is False:
            print("****** FAILED to find chessboard in", fname)

    return cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)


def test_calibration(image, mtx, dist, folderPath='camera_cal'):
    img = cv2.imread(folderPath + '/' + image)
    undist = cv2.undistort(img, mtx, dist, None, mtx)
    cv2.imwrite('output_images/calibrated_' + image, undist)

def threshold_image(img):
    # color transform and binary combine with color threshold and sobel_x threshold
    return img

def perspective_transform(img, M):
    return img

def undistort_image(img, mtx, dist):
    return cv2.undistort(img, mtx, dist, None, mtx)

def get_unwarp_params(image_path):

    # coordinates of road in normal image
    # (184, 659) (581, 433) (684, 433) (1111, 659)

    # transferring to:
    # (150, 700) (150, 20) (1130, 20) (1130, 700)
    src = np.float32([[184, 659], [581, 433], [684, 433], [1111, 659]])
    # c) define 4 destination points dst = np.float32([[,],[,],[,],[,]])
    dst = np.float32([[150, 700], [150, 20], [1130, 20], [1130, 700]])
    # d) use cv2.getPerspectiveTransform() to get M, the transform matrix
    M = cv2.getPerspectiveTransform(src, dst)
    Minv = cv2.getPerspectiveTransform(dst, src)
    return M, Minv

def test_perspective():
    images = glob.glob("test_images/*.*")

def unwarp_image(img, M):
    return img

def process_frame(img, mtx, dist, M, Minv):
    undistorted_image = undistort_image(img, mtx, dist)
    threshold_image = threshold_image(undistorted_image)
    unwarped_image = unwarp_image(threshold_image)
    final = unwarped_image
    return final

def get_calibration_data():
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
    return mtx, dist

def main():
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
    mtx, dist = get_calibration_data()
    M, Minv = get_unwarp_params("test_images/straight_lines2.jpg")

    test_calibration('calibration1.jpg', mtx, dist)
    test_calibration('calibration5.jpg', mtx, dist)


    test_videos = ['project_video.mp4'] # 'challenge_video.mp4', 'harder_challenge_video.mp4'

    for vid_file in test_videos:
        clip = VideoFileClip(vid_file)
        output_clip = clip.fl_image(lambda img: process_frame(img, mtx, dist, M, Minv))
        output_clip.write_videofile('output_'+vid_file, audio=False, threads=4)

main()
