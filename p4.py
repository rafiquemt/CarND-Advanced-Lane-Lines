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
import p4sobel as sobel


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


def undistort_image(img, mtx, dist):
    return cv2.undistort(img, mtx, dist, None, mtx)


def get_warp_params():

    # coordinates of road in normal image
    src = np.float32([[184, 645], [583, 445], [697, 445], [1111, 645]])
    # transferring to:
    dst = np.float32([[150, 700], [150, 20], [1130, 20], [1130, 700]])
    M = cv2.getPerspectiveTransform(src, dst)
    Minv = cv2.getPerspectiveTransform(dst, src)
    # return inverse which is used later
    return M, Minv


def warp_image(img, M):
    warped = cv2.warpPerspective(
        img, M, (img.shape[1], img.shape[0]), flags=cv2.INTER_CUBIC)
    return warped


def test_perspective():
    mtx, dist = get_calibration_data()
    M, Minv = get_warp_params()
    images = glob.glob("test_images/*.*")
    for fname in images:
        img = cv2.imread(fname)
        undistorted = cv2.undistort(img, mtx, dist, None, mtx)
        warped = warp_image(undistorted, M)
        cv2.imwrite("output_images/unwarped_" + fname, warped)

def test_threshold():
    mtx, dist = get_calibration_data()
    M, Minv = get_warp_params()
    images = glob.glob("test_images/*.*")
    for fname in images:
        img = cv2.imread(fname)
        undistorted = undistort_image(img, mtx, dist)
        thresholded = color_sobel_threshold(undistorted)
        warped = warp_image(thresholded, M)
        cv2.imwrite("output_images/threshold_" + fname, thresholded)


def color_sobel_threshold(img, s_thresh=(110, 255), sx_thresh=(20, 100)):
    img = np.copy(img)
    # Convert to HSV color space and separate the V channel
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HLS).astype(np.float)
    l_channel = hsv[:, :, 1]
    s_channel = hsv[:, :, 2]
    # Sobel x
    sobelx = cv2.Sobel(l_channel, cv2.CV_64F, 1, 0)  # Take the derivative in x
    # Absolute x derivative to accentuate lines away from horizontal
    abs_sobelx = np.absolute(sobelx)
    scaled_sobel = np.uint8(255 * abs_sobelx / np.max(abs_sobelx))

    # Threshold x gradient
    sxbinary = np.zeros_like(scaled_sobel)
    sxbinary[(scaled_sobel >= sx_thresh[0]) &
             (scaled_sobel <= sx_thresh[1])] = 1

    # Threshold color channel
    s_binary = np.zeros_like(s_channel)
    s_binary[(s_channel >= s_thresh[0]) & (s_channel <= s_thresh[1])] = 1

    # combined = np.zeros_like(s_binary)
    # combined[sxbinary == 1 | s_binary == 1] = 1
    # Stack each channel
    # Note color_binary[:, :, 0] is all 0s, effectively an all black image. It might
    # be beneficial to replace this channel with something else.
    color_binary = np.dstack((sxbinary, sxbinary, s_binary))
    return color_binary * 255


def process_frame(img, mtx, dist, M, Minv):
    undistorted_image = undistort_image(img, mtx, dist)
    thresholded_image = color_sobel_threshold(undistorted_image)
    warped_image = warp_image(thresholded_image, M)
    final = warped_image
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
    M, Minv = get_warp_params()

    test_calibration('calibration1.jpg', mtx, dist)
    test_calibration('calibration5.jpg', mtx, dist)

    # 'challenge_video.mp4', 'harder_challenge_video.mp4'
    test_videos = ['project_video.mp4']

    for vid_file in test_videos:
        clip = VideoFileClip(vid_file)
        output_clip = clip.fl_image(
            lambda img: process_frame(img, mtx, dist, M, Minv))
        output_clip.write_videofile(
            'output_' + vid_file, audio=False, threads=4)



# main()
#test_perspective()
test_threshold()
