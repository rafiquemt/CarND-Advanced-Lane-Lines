"""
   Project 4 - Advanced Lane Finding
   Tariq Rafique
"""
# pylint: disable=I0011,E1101,C0111,C0103,C0301,W0611,R0914,R0902,R0903,R0913,E0401
import glob
import os.path
import pickle

import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
from moviepy.editor import VideoFileClip

import cv2


class Line():
    def __init__(self):
        # was the line detected in the last iteration?
        self.detected = False
        # x values of the last n fits of the line
        self.recent_xfitted = []
        # average x values of the fitted line over the last n iterations
        self.bestx = None
        # polynomial coefficients averaged over the last n iterations
        self.best_fit = None
        # polynomial coefficients for the most recent fit
        self.current_fit = [np.array([False])]
        # radius of curvature of the line in some units
        self.radius_of_curvature = []
        # distance in meters of vehicle center from the line
        self.line_base_pos = []
        # difference in fit coefficients between last and new fits
        self.diffs = np.array([0, 0, 0], dtype='float')
        # x values for detected line pixels
        self.allx = None
        # y values for detected line pixels
        self.ally = None
        # history of all fits
        self.all_fits = []

    def add_fit(self, new_fit):
        self.current_fit = new_fit
        self.all_fits.append(new_fit)
        self.best_fit = np.mean(self.all_fits[-5:], axis=0)
        #self.best_fit =

    def get_curvature(self):
        return np.mean(self.radius_of_curvature[-5:])


def sliding_window_histogram_new(binary_warped, left, right):
    # Choose the number of sliding windows
    nwindows = 9
    # Set height of windows
    window_height = np.int(binary_warped.shape[0] / nwindows)
    # Set the width of the windows +/- margin
    margin = 50
    # Set minimum number of pixels found to recenter window
    minpix = 25
    # Create empty lists to receive left and right lane pixel indices
    left_lane_inds = []
    right_lane_inds = []

    # Identify the x and y positions of all nonzero pixels in the image
    nonzero = binary_warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])

    # Create an output image to draw on and  visualize the result
    out_img = np.dstack((binary_warped, binary_warped, binary_warped)) * 255

    if should_restart_line_search(left, right):
        # Assuming you have created a warped binary image called "binary_warped"
        # Take a histogram of the bottom half of the image
        histogram = np.sum(
            binary_warped[3*binary_warped.shape[0] / 4:, :], axis=0)
        # Find the peak of the left and right halves of the histogram
        # These will be the starting point for the left and right lines
        midpoint = np.int(histogram.shape[0] / 2)
        leftx_base = np.argmax(histogram[:midpoint])
        rightx_base = np.argmax(histogram[midpoint:]) + midpoint

        # Current positions to be updated for each window
        leftx_current = leftx_base
        rightx_current = rightx_base

        # Step through the windows one by one
        for window in range(nwindows):
            # Identify window boundaries in x and y (and right and left)
            win_y_low = binary_warped.shape[0] - (window + 1) * window_height
            win_y_high = binary_warped.shape[0] - window * window_height
            win_xleft_low = leftx_current - margin
            win_xleft_high = leftx_current + margin
            win_xright_low = rightx_current - margin
            win_xright_high = rightx_current + margin
            # Draw the windows on the visualization image
            cv2.rectangle(out_img, (win_xleft_low, win_y_low),
                          (win_xleft_high, win_y_high), (0, 255, 0), 2)
            cv2.rectangle(out_img, (win_xright_low, win_y_low),
                          (win_xright_high, win_y_high), (0, 255, 0), 2)
            # Identify the nonzero pixels in x and y within the window
            good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) &
                              (nonzerox >= win_xleft_low) & (nonzerox < win_xleft_high)).nonzero()[0]
            good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) &
                               (nonzerox >= win_xright_low) & (nonzerox < win_xright_high)).nonzero()[0]
            # Append these indices to the lists
            left_lane_inds.append(good_left_inds)
            right_lane_inds.append(good_right_inds)
            # If you found > minpix pixels, recenter next window on their mean
            # position
            if len(good_left_inds) > minpix:
                leftx_current = np.int(np.mean(nonzerox[good_left_inds]))
            if len(good_right_inds) > minpix:
                rightx_current = np.int(np.mean(nonzerox[good_right_inds]))

        # Concatenate the arrays of indices
        left_lane_inds = np.concatenate(left_lane_inds)
        right_lane_inds = np.concatenate(right_lane_inds)
    else:
        left_fit = left.best_fit
        right_fit = right.best_fit
        left_lane_inds = (
            (nonzerox > (left_fit[0] * (nonzeroy**2) + left_fit[1] * nonzeroy + left_fit[2] - margin)) &
            (nonzerox < (left_fit[0] * (nonzeroy**2) + left_fit[1] * nonzeroy + left_fit[2] + margin)))
        right_lane_inds = (
            (nonzerox > (right_fit[0] * (nonzeroy**2) + right_fit[1] * nonzeroy + right_fit[2] - margin)) &
            (nonzerox < (right_fit[0] * (nonzeroy**2) + right_fit[1] * nonzeroy + right_fit[2] + margin)))

    # Extract left and right line pixel positions
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds]
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]

    # Fit a second order polynomial to each
    left_fit = np.polyfit(lefty, leftx, 2)
    right_fit = np.polyfit(righty, rightx, 2)
    left.add_fit(left_fit)
    right.add_fit(right_fit)

    left.allx = leftx
    left.ally = lefty
    right.allx = rightx
    right.ally = righty

    left.detected = True
    right.detected = True

    # Generate x and y values for plotting
    ploty = np.linspace(0, binary_warped.shape[0] - 1, binary_warped.shape[0])
    left_fitx = left_fit[0] * ploty**2 + left_fit[1] * ploty + left_fit[2]
    right_fitx = right_fit[0] * ploty**2 + right_fit[1] * ploty + right_fit[2]

    out_img[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [255, 0, 0]
    out_img[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [0, 0, 255]

    result = out_img

    return result, left_fitx, right_fitx, ploty

def should_restart_line_search(left, right):
    return left.detected is False and right.detected is False


def draw_lines(raw_image, warped, Minv, left_fitx, right_fitx, ploty):
    # Create an image to draw the lines on
    warp_zero = np.zeros_like(warped).astype(np.uint8)
    color_warp = np.dstack((warp_zero, warp_zero, warp_zero))

    # Recast the x and y points into usable format for cv2.fillPoly()
    pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
    pts_right = np.array(
        [np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
    pts = np.hstack((pts_left, pts_right))
    # Draw the lane onto the warped blank image
    cv2.fillPoly(color_warp, np.int_([pts]), (0, 255, 0))

    # Warp the blank back to original image space using inverse perspective
    # matrix (Minv)
    newwarp = cv2.warpPerspective(
        color_warp, Minv, (raw_image.shape[1], raw_image.shape[0]))
    # Combine the result with the original image
    result = cv2.addWeighted(raw_image, 1, newwarp, 0.3, 0)
    return result


def draw_radius_of_curvature_center_location(img, left, right, ploty):
    # Define y-value where we want radius of curvature
    # I'll choose the maximum y-value, corresponding to the bottom of the image
    y_eval = np.max(ploty)
    leftx = left.allx
    lefty = left.ally

    rightx = right.allx
    righty = right.ally

    # Define conversions in x and y from pixels space to meters
    ym_per_pix = 46 / 720  # meters per pixel in y dimension
    xm_per_pix = 3.7 / 711  # meters per pixel in x dimension

    # Fit new polynomials to x,y in world space
    left_fit_cr = np.polyfit(lefty * ym_per_pix, leftx * xm_per_pix, 2)
    right_fit_cr = np.polyfit(righty * ym_per_pix, rightx * xm_per_pix, 2)
    # Calculate the new radii of curvature
    left_curverad = ((1 + (2 * left_fit_cr[0] * y_eval * ym_per_pix +
                           left_fit_cr[1])**2)**1.5) / np.absolute(2 * left_fit_cr[0])
    right_curverad = ((1 + (2 * right_fit_cr[0] * y_eval * ym_per_pix +
                            right_fit_cr[1])**2)**1.5) / np.absolute(2 * right_fit_cr[0])

    left.radius_of_curvature.append(left_curverad)
    right.radius_of_curvature.append(right_curverad)

    curvature_average = (left.get_curvature() + right.get_curvature()) / 2

    cv2.putText(img, "Estimated curvature: {:6.2f}m".format(curvature_average),
                (200, 50), cv2.FONT_HERSHEY_PLAIN, fontScale=2.5, thickness=3, color=(0, 0, 0))
    x_left = left.best_fit[0]*y_eval**2 + left.best_fit[1]*y_eval + left.best_fit[2]
    x_right = right.best_fit[0]*y_eval**2 + right.best_fit[1]*y_eval + right.best_fit[2]
    mid_lane = x_left + ((x_right - x_left)/2)
    camera_offset_from_mid = ((img.shape[1] / 2) - mid_lane) * xm_per_pix

    cv2.putText(img, "Offset {:6.2f}m".format(camera_offset_from_mid),
                (400, 100), cv2.FONT_HERSHEY_PLAIN, fontScale=2.5, thickness=3, color=(0, 0, 0))

    return img


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
    src = np.float32([[289, 661], [595, 450], [684, 450], [1017, 661]])
    # transferring to:
    dst = np.float32([[250, 700], [250, 20], [950, 20], [950, 700]])
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
    M = get_warp_params()
    images = glob.glob("test_images/*.*")
    for fname in images:
        img = cv2.imread(fname)
        undistorted = undistort_image(img, mtx, dist)
        thresholded = color_sobel_threshold(undistorted) * 255
        warped = warp_image(thresholded, M)
        cv2.imwrite("output_images/threshold_" + fname, warped)


def color_sobel_threshold(img, s_thresh=(130, 175), sx_thresh=(20, 150), colored=False):
    img = np.copy(img)
    # Convert to HSV color space and separate the V channel
    hls = cv2.cvtColor(img, cv2.COLOR_BGR2HLS).astype(np.float)
    l_channel = hls[:, :, 1]
    s_channel = hls[:, :, 2]
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

    combined = np.zeros_like(s_binary)
    combined[(s_binary == 1) | (sxbinary == 1)] = 1
    # Stack each channel
    # Note color_binary[:, :, 0] is all 0s, effectively an all black image. It might
    # be beneficial to replace this channel with something else.
    color_binary = np.dstack((sxbinary, sxbinary, s_binary))
    if colored is True:
        return color_binary, combined
    else:
        return combined


def process_frame(img, mtx, dist, M, Minv, left, right):
    undistorted = undistort_image(img, mtx, dist)
    thresholded = color_sobel_threshold(undistorted)
    warped = warp_image(thresholded, M)
    warped_with_lines, left_fitx, right_fitx, ploty = sliding_window_histogram_new(
        warped, left, right)
    img_with_lines = draw_lines(
        undistorted, warped, Minv, left_fitx, right_fitx, ploty)
    img_annotated = draw_radius_of_curvature_center_location(img_with_lines, left, right, ploty)
    final = img_annotated
    return final


def test_full_pipeline():
    mtx, dist = get_calibration_data()
    M, Minv = get_warp_params()
    images = glob.glob("test_images/*.*")

    for fname in images:
        left = Line()
        right = Line()
        img = cv2.imread(fname)
        undistorted = undistort_image(img, mtx, dist)
        thresholded, thresholded_combined = color_sobel_threshold(undistorted, colored=True)
        warped = warp_image(thresholded_combined, M)
        warped_with_lines, left_fitx, right_fitx, ploty = sliding_window_histogram_new(
            warped, left, right)
        img_with_lines = draw_lines(
            undistorted, warped, Minv, left_fitx, right_fitx, ploty)
        cv2.imwrite("output_images/lines_" + fname, warped_with_lines)
        cv2.imwrite("output_images/threshold_" + fname, thresholded * 255)
        cv2.imwrite("output_images/final_" + fname, img_with_lines)
        cv2.imwrite("output_images/undistorted_" + fname, undistorted)


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

    # test_calibration('calibration1.jpg', mtx, dist)
    # test_calibration('calibration5.jpg', mtx, dist)

    # 'challenge_video.mp4', 'harder_challenge_video.mp4'
    test_videos = ['project_video.mp4']
    #test_videos = ['smaller_first_bridge.mp4']
    left = Line()
    right = Line()
    for vid_file in test_videos:
        clip = VideoFileClip(vid_file)
        output_clip = clip.fl_image(
            lambda img: process_frame(img, mtx, dist, M, Minv, left, right))
        output_clip.write_videofile(
            'output_' + vid_file, audio=False, threads=4)


# test_perspective()
# test_threshold()
# test_full_pipeline()

main()
